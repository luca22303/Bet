"""Point-in-time integrity.

The most important tests in the project. If the store ever hands a model a fact
that did not exist yet, every downstream number is fiction — and it is fiction
that looks like success, which is why it has to be caught here rather than
noticed later.
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from bet.config import SETTINGS


def test_results_are_invisible_before_kickoff(populated_store):
    store = populated_store
    row = store.con.execute(
        "SELECT match_id, kickoff_utc FROM match ORDER BY kickoff_utc LIMIT 1"
    ).fetchone()
    match_id, kickoff = row

    before = store.matches_as_of(kickoff - timedelta(hours=1))
    assert match_id not in set(before["match_id"])

    after = store.matches_as_of(kickoff + timedelta(hours=3))
    assert match_id in set(after["match_id"])


def test_training_set_grows_monotonically_through_time(populated_store):
    store = populated_store
    counts = [
        len(store.matches_as_of(datetime(year, 6, 1)))
        for year in (2020, 2021, 2022, 2023, 2024)
    ]
    assert counts == sorted(counts)
    assert counts[0] < counts[-1]


def test_closing_odds_are_not_visible_to_a_pre_match_model(populated_store):
    """Closing prices exist for scoring, but a model asking `as_of` must not see them."""
    store = populated_store
    kickoff = store.con.execute("SELECT MIN(kickoff_utc) FROM match").fetchone()[0]

    visible = store.odds_as_of(kickoff - timedelta(hours=1))
    assert visible.empty

    assert not store.closing_odds().empty


def test_leakage_report_is_clean_on_correctly_built_data(populated_store):
    assert int(populated_store.leakage_report()["violations"].sum()) == 0


def test_leakage_report_catches_an_injected_leak(populated_store):
    """Deliberately corrupt the store and confirm the guard fires.

    A check that has never been seen to fail is not a check.
    """
    store = populated_store
    match_id, kickoff = store.con.execute(
        "SELECT match_id, kickoff_utc FROM match LIMIT 1").fetchone()

    store.con.execute(
        "UPDATE match_result SET known_at = ? WHERE match_id = ?",
        [kickoff - timedelta(days=1), match_id],
    )

    report = store.leakage_report()
    violations = int(report.loc[report["table"] == "match_result", "violations"].iloc[0])
    assert violations == 1


def test_upsert_is_idempotent(store):
    frame = pd.DataFrame([{
        "match_id": "x", "source": "t", "league": "bundesliga", "season": "2024-25",
        "kickoff_utc": datetime(2024, 8, 24, 15, 30),
        "home_team_id": "bayern_munich", "away_team_id": "sc_freiburg",
        "known_at": datetime(2024, 7, 1),
    }])
    for _ in range(3):
        store.upsert("match", frame, ["match_id"])
    assert store.con.execute("SELECT COUNT(*) FROM match").fetchone()[0] == 1


def test_upsert_refuses_rows_without_known_at(store):
    frame = pd.DataFrame([{
        "match_id": "y", "source": "t", "league": "bundesliga", "season": "2024-25",
        "kickoff_utc": datetime(2024, 8, 24, 15, 30),
        "home_team_id": "bayern_munich", "away_team_id": "sc_freiburg",
        "known_at": None,
    }])
    with pytest.raises(ValueError, match="known_at"):
        store.upsert("match", frame, ["match_id"])


def test_odds_as_of_returns_the_latest_price_not_all_of_them(store):
    kickoff = datetime(2024, 8, 24, 15, 30)
    store.upsert("match", pd.DataFrame([{
        "match_id": "m1", "source": "t", "league": "bundesliga", "season": "2024-25",
        "kickoff_utc": kickoff, "home_team_id": "bayern_munich",
        "away_team_id": "sc_freiburg", "known_at": kickoff - timedelta(days=30),
    }]), ["match_id"])

    quotes = pd.DataFrame([
        {"match_id": "m1", "book": "b", "market": "1x2", "selection": "H",
         "decimal_odds": price, "quoted_at": kickoff - timedelta(days=days),
         "is_closing": False, "known_at": kickoff - timedelta(days=days)}
        for price, days in ((1.80, 3), (1.75, 2), (1.70, 1))
    ])
    store.upsert("odds_quote", quotes, ["match_id", "book", "market", "selection", "quoted_at"])

    latest = store.odds_as_of(kickoff - timedelta(hours=12))
    assert len(latest) == 1
    assert latest["decimal_odds"].iloc[0] == pytest.approx(1.70)

    earlier = store.odds_as_of(kickoff - timedelta(days=2, hours=12))
    assert earlier["decimal_odds"].iloc[0] == pytest.approx(1.80)


# ------------------------------------------------------------------ upsert

def _match_rows(ids, **overrides):
    import pandas as pd

    base = {
        "source": "football_data", "league": "bundesliga", "season": "2026-27",
        "kickoff_utc": pd.Timestamp("2026-08-28 19:30:00"),
        "home_team_id": "a", "away_team_id": "b",
        "known_at": pd.Timestamp("2026-07-01"),
    }
    base.update(overrides)
    return pd.DataFrame([dict(base, match_id=i) for i in ids])


def test_upsert_replaces_rows_that_collide_on_the_key(store):
    store.init_schema()
    store.upsert("match", _match_rows(["m1", "m2"]), ["match_id"])
    store.upsert("match", _match_rows(["m2", "m3"], home_team_id="z"), ["match_id"])

    rows = store.con.execute(
        "SELECT match_id, home_team_id FROM match ORDER BY match_id").fetchall()
    assert rows == [("m1", "a"), ("m2", "z"), ("m3", "z")]


def test_two_sources_writing_the_same_matches_does_not_break_the_index(store):
    """The refresh that invalidated the database did exactly this.

    football-data.co.uk writes the played fixtures, then OpenLigaDB writes the
    whole season over the top. Driving the primary key's index with a bulk
    DELETE for that overlap is what failed with "Failed to delete all rows
    from index. Only deleted 0 out of 36 rows".
    """
    store.init_schema()
    played = [f"bundesliga:2026-27:t{i}:t{i + 1}" for i in range(36)]
    season = [f"bundesliga:2026-27:t{i}:t{i + 1}" for i in range(306)]

    for _ in range(5):                       # several refreshes in one session
        store.upsert("match", _match_rows(played, source="football_data"), ["match_id"])
        store.upsert("match", _match_rows(season, source="openligadb"), ["match_id"])

    assert store.con.execute("SELECT count(*) FROM match").fetchone()[0] == 306
    assert store.con.execute(
        "SELECT count(DISTINCT match_id) FROM match").fetchone()[0] == 306


def test_upsert_keeps_columns_the_incoming_frame_omits(store):
    """A partial scrape must not erase what another source established."""
    store.init_schema()
    store.upsert("match", _match_rows(["m1"], home_team_id="hertha"), ["match_id"])

    import pandas as pd
    partial = pd.DataFrame([{"match_id": "m1", "source": "openligadb",
                             "known_at": pd.Timestamp("2026-07-02")}])
    store.upsert("match", partial, ["match_id"])

    row = store.con.execute(
        "SELECT source, home_team_id FROM match WHERE match_id = 'm1'").fetchone()
    assert row == ("openligadb", "hertha")


def test_upsert_collapses_duplicate_keys_within_one_batch(store):
    """Two rows for one key in a single batch would otherwise raise."""
    store.init_schema()
    frame = _match_rows(["m1", "m1"])
    frame.loc[1, "home_team_id"] = "last-one-wins"
    written = store.upsert("match", frame, ["match_id"])

    assert written == 1
    assert store.con.execute(
        "SELECT home_team_id FROM match").fetchone() == ("last-one-wins",)


def test_upsert_rejects_a_key_column_the_frame_does_not_have(store):
    import pytest

    store.init_schema()
    with pytest.raises(ValueError, match="not in the frame"):
        store.upsert("match", _match_rows(["m1"]), ["match_id", "nonexistent"])


def test_a_failed_upsert_leaves_the_existing_rows_alone(store):
    """DELETE-then-INSERT lost the rows outright when the INSERT failed."""
    import pandas as pd
    import pytest

    store.init_schema()
    store.upsert("match", _match_rows(["m1", "m2"]), ["match_id"])

    broken = _match_rows(["m1", "m2"])
    broken["kickoff_utc"] = "not a timestamp at all"
    with pytest.raises(Exception):
        store.upsert("match", broken, ["match_id"])

    assert store.con.execute("SELECT count(*) FROM match").fetchone()[0] == 2


def test_upserting_on_part_of_the_key_is_refused_with_a_clear_message(store):
    """`match_result` is keyed by source so two scrapes can disagree.

    Upserting it on match_id alone would mean "replace every source's result
    for this match", quietly undoing that. DuckDB reports it as a binder
    error about conflict targets, which says nothing about the actual mistake.
    """
    import pandas as pd
    import pytest

    store.init_schema()
    frame = pd.DataFrame([{
        "match_id": "m1", "source": "football_data", "home_goals": 2,
        "away_goals": 1, "outcome": "H", "ht_home": None, "ht_away": None,
        "known_at": pd.Timestamp("2026-08-28"),
    }])
    with pytest.raises(ValueError, match="not a unique constraint"):
        store.upsert("match_result", frame, ["match_id"])
