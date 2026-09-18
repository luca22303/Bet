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
