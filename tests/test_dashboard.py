"""Dashboard rendering and the chart primitives.

The visual checks that matter here are the ones a markup grep misses: that a
formation places each player exactly once, and that a scoreline is withheld when
the model cannot actually call it.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from bet.dashboard import build, most_likely_score, top_scorelines
from bet.models.dixon_coles import score_matrix
from bet.viz import (
    Series,
    assign_to_formation,
    diverging_bar,
    hbar_chart,
    line_chart,
    parse_formation_rows,
    pitch,
    scatter,
    stat_tile,
)


# ----------------------------------------------------------------- formation


@pytest.mark.parametrize("formation,rows", [
    ("4-2-3-1", [4, 2, 3, 1]), ("4-3-3", [4, 3, 3]), ("3-5-2", [3, 5, 2]),
])
def test_formation_parses_to_its_lines(formation, rows):
    """The literal lines, not the group totals.

    4-2-3-1 and 4-3-3 field the same five midfielders in visibly different
    shapes, and the shape is the point of drawing a pitch at all.
    """
    assert parse_formation_rows(formation) == rows


def test_unparseable_formation_falls_back():
    assert parse_formation_rows("nonsense") == [4, 2, 3, 1]
    assert parse_formation_rows("9-9-9") == [4, 2, 3, 1]


def _squad(defenders=4, midfielders=5, forwards=1):
    players = [{"name": "Keeper", "group": "goalkeeper", "short": "GK"}]
    for group, count in (("defender", defenders), ("midfielder", midfielders),
                         ("forward", forwards)):
        players += [{"name": f"{group[:3].title()}{i}", "group": group,
                     "short": group[:2].upper()} for i in range(count)]
    return players


def test_every_player_is_placed_exactly_once():
    """The bug a screenshot caught and the markup did not.

    The obvious implementation keeps per-group pools plus a shared spare pool,
    but those hold the same objects: taking a midfielder from his group leaves
    him in the spares, and he appears twice on the pitch.
    """
    placed = assign_to_formation("4-2-3-1", _squad())
    names = [p["name"] for p in placed]
    assert len(placed) == 11
    assert len(set(names)) == 11


def test_a_thin_squad_still_produces_a_pitch():
    """An incomplete line-up is more useful than an empty one."""
    placed = assign_to_formation("4-2-3-1", _squad(defenders=2, midfielders=1, forwards=0))
    names = [p["name"] for p in placed]
    assert len(set(names)) == len(names)
    assert len(placed) == 4


def test_a_squad_short_of_forwards_borrows_from_midfield():
    placed = assign_to_formation("4-3-3", _squad(defenders=4, midfielders=6, forwards=0))
    assert len(placed) == 11
    assert len({p["name"] for p in placed}) == 11


def test_goalkeeper_sits_behind_every_other_line():
    placed = assign_to_formation("4-4-2", _squad(midfielders=4, forwards=2))
    keeper = next(p for p in placed if p["group"] == "goalkeeper")
    assert keeper["row"] == -1
    assert all(p["row"] >= 0 for p in placed if p is not keeper)


def test_pitch_renders_one_marker_per_player():
    svg = pitch("4-3-3", _squad(midfielders=3, forwards=3), team_name="Test")
    assert svg.count('class="player') == 11
    assert "<svg" in svg and "</svg>" in svg


# -------------------------------------------------------------- score calling


def test_a_clear_favourite_gets_a_scoreline():
    result = most_likely_score(score_matrix(2.4, 0.6, -0.13))
    assert result["confident"]
    assert result["score"] == "2-0"


def test_an_unpredictable_match_is_left_blank():
    """Blank is the honest output.

    In a low-scoring even match 0-0 and 1-1 sit within a point of each other;
    printing whichever leads would dress a coin-flip as a prediction.
    """
    result = most_likely_score(score_matrix(0.9, 0.95, -0.13))
    assert not result["confident"]
    assert result["score"] is None
    assert result["reason"]


def test_a_flat_distribution_is_rejected_on_probability():
    flat = np.full((6, 6), 1 / 36)
    result = most_likely_score(flat)
    assert result["score"] is None
    assert "too flat" in result["reason"]


def test_scorelines_are_ranked_and_sum_below_one():
    lines = top_scorelines(score_matrix(1.6, 1.1, -0.13), count=6)
    probabilities = [p for _, p in lines]
    assert probabilities == sorted(probabilities, reverse=True)
    assert 0 < sum(probabilities) < 1


# ------------------------------------------------------------------- charts


def test_diverging_bar_labels_only_readable_shares():
    """A segment too small to hold a legible label is left bare, not clipped."""
    html = diverging_bar({"H": 0.90, "D": 0.07, "A": 0.03},
                         {"H": "Home", "D": "Draw", "A": "Away"})
    assert "90%" in html
    assert "3%" not in html
    assert html.count("data-tip") == 3


def test_charts_degrade_on_empty_input():
    assert "not enough data" in line_chart([])
    assert "no results" in hbar_chart([])
    assert "no data" in scatter([])


def test_line_chart_direct_labels_each_series():
    """Identity never rests on colour alone."""
    svg = line_chart([Series("xG for", [(0, 1.0), (1, 1.4)], 1),
                      Series("xG against", [(0, 0.9), (1, 1.1)], 2)])
    assert "xG for" in svg and "xG against" in svg
    assert 'class="line s1"' in svg and 'class="line s2"' in svg


def test_hbar_marks_the_benchmark_with_a_word_not_a_hue():
    svg = hbar_chart([("market", 0.195), ("dixon_coles", 0.199)], highlight="market")
    assert "benchmark" in svg
    # One series, one colour: shading each bar by its own value would burn the
    # only free channel restating the length the bar already shows.
    assert svg.count('class="bar"') == 1
    assert svg.count('class="bar benchmark"') == 1


def test_stat_tile_pairs_status_with_an_icon_and_word():
    """A status colour never carries meaning alone."""
    html = stat_tile("3", "errors", status="critical")
    assert "critical" in html
    assert "&#10007;" in html


# ---------------------------------------------------------------- full page


def test_page_builds_and_is_self_contained(store_with_players):
    page = build(store_with_players, datetime(2024, 1, 5), days=10)
    assert page.startswith("<!DOCTYPE html>")
    # No network: everything inline, so it opens from disk.
    assert "<script src=" not in page
    assert "<link rel=\"stylesheet\"" not in page
    assert "http://" not in page.replace("http://www.w3.org", "")


def test_page_has_every_view_and_clickable_fixtures(store_with_players):
    page = build(store_with_players, datetime(2024, 1, 5), days=10)
    for view in ("overview", "fixtures", "model", "props"):
        assert f'id="{view}"' in page
    assert 'role="button"' in page
    assert 'class="pitch"' in page


def test_page_is_light_mode_throughout(store_with_players):
    """No dark scopes and no toggle: one appearance, everywhere."""
    page = build(store_with_players, datetime(2024, 1, 5), days=10)
    assert "prefers-color-scheme" not in page
    assert "data-theme" not in page
    assert "themer" not in page


# ---------------------------------------------------------------- provenance


def test_synthetic_data_is_declared_loudly(store_with_players):
    """A dashboard built from fixtures looks exactly like one built from real
    results. That resemblance is how a demo gets read as a forecast, so the
    page says so before anything else."""
    page = build(store_with_players, datetime(2024, 1, 5), days=10)
    assert "Synthetic data" in page
    assert "not a real forecast" in page
    assert "provenance synthetic" in page


def test_real_sources_are_not_flagged_as_synthetic(store):
    """A store from genuine ingests must not carry the warning."""
    kickoff = datetime(2024, 3, 1, 15, 30)
    as_of = kickoff + timedelta(days=1)
    store.upsert("match", pd.DataFrame([{
        "match_id": "m1", "source": "football_data", "league": "bundesliga",
        "season": "2023-24", "kickoff_utc": kickoff,
        "home_team_id": "bayern_munich", "away_team_id": "sc_freiburg",
        "known_at": kickoff - timedelta(days=30)}]), ["match_id"])
    store.upsert("match_result", pd.DataFrame([{
        "match_id": "m1", "source": "football_data", "home_goals": 2,
        "away_goals": 1, "outcome": "H", "ht_home": None, "ht_away": None,
        "known_at": kickoff + timedelta(hours=2)}]), ["match_id", "source"])

    from bet.dashboard import data_provenance
    provenance = data_provenance(store, as_of)
    assert not provenance["synthetic"]
    assert provenance["sources"] == ["football_data"]


def test_provenance_respects_the_point_in_time_cut(store_with_players):
    """Ages are measured against what was knowable, like every other read.

    Without the filter a store holding future fixtures reports a negative age,
    which is both wrong and obviously wrong.
    """
    from bet.dashboard import data_provenance
    provenance = data_provenance(store_with_players, datetime(2024, 1, 5))
    ages = [f["age_days"] for f in provenance["freshness"] if f["age_days"] is not None]
    assert ages
    assert all(age >= 0 for age in ages)


def test_stale_data_is_flagged(store_with_players):
    """Catches an ingest that stopped working weeks ago."""
    from bet.dashboard import data_provenance
    provenance = data_provenance(store_with_players, datetime(2026, 1, 1))
    assert provenance["stale"]
    assert provenance["worst_age_days"] > 100


def test_an_empty_store_says_so(store):
    from bet.dashboard import data_provenance
    assert data_provenance(store, datetime(2024, 1, 1))["empty"]


def test_player_table_shows_days_since_last_start(store_with_players):
    """A per-90 rate carries no date.

    Without this column a striker who stopped playing in September sits in the
    table looking identical to one who played on Saturday.
    """
    page = build(store_with_players, datetime(2024, 1, 5), days=10)
    assert ">last<" in page
    assert "days since" in page


def test_model_health_says_so_when_no_backtest_has_run(store_with_players):
    page = build(store_with_players, datetime(2024, 1, 5), days=10)
    assert "backtest-from" in page


def test_empty_store_still_renders(store):
    page = build(store, datetime(2024, 1, 1), days=7)
    assert "<!DOCTYPE html>" in page
    assert "No fixtures" in page or "no fixtures" in page.lower()


def _thin_store(store, *, played):
    """A store refreshed moments ago that still holds almost no history."""
    import pandas as pd
    from datetime import timedelta

    store.init_schema()
    now = datetime(2026, 9, 22)
    matches, results = [], []
    for i in range(played):
        day = now - timedelta(days=played - i)
        matches.append({"match_id": f"t{i}", "source": "fd", "league": "bundesliga",
                        "season": "2026-27", "kickoff_utc": day,
                        "home_team_id": "bayern_munich", "away_team_id": "rb_leipzig",
                        "known_at": day - timedelta(days=30)})
        results.append({"match_id": f"t{i}", "source": "fd", "home_goals": 2,
                        "away_goals": 1, "outcome": "H", "ht_home": 1, "ht_away": 0,
                        "known_at": day + timedelta(hours=2)})
    store.upsert("match", pd.DataFrame(matches), ["match_id"])
    store.upsert("match_result", pd.DataFrame(results), ["match_id", "source"])
    return now


def test_a_store_with_one_season_says_why_nothing_is_priced(store):
    """Fixtures with no expectations and no explanation looks broken.

    A refresh fetched only the season in progress, so the page rendered with
    every expectation blank and nothing saying that a backfill was missing.
    """
    from bet.dashboard import build, data_provenance

    now = _thin_store(store, played=36)
    provenance = data_provenance(store, now)
    assert provenance["thin"] is True
    assert provenance["empty"] is False
    assert provenance["played_matches"] == 36

    page = build(store, now, days=8, league="bundesliga")
    assert "Not enough history to price anything" in page
    assert "36 played match(es)" in page
    assert "2015-2026" in page          # the command that fixes it


def test_a_stocked_store_shows_no_such_warning(store):
    from bet.dashboard import build, data_provenance

    now = _thin_store(store, played=700)
    assert data_provenance(store, now)["thin"] is False
    assert "Not enough history" not in build(store, now, days=8, league="bundesliga")


def test_an_empty_store_points_at_the_button_not_only_a_command(store):
    """'Run bet ingest --source all' was not a command this project has."""
    from bet.dashboard import build

    store.init_schema()
    page = build(store, datetime(2026, 9, 22), days=8, league="bundesliga")
    assert "Refresh data" in page
    assert "--source all" not in page


# --------------------------------------------------------- matchday scoping

def test_the_page_shows_a_previous_matchday_panel(store_with_players):
    """Predicted-vs-actual for the last completed round, collapsed by default."""
    page = build(store_with_players, datetime(2024, 1, 5), days=8)
    assert '<details class="history">' in page
    assert "Previous matchday" in page
    # Collapsed: no `open` attribute, so it does not show by default.
    assert '<details class="history" open' not in page
    assert "full time" in page
    assert "correct" in page or "missed" in page


def test_the_next_matchday_fixtures_share_one_kickoff(store_with_players):
    """Matchday-scoped, not a day window: every card in the tab is one round."""
    from bet.recommend import next_matchday_brief

    brief = next_matchday_brief(store_with_players, datetime(2024, 1, 5))
    assert brief.matches
    assert len({m.kickoff for m in brief.matches}) == 1


def test_next_matchday_does_not_go_blank_in_a_schedule_gap(store):
    """The reported bug: a fixed 8-day window landed in a gap and showed
    nothing, even though the season was fully loaded."""
    import numpy as np

    from conftest import generate_season

    store.init_schema()
    rng = np.random.default_rng(3)
    matches, results, quotes = generate_season(
        datetime(2022, 8, 10, 15, 30), "2022-23", rng)
    store.upsert("match", matches, ["match_id"])
    store.upsert("match_result", results, ["match_id", "source"])
    store.upsert("odds_quote", quotes,
                 ["match_id", "book", "market", "selection", "quoted_at"])

    as_of = datetime(2024, 9, 20)
    far_next = datetime(2024, 10, 4)          # a two-week gap, an 8-day
                                              # window would show nothing
    store.upsert("match", pd.DataFrame([{
        "match_id": "future1", "source": "t", "league": "bundesliga",
        "season": "2024-25", "kickoff_utc": far_next,
        "home_team_id": "bayern_munich", "away_team_id": "sc_freiburg",
        "known_at": as_of - timedelta(days=5),
    }]), ["match_id"])

    page = build(store, as_of, days=8, include_quality=False)
    assert "No fixtures in the window" not in page
    assert "bayern" in page.lower() and "freiburg" in page.lower()


def test_no_history_panel_when_nothing_has_been_played(store):
    store.init_schema()
    page = build(store, datetime(2024, 9, 20), days=8, include_quality=False)
    assert '<details class="history">' not in page


def test_a_missed_prediction_is_tagged_distinctly_from_a_correct_one(store_with_players):
    """The visual language for right/wrong must actually differ, not just the
    words -- 'correct' and 'missed' are also different CSS classes."""
    page = build(store_with_players, datetime(2024, 1, 5), days=8)
    if '"tag ok">correct' in page:
        assert 'class="tag ok"' in page
    if '"tag miss">missed' in page:
        assert 'class="tag miss"' in page


def test_a_finished_match_in_the_current_matchday_is_coloured(store):
    """A round is a weekend: Friday's match can be over while Saturday's is
    still to come, and Friday's card must show the verdict and be coloured,
    not silently drop out of the fixtures tab or sit there unexplained."""
    import numpy as np

    from conftest import TEAMS, generate_season

    store.init_schema()
    rng = np.random.default_rng(9)
    matches, results, quotes = generate_season(
        datetime(2022, 8, 10, 15, 30), "2022-23", rng)
    store.upsert("match", matches, ["match_id"])
    store.upsert("match_result", results, ["match_id", "source"])
    store.upsert("odds_quote", quotes,
                 ["match_id", "book", "market", "selection", "quoted_at"])

    friday = datetime(2024, 9, 20, 18, 30)
    saturday = friday + timedelta(hours=20)
    as_of = friday + timedelta(hours=3)          # Friday's match has finished

    store.upsert("match", pd.DataFrame([
        {"match_id": "friday", "source": "t", "league": "bundesliga",
         "season": "2024-25", "kickoff_utc": friday,
         "home_team_id": TEAMS[0], "away_team_id": TEAMS[1],
         "known_at": friday - timedelta(days=30)},
        {"match_id": "saturday", "source": "t", "league": "bundesliga",
         "season": "2024-25", "kickoff_utc": saturday,
         "home_team_id": TEAMS[2], "away_team_id": TEAMS[3],
         "known_at": saturday - timedelta(days=30)},
    ]), ["match_id"])
    store.upsert("match_result", pd.DataFrame([{
        "match_id": "friday", "source": "t", "home_goals": 2, "away_goals": 0,
        "outcome": "H", "ht_home": 1, "ht_away": 0,
        "known_at": friday + timedelta(hours=2),
    }]), ["match_id", "source"])

    page = build(store, as_of, days=8, include_quality=False, served=False)

    assert "full time" in page                    # Friday's card is resolved
    assert 'class="card correct"' in page or 'class="card miss"' in page
    # Saturday's card is still a plain, unresolved one.
    assert 'class="card"' in page or 'class="card value"' in page


# --------------------------------------------------- missing player data

def test_no_player_data_is_explained_not_shown_as_an_empty_pitch(store):
    """Reported as 'no predicted formations for upcoming matches'.

    The real cause: football_data + openligadb + clubelo alone -- what a
    default `bet serve --refresh` fetches -- carry no player-level data at
    all. `predict_formation` honestly falls back to a 0%-confidence default
    formation with nobody selected, and the page drew that as an empty pitch
    under a formation label, which reads as broken rather than "not
    ingested". It is not about how far in the future the match is.
    """
    import numpy as np

    from conftest import generate_season

    store.init_schema()
    rng = np.random.default_rng(6)
    matches, results, quotes = generate_season(
        datetime(2023, 8, 10, 15, 30), "2023-24", rng)
    store.upsert("match", matches, ["match_id"])
    store.upsert("match_result", results, ["match_id", "source"])
    store.upsert("odds_quote", quotes,
                 ["match_id", "book", "market", "selection", "quoted_at"])
    # Deliberately no player / player_match_stat / lineup rows at all.

    page = build(store, datetime(2024, 1, 5), days=8, include_quality=False)
    assert "No player data for" in page
    assert "predicted 0%" not in page
    assert 'class="pitch"' not in page


def test_a_real_low_confidence_prediction_still_shows_the_pitch(store_with_players):
    """The fallback message must not swallow a genuine, if uncertain, guess."""
    page = build(store_with_players, datetime(2024, 1, 5), days=8)
    assert 'class="pitch"' in page
    assert "No player data for" not in page


# --------------------------------------------------- previous-matchday detail

def test_a_previous_matchday_card_opens_into_its_own_detail(store_with_players):
    """Reported: 'for past matches, I cannot see details of the match.'

    A played fixture should open into the same formation/lineup/scoreline
    detail a live one does -- there is no reason the confirmed XI that
    actually played is any less available than a predicted one.
    """
    page = build(store_with_players, datetime(2024, 1, 5), days=8)
    m = __import__("re").search(r'<details class="history">(.*?)</details>', page, __import__("re").S)
    history_block = m.group(1)
    assert 'role="button"' in history_block
    assert 'class="pitch"' in history_block
    assert "Scorelines" in history_block


def test_a_previous_matchday_card_explains_missing_player_data_too(store):
    """The same honest fallback applies looking backward as forward."""
    import numpy as np

    from conftest import generate_season

    store.init_schema()
    rng = np.random.default_rng(7)
    matches, results, quotes = generate_season(
        datetime(2022, 8, 10, 15, 30), "2022-23", rng)
    store.upsert("match", matches, ["match_id"])
    store.upsert("match_result", results, ["match_id", "source"])
    store.upsert("odds_quote", quotes,
                 ["match_id", "book", "market", "selection", "quoted_at"])

    friday = datetime(2024, 9, 20, 18, 30)
    as_of = friday + timedelta(hours=6)
    store.upsert("match", pd.DataFrame([{
        "match_id": "friday", "source": "t", "league": "bundesliga",
        "season": "2024-25", "kickoff_utc": friday,
        "home_team_id": "bayern_munich", "away_team_id": "rb_leipzig",
        "known_at": friday - timedelta(days=30),
    }]), ["match_id"])
    store.upsert("match_result", pd.DataFrame([{
        "match_id": "friday", "source": "t", "home_goals": 1, "away_goals": 1,
        "outcome": "D", "ht_home": 0, "ht_away": 0,
        "known_at": friday + timedelta(hours=2),
    }]), ["match_id", "source"])

    page = build(store, as_of, days=8, include_quality=False)
    m = __import__("re").search(r'<details class="history">(.*?)</details>', page, __import__("re").S)
    assert "No player data for" in m.group(1)


def test_one_side_missing_player_data_still_shows_the_others_pitch(store):
    """A newly promoted club with no fbref history yet, playing an established
    one that has it -- the common real case, not an all-or-nothing one."""
    import numpy as np

    from conftest import (TEAMS, generate_player_stats, generate_season,
                          lineup_rows_from_stats)

    store.init_schema()
    rng = np.random.default_rng(8)
    matches, results, quotes = generate_season(
        datetime(2023, 8, 10, 15, 30), "2023-24", rng)
    store.upsert("match", matches, ["match_id"])
    store.upsert("match_result", results, ["match_id", "source"])
    store.upsert("odds_quote", quotes,
                 ["match_id", "book", "market", "selection", "quoted_at"])

    players, stats = generate_player_stats(matches, rng)
    without_one_team = stats[stats["team_id"] != TEAMS[0]]
    store.upsert("player", players.drop_duplicates("player_id"), ["player_id"])
    store.upsert("player_match_stat", without_one_team, ["match_id", "player_id", "source"])
    store.upsert("lineup", lineup_rows_from_stats(without_one_team, matches),
                 ["match_id", "player_id", "source", "known_at"])

    page = build(store, datetime(2024, 1, 5), days=8, include_quality=False)
    assert "No player data for" in page
    assert 'class="pitch"' in page          # the other side still gets one


# ---------------------------------------------------------- player modal

def test_player_modal_shows_the_real_box_score_for_a_played_match():
    from bet.dashboard import _player_modal

    modal = _player_modal(
        "p1", "Kramer", "VfL Bochum", "defender", "DF", 0.9, 3,
        {"minutes": 79, "goals": 0, "assists": 0, "shots": 1,
         "shots_on_target": 0, "xg": 0.08, "tackles": 2, "interceptions": 1,
         "yellow_cards": 1, "red_cards": 0},
        played=True)

    rows = dict(modal["rows"])
    assert rows["Minutes"] == "79"
    assert rows["xG"] == "0.08"
    assert rows["Cards"] == "1 yellow"
    assert modal["note"] == ""


def test_player_modal_notes_missing_keeper_stats_rather_than_faking_them():
    """A box score without the goalkeeper table's own fields (an older
    ingest, or a match it was never parsed for) must say so, not show
    nothing with no explanation."""
    from bet.dashboard import _player_modal

    modal = _player_modal(
        "gk1", "Neuer", "Bayern", "goalkeeper", "GK", 1.0, 0,
        {"minutes": 90, "goals": 0, "assists": 0}, played=True)

    rows = dict(modal["rows"])
    assert "Shots" not in rows           # not a keeper stat
    assert "Saves" not in rows
    assert "not ingested" in modal["note"].lower()


def test_player_modal_shows_real_keeper_stats_when_ingested():
    from bet.dashboard import _player_modal

    modal = _player_modal(
        "gk1", "Neuer", "Bayern", "goalkeeper", "GK", 1.0, 0,
        {"minutes": 90, "goals": 0, "assists": 0, "gk_shots_faced": 6,
         "gk_goals_against": 1, "gk_saves": 5, "gk_save_pct": 83.3}, played=True)

    rows = dict(modal["rows"])
    assert rows["Shots faced"] == "6"
    assert rows["Goals conceded"] == "1"
    assert rows["Saves"] == "5"
    assert rows["Save %"] == "83.3%"
    assert modal["note"] == ""


def test_player_modal_shows_the_richer_outfield_stats_when_present():
    from bet.dashboard import _player_modal

    modal = _player_modal(
        "p1", "Kimmich", "Bayern", "midfielder", "MF", 1.0, 0,
        {"minutes": 90, "goals": 0, "assists": 1, "shots": 2, "shots_on_target": 1,
         "avg_shot_distance": 18.288, "xg": 0.15, "dribbles_attempted": 4,
         "dribbles_completed": 3, "tackles": 3, "tackles_won": 2, "interceptions": 1,
         "aerials_won": 2, "aerials_lost": 1, "fouls_drawn": 2}, played=True)

    rows = dict(modal["rows"])
    assert rows["Avg shot distance"] == "18.3m"
    assert rows["Dribbles"] == "3/4"
    assert rows["Tackles"] == "3 (2 won)"
    assert rows["Aerials won"] == "2/3"
    assert rows["Fouls drawn"] == "2"


def test_player_modal_skips_richer_stats_never_ingested_for_this_row():
    """A box score predating these columns must not show fabricated zeroes."""
    from bet.dashboard import _player_modal

    modal = _player_modal(
        "p1", "Kimmich", "Bayern", "midfielder", "MF", 1.0, 0,
        {"minutes": 90, "goals": 0, "assists": 0, "shots": 1}, played=True)

    rows = dict(modal["rows"])
    assert "Avg shot distance" not in rows
    assert "Dribbles" not in rows
    assert "Aerials won" not in rows
    assert "Fouls drawn" not in rows


def test_team_richer_tiles_aggregate_across_the_squad():
    from bet.dashboard import _team_richer_tiles

    rows = pd.DataFrame({
        "shots": [3, 1], "avg_shot_distance": [18.0, 12.0],
        "dribbles_completed": [2, 1], "dribbles_attempted": [3, 2],
        "tackles_won": [2, 1], "aerials_won": [3, 0], "aerials_lost": [1, 0],
        "gk_saves": [None, 4], "gk_goals_against": [None, 1],
    })
    html = _team_richer_tiles(rows)
    assert "dribbles completed" in html
    assert "3/5" in html                     # 2+1 completed of 3+2 attempted
    assert "tackles won" in html
    assert "goalkeeper saves" in html
    assert "1 conceded" in html
    # Weighted by shots: (18*3 + 12*1) / 4 = 16.5
    assert "16.5m" in html


def test_team_richer_tiles_blank_when_never_ingested():
    from bet.dashboard import _team_richer_tiles

    assert _team_richer_tiles(pd.DataFrame()) == ""
    assert _team_richer_tiles(pd.DataFrame({"other": [1, 2]})) == ""


def test_render_ticker_is_blank_with_no_events():
    from bet.dashboard import _render_ticker

    assert _render_ticker(pd.DataFrame(), "bayern_munich", "borussia_dortmund") == ""


def test_render_ticker_shows_minute_label_and_side():
    from bet.dashboard import _render_ticker

    events = pd.DataFrame([
        {"minute": 1, "stoppage": None, "event_type": "kickoff", "team_id": None,
         "player_name": None, "description": "Anpfiff"},
        {"minute": 23, "stoppage": None, "event_type": "goal", "team_id": "borussia_dortmund",
         "player_name": "Guirassy", "description": "Tor fuer Borussia Dortmund"},
        {"minute": 45, "stoppage": 2, "event_type": "yellow_card", "team_id": "bayern_munich",
         "player_name": "Kimmich", "description": "Gelbe Karte fuer Kimmich"},
        {"minute": 78, "stoppage": None, "event_type": "note", "team_id": None,
         "player_name": None, "description": "Ein Kommentar ohne Klassifikation"},
    ])
    html = _render_ticker(events, "bayern_munich", "borussia_dortmund")

    assert "45+2&rsquo;" in html
    assert '<span class="label">Goal</span>' in html
    assert "Guirassy" in html
    assert '<div class="tickerrow goal away">' in html
    assert '<div class="tickerrow yellow_card home">' in html
    # An unclassified line still shows its own text, with no fabricated label.
    assert "Ein Kommentar ohne Klassifikation" in html
    assert '<span class="label"></span>' not in html


def test_render_ticker_does_not_leak_pandas_nan_for_a_missing_player():
    """A DataFrame column mixing None and real strings coerces the None to a
    float NaN, not Python's None -- `if event.player_name` alone is truthy for
    NaN, so a naive check would print the literal text "nan" for every event
    with no resolved player once the frame also holds a resolved one."""
    from bet.dashboard import _render_ticker

    events = pd.DataFrame([
        {"minute": 1, "stoppage": None, "event_type": "kickoff", "team_id": None,
         "player_name": None, "description": "Anpfiff"},
        {"minute": 23, "stoppage": None, "event_type": "goal", "team_id": "borussia_dortmund",
         "player_name": "Guirassy", "description": "Tor fuer Borussia Dortmund"},
    ])
    html = _render_ticker(events, "bayern_munich", "borussia_dortmund")
    assert "nan" not in html
    assert "Guirassy" in html


def test_ticker_tab_falls_back_to_an_honest_message_with_no_events(store_with_players):
    """No live source ingested for this match must not look like a broken
    feature -- it should say plainly that nothing has been fetched."""
    from bet.dashboard import build

    page = build(store_with_players, datetime(2024, 1, 5), days=8)
    assert "No live play-by-play source is ingested yet" in page


def test_player_modal_on_an_upcoming_fixture_shows_context_not_a_fake_score():
    from bet.dashboard import _player_modal

    modal = _player_modal(
        "p2", "Kane", "Bayern", "forward", "FW", 0.95, 4, None, played=False)

    rows = dict(modal["rows"])
    assert rows["Start confidence"] == "95%"
    assert "Goals" not in rows
    assert "not this match" in modal["note"] or "rolling per-90" in modal["note"]


def test_a_named_but_unused_substitute_says_so(): 
    from bet.dashboard import _player_modal

    modal = _player_modal(
        "p3", "Sub", "Bayern", "midfielder", "MF", 0.4, 10, None, played=True)
    rows = dict(modal["rows"])
    assert rows["Minutes"] == "0"
    assert "did not play" in modal["note"]


def test_pitch_marks_are_clickable_with_an_embedded_payload():
    from bet.viz import pitch

    players = [{"name": "Test Player", "short": "TP", "row": 0, "row_size": 1,
               "modal": {"name": "Test Player", "team": "X", "rows": [["Minutes", "90"]], "note": ""}}]
    svg = pitch("4-3-3", players, team_name="X")
    assert 'class="playermark"' in svg
    assert 'data-player=' in svg
    assert 'role="button"' in svg


def test_a_mark_with_no_modal_data_is_not_clickable():
    """A pitch built without per-player detail (e.g. a stripped-down caller)
    must not claim to be clickable with nothing behind it."""
    from bet.viz import pitch

    players = [{"name": "Test Player", "short": "TP", "row": 0, "row_size": 1}]
    svg = pitch("4-3-3", players, team_name="X")
    assert 'class="playermark"' not in svg


def test_match_detail_has_a_formation_heatmap_ticker_switcher(store_with_players):
    page = build(store_with_players, datetime(2024, 1, 5), days=8)
    assert 'class="subnav"' in page
    assert ">Formation</button>" in page
    assert ">Heatmaps</button>" in page
    assert ">Ticker</button>" in page
    # Real data source, not a fabricated one.
    assert "Sofascore" not in page or "unverified" in page.lower()


def test_each_matchs_subtabs_have_unique_ids(store_with_players):
    """Two fixtures on one page must not share a sub-tab switcher."""
    import re

    page = build(store_with_players, datetime(2024, 1, 5), days=8)
    ids = re.findall(r'id="([\w-]+-formation)"', page)
    assert len(ids) == len(set(ids)), "duplicate sub-tab ids across fixtures"


def test_a_finished_match_in_an_unfinished_round_is_not_shown_twice(store):
    """The reported bug: the same 'No player data for X and Y' message
    appeared twice on the page. Friday's match, already finished, was being
    claimed by both the current-round view and the previous-matchday panel
    at once, because the previous-matchday lookup could not see that
    Saturday's fixture in the same round had not been played yet."""
    import numpy as np

    from conftest import generate_season

    store.init_schema()
    rng = np.random.default_rng(12)
    matches, results, quotes = generate_season(
        datetime(2022, 8, 10, 15, 30), "2022-23", rng)
    store.upsert("match", matches, ["match_id"])
    store.upsert("match_result", results, ["match_id", "source"])
    store.upsert("odds_quote", quotes,
                 ["match_id", "book", "market", "selection", "quoted_at"])

    friday = datetime(2024, 9, 20, 18, 30)
    saturday = friday + timedelta(hours=20)
    as_of = friday + timedelta(hours=3)

    store.upsert("match", pd.DataFrame([
        {"match_id": "friday", "source": "t", "league": "bundesliga",
         "season": "2024-25", "kickoff_utc": friday,
         "home_team_id": "borussia_dortmund", "away_team_id": "sv_werder_bremen",
         "known_at": friday - timedelta(days=30)},
        {"match_id": "saturday", "source": "t", "league": "bundesliga",
         "season": "2024-25", "kickoff_utc": saturday,
         "home_team_id": "bayern_munich", "away_team_id": "rb_leipzig",
         "known_at": saturday - timedelta(days=30)},
    ]), ["match_id"])
    store.upsert("match_result", pd.DataFrame([{
        "match_id": "friday", "source": "t", "home_goals": 2, "away_goals": 0,
        "outcome": "H", "ht_home": 1, "ht_away": 0,
        "known_at": friday + timedelta(hours=2),
    }]), ["match_id", "source"])

    page = build(store, as_of, days=8, include_quality=False)
    # Dortmund vs Bremen exists exactly once. Before the fix it was claimed by
    # both the current round and the previous-matchday panel at once.
    assert page.count("vs</span> Werder Bremen") == 1
    # The round is not finished (Saturday has not been played), so there is
    # no previous matchday yet at all.
    assert '<details class="history">' not in page


def test_scorelines_still_render_with_no_player_data_at_all(store):
    """The early return that produced the previous bug also threw away
    Scorelines and every other model-derived number, which needs no player
    data at all -- 'nothing is visible, no data at all, no match details'."""
    import numpy as np

    from conftest import generate_season

    store.init_schema()
    rng = np.random.default_rng(13)
    matches, results, quotes = generate_season(
        datetime(2023, 8, 10, 15, 30), "2023-24", rng)
    store.upsert("match", matches, ["match_id"])
    store.upsert("match_result", results, ["match_id", "source"])
    store.upsert("odds_quote", quotes,
                 ["match_id", "book", "market", "selection", "quoted_at"])
    # No player data at all -- the exact reported situation.

    page = build(store, datetime(2024, 1, 5), days=8, include_quality=False)
    assert "No player data for" in page
    assert "Scorelines" in page
    assert 'class="subnav"' in page      # the tabs are still there


# ---------------------------------------------------------- zone heatmap

def test_zone_heatmap_draws_three_bands_and_both_boxes():
    from bet.viz import zone_heatmap

    svg = zone_heatmap({"thirds": {"def": 0.3, "mid": 0.45, "att": 0.25},
                        "penalty": {"def_pen": 4, "att_pen": 9}, "touches": 120},
                       team_name="Bayern Munich")
    assert svg.count("data-tip=") == 3          # one per band
    assert "25%" in svg and "45%" in svg and "30%" in svg
    assert "4 in the box" in svg
    assert "9 in the box" in svg
    assert 'aria-label="Bayern Munich touch zones"' in svg


def test_zone_heatmap_omits_a_penalty_label_that_was_never_ingested():
    from bet.viz import zone_heatmap

    svg = zone_heatmap({"thirds": {"def": 0.5, "mid": 0.3, "att": 0.2}, "penalty": {}},
                       team_name="X")
    assert "in the box" not in svg


def test_zone_heatmap_text_stays_legible_at_both_ends_of_the_scale():
    """The scale runs light-to-dark; white text on the lightest band would be
    unreadable, and dark ink on the darkest band would be little better."""
    from bet.viz import zone_heatmap

    heavy = zone_heatmap({"thirds": {"def": 0.05, "mid": 0.05, "att": 0.9},
                         "penalty": {}})
    # The heaviest band (90%, saturating well past the 55% reference) must
    # not still be using dark-on-dark text.
    assert 'fill="var(--ink)">90%' not in heavy
    assert 'fill="#fff">90%' in heavy
    light = zone_heatmap({"thirds": {"def": 0.02, "mid": 0.02, "att": 0.96},
                         "penalty": {}})
    assert 'fill="#fff">2%' not in light


# --------------------------------------------------- top-level nav collision

def test_a_matchs_subnav_is_not_matched_by_the_top_level_tab_selector(store_with_players):
    """Reported: clicking Formation/Heatmaps/Ticker closed the whole fixture,
    and the team stats below it vanished too.

    The top-level tab bar's own script binds a click handler to every literal
    `<nav>` element on the page via `document.querySelectorAll('nav button')`,
    on the assumption there is exactly one. The per-fixture sub-tab switcher
    was also marked up as a `<nav>`, so that same query matched it too, and
    its handler reset every `.view` -- including `#fixtures` itself, with the
    fixture already open -- by the top-level convention this button's
    `data-subview` does not carry. That query is a stand-in for the real
    `document.querySelectorAll('nav button')` the page runs; it must return
    only the site's own tab bar, never a fixture's internal switcher.
    """
    from bs4 import BeautifulSoup

    page = build(store_with_players, datetime(2024, 1, 5), days=8)
    soup = BeautifulSoup(page, "lxml")

    top_level = soup.select("nav button")
    assert {b.get("data-view") for b in top_level} == {
        "overview", "fixtures", "model", "props"}

    # The sub-tab switcher must still exist and still be labelled a tablist --
    # it is just not allowed to be a second <nav> landmark.
    assert soup.select_one(".subnav[role='tablist']") is not None
    assert soup.select_one("nav .subnav") is None
