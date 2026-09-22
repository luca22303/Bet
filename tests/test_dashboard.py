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
