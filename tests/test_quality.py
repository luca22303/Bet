"""Data quality checks.

Scrapers fail silently. These verify that the checks actually fire on broken
data rather than only passing on clean data.
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from bet.quality import (
    check_odds_sanity,
    check_staleness,
    check_team_identity,
    reconcile_results,
    run_quality_checks,
)


def test_clean_store_passes(populated_store):
    report = run_quality_checks(populated_store, as_of=datetime(2024, 6, 1))
    assert report.passed
    assert not report.errors


def test_coverage_shows_gaps_per_season(populated_store):
    report = run_quality_checks(populated_store, as_of=datetime(2024, 6, 1))
    assert not report.coverage.empty
    assert "with_player_stats" in report.coverage.columns
    # These seasons have results and odds but no player data.
    assert (report.coverage["with_player_stats"] == 0).all()
    assert (report.coverage["results"] > 0).all()


def test_recovers_the_synthetic_book_margin(populated_store):
    """The generator applies a 6% margin; the check must measure it back."""
    issues = check_odds_sanity(populated_store)
    overround = next(i for i in issues if "overround" in i.detail)
    assert "6.0" in overround.detail or "5.9" in overround.detail


def test_impossible_odds_are_an_error(store):
    store.upsert("odds_quote", pd.DataFrame([{
        "match_id": "m", "book": "b", "market": "1x2", "selection": "H",
        "decimal_odds": 0.4, "quoted_at": datetime(2024, 1, 1),
        "is_closing": True, "known_at": datetime(2024, 1, 1),
    }]), ["match_id", "book", "market", "selection", "quoted_at"])
    assert any(i.severity == "error" for i in check_odds_sanity(store))


def test_a_book_summing_below_one_is_flagged_as_mis_parsed(store):
    """An arbitrage is far more likely to be a parser bug than a real price."""
    rows = [{
        "match_id": "m", "book": "b", "market": "1x2", "selection": sel,
        "decimal_odds": 4.0, "quoted_at": datetime(2024, 1, 1),
        "is_closing": True, "known_at": datetime(2024, 1, 1),
    } for sel in ("H", "D", "A")]           # booksum 0.75
    store.upsert("odds_quote", pd.DataFrame(rows),
                 ["match_id", "book", "market", "selection", "quoted_at"])
    issues = check_odds_sanity(store)
    assert any(i.severity == "error" and "below 1.0" in i.detail for i in issues)


def test_single_source_results_are_reported_as_uncheckable(populated_store):
    """Honest about the limit: one source cannot be cross-checked."""
    issues = reconcile_results(populated_store)
    assert any("cannot be cross-checked" in i.detail for i in issues)


def test_conflicting_sources_are_an_error(store):
    """The strongest check available: two scrapes either agree or one is wrong.

    This only works because `match_result` is keyed on (match_id, source). Keyed
    on match_id alone, the second ingest would overwrite the first and the
    disagreement would be invisible.
    """
    kickoff = datetime(2024, 3, 1, 15, 30)
    store.upsert("match", pd.DataFrame([{
        "match_id": "m1", "source": "football_data", "league": "bundesliga",
        "season": "2023-24", "kickoff_utc": kickoff,
        "home_team_id": "bayern_munich", "away_team_id": "sc_freiburg",
        "known_at": kickoff - timedelta(days=30),
    }]), ["match_id"])

    store.upsert("match_result", pd.DataFrame([
        {"match_id": "m1", "source": "football_data", "home_goals": 2, "away_goals": 1,
         "outcome": "H", "ht_home": None, "ht_away": None,
         "known_at": kickoff + timedelta(hours=2)},
        {"match_id": "m1", "source": "openligadb", "home_goals": 3, "away_goals": 0,
         "outcome": "H", "ht_home": None, "ht_away": None,
         "known_at": kickoff + timedelta(hours=2)},
    ]), ["match_id", "source"])

    issues = reconcile_results(store)
    assert any(i.severity == "error" and "disagree" in i.detail for i in issues)


def test_agreeing_sources_are_reported_as_verified(store):
    kickoff = datetime(2024, 3, 1, 15, 30)
    store.upsert("match", pd.DataFrame([{
        "match_id": "m1", "source": "football_data", "league": "bundesliga",
        "season": "2023-24", "kickoff_utc": kickoff,
        "home_team_id": "bayern_munich", "away_team_id": "sc_freiburg",
        "known_at": kickoff - timedelta(days=30),
    }]), ["match_id"])
    store.upsert("match_result", pd.DataFrame([
        {"match_id": "m1", "source": src, "home_goals": 2, "away_goals": 1,
         "outcome": "H", "ht_home": None, "ht_away": None,
         "known_at": kickoff + timedelta(hours=2)}
        for src in ("football_data", "openligadb")
    ]), ["match_id", "source"])

    issues = reconcile_results(store)
    assert any("all sources agree" in i.detail for i in issues)
    assert not any(i.severity == "error" for i in issues)


def test_reads_pick_one_source_deterministically(store):
    """Two sources must not produce two training rows for one match."""
    kickoff = datetime(2024, 3, 1, 15, 30)
    store.upsert("match", pd.DataFrame([{
        "match_id": "m1", "source": "football_data", "league": "bundesliga",
        "season": "2023-24", "kickoff_utc": kickoff,
        "home_team_id": "bayern_munich", "away_team_id": "sc_freiburg",
        "known_at": kickoff - timedelta(days=30),
    }]), ["match_id"])
    store.upsert("match_result", pd.DataFrame([
        {"match_id": "m1", "source": "openligadb", "home_goals": 3, "away_goals": 0,
         "outcome": "H", "ht_home": None, "ht_away": None,
         "known_at": kickoff + timedelta(hours=2)},
        {"match_id": "m1", "source": "football_data", "home_goals": 2, "away_goals": 1,
         "outcome": "H", "ht_home": None, "ht_away": None,
         "known_at": kickoff + timedelta(hours=2)},
    ]), ["match_id", "source"])

    matches = store.matches_as_of(kickoff + timedelta(days=1))
    assert len(matches) == 1
    # football_data wins the preference, whichever order the ingests ran in.
    assert int(matches.iloc[0]["home_goals"]) == 2


def test_stale_data_is_flagged(populated_store):
    """Catches ingestion having silently stopped weeks ago."""
    issues = check_staleness(populated_store, as_of=datetime(2030, 1, 1))
    assert any("days old" in i.detail for i in issues)


def test_fresh_data_is_not_flagged(store):
    now = datetime(2024, 6, 1)
    store.upsert("match_result", pd.DataFrame([{
        "match_id": "m", "source": "football_data",
        "home_goals": 1, "away_goals": 0, "outcome": "H",
        "ht_home": None, "ht_away": None, "known_at": now - timedelta(days=1),
    }]), ["match_id", "source"])
    issues = check_staleness(store, as_of=now)
    assert not any(i.check == "staleness" and "match_result" in i.detail
                   and "days old" in i.detail for i in issues)


def test_rare_teams_signal_a_name_resolution_failure(store):
    """A club splitting into a canonical id and a near-duplicate."""
    kickoff = datetime(2024, 3, 1, 15, 30)
    rows = [{
        "match_id": f"m{i}", "source": "t", "league": "bundesliga", "season": "2023-24",
        "kickoff_utc": kickoff, "home_team_id": "bayern_munich",
        "away_team_id": "sc_freiburg" if i else "a_typo_team",
        "known_at": kickoff - timedelta(days=30),
    } for i in range(30)]
    store.upsert("match", pd.DataFrame(rows), ["match_id"])
    issues = check_team_identity(store)
    assert any("a_typo_team" in i.detail for i in issues)


def test_point_in_time_violation_is_an_error(populated_store):
    """The one failure that invalidates every other number."""
    store = populated_store
    match_id, kickoff = store.con.execute(
        "SELECT match_id, kickoff_utc FROM match LIMIT 1").fetchone()
    store.con.execute("UPDATE match_result SET known_at = ? WHERE match_id = ?",
                      [kickoff - timedelta(days=1), match_id])

    report = run_quality_checks(store, as_of=datetime(2024, 6, 1))
    assert not report.passed
    assert any(i.check == "point_in_time" for i in report.errors)


def test_empty_store_is_an_error(store):
    report = run_quality_checks(store, as_of=datetime(2024, 1, 1))
    assert not report.passed


def test_report_renders_as_text(populated_store):
    text = run_quality_checks(populated_store, as_of=datetime(2024, 6, 1)).to_text()
    assert "DATA QUALITY" in text
    assert "coverage" in text
