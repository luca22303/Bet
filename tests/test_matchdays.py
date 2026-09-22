"""Grouping fixtures into matchdays, and picking the previous/next one."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from bet.matchdays import group_by_matchday, next_matchday, previous_matchday


def _fixture(match_id, kickoff, home="a", away="b", **extra):
    row = {"match_id": match_id, "league": "bundesliga", "season": "2024-25",
          "kickoff_utc": kickoff, "home_team_id": home, "away_team_id": away,
          "home_goals": None, "away_goals": None, "outcome": None}
    row.update(extra)
    return row


def test_group_by_matchday_clusters_a_weekend_together():
    base = datetime(2024, 9, 20, 18, 30)
    fixtures = pd.DataFrame([
        _fixture("m1", base),
        _fixture("m2", base + timedelta(hours=20)),
        _fixture("m3", base + timedelta(hours=44)),          # same weekend
        _fixture("m4", base + timedelta(days=7)),             # next matchday
    ])
    groups = group_by_matchday(fixtures)
    assert len(groups) == 2
    assert set(groups[0]["match_id"]) == {"m1", "m2", "m3"}
    assert set(groups[1]["match_id"]) == {"m4"}


def test_group_by_matchday_handles_an_empty_frame():
    assert group_by_matchday(pd.DataFrame()) == []


def test_next_matchday_returns_the_whole_cluster_not_just_one_fixture(store):
    store.init_schema()
    as_of = datetime(2024, 9, 15)
    base = datetime(2024, 9, 20, 18, 30)
    store.upsert("match", pd.DataFrame([
        {"match_id": "m1", "source": "t", "league": "bundesliga", "season": "2024-25",
         "kickoff_utc": base, "home_team_id": "a", "away_team_id": "b",
         "known_at": as_of - timedelta(days=1)},
        {"match_id": "m2", "source": "t", "league": "bundesliga", "season": "2024-25",
         "kickoff_utc": base + timedelta(hours=20), "home_team_id": "c", "away_team_id": "d",
         "known_at": as_of - timedelta(days=1)},
        {"match_id": "m3", "source": "t", "league": "bundesliga", "season": "2024-25",
         "kickoff_utc": base + timedelta(days=7), "home_team_id": "a", "away_team_id": "c",
         "known_at": as_of - timedelta(days=1)},
    ]), ["match_id"])

    fixtures = next_matchday(store, as_of, league="bundesliga")
    assert set(fixtures["match_id"]) == {"m1", "m2"}


def test_next_matchday_is_empty_when_nothing_is_scheduled(store):
    store.init_schema()
    assert next_matchday(store, datetime(2024, 9, 15), league="bundesliga").empty


def test_previous_matchday_only_counts_played_matches(store):
    """A match that has kicked off but has no result yet is not 'previous' --
    there is nothing yet to compare a prediction against."""
    store.init_schema()
    as_of = datetime(2024, 9, 22)
    played_day = datetime(2024, 9, 13, 15, 30)
    unresolved_day = datetime(2024, 9, 20, 15, 30)          # kicked off, no result

    store.upsert("match", pd.DataFrame([
        {"match_id": "played", "source": "t", "league": "bundesliga", "season": "2024-25",
         "kickoff_utc": played_day, "home_team_id": "a", "away_team_id": "b",
         "known_at": played_day - timedelta(days=30)},
        {"match_id": "unresolved", "source": "t", "league": "bundesliga", "season": "2024-25",
         "kickoff_utc": unresolved_day, "home_team_id": "c", "away_team_id": "d",
         "known_at": unresolved_day - timedelta(days=30)},
    ]), ["match_id"])
    store.upsert("match_result", pd.DataFrame([{
        "match_id": "played", "source": "t", "home_goals": 2, "away_goals": 1,
        "outcome": "H", "ht_home": 1, "ht_away": 0,
        "known_at": played_day + timedelta(hours=2),
    }]), ["match_id", "source"])

    fixtures = previous_matchday(store, as_of, league="bundesliga")
    assert set(fixtures["match_id"]) == {"played"}


def test_previous_matchday_is_empty_when_nothing_has_been_played(store):
    store.init_schema()
    assert previous_matchday(store, datetime(2024, 9, 22), league="bundesliga").empty
