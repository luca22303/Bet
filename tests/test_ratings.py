"""Derived power ratings, and choosing between rating providers.

ClubElo going down took the promoted-team prior with it. Ratings do not have to
be fetched, though: Elo is computed from results, and results are the one thing
this store holds from two independent sources.
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from bet.ratings import (DERIVED_SOURCE, derive_and_store, elo_timeline,
                         final_elo)

NOW = datetime(2026, 9, 22)


def _history(store, *, n=120):
    """A season or two where Bayern win and Bochum lose."""
    teams = ["bayern_munich", "borussia_dortmund", "rb_leipzig", "vfl_bochum"]
    matches, results = [], []
    for i in range(n):
        day = NOW - timedelta(days=(n - i) * 3)
        home, away = teams[i % 4], teams[(i + 1) % 4]
        if home == "bayern_munich":
            home_goals, away_goals = 3, 0
        elif away == "bayern_munich":
            home_goals, away_goals = 0, 3
        elif home == "vfl_bochum":
            home_goals, away_goals = 0, 2
        else:
            home_goals, away_goals = 1, 1
        matches.append({
            "match_id": f"m{i}", "source": "fd", "league": "bundesliga",
            "season": "2025-26", "kickoff_utc": day, "home_team_id": home,
            "away_team_id": away, "known_at": day - timedelta(days=30)})
        results.append({
            "match_id": f"m{i}", "source": "fd",
            "home_goals": home_goals, "away_goals": away_goals,
            "outcome": ("H" if home_goals > away_goals
                        else "D" if home_goals == away_goals else "A"),
            "ht_home": 0, "ht_away": 0, "known_at": day + timedelta(hours=2)})

    store.init_schema()
    store.upsert("match", pd.DataFrame(matches), ["match_id"])
    store.upsert("match_result", pd.DataFrame(results), ["match_id", "source"])


def _clubelo_rows(valid_from, ratings):
    return pd.DataFrame([{
        "team_id": team, "source": "clubelo", "rating": rating,
        "valid_from": valid_from.date(), "valid_to": None,
        "known_at": valid_from,
    } for team, rating in ratings.items()])


# ------------------------------------------------------------- the walk

def test_ratings_separate_teams_by_results(store):
    _history(store)
    ratings = final_elo(store.matches_as_of(NOW))

    assert ratings["bayern_munich"] > 1500 > ratings["vfl_bochum"]


def test_an_empty_history_produces_nothing_rather_than_raising(store):
    store.init_schema()
    assert elo_timeline(pd.DataFrame()) == []
    assert final_elo(pd.DataFrame()) == {}


def test_unplayed_fixtures_are_skipped(store):
    """A future fixture carries no result and must not move a rating."""
    frame = pd.DataFrame([{
        "match_id": "f1", "kickoff_utc": NOW + timedelta(days=3),
        "home_team_id": "a", "away_team_id": "b",
        "home_goals": None, "away_goals": None, "result_known_at": None,
    }])
    assert elo_timeline(frame) == []


def test_a_bigger_win_moves_the_rating_further():
    """A 4-0 is more evidence than a 1-0."""
    def move(home_goals, away_goals):
        frame = pd.DataFrame([{
            "match_id": "m", "kickoff_utc": NOW, "home_team_id": "a",
            "away_team_id": "b", "home_goals": home_goals,
            "away_goals": away_goals, "result_known_at": NOW,
        }])
        return final_elo(frame)["a"]

    assert move(4, 0) > move(1, 0) > 1500.0


# -------------------------------------------------------- point in time

def test_a_derived_rating_is_not_visible_before_its_match(store):
    """The whole store rests on this; a fallback may not be the hole in it."""
    _history(store)
    derive_and_store(store, league="bundesliga")

    rows = store.con.execute(
        "SELECT r.known_at, r.valid_from FROM team_rating r "
        f"WHERE r.source = '{DERIVED_SOURCE}'").df()
    assert not rows.empty
    # known_at is the final whistle; valid_from is the match date.
    assert (pd.to_datetime(rows["known_at"]).dt.date
            >= pd.to_datetime(rows["valid_from"]).dt.date).all()


def test_ratings_as_of_excludes_matches_that_had_not_happened(store):
    _history(store)
    derive_and_store(store, league="bundesliga")

    early = store.ratings_as_of(NOW - timedelta(days=300))
    late = store.ratings_as_of(NOW)
    assert not early.empty
    assert early["valid_from"].max() < late["valid_from"].max()


def test_derived_ratings_pass_the_leakage_guard(store):
    _history(store)
    derive_and_store(store, league="bundesliga")

    report = store.leakage_report()
    violations = report[report["table"] == "team_rating"]["violations"].sum()
    assert int(violations) == 0


# ------------------------------------------------------ choosing a source

def test_clubelo_wins_while_it_is_current(store):
    """The derived walk only knows this league; ClubElo places it in Europe."""
    _history(store)
    derive_and_store(store, league="bundesliga")
    store.upsert("team_rating",
                 _clubelo_rows(NOW - timedelta(days=2), {"bayern_munich": 2001.0}),
                 ["team_id", "source", "valid_from"])

    ratings = store.ratings_as_of(NOW)
    assert list(ratings["team_id"]) == ["bayern_munich"]
    assert ratings["rating"].iloc[0] == 2001.0


def test_a_stale_clubelo_snapshot_gives_way_to_the_derived_walk(store):
    """This is the fallback working: ClubElo went down months ago.

    Serving its last snapshot indefinitely would quietly pass off a rating
    from another season as current.
    """
    _history(store)
    derive_and_store(store, league="bundesliga")
    store.upsert("team_rating",
                 _clubelo_rows(NOW - timedelta(days=200), {"bayern_munich": 2001.0}),
                 ["team_id", "source", "valid_from"])

    ratings = store.ratings_as_of(NOW)
    assert 2001.0 not in set(ratings["rating"])
    assert len(ratings) == 4                       # the whole league, not one club


def test_a_single_provider_supplies_the_whole_table(store):
    """Scales differ between providers, so a mixed column is meaningless.

    ClubElo spans roughly 1300-2100 across Europe; a league-local walk
    starting everyone at 1500 spans far less. A table with some rows from
    each has a column where a number means something different per row.
    """
    _history(store)
    derive_and_store(store, league="bundesliga")
    # ClubElo knows only one of the four clubs.
    store.upsert("team_rating",
                 _clubelo_rows(NOW - timedelta(days=1), {"bayern_munich": 2001.0}),
                 ["team_id", "source", "valid_from"])

    ratings = store.ratings_as_of(NOW)
    assert len(ratings) == 1, "fell back per team and mixed two rating scales"


def test_an_explicit_source_still_overrides_the_choice(store):
    _history(store)
    derive_and_store(store, league="bundesliga")

    assert store.ratings_as_of(NOW, source="clubelo").empty
    assert not store.ratings_as_of(NOW, source=DERIVED_SOURCE).empty


def test_no_ratings_at_all_returns_an_empty_frame_with_the_right_columns(store):
    store.init_schema()
    frame = store.ratings_as_of(NOW)
    assert frame.empty
    assert list(frame.columns) == ["team_id", "rating", "valid_from"]
