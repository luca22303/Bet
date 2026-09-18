"""Predicted line-ups.

The headline behaviour: a player who has not featured in weeks must not be
priced as a starter, however many minutes he banked earlier in the season.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from bet.availability import AvailabilityStatus
from bet.lineups import (
    DEFAULT_HALF_LIFE_DAYS,
    infer_formation,
    lineup_strength,
    lineup_strength_shift,
    predict_formation,
    predict_lineup,
    recency_weights,
    start_propensity,
)


# ------------------------------------------------------------------ mechanics


def test_recency_weights_decay_by_half_life():
    as_of = datetime(2024, 3, 1)
    weights = recency_weights(
        [as_of, as_of - timedelta(days=21), as_of - timedelta(days=42)],
        as_of, half_life_days=21.0)
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(0.5, abs=0.01)
    assert weights[2] == pytest.approx(0.25, abs=0.01)


@pytest.mark.parametrize("groups,expected", [
    (["defender"] * 4 + ["midfielder"] * 5 + ["forward"], "4-2-3-1"),
    (["defender"] * 4 + ["midfielder"] * 3 + ["forward"] * 3, "4-3-3"),
    (["defender"] * 3 + ["midfielder"] * 5 + ["forward"] * 2, "3-5-2"),
])
def test_formation_inferred_from_who_is_on_the_pitch(groups, expected):
    """Sources disagree on formation strings and FBref supplies none."""
    assert infer_formation(groups) == expected


def test_unrecognised_shape_is_reported_verbatim():
    assert infer_formation(["defender"] * 6 + ["midfielder"] * 3 + ["forward"]) == "6-3-1"


# ------------------------------------------------------------- propensity


def test_recent_starters_rank_above_absentees(store_with_players):
    as_of = datetime(2024, 1, 5)
    frame = start_propensity(store_with_players, "bayern_munich", as_of)
    assert not frame.empty
    assert frame["propensity"].is_monotonic_decreasing
    # Regulars in the generator start 85-100% of the time; fringe players ~10%.
    assert frame["propensity"].iloc[0] > 0.7
    assert frame["propensity"].iloc[-1] < 0.4


def test_a_player_who_stopped_playing_loses_propensity(store):
    """The bug this module exists to fix.

    A striker who played every week until October and has not appeared since
    still holds a large share of the season's minutes. Weighted by cumulative
    minutes he looks like a first-choice starter all year; weighted by recency
    he does not.
    """
    as_of = datetime(2024, 3, 1)
    rows, matches = [], []
    for i in range(20):
        day = as_of - timedelta(days=7 * (20 - i))
        match_id = f"m{i}"
        matches.append({
            "match_id": match_id, "source": "t", "league": "bundesliga",
            "season": "2023-24", "kickoff_utc": day,
            "home_team_id": "bayern_munich", "away_team_id": "sc_freiburg",
            "known_at": day - timedelta(days=30),
        })
        # The star starts the first ten matches, then disappears.
        for player, started in (("star", i < 10), ("squad_player", i >= 10)):
            if not started:
                continue
            rows.append({
                "match_id": match_id, "player_id": player, "team_id": "bayern_munich",
                "source": "t", "position": "FW", "started": True, "minutes": 90.0,
                "known_at": day + timedelta(hours=2),
            })

    store.upsert("match", pd.DataFrame(matches), ["match_id"])
    store.upsert("player_match_stat", pd.DataFrame(rows), ["match_id", "player_id", "source"])

    frame = start_propensity(store, "bayern_munich", as_of).set_index("player_id")
    star = float(frame.loc["star", "propensity"])
    incumbent = float(frame.loc["squad_player", "propensity"])

    # The ordering is the property that matters, and the gap should be large:
    # weighted by cumulative minutes these two would be near-equal, since each
    # started ten matches.
    assert incumbent > star
    assert star < incumbent / 5
    assert frame.loc["star", "days_since_start"] > 60


def test_propensity_is_empty_for_an_unknown_team(store_with_players):
    assert start_propensity(store_with_players, "a_team_that_never_played",
                            datetime(2024, 1, 5)).empty


# -------------------------------------------------------------- formation


def test_formation_and_rotation_are_predicted(store_with_players):
    formation, confidence, rotation = predict_formation(
        store_with_players, "bayern_munich", datetime(2024, 1, 5))
    assert "-" in formation
    assert 0.0 <= confidence <= 1.0
    assert 0.0 <= rotation <= 1.0


def test_a_settled_side_scores_low_rotation(store):
    """Rotation is the honest measure of how far to trust a predicted XI."""
    as_of = datetime(2024, 3, 1)
    matches, rows = [], []
    squad = [("gk", "GK")] + [(f"d{i}", "DF") for i in range(4)] \
        + [(f"m{i}", "MF") for i in range(3)] + [(f"f{i}", "FW") for i in range(3)]

    for i in range(10):
        day = as_of - timedelta(days=7 * (10 - i))
        matches.append({
            "match_id": f"m{i}", "source": "t", "league": "bundesliga",
            "season": "2023-24", "kickoff_utc": day,
            "home_team_id": "bayern_munich", "away_team_id": "sc_freiburg",
            "known_at": day - timedelta(days=30),
        })
        for player, position in squad:          # identical XI every week
            rows.append({
                "match_id": f"m{i}", "player_id": player, "team_id": "bayern_munich",
                "source": "t", "position": position, "started": True, "minutes": 90.0,
                "known_at": day + timedelta(hours=2),
            })

    store.upsert("match", pd.DataFrame(matches), ["match_id"])
    store.upsert("player_match_stat", pd.DataFrame(rows), ["match_id", "player_id", "source"])

    formation, confidence, rotation = predict_formation(store, "bayern_munich", as_of)
    assert formation == "4-3-3"
    assert confidence == pytest.approx(1.0)
    assert rotation == pytest.approx(0.0)


# ------------------------------------------------------------------- XI


def test_predicted_xi_has_eleven_players_in_shape(store_with_players):
    lineup = predict_lineup(store_with_players, "bayern_munich", datetime(2024, 1, 5))
    assert len(lineup.starters) == 11
    assert len(set(lineup.starters)) == 11
    assert not lineup.is_confirmed
    assert 0.0 < lineup.confidence <= 1.0


def test_unavailable_players_are_excluded_from_the_xi(store_with_players):
    """Scaling propensity to zero is not enough.

    Sorting alone does not remove a ruled-out player: when every forward is out,
    taking the top three still returns three ruled-out forwards and the absence
    silently has no effect.
    """
    store = store_with_players
    as_of = datetime(2024, 1, 5)
    baseline = predict_lineup(store, "bayern_munich", as_of)
    out = {p: AvailabilityStatus.OUT for p in baseline.starters[:3]}

    weakened = predict_lineup(store, "bayern_munich", as_of, absences=out)
    assert len(weakened.starters) == 11
    for player_id in out:
        assert player_id not in weakened.starters


def test_a_confirmed_lineup_replaces_the_prediction(store_with_players):
    """A confirmed XI is the answer, not evidence to blend with a guess."""
    store = store_with_players
    as_of = datetime(2024, 1, 5)
    fixture = store.fixtures_between(as_of, as_of + timedelta(days=30)).iloc[0]

    chosen = (store.player_rates_as_of(as_of, min_minutes=0.0)
              .query("team_id == @fixture.home_team_id")
              .nsmallest(11, "minutes")["player_id"].tolist())

    store.upsert("lineup", pd.DataFrame([{
        "match_id": fixture["match_id"], "player_id": p,
        "team_id": fixture["home_team_id"], "source": "confirmed_test",
        "is_starter": True, "is_confirmed": True, "shirt_number": None,
        "formation": None, "known_at": as_of,
    } for p in chosen]), ["match_id", "player_id", "source", "known_at"])

    lineup = predict_lineup(store, fixture["home_team_id"], as_of,
                            match_id=fixture["match_id"])
    assert lineup.is_confirmed
    assert lineup.confidence == 1.0
    assert set(lineup.starters) == set(chosen)


# -------------------------------------------------------------- strength


def test_a_stronger_xi_has_more_attacking_output(store_with_players):
    store = store_with_players
    as_of = datetime(2024, 1, 5)
    rates = store.player_rates_as_of(as_of, min_minutes=0.0)
    squad = rates[rates["team_id"] == "bayern_munich"]

    best = squad.nlargest(11, "xg_p90")["player_id"].tolist()
    worst = squad.nsmallest(11, "xg_p90")["player_id"].tolist()

    assert lineup_strength(rates, best)["attack"] > lineup_strength(rates, worst)["attack"]


def test_strength_shift_is_measured_against_the_team_s_own_recent_elevens(store_with_players):
    """Not against an absolute scale.

    The fitted parameters already encode how good a side usually is, so
    comparing to an absolute would double-count team quality and move every
    fixture.
    """
    store = store_with_players
    as_of = datetime(2024, 1, 5)
    rates = store.player_rates_as_of(as_of, min_minutes=0.0)

    lineup = predict_lineup(store, "bayern_munich", as_of)
    shift = lineup_strength_shift(store, "bayern_munich", as_of, rates, lineup)

    assert shift["baseline_matches"] > 0
    # A typical XI should sit close to typical.
    assert abs(shift["attack_shift"]) < 0.25


def test_losing_the_front_line_cuts_attacking_strength(store_with_players):
    store = store_with_players
    as_of = datetime(2024, 1, 5)
    rates = store.player_rates_as_of(as_of, min_minutes=0.0)

    baseline = predict_lineup(store, "bayern_munich", as_of)
    forwards = [p for p in baseline.starters if "st" in p or "lw" in p or "rw" in p]
    out = {p: AvailabilityStatus.OUT for p in forwards}
    weakened = predict_lineup(store, "bayern_munich", as_of, absences=out)

    assert (lineup_strength(rates, weakened.starters)["attack"]
            < lineup_strength(rates, baseline.starters)["attack"])


def test_strength_degrades_quietly_without_data(store):
    lineup = predict_lineup(store, "bayern_munich", datetime(2024, 1, 5))
    assert lineup.starters == []
    assert any("no recent appearance" in n for n in lineup.notes)
    assert lineup_strength(pd.DataFrame(), [])["players"] == 0


def test_strength_shift_is_reproducible(store_with_players):
    """Identical inputs must give an identical shift, every time.

    The baseline is the team's most recent line-ups ordered by `known_at`, and
    every fixture on a matchday shares a kickoff. Without a tiebreaker the set
    of "recent" matches was decided by whatever row order the database returned,
    so the shift -- and every price built on it -- differed between runs on the
    same data. A model that is not reproducible cannot be backtested.
    """
    store = store_with_players
    as_of = datetime(2024, 1, 5)
    rates = store.player_rates_as_of(as_of, min_minutes=0.0)
    lineup = predict_lineup(store, "bayern_munich", as_of)

    shifts = [
        lineup_strength_shift(store, "bayern_munich", as_of, rates, lineup)["attack_shift"]
        for _ in range(5)
    ]
    assert len(set(shifts)) == 1


def test_passing_a_match_id_does_not_change_an_unconfirmed_prediction(store_with_players):
    """With no confirmed XI published, both call shapes must agree.

    This is what the matchday 'before' price relies on: the only thing allowed
    to differ between the two prices is the team news itself.
    """
    store = store_with_players
    as_of = datetime(2024, 1, 5)
    fixture = store.fixtures_between(as_of, as_of + timedelta(days=30)).iloc[0]
    rates = store.player_rates_as_of(as_of, min_minutes=0.0)
    team = fixture["home_team_id"]

    without = predict_lineup(store, team, as_of, match_id=None, use_confirmed=False)
    with_id = predict_lineup(store, team, as_of, match_id=fixture["match_id"],
                             use_confirmed=True)

    assert not with_id.is_confirmed
    assert set(without.starters) == set(with_id.starters)
    assert (lineup_strength_shift(store, team, as_of, rates, without)["attack_shift"]
            == lineup_strength_shift(store, team, as_of, rates, with_id)["attack_shift"])
