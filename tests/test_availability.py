"""Absence impact estimation.

The point of these tests: the adjustment must come from the data. A hand-tuned
constant would pass none of them, because they check that the *size* of the
effect tracks the player's actual contribution and the quality of his
replacement.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from bet.availability import (
    MAX_LOG_ADJUSTMENT,
    AvailabilityStatus,
    absences_from_store,
    estimate_absence_impact,
)


@pytest.fixture
def squad_rates():
    """A squad where the striker carries most of the attacking output."""
    return pd.DataFrame([
        {"player_id": "st", "team_id": "t", "position": "FW", "minutes": 2700,
         "xg_p90": 0.75, "xa_p90": 0.20, "tackles_p90": 0.3,
         "interceptions_p90": 0.2, "blocks_p90": 0.1},
        {"player_id": "backup_st", "team_id": "t", "position": "FW", "minutes": 600,
         "xg_p90": 0.30, "xa_p90": 0.10, "tackles_p90": 0.3,
         "interceptions_p90": 0.2, "blocks_p90": 0.1},
        {"player_id": "am", "team_id": "t", "position": "MF", "minutes": 2400,
         "xg_p90": 0.25, "xa_p90": 0.35, "tackles_p90": 1.2,
         "interceptions_p90": 0.8, "blocks_p90": 0.3},
        {"player_id": "cb", "team_id": "t", "position": "DF", "minutes": 2900,
         "xg_p90": 0.05, "xa_p90": 0.02, "tackles_p90": 2.4,
         "interceptions_p90": 1.9, "blocks_p90": 1.1},
        {"player_id": "backup_cb", "team_id": "t", "position": "DF", "minutes": 900,
         "xg_p90": 0.03, "xa_p90": 0.01, "tackles_p90": 1.6,
         "interceptions_p90": 1.2, "blocks_p90": 0.7},
    ])


def test_no_absences_means_no_adjustment(squad_rates):
    impact = estimate_absence_impact(squad_rates, "t", {})
    assert impact.attack_multiplier == 1.0
    assert impact.attack_log_shift == 0.0


def test_losing_the_main_striker_cuts_attack(squad_rates):
    impact = estimate_absence_impact(squad_rates, "t", {"st": AvailabilityStatus.OUT})
    assert impact.attack_multiplier < 1.0
    assert impact.attack_log_shift < 0
    assert "st" in impact.absent_players


def test_impact_scales_with_the_player_lost(squad_rates):
    """A first-choice striker must cost more than a squad midfielder.

    This is what an estimated adjustment buys over a hand-tuned one.
    """
    striker = estimate_absence_impact(squad_rates, "t", {"st": AvailabilityStatus.OUT})
    midfielder = estimate_absence_impact(squad_rates, "t", {"am": AvailabilityStatus.OUT})
    assert striker.attack_multiplier < midfielder.attack_multiplier


def test_a_good_replacement_softens_the_blow(squad_rates):
    """Replacement level is squad-specific, not a league average.

    A club with a strong bench loses less to the same injury, and the model has
    to see that.
    """
    with_backup = estimate_absence_impact(squad_rates, "t", {"st": AvailabilityStatus.OUT})

    weak_bench = squad_rates.copy()
    weak_bench.loc[weak_bench["player_id"] == "backup_st", ["xg_p90", "xa_p90"]] = [0.05, 0.02]
    without_backup = estimate_absence_impact(weak_bench, "t", {"st": AvailabilityStatus.OUT})

    assert without_backup.attack_multiplier < with_backup.attack_multiplier


def test_doubtful_counts_as_a_partial_absence(squad_rates):
    out = estimate_absence_impact(squad_rates, "t", {"st": AvailabilityStatus.OUT})
    doubtful = estimate_absence_impact(squad_rates, "t", {"st": AvailabilityStatus.DOUBTFUL})
    assert out.attack_multiplier < doubtful.attack_multiplier < 1.0


def test_defender_absence_hits_defence_not_attack(squad_rates):
    impact = estimate_absence_impact(squad_rates, "t", {"cb": AvailabilityStatus.OUT})
    assert impact.defence_multiplier < impact.attack_multiplier


def test_adjustment_is_capped(squad_rates):
    """A long injury list must not produce an absurd price."""
    everyone = {pid: AvailabilityStatus.OUT for pid in squad_rates["player_id"]}
    impact = estimate_absence_impact(squad_rates, "t", everyone)
    assert impact.capped
    assert abs(impact.attack_log_shift) <= MAX_LOG_ADJUSTMENT + 1e-9


def test_unknown_player_produces_no_adjustment_and_says_so(squad_rates):
    impact = estimate_absence_impact(squad_rates, "t", {"ghost": AvailabilityStatus.OUT})
    assert impact.attack_multiplier == 1.0
    assert any("ghost" in note for note in impact.notes)


def test_fit_players_are_ignored(squad_rates):
    impact = estimate_absence_impact(squad_rates, "t", {"st": AvailabilityStatus.FIT})
    assert impact.attack_multiplier == 1.0
    assert impact.absent_players == []


def test_empty_squad_data_degrades_quietly(squad_rates):
    impact = estimate_absence_impact(pd.DataFrame(), "t", {"st": AvailabilityStatus.OUT})
    assert impact.attack_multiplier == 1.0


# ------------------------------------------------------------ store integration


def _write_availability(store, player_id, team_id, status, known_at, expected_return=None):
    store.upsert("player_availability", pd.DataFrame([{
        "player_id": player_id, "team_id": team_id, "source": "test",
        "status": status, "reason": "test", "expected_return": expected_return,
        "confidence": 0.9, "known_at": known_at,
    }]), ["player_id", "source", "known_at"])


def test_absences_read_respects_point_in_time(store):
    reported = datetime(2024, 3, 10)
    _write_availability(store, "p1", "t", "OUT", reported)

    assert absences_from_store(store, reported - timedelta(days=1), "t") == {}
    assert absences_from_store(store, reported + timedelta(days=1), "t") == {
        "p1": AvailabilityStatus.OUT}


def test_a_later_report_supersedes_an_earlier_one(store):
    """A Tuesday doubt is irrelevant once Friday says fit."""
    _write_availability(store, "p1", "t", "OUT", datetime(2024, 3, 10))
    _write_availability(store, "p1", "t", "FIT", datetime(2024, 3, 14))
    assert absences_from_store(store, datetime(2024, 3, 15), "t") == {}


def test_stale_reports_are_ignored(store):
    """A six-week-old injury note is not evidence the player is still out."""
    _write_availability(store, "p1", "t", "OUT", datetime(2024, 1, 1))
    assert absences_from_store(store, datetime(2024, 3, 1), "t") == {}


def test_a_passed_return_date_clears_the_absence(store):
    _write_availability(store, "p1", "t", "OUT", datetime(2024, 3, 10),
                        expected_return=datetime(2024, 3, 12).date())
    assert absences_from_store(store, datetime(2024, 3, 14), "t") == {}
    assert "p1" in absences_from_store(store, datetime(2024, 3, 11), "t")


def test_a_fringe_player_absence_barely_moves_the_forecast(squad_rates):
    """The XI is unchanged, so the forecast should be too.

    Without weighting by playing time, ruling out a reserve shifts the model as
    if a starter had gone -- and can even improve the team, because the nominal
    'replacement' is the starter who was already playing.
    """
    starter = estimate_absence_impact(squad_rates, "t", {"st": AvailabilityStatus.OUT})
    fringe = estimate_absence_impact(squad_rates, "t", {"backup_st": AvailabilityStatus.OUT})

    starter_effect = abs(1.0 - starter.attack_multiplier)
    fringe_effect = abs(1.0 - fringe.attack_multiplier)

    assert fringe_effect < 0.03
    # A relative check rather than an absolute one: contributions are shrunk
    # toward the squad mean, so the magnitude depends on how much evidence each
    # player has, while the ordering must hold regardless.
    assert starter_effect > 4 * max(fringe_effect, 0.01)


def test_impact_ranks_players_by_how_much_they_actually_contribute(squad_rates):
    """End-to-end ordering check on the whole squad."""
    ranked = {
        pid: estimate_absence_impact(squad_rates, "t", {pid: AvailabilityStatus.OUT})
        for pid in ("st", "am", "cb", "backup_st")
    }
    # Attack: striker hurts most, then the creative midfielder, then a defender.
    assert (ranked["st"].attack_multiplier
            < ranked["am"].attack_multiplier
            < ranked["cb"].attack_multiplier)
    # Defence: the centre-back hurts most.
    assert ranked["cb"].defence_multiplier < ranked["am"].defence_multiplier
    assert ranked["cb"].defence_multiplier < ranked["st"].defence_multiplier
