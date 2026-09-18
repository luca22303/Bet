"""Player prop modelling.

The generator uses fixed per-90 rates (a striker at 3.4 shots, a centre-back at
0.35), so the model can be checked for recovering them rather than merely for
returning a number.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from bet.models.props import (
    PlayerPropModel,
    SUPPORTED_STATS,
    estimate_priors,
    negative_binomial_over,
    poisson_over,
    shrink_rate,
)


# ------------------------------------------------------------------ shrinkage


def test_no_evidence_returns_the_prior():
    assert shrink_rate(0.0, 0.0, prior_rate=2.0, prior_strength=5.0) == pytest.approx(2.0)


def test_shrinkage_moves_toward_the_observed_rate_as_evidence_accumulates():
    """A striker with one good match must not be priced as a 5-shots-a-game player."""
    steps = [shrink_rate(5.0 * n, n, prior_rate=2.0, prior_strength=5.0)
             for n in (1, 3, 10, 30, 100)]
    assert steps == sorted(steps)
    assert steps[0] < 3.0        # one ninety is barely trusted
    assert steps[-1] > 4.7       # a hundred is trusted almost fully


def test_shrinkage_never_overshoots_the_observed_rate():
    for n in (1, 5, 20, 100):
        assert shrink_rate(5.0 * n, n, prior_rate=2.0, prior_strength=5.0) < 5.0


# --------------------------------------------------------------- distribution


def test_overdispersion_raises_the_upper_tail_and_lowers_the_middle():
    """Overdispersion moves mass out of the middle into both tails.

    Which way it moves a price therefore depends on where the line sits, and the
    upper-tail divergence is large: at an expected 2.0, over 5.5 is three times
    likelier under a negative binomial than under Poisson.
    """
    expected, r = 2.0, 3.0
    assert negative_binomial_over(expected, r, 1.5) < poisson_over(expected, 1.5)
    assert negative_binomial_over(expected, r, 5.5) > poisson_over(expected, 5.5)
    assert negative_binomial_over(expected, r, 5.5) > 2.5 * poisson_over(expected, 5.5)


def test_high_dispersion_converges_to_poisson():
    for line in (0.5, 1.5, 3.5):
        assert negative_binomial_over(2.0, 5000.0, line) == pytest.approx(
            poisson_over(2.0, line), abs=1e-3)


def test_probabilities_are_monotone_in_the_line():
    probs = [negative_binomial_over(2.0, 4.0, line) for line in (0.5, 1.5, 2.5, 3.5, 4.5)]
    assert probs == sorted(probs, reverse=True)


def test_zero_expectation_is_handled():
    assert negative_binomial_over(0.0, 4.0, 1.5) == 0.0


# ------------------------------------------------------------------- priors


def test_priors_are_estimated_per_position_group(store_with_players):
    rates = store_with_players.player_rates_as_of(datetime(2024, 1, 1), min_minutes=0.0)
    priors = estimate_priors(rates, "shots")

    assert priors.league_mean > 0
    assert priors.dispersion > 0
    # Forwards shoot more than defenders; if the grouping does not show that,
    # positions are not being read correctly.
    assert priors.group_means["forward"] > priors.group_means["defender"]
    assert priors.group_means["defender"] > priors.group_means["goalkeeper"]


def test_priors_degrade_gracefully_on_empty_input():
    priors = estimate_priors(pd.DataFrame(), "shots")
    assert priors.league_mean == 0.0


# -------------------------------------------------------------------- model


def test_model_recovers_known_shot_rates(store_with_players):
    """Strikers are generated at 3.4 shots per 90 and centre-backs at 0.35."""
    model = PlayerPropModel(stat="shots").fit(store_with_players, datetime(2024, 1, 1))

    striker = model.predict("bayern_munich_st", minutes_override=90.0)
    defender = model.predict("bayern_munich_cb1", minutes_override=90.0)

    assert striker.rate_per90 > defender.rate_per90
    assert 2.5 < striker.rate_per90 < 4.3
    assert defender.rate_per90 < 1.2


def test_expected_count_scales_with_minutes(store_with_players):
    model = PlayerPropModel(stat="shots").fit(store_with_players, datetime(2024, 1, 1))
    full = model.predict("bayern_munich_st", minutes_override=90.0)
    half = model.predict("bayern_munich_st", minutes_override=45.0)
    assert half.expected_count == pytest.approx(full.expected_count / 2, rel=1e-6)


def test_confirmed_starter_raises_expected_minutes(store_with_players):
    """The confirmed XI collapses the biggest source of uncertainty in a prop."""
    model = PlayerPropModel(stat="shots").fit(store_with_players, datetime(2024, 1, 1))
    starting = model.expected_minutes("bayern_munich_sub2", is_starter=True)
    benched = model.expected_minutes("bayern_munich_sub2", is_starter=False)
    unknown = model.expected_minutes("bayern_munich_sub2")
    assert starting > unknown > benched


def test_team_volume_raises_expected_count(store_with_players):
    """A side expected to score more takes more shots, but sub-linearly."""
    model = PlayerPropModel(stat="shots").fit(store_with_players, datetime(2024, 1, 1))
    base = model.predict("bayern_munich_st", minutes_override=90.0, team_expected_goals=1.5)
    high = model.predict("bayern_munich_st", minutes_override=90.0, team_expected_goals=3.0)
    assert high.expected_count > base.expected_count
    # Elasticity below one: doubling expected goals must not double the shots.
    assert high.expected_count < 2 * base.expected_count


def test_probabilities_are_coherent(store_with_players):
    model = PlayerPropModel(stat="shots").fit(store_with_players, datetime(2024, 1, 1))
    prediction = model.predict("bayern_munich_st", minutes_override=90.0)
    assert prediction.p_over + prediction.p_under == pytest.approx(1.0)
    assert prediction.fair_odds_over > 1.0
    assert 0.0 < prediction.p_over < 1.0


def test_unknown_player_falls_back_to_the_prior(store_with_players):
    """A new signing has no history; the model must price him, not crash."""
    model = PlayerPropModel(stat="shots").fit(store_with_players, datetime(2024, 1, 1))
    prediction = model.predict("a_player_who_never_played", minutes_override=90.0)
    assert prediction.sample_90s == 0.0
    assert prediction.rate_per90 == pytest.approx(model.priors.league_mean)


def test_opponent_factor_is_shrunk_toward_one(store_with_players):
    """Ten matches of defensive data is a small sample.

    Without shrinkage the opponent adjustment swings prop prices far harder than
    the evidence supports.
    """
    model = PlayerPropModel(stat="shots").fit(store_with_players, datetime(2024, 1, 1))
    factors = np.array(list(model.opponent_factors.values()))
    assert len(factors) > 0
    assert np.all(factors > 0.4)
    assert np.all(factors < 1.8)


def test_match_pricing_covers_both_squads(store_with_players):
    store = store_with_players
    as_of = datetime(2024, 1, 1)
    model = PlayerPropModel(stat="shots").fit(store, as_of)

    row = store.con.execute(
        "SELECT match_id, home_team_id, away_team_id FROM match "
        "WHERE kickoff_utc > ? ORDER BY kickoff_utc LIMIT 1", [as_of]).fetchone()
    match_id, home, away = row

    frame = model.predict_match(store, match_id, home, away, as_of,
                                home_expected_goals=1.8, away_expected_goals=1.1)
    assert not frame.empty
    assert set(frame["team_id"]) == {home, away}
    assert frame["expected"].is_monotonic_decreasing
    assert (frame["p_over"].between(0, 1)).all()


def test_point_in_time_is_respected(store_with_players):
    """A model fitted early must see less evidence than one fitted late."""
    store = store_with_players
    early = PlayerPropModel(stat="shots").fit(store, datetime(2022, 1, 1))
    late = PlayerPropModel(stat="shots").fit(store, datetime(2024, 1, 1))
    assert early.predict("bayern_munich_st").sample_90s < late.predict("bayern_munich_st").sample_90s


@pytest.mark.parametrize("stat", sorted(SUPPORTED_STATS))
def test_every_supported_stat_prices(store_with_players, stat):
    model = PlayerPropModel(stat=stat).fit(store_with_players, datetime(2024, 1, 1))
    prediction = model.predict("bayern_munich_st", minutes_override=90.0)
    assert 0.0 <= prediction.p_over <= 1.0


def test_unsupported_stat_rejected():
    with pytest.raises(ValueError, match="unsupported stat"):
        PlayerPropModel(stat="corners_won_while_facing_north")
