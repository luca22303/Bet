"""The promoted-team prior.

Every August three sides arrive with no Bundesliga record. A results-only model
gives them the league average, which is wrong, and stays wrong into October --
precisely the fixtures where the model will believe it has found the most value.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from bet.models.dixon_coles import DixonColesModel, fit_dixon_coles
from bet.models.promoted import (
    PromotedTeamPrior,
    apply_prior,
    count_appearances,
    fit_promoted_prior,
)
from conftest import TEAMS, generate_season


@pytest.fixture(scope="module")
def fitted_with_ratings():
    rng = np.random.default_rng(5)
    frames = []
    for offset in range(4):
        matches, results, _ = generate_season(
            datetime(2019 + offset, 8, 10, 15, 30), f"{2019 + offset}", rng)
        frames.append(matches.merge(results[["match_id", "home_goals", "away_goals"]],
                                    on="match_id"))
    training = pd.concat(frames, ignore_index=True)
    params = fit_dixon_coles(training, as_of=datetime(2024, 1, 1), xi=0.0)

    # Ratings ordered like the true strengths, as ClubElo's would be.
    ratings = pd.DataFrame({
        "team_id": TEAMS,
        "rating": [1850 - 30 * i for i in range(len(TEAMS))],
    })
    return params, ratings, training


def test_prior_learns_the_rating_to_strength_relationship(fitted_with_ratings):
    params, ratings, training = fitted_with_ratings
    prior = fit_promoted_prior(params, ratings, count_appearances(training))

    assert prior.fitted
    assert prior.n_reference_teams >= 6
    # Higher rating must imply more attack and a better (more negative) defence.
    assert prior.attack_slope > 0
    assert prior.defence_slope < 0


def test_prior_ranks_a_strong_and_weak_promoted_side_correctly(fitted_with_ratings):
    params, ratings, training = fitted_with_ratings
    prior = fit_promoted_prior(params, ratings, count_appearances(training))
    assert prior.attack_for(1800) > prior.attack_for(1400)
    assert prior.defence_for(1800) < prior.defence_for(1400)


def test_prior_needs_enough_reference_teams():
    """Two points define a line but tell you nothing."""
    params = type("P", (), {
        "teams": ("a", "b"),
        "attack": {"a": 0.1, "b": -0.1},
        "defence": {"a": -0.1, "b": 0.1},
    })()
    ratings = pd.DataFrame({"team_id": ["a", "b"], "rating": [1700, 1500]})
    prior = fit_promoted_prior(params, ratings, pd.Series({"a": 100, "b": 100}))
    assert not prior.fitted


def test_unfitted_prior_is_neutral():
    prior = PromotedTeamPrior()
    assert prior.attack_for(1800) == 0.0
    assert prior.defence_for(1200) == 0.0


def test_thin_data_teams_are_pulled_toward_the_prior(fitted_with_ratings):
    params, ratings, training = fitted_with_ratings
    appearances = count_appearances(training)
    prior = fit_promoted_prior(params, ratings, appearances)

    # A team with almost no history: its fitted strength is noise.
    thin = pd.Series(appearances).copy()
    thin["fc_st_pauli"] = 4

    adjusted = apply_prior(params, prior, ratings, thin, full_confidence_matches=60)
    moved = abs(adjusted["attack"]["fc_st_pauli"] - params.attack["fc_st_pauli"])
    assert moved > 1e-6

    # A team with a full record keeps its own estimate.
    assert adjusted["attack"]["bayern_munich"] == pytest.approx(params.attack["bayern_munich"])


def test_blending_is_gradual_not_a_step_change(fitted_with_ratings):
    """A hard threshold would put a visible discontinuity mid-season."""
    params, ratings, training = fitted_with_ratings
    appearances = count_appearances(training)
    prior = fit_promoted_prior(params, ratings, appearances)

    distances = []
    for played in (5, 20, 40, 59):
        counts = pd.Series(appearances).copy()
        counts["fc_st_pauli"] = played
        adjusted = apply_prior(params, prior, ratings, counts, full_confidence_matches=60)
        distances.append(abs(adjusted["attack"]["fc_st_pauli"] - params.attack["fc_st_pauli"]))

    # Influence of the prior must fall monotonically as evidence accumulates.
    assert distances == sorted(distances, reverse=True)


def test_prior_is_a_no_op_without_ratings(fitted_with_ratings):
    params, _, training = fitted_with_ratings
    empty = pd.DataFrame(columns=["team_id", "rating"])
    adjusted = apply_prior(params, PromotedTeamPrior(), empty, count_appearances(training))
    assert adjusted["attack"] == params.attack


def test_model_runs_with_the_prior_enabled_and_no_rating_data(populated_store):
    """The store may have no ClubElo data yet; that must degrade, not crash."""
    model = DixonColesModel(xi=0.0, use_promoted_prior=True)
    model.fit(populated_store, datetime(2023, 1, 1))
    fixtures = populated_store.fixtures_between(datetime(2023, 1, 1), datetime(2023, 2, 1))
    probs = model.predict(populated_store, fixtures, datetime(2023, 1, 1))
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_count_appearances_counts_both_home_and_away():
    matches = pd.DataFrame({
        "home_team_id": ["a", "b", "a"],
        "away_team_id": ["b", "a", "c"],
    })
    counts = count_appearances(matches)
    assert counts["a"] == 3
    assert counts["b"] == 2
    assert counts["c"] == 1
