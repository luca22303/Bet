"""Dixon-Coles correctness, including the failure modes the paper leaves implicit."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from bet.models.dixon_coles import (
    DixonColesModel,
    both_teams_to_score,
    fit_dixon_coles,
    fit_log_rates,
    match_probabilities,
    outcome_probabilities,
    over_under,
    rho_bounds,
    score_matrix,
    tau,
    time_weights,
)
from conftest import TRUE_ATTACK, TRUE_DEFENCE, TRUE_HOME_ADVANTAGE, generate_season


@pytest.fixture(scope="module")
def training_frame():
    rng = np.random.default_rng(1)
    frames = []
    for offset in range(6):
        matches, results, _ = generate_season(
            datetime(2019 + offset, 8, 10, 15, 30), f"{2019 + offset}", rng)
        frames.append(matches.merge(results[["match_id", "home_goals", "away_goals"]],
                                    on="match_id"))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def fitted(training_frame):
    return fit_dixon_coles(training_frame, as_of=datetime(2026, 1, 1), xi=0.0)


# ---------------------------------------------------------------- score matrix


def test_score_matrix_is_a_distribution():
    matrix = score_matrix(1.6, 1.1, -0.13)
    assert matrix.sum() == pytest.approx(1.0)
    assert np.all(matrix >= 0)


def test_negative_rho_inflates_draws():
    """The entire point of the tau correction.

    Independent Poissons underpredict 0-0 and 1-1. A negative rho corrects that,
    and if it does not, the correction is wired up backwards.
    """
    corrected = outcome_probabilities(score_matrix(1.6, 1.1, -0.13))
    uncorrected = outcome_probabilities(score_matrix(1.6, 1.1, 0.0))
    assert corrected[1] > uncorrected[1]


def test_tau_only_touches_the_four_low_score_cells():
    lam, mu, rho = 1.5, 1.2, -0.1
    for x, y in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        assert tau(x, y, lam, mu, rho) != pytest.approx(1.0)
    for x, y in [(2, 0), (0, 2), (2, 2), (3, 1), (1, 3)]:
        assert tau(x, y, lam, mu, rho) == pytest.approx(1.0)


def test_rho_bounds_keep_every_corrected_cell_positive():
    lam, mu = 1.6, 1.1
    lower, upper = rho_bounds(lam, mu)
    for rho in (lower + 1e-6, 0.0, upper - 1e-6):
        for x, y in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            assert tau(x, y, lam, mu, rho) > 0


def test_extreme_rho_is_clipped_rather_than_producing_negative_probabilities():
    """An unbounded rho can make the (0,0) cell negative. It must not escape."""
    matrix = score_matrix(1.6, 1.1, rho=0.95)
    assert np.all(matrix >= 0)
    assert matrix.sum() == pytest.approx(1.0)


def test_stronger_home_side_gets_a_higher_win_probability():
    strong = match_probabilities(2.4, 0.7, -0.1)
    even = match_probabilities(1.3, 1.3, -0.1)
    assert strong[0] > even[0]
    assert strong[2] < even[2]


def test_derived_markets_are_consistent_with_the_matrix():
    matrix = score_matrix(1.6, 1.1, -0.13)
    over, under = over_under(matrix, 2.5)
    assert over + under == pytest.approx(1.0)
    assert 0.0 < both_teams_to_score(matrix) < 1.0


# ------------------------------------------------------------------- fitting


def test_fit_recovers_the_true_home_advantage(fitted):
    assert fitted.home_advantage == pytest.approx(TRUE_HOME_ADVANTAGE, abs=0.06)


def test_fit_recovers_the_true_team_strengths(fitted):
    """Synthetic seasons are generated from known strengths, so recovery is checkable."""
    true_attack = np.array([TRUE_ATTACK[t] for t in fitted.teams])
    true_attack -= true_attack.mean()          # match the sum-to-zero constraint
    fitted_attack = np.array([fitted.attack[t] for t in fitted.teams])
    assert np.corrcoef(true_attack, fitted_attack)[0, 1] > 0.93

    true_defence = np.array([TRUE_DEFENCE[t] for t in fitted.teams])
    true_defence -= true_defence.mean()
    fitted_defence = np.array([fitted.defence[t] for t in fitted.teams])
    assert np.corrcoef(true_defence, fitted_defence)[0, 1] > 0.93


def test_parameters_satisfy_both_sum_to_zero_constraints(fitted):
    """The identifiability fix.

    Constraining attack alone leaves a flat ridge: adding k to every attack and
    subtracting k from every defence changes neither rate. Both must be pinned.
    """
    assert sum(fitted.attack.values()) == pytest.approx(0.0, abs=1e-9)
    assert sum(fitted.defence.values()) == pytest.approx(0.0, abs=1e-9)


def test_refitting_the_same_data_gives_the_same_parameters(training_frame, fitted):
    """What the ridge would break.

    On an unidentified likelihood the optimiser stops at an arbitrary point
    along the flat direction, so parameters drift between refits even though the
    data has not changed.
    """
    shuffled = training_frame.sample(frac=1.0, random_state=9)
    refit = fit_dixon_coles(shuffled, as_of=datetime(2026, 1, 1), xi=0.0)
    drift = max(abs(fitted.attack[t] - refit.attack[t]) for t in fitted.teams)
    assert drift < 1e-4


def test_rates_are_plausible_football_scores(fitted):
    lam, mu = fitted.rates("bayern_munich", "1_fc_heidenheim")
    assert 0.5 < lam < 6.0
    assert 0.1 < mu < 3.0
    assert lam > mu


def test_rho_is_near_zero_when_goals_are_genuinely_independent(fitted):
    """The synthetic generator uses independent Poissons, so there is no
    low-score dependence to find. A model reporting a large rho here would be
    fitting noise."""
    assert abs(fitted.rho) < 0.1


def test_empty_training_data_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        fit_dixon_coles(pd.DataFrame(columns=["home_team_id", "away_team_id", "home_goals",
                                              "away_goals", "kickoff_utc"]),
                        as_of=datetime(2024, 1, 1))


def test_missing_columns_are_rejected():
    with pytest.raises(ValueError, match="missing columns"):
        fit_dixon_coles(pd.DataFrame({"home_team_id": ["a"]}), as_of=datetime(2024, 1, 1))


# ------------------------------------------------------------- time weighting


def test_time_weights_decay_with_age():
    kickoffs = pd.Series([datetime(2024, 1, 1), datetime(2024, 6, 1), datetime(2024, 12, 1)])
    weights = time_weights(kickoffs, datetime(2025, 1, 1), xi=0.0018)
    assert weights[0] < weights[1] < weights[2]
    assert np.all(weights > 0)


def test_zero_decay_weights_all_history_equally():
    kickoffs = pd.Series([datetime(2020, 1, 1), datetime(2024, 1, 1)])
    assert time_weights(kickoffs, datetime(2025, 1, 1), xi=0.0) == pytest.approx([1.0, 1.0])


def test_decay_shifts_the_fit_toward_recent_form(training_frame):
    slow = fit_dixon_coles(training_frame, as_of=datetime(2026, 1, 1), xi=0.0)
    fast = fit_dixon_coles(training_frame, as_of=datetime(2026, 1, 1), xi=0.01)
    assert any(abs(slow.attack[t] - fast.attack[t]) > 0.01 for t in slow.teams)


# ----------------------------------------------------------------- xG fitting


def test_log_rate_fit_recovers_strengths_from_continuous_targets(training_frame):
    """xG arrives as a continuous value, so it cannot use the Poisson likelihood.

    Weighted least squares on log rates recovers the same structure.
    """
    frame = training_frame.copy()
    rng = np.random.default_rng(3)
    # Stand-in xG: the true rates plus noise, which is what xG effectively is.
    frame["home_xg"] = np.clip(frame["home_goals"] + rng.normal(0, 0.3, len(frame)), 0.05, None)
    frame["away_xg"] = np.clip(frame["away_goals"] + rng.normal(0, 0.3, len(frame)), 0.05, None)

    params = fit_log_rates(frame, as_of=datetime(2026, 1, 1), xi=0.0)
    true_attack = np.array([TRUE_ATTACK[t] for t in params.teams])
    true_attack -= true_attack.mean()
    fitted_attack = np.array([params.attack[t] for t in params.teams])
    assert np.corrcoef(true_attack, fitted_attack)[0, 1] > 0.85
    assert sum(params.attack.values()) == pytest.approx(0.0, abs=1e-8)


# ------------------------------------------------------------------ the model


def test_model_produces_valid_probabilities(populated_store):
    model = DixonColesModel(xi=0.0).fit(populated_store, datetime(2023, 1, 1))
    fixtures = populated_store.fixtures_between(datetime(2023, 1, 1), datetime(2023, 3, 1))
    probs = model.predict(populated_store, fixtures, datetime(2023, 1, 1))
    assert probs.shape == (len(fixtures), 3)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_model_falls_back_gracefully_without_enough_history(populated_store):
    model = DixonColesModel(min_training_matches=100).fit(populated_store, datetime(2019, 8, 12))
    fixtures = populated_store.fixtures_between(datetime(2019, 8, 12), datetime(2019, 9, 1))
    probs = model.predict(populated_store, fixtures, datetime(2019, 8, 12))
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_model_beats_elo_on_data_generated_from_team_strengths(populated_store):
    from bet.evaluation.backtest import walk_forward
    from bet.models.baselines import EloModel

    start = datetime(2022, 1, 1)
    dc = walk_forward(populated_store, DixonColesModel(xi=0.0), start=start)
    elo = walk_forward(populated_store, EloModel(), start=start)
    assert dc.metrics["rps"] < elo.metrics["rps"]


def test_derived_markets_are_exposed(populated_store):
    model = DixonColesModel(xi=0.0).fit(populated_store, datetime(2023, 1, 1))
    markets = model.predict_markets("bayern_munich", "fc_augsburg")
    assert markets["p_h"] + markets["p_d"] + markets["p_a"] == pytest.approx(1.0)
    assert markets["p_over_2_5"] + markets["p_under_2_5"] == pytest.approx(1.0)
    assert markets["expected_home_goals"] > markets["expected_away_goals"]


def test_scoreline_table_is_a_distribution(populated_store):
    model = DixonColesModel(xi=0.0).fit(populated_store, datetime(2023, 1, 1))
    table = model.predict_scorelines("bayern_munich", "fc_augsburg")
    assert table.to_numpy().sum() == pytest.approx(1.0)


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError, match="target"):
        DixonColesModel(target="nonsense")
    with pytest.raises(ValueError, match="blend_weight"):
        DixonColesModel(blend_weight=1.5)
