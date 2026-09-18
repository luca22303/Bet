import numpy as np
import pytest

from bet.evaluation.metrics import (
    brier_score,
    calibration_table,
    expected_calibration_error,
    log_loss,
    ranked_probability_score,
    score_predictions,
)


def test_perfect_forecast_scores_zero():
    assert ranked_probability_score([[1.0, 0.0, 0.0]], ["H"]) == pytest.approx(0.0)
    assert brier_score([[1.0, 0.0, 0.0]], ["H"]) == pytest.approx(0.0)


def test_rps_respects_outcome_ordering():
    """The property that makes RPS the right metric for 1X2.

    Predicting an away win when the home side won is a worse error than
    predicting a draw. Log loss and Brier call these two identical; RPS does not.
    """
    predicted_draw = ranked_probability_score([[0.0, 1.0, 0.0]], ["H"])
    predicted_away = ranked_probability_score([[0.0, 0.0, 1.0]], ["H"])
    assert predicted_draw < predicted_away
    assert predicted_draw == pytest.approx(0.5)
    assert predicted_away == pytest.approx(1.0)

    # Brier, by contrast, cannot tell them apart.
    assert brier_score([[0.0, 1.0, 0.0]], ["H"]) == pytest.approx(
        brier_score([[0.0, 0.0, 1.0]], ["H"]))


def test_log_loss_punishes_confident_errors():
    assert log_loss([[0.99, 0.005, 0.005]], ["A"]) > log_loss([[0.4, 0.3, 0.3]], ["A"])


def test_probabilities_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        ranked_probability_score([[0.5, 0.3, 0.1]], ["H"])


def test_negative_probabilities_rejected():
    with pytest.raises(ValueError):
        ranked_probability_score([[1.2, -0.1, -0.1]], ["H"])


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError, match="outcomes"):
        ranked_probability_score([[0.4, 0.3, 0.3], [0.4, 0.3, 0.3]], ["H"])


def test_calibration_detects_a_miscalibrated_model():
    """A model claiming 80% that lands 50% of the time must be flagged.

    This is the failure mode that produces confident, losing bets while the
    aggregate scores still look acceptable.
    """
    rng = np.random.default_rng(7)
    n = 2000
    outcomes = ["H" if u < 0.5 else "A" for u in rng.random(n)]
    probs = np.tile([0.8, 0.1, 0.1], (n, 1))

    table = calibration_table(probs, outcomes)
    overconfident = table[(table["bin_low"] <= 0.8) & (table["bin_high"] > 0.8)]
    assert not overconfident.empty
    assert overconfident["gap"].iloc[0] < -0.2          # realises far below claim
    assert expected_calibration_error(probs, outcomes) > 0.1


def test_well_calibrated_model_has_small_error():
    rng = np.random.default_rng(11)
    n = 5000
    p = np.tile([0.5, 0.25, 0.25], (n, 1))
    draws = rng.random(n)
    outcomes = ["H" if u < 0.5 else ("D" if u < 0.75 else "A") for u in draws]
    assert expected_calibration_error(p, outcomes) < 0.03


def test_score_predictions_reports_every_metric():
    keys = score_predictions([[0.5, 0.3, 0.2]], ["H"]).keys()
    assert {"n", "rps", "log_loss", "brier", "accuracy"} <= set(keys)
