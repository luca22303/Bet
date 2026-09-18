"""Scoring rules for 1X2 forecasts.

The headline metric is the ranked probability score. Football outcomes are
ordered (home win, draw, away win): a model that predicts an away win when the
home side wins is more wrong than one that predicted a draw. Log loss and Brier
treat those errors as identical; RPS does not, which is why it is the standard
for 1X2 forecast comparison (Constantinou & Fenton, 2012).

All three are reported, because they fail differently. Log loss is unbounded and
one confident miss dominates it. Brier is bounded and forgiving. Disagreement
between them is informative, so none of them is dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bet.config import OUTCOMES

_OUTCOME_INDEX = {o: i for i, o in enumerate(OUTCOMES)}


def _prepare(probs, outcomes) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probs, dtype=float)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    if p.shape[1] != len(OUTCOMES):
        raise ValueError(f"expected {len(OUTCOMES)} columns ordered {OUTCOMES}, got shape {p.shape}")

    if np.any(p < -1e-9):
        raise ValueError("probabilities must be non-negative")
    row_sums = p.sum(axis=1)
    if np.any(np.abs(row_sums - 1.0) > 1e-6):
        raise ValueError(f"probability rows must sum to 1, worst row sums to {row_sums[np.argmax(np.abs(row_sums - 1))]:.6f}")

    outcomes = np.asarray(outcomes)
    if outcomes.ndim == 0:
        outcomes = outcomes.reshape(1)
    try:
        idx = np.array([_OUTCOME_INDEX[str(o)] for o in outcomes], dtype=int)
    except KeyError as exc:
        raise ValueError(f"outcome must be one of {OUTCOMES}, got {exc}") from exc

    if idx.size != p.shape[0]:
        raise ValueError(f"got {p.shape[0]} probability rows but {idx.size} outcomes")
    return p, idx


def ranked_probability_score(probs, outcomes) -> float:
    """Mean RPS. Lower is better; 0 is perfect.

    RPS = 1/(r-1) * sum_{i=1}^{r-1} (sum_{j<=i} p_j - sum_{j<=i} e_j)^2
    """
    p, idx = _prepare(probs, outcomes)
    observed = np.zeros_like(p)
    observed[np.arange(p.shape[0]), idx] = 1.0
    cum_diff = np.cumsum(p, axis=1) - np.cumsum(observed, axis=1)
    # The final cumulative difference is always zero, so it is excluded.
    return float(np.mean(np.sum(cum_diff[:, :-1] ** 2, axis=1) / (p.shape[1] - 1)))


def log_loss(probs, outcomes, eps: float = 1e-15) -> float:
    """Mean negative log likelihood of the realised outcomes."""
    p, idx = _prepare(probs, outcomes)
    picked = np.clip(p[np.arange(p.shape[0]), idx], eps, 1.0)
    return float(-np.mean(np.log(picked)))


def brier_score(probs, outcomes) -> float:
    """Multi-class Brier score (mean squared error over the full vector)."""
    p, idx = _prepare(probs, outcomes)
    observed = np.zeros_like(p)
    observed[np.arange(p.shape[0]), idx] = 1.0
    return float(np.mean(np.sum((p - observed) ** 2, axis=1)))


def accuracy(probs, outcomes) -> float:
    """Share of matches where the highest-probability outcome occurred.

    Reported only for intuition. Never optimise it: a model can raise accuracy
    by getting more confident while becoming worse calibrated, and calibration
    is what decides whether a bet is profitable.
    """
    p, idx = _prepare(probs, outcomes)
    return float(np.mean(np.argmax(p, axis=1) == idx))


def score_predictions(probs, outcomes) -> dict[str, float]:
    p, idx = _prepare(probs, outcomes)
    return {
        "n": int(p.shape[0]),
        "rps": ranked_probability_score(p, [OUTCOMES[i] for i in idx]),
        "log_loss": log_loss(p, [OUTCOMES[i] for i in idx]),
        "brier": brier_score(p, [OUTCOMES[i] for i in idx]),
        "accuracy": accuracy(p, [OUTCOMES[i] for i in idx]),
    }


def calibration_table(probs, outcomes, bins: int = 10) -> pd.DataFrame:
    """Predicted vs realised frequency, pooled across all three outcomes.

    A model can have a good RPS and still be miscalibrated in the region you
    actually bet. Read this before believing any expected-value number: if the
    0.6-0.7 bucket realises at 0.52, every 'value' bet in that band is a loser.
    """
    p, idx = _prepare(probs, outcomes)
    observed = np.zeros_like(p)
    observed[np.arange(p.shape[0]), idx] = 1.0

    flat_p = p.ravel()
    flat_o = observed.ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.clip(np.digitize(flat_p, edges, right=False) - 1, 0, bins - 1)

    rows = []
    for b in range(bins):
        mask = bucket == b
        if not mask.any():
            continue
        rows.append({
            "bin_low": edges[b],
            "bin_high": edges[b + 1],
            "n": int(mask.sum()),
            "mean_predicted": float(flat_p[mask].mean()),
            "observed_rate": float(flat_o[mask].mean()),
        })
    table = pd.DataFrame(rows)
    if not table.empty:
        table["gap"] = table["observed_rate"] - table["mean_predicted"]
    return table


def expected_calibration_error(probs, outcomes, bins: int = 10) -> float:
    table = calibration_table(probs, outcomes, bins=bins)
    if table.empty:
        return float("nan")
    weights = table["n"] / table["n"].sum()
    return float((weights * table["gap"].abs()).sum())
