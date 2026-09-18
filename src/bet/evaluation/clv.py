"""Closing line value.

The primary evidence that a model has an edge, and the reason this module
exists rather than a profit-and-loss chart.

Return on investment over a Bundesliga season is roughly 300 bets. At a true 2%
edge the standard error on ROI over 300 bets is larger than the edge itself, so
a losing season and a winning season are both consistent with having an edge and
with not having one. You would need several seasons to tell them apart.

Closing line value converges far faster. If you consistently take prices better
than the closing line, you are beating the most efficient estimate available,
and profit follows given enough volume. If your CLV is negative, you are losing,
however the season's profit happens to look.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bet.config import OUTCOMES
from bet.odds.devig import DevigMethod, devig


def closing_line_value(bet_odds: float, closing_odds: float) -> float:
    """Fractional advantage of the taken price over the close.

    +0.03 means the price taken was 3% better than the closing price.
    """
    if bet_odds <= 1.0 or closing_odds <= 1.0:
        raise ValueError("decimal odds must exceed 1.0")
    return bet_odds / closing_odds - 1.0


def clv_report(bets: pd.DataFrame, *, odds_col: str = "decimal_odds",
               close_col: str = "closing_odds") -> dict[str, float]:
    """Aggregate CLV across a set of placed bets.

    `beat_close_rate` is the headline. Above 0.5 sustained across a few hundred
    bets is real evidence; a positive ROI with a rate below 0.5 is luck.
    """
    if bets.empty:
        return {"n": 0}

    valid = bets[(bets[odds_col] > 1.0) & (bets[close_col] > 1.0)].copy()
    if valid.empty:
        return {"n": 0}

    valid["clv"] = valid[odds_col] / valid[close_col] - 1.0
    n = len(valid)
    mean_clv = float(valid["clv"].mean())
    std = float(valid["clv"].std(ddof=1)) if n > 1 else float("nan")

    return {
        "n": n,
        "mean_clv": mean_clv,
        "median_clv": float(valid["clv"].median()),
        "beat_close_rate": float((valid["clv"] > 0).mean()),
        "clv_std": std,
        # How many standard errors the mean CLV sits above zero.
        "clv_t_stat": float(mean_clv / (std / np.sqrt(n))) if n > 1 and std > 0 else float("nan"),
    }


def implied_clv_from_probabilities(model_probs: np.ndarray, closing_odds: np.ndarray,
                                   method: DevigMethod | str = DevigMethod.SHIN) -> pd.DataFrame:
    """Compare model probabilities against devigged closing probabilities.

    Usable without placing a single bet: if the model's probabilities are no
    closer to the truth than the closing line's, there is nothing to bet on,
    and this says so before any money moves.
    """
    model_probs = np.atleast_2d(np.asarray(model_probs, dtype=float))
    closing_odds = np.atleast_2d(np.asarray(closing_odds, dtype=float))
    if model_probs.shape != closing_odds.shape:
        raise ValueError("model probabilities and closing odds must have the same shape")

    rows = []
    for probs, odds in zip(model_probs, closing_odds):
        if np.any(odds <= 1.0) or np.any(~np.isfinite(odds)):
            continue
        market = devig(odds, method=method)
        rows.append({
            **{f"model_{o.lower()}": p for o, p in zip(OUTCOMES, probs)},
            **{f"market_{o.lower()}": p for o, p in zip(OUTCOMES, market)},
            "max_abs_divergence": float(np.max(np.abs(probs - market))),
            "total_variation": float(0.5 * np.sum(np.abs(probs - market))),
        })
    return pd.DataFrame(rows)
