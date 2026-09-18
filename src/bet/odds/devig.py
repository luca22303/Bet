"""Turning quoted odds into probabilities.

Raw inverse odds sum to more than one; the excess is the bookmaker's margin.
Removing it is not a detail. The four standard methods disagree by several
percentage points on longshots, and that disagreement is the same size as any
edge you are likely to find, so the choice of method can manufacture or erase
an apparent edge on its own.

Multiplicative is the default in most tutorials and the worst of the four: it
removes margin proportionally, which understates favourites and overstates
longshots, because books load margin disproportionately onto longshots
(the favourite-longshot bias). Shin and power both model that loading. Prefer
Shin, and never compare a model against a multiplicatively devigged line
without checking the conclusion survives under Shin.
"""

from __future__ import annotations

from enum import Enum

import numpy as np


class DevigMethod(str, Enum):
    MULTIPLICATIVE = "multiplicative"
    ADDITIVE = "additive"
    POWER = "power"
    SHIN = "shin"


def _as_array(odds) -> np.ndarray:
    arr = np.asarray(odds, dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError("expected a 1-D array of at least two decimal odds")
    if np.any(~np.isfinite(arr)) or np.any(arr <= 1.0):
        raise ValueError(f"decimal odds must be finite and > 1.0, got {arr.tolist()}")
    return arr


def booksum(odds) -> float:
    """Sum of inverse odds. 1.0 is a fair book; 1.07 is a 7% overround."""
    return float(np.sum(1.0 / _as_array(odds)))


def overround(odds) -> float:
    """Bookmaker margin as a fraction of the fair book."""
    return booksum(odds) - 1.0


def _multiplicative(inv: np.ndarray) -> np.ndarray:
    return inv / inv.sum()


def _additive(inv: np.ndarray) -> np.ndarray:
    """Spread the margin equally in probability terms across selections."""
    excess = inv.sum() - 1.0
    probs = inv - excess / inv.size
    if np.any(probs <= 0):
        # Happens on lopsided books where a longshot's whole probability is
        # smaller than its equal share of the margin. Fall back rather than
        # returning a negative probability.
        return _multiplicative(inv)
    return probs / probs.sum()


def _power(inv: np.ndarray, tol: float = 1e-12, max_iter: int = 200) -> np.ndarray:
    """Find k with sum(inv_i ** k) == 1.

    Because every inv_i < 1, raising to k > 1 shrinks small probabilities more
    than large ones, which is the favourite-longshot loading we want to undo.
    """
    lo, hi = 1.0, 1.0
    for _ in range(100):
        if np.sum(inv ** hi) <= 1.0:
            break
        hi *= 2.0
    else:  # pragma: no cover - only for pathological books
        return _multiplicative(inv)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        total = np.sum(inv ** mid)
        if abs(total - 1.0) < tol:
            break
        if total > 1.0:
            lo = mid
        else:
            hi = mid
    probs = inv ** (0.5 * (lo + hi))
    return probs / probs.sum()


def _shin(inv: np.ndarray, tol: float = 1e-12, max_iter: int = 200) -> np.ndarray:
    """Shin (1993): margin as compensation for insider trading.

    `z` is the implied proportion of informed money. Solving for the z that
    makes the recovered probabilities sum to one both removes the margin and
    corrects the longshot bias, which is why this is the default.
    """
    total_inv = inv.sum()

    def probs_for(z: float) -> np.ndarray:
        if z <= 0.0:
            return inv / np.sqrt(total_inv)
        disc = z * z + 4.0 * (1.0 - z) * (inv * inv) / total_inv
        return (np.sqrt(disc) - z) / (2.0 * (1.0 - z))

    lo, hi = 0.0, 0.99
    if probs_for(lo).sum() <= 1.0:
        return _multiplicative(inv)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        total = probs_for(mid).sum()
        if abs(total - 1.0) < tol:
            break
        if total > 1.0:
            lo = mid
        else:
            hi = mid

    probs = probs_for(0.5 * (lo + hi))
    probs = np.clip(probs, 1e-12, None)
    return probs / probs.sum()


_METHODS = {
    DevigMethod.MULTIPLICATIVE: _multiplicative,
    DevigMethod.ADDITIVE: _additive,
    DevigMethod.POWER: _power,
    DevigMethod.SHIN: _shin,
}


def devig(odds, method: DevigMethod | str = DevigMethod.SHIN) -> np.ndarray:
    """Convert decimal odds into probabilities summing to one."""
    inv = 1.0 / _as_array(odds)
    if inv.sum() <= 1.0:
        # No margin to remove (an exchange, or arbitrage across books).
        return inv / inv.sum()
    return _METHODS[DevigMethod(method)](inv)


def devig_frame(frame, odds_col: str = "decimal_odds", group_cols=("match_id", "book"),
                selection_col: str = "selection", method: DevigMethod | str = DevigMethod.SHIN):
    """Devig a long-format odds table, one book and market at a time."""
    import pandas as pd

    out = []
    for keys, group in frame.groupby(list(group_cols), sort=False):
        group = group.sort_values(selection_col)
        try:
            probs = devig(group[odds_col].to_numpy(), method=method)
        except ValueError:
            continue
        block = group.copy()
        block["prob"] = probs
        block["overround"] = overround(group[odds_col].to_numpy())
        out.append(block)
    return pd.concat(out, ignore_index=True) if out else frame.assign(prob=[], overround=[])
