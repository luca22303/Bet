"""The contract every forecasting model implements.

Deliberately narrow: `fit` is handed a store and a timestamp, and may read
nothing the store does not consider knowable at that timestamp. `predict` gets
fixtures and returns a probability per outcome. The backtest engine drives both,
so no model can accidentally acquire a wider view of the world than the one it
will have on a Saturday morning.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import pandas as pd

from bet.config import OUTCOMES

Prediction = np.ndarray  # shape (n_fixtures, 3), columns ordered as OUTCOMES


class Model(ABC):
    """Base class for 1X2 forecasters."""

    name: str = "model"

    @abstractmethod
    def fit(self, store, as_of: datetime) -> "Model":
        """Estimate parameters using only facts knowable at `as_of`."""

    @abstractmethod
    def predict(self, store, fixtures: pd.DataFrame, as_of: datetime) -> Prediction:
        """Return an (n, 3) array of probabilities ordered (H, D, A)."""

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _validate(probs: np.ndarray, n: int) -> np.ndarray:
        arr = np.asarray(probs, dtype=float)
        if arr.shape != (n, len(OUTCOMES)):
            raise ValueError(f"{arr.shape} predictions returned, expected {(n, len(OUTCOMES))}")
        if np.any(~np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError("predictions must be finite and non-negative")
        sums = arr.sum(axis=1, keepdims=True)
        if np.any(sums <= 0):
            raise ValueError("a prediction row summed to zero")
        return arr / sums

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
