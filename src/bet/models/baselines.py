"""Baselines. A forecast is only interesting relative to these.

Three of them, in ascending order of difficulty:

  HomePriorModel  the base rates. Beating it proves nothing but is a floor;
                  failing to beat it means something is broken.
  EloModel        a respectable public power rating. A real model should win.
  MarketModel     devigged closing prices. This is the one that matters. If a
                  model cannot beat the devigged close out of sample, it has no
                  betting edge, whatever its expected-value report claims.

MarketModel deliberately reads closing odds, which are not knowable before
kickoff. It is a scoring benchmark and must never be used to generate a signal.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from bet.config import OUTCOMES
from bet.models.base import Model, Prediction
from bet.odds.devig import DevigMethod, devig


class HomePriorModel(Model):
    """Long-run base rates of home win / draw / away win, ignoring the teams."""

    name = "home_prior"

    def __init__(self) -> None:
        self.rates = np.array([0.45, 0.25, 0.30])

    def fit(self, store, as_of: datetime) -> "HomePriorModel":
        history = store.matches_as_of(as_of)
        if len(history) >= 50:
            counts = history["outcome"].value_counts()
            total = counts.sum()
            self.rates = np.array([counts.get(o, 0) / total for o in OUTCOMES], dtype=float)
        return self

    def predict(self, store, fixtures: pd.DataFrame, as_of: datetime) -> Prediction:
        return self._validate(np.tile(self.rates, (len(fixtures), 1)), len(fixtures))


class EloModel(Model):
    """Elo ratings fitted on results, with a draw model and home advantage.

    Self-contained rather than reading ClubElo, so the backtest has a baseline
    even before any external rating feed is ingested. `EloModel(source=...)`
    would be the variant that reads `team_rating`; this one learns from results
    the store already holds.

    The draw probability uses the standard logistic-in-rating-gap shape: draws
    peak when sides are evenly matched and decay as the gap widens.
    """

    name = "elo"

    def __init__(self, k: float = 20.0, home_advantage: float = 65.0,
                 draw_base: float = 0.28, draw_decay: float = 0.0016,
                 initial: float = 1500.0) -> None:
        self.k = k
        self.home_advantage = home_advantage
        self.draw_base = draw_base
        self.draw_decay = draw_decay
        self.initial = initial
        self.ratings: dict[str, float] = {}

    def fit(self, store, as_of: datetime) -> "EloModel":
        # The walk itself lives in bet.ratings, which is also what writes the
        # derived ratings into the store. Two copies of "our Elo" would drift
        # apart and nobody would notice until a backtest disagreed with a
        # dashboard.
        from bet.ratings import final_elo

        self.ratings = final_elo(
            store.matches_as_of(as_of), k=self.k,
            home_advantage=self.home_advantage, initial=self.initial)
        return self

    def predict(self, store, fixtures: pd.DataFrame, as_of: datetime) -> Prediction:
        out = np.zeros((len(fixtures), 3))
        for i, row in enumerate(fixtures.itertuples(index=False)):
            home = self.ratings.get(row.home_team_id, self.initial)
            away = self.ratings.get(row.away_team_id, self.initial)
            gap = home + self.home_advantage - away

            expected = 1.0 / (1.0 + 10 ** (-gap / 400.0))
            p_draw = self.draw_base * np.exp(-self.draw_decay * abs(gap))
            # Split the non-draw mass in proportion to the Elo expectation.
            p_home = (1.0 - p_draw) * expected
            p_away = (1.0 - p_draw) * (1.0 - expected)
            out[i] = (p_home, p_draw, p_away)
        return self._validate(out, len(fixtures))


class MarketModel(Model):
    """Devigged bookmaker prices. The benchmark that decides viability.

    Reads closing odds by default, which no pre-match model can see. That is
    intentional: it is the hardest honest benchmark. `use_closing=False` reads
    the latest price knowable at `as_of` instead, which is what a real-time
    system would face.
    """

    name = "market"

    def __init__(self, book: str | None = None, method: DevigMethod | str = DevigMethod.SHIN,
                 use_closing: bool = True, fallback: np.ndarray | None = None) -> None:
        self.book = book
        self.method = method
        self.use_closing = use_closing
        self.fallback = fallback if fallback is not None else np.array([0.45, 0.25, 0.30])

    def fit(self, store, as_of: datetime) -> "MarketModel":
        return self  # nothing to estimate

    def predict(self, store, fixtures: pd.DataFrame, as_of: datetime) -> Prediction:
        match_ids = fixtures["match_id"].tolist()
        if self.use_closing:
            odds = store.closing_odds(match_ids, book=self.book)
        else:
            odds = store.odds_as_of(as_of, match_ids, book=self.book)

        out = np.tile(self.fallback, (len(fixtures), 1))
        if odds.empty:
            return self._validate(out, len(fixtures))

        # Consensus across books: average the inverse odds per selection, which
        # is closer to a fair price than any single fat retail book.
        odds = odds.copy()
        odds["inv"] = 1.0 / odds["decimal_odds"]
        pivot = odds.pivot_table(index="match_id", columns="selection", values="inv", aggfunc="mean")

        for sel in OUTCOMES:
            if sel not in pivot.columns:
                return self._validate(out, len(fixtures))

        index = {mid: i for i, mid in enumerate(match_ids)}
        for match_id, row in pivot.iterrows():
            if match_id not in index:
                continue
            inv = np.array([row[s] for s in OUTCOMES], dtype=float)
            if np.any(~np.isfinite(inv)) or np.any(inv <= 0):
                continue
            out[index[match_id]] = devig(1.0 / inv, method=self.method)

        return self._validate(out, len(fixtures))
