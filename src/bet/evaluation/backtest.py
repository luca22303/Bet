"""Walk-forward backtesting.

One rule, enforced structurally rather than by discipline: for each matchday the
engine computes `as_of = earliest kickoff - lead time`, refits the model against
the store as it stood at that instant, and only then reveals the fixtures. A
model physically cannot see a result it is about to predict, because the store
filters on `known_at` and the engine never passes it anything else.

Refitting per matchday rather than per match is the usual compromise: fitting
per match is ~9x the work for information that barely changes within a weekend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from bet.config import OUTCOMES, SETTINGS
from bet.evaluation.metrics import calibration_table, expected_calibration_error, score_predictions
from bet.matchdays import group_by_matchday
from bet.models.base import Model


@dataclass
class BacktestResult:
    model_name: str
    predictions: pd.DataFrame
    metrics: dict[str, float] = field(default_factory=dict)
    calibration: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary_row(self) -> dict:
        return {"model": self.model_name, **self.metrics}

    def __str__(self) -> str:
        m = self.metrics
        return (
            f"{self.model_name:<16} n={m.get('n', 0):>5}  "
            f"RPS={m.get('rps', float('nan')):.5f}  "
            f"logloss={m.get('log_loss', float('nan')):.5f}  "
            f"brier={m.get('brier', float('nan')):.5f}  "
            f"ECE={m.get('ece', float('nan')):.5f}"
        )


def _matchday_groups(fixtures: pd.DataFrame, gap_hours: int = 36) -> list[pd.DataFrame]:
    """Split fixtures into refit blocks, breaking wherever a gap appears.

    Bundesliga weekends cluster Friday to Sunday, with midweek rounds appearing
    irregularly, so clustering on gaps is more robust than assuming a calendar.

    Shares its definition of "a matchday" with `bet.matchdays`, which the
    dashboard uses to find the previous and next one to display -- one
    clustering rule, not two that could quietly disagree.
    """
    return group_by_matchday(fixtures, gap_hours=gap_hours)


def walk_forward(store, model: Model, *, start: datetime, end: datetime | None = None,
                 league: str | None = None, lead_seconds: int | None = None,
                 min_training_matches: int = 100, verbose: bool = False) -> BacktestResult:
    """Run `model` forward through time and score it against what happened."""
    end = end or datetime.utcnow()
    lead = timedelta(seconds=lead_seconds if lead_seconds is not None
                     else SETTINGS.prediction_lead_seconds)

    fixtures = store.fixtures_between(start, end, league=league)
    fixtures = fixtures[fixtures["outcome"].notna()].copy()
    if fixtures.empty:
        return BacktestResult(model.name, pd.DataFrame())

    rows: list[pd.DataFrame] = []
    for block in _matchday_groups(fixtures):
        as_of = pd.to_datetime(block["kickoff_utc"]).min().to_pydatetime() - lead

        training = store.matches_as_of(as_of, league=league)
        if len(training) < min_training_matches:
            if verbose:
                print(f"  skip {as_of:%Y-%m-%d}: only {len(training)} training matches")
            continue

        model.fit(store, as_of)
        probs = model.predict(store, block, as_of)

        block = block.copy()
        block["as_of"] = as_of
        block["p_home"] = probs[:, 0]
        block["p_draw"] = probs[:, 1]
        block["p_away"] = probs[:, 2]
        block["model"] = model.name
        rows.append(block)

        if verbose:
            print(f"  {as_of:%Y-%m-%d}: {len(block)} fixtures, trained on {len(training)}")

    if not rows:
        return BacktestResult(model.name, pd.DataFrame())

    predictions = pd.concat(rows, ignore_index=True)
    probs = predictions[["p_home", "p_draw", "p_away"]].to_numpy()
    outcomes = predictions["outcome"].tolist()

    metrics = score_predictions(probs, outcomes)
    metrics["ece"] = expected_calibration_error(probs, outcomes)

    return BacktestResult(
        model_name=model.name,
        predictions=predictions,
        metrics=metrics,
        calibration=calibration_table(probs, outcomes),
    )


def compare(store, models: list[Model], *, start: datetime, end: datetime | None = None,
            league: str | None = None, verbose: bool = False) -> tuple[pd.DataFrame, dict[str, BacktestResult]]:
    """Backtest several models over identical fixtures and rank them by RPS."""
    results: dict[str, BacktestResult] = {}
    for model in models:
        if verbose:
            print(f"backtesting {model.name}...")
        results[model.name] = walk_forward(
            store, model, start=start, end=end, league=league, verbose=verbose
        )

    table = pd.DataFrame([r.summary_row() for r in results.values() if not r.predictions.empty])
    if not table.empty:
        table = table.sort_values("rps").reset_index(drop=True)
        # Skill relative to the market benchmark: positive means better.
        if "market" in table["model"].values:
            market_rps = float(table.loc[table["model"] == "market", "rps"].iloc[0])
            table["rps_skill_vs_market"] = (market_rps - table["rps"]) / market_rps
    return table, results
