from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from bet.evaluation.backtest import compare, walk_forward
from bet.evaluation.clv import clv_report, closing_line_value
from bet.models.base import Model
from bet.models.baselines import EloModel, HomePriorModel, MarketModel


class PeekingModel(Model):
    """A model that tries to cheat, used to prove the engine does not let it."""

    name = "peeker"

    def __init__(self):
        self.seen_outcomes = []

    def fit(self, store, as_of):
        self.training = store.matches_as_of(as_of)
        self.as_of = as_of
        return self

    def predict(self, store, fixtures, as_of):
        # Whatever the model asks for, the store must not reveal a result that
        # had not happened by `as_of`.
        visible = store.matches_as_of(as_of)
        self.seen_outcomes = visible["match_id"].tolist()
        overlap = set(visible["match_id"]) & set(fixtures["match_id"])
        assert not overlap, f"engine leaked {len(overlap)} results being predicted"
        return np.tile([1 / 3, 1 / 3, 1 / 3], (len(fixtures), 1))


def test_engine_never_reveals_the_fixtures_it_is_predicting(populated_store):
    result = walk_forward(populated_store, PeekingModel(), start=datetime(2021, 1, 1))
    assert not result.predictions.empty


def test_walk_forward_produces_scored_predictions(populated_store):
    result = walk_forward(populated_store, EloModel(), start=datetime(2021, 1, 1))
    assert not result.predictions.empty
    assert 0.0 < result.metrics["rps"] < 0.35
    probs = result.predictions[["p_home", "p_draw", "p_away"]].to_numpy()
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_elo_beats_the_base_rate_baseline(populated_store):
    """Sanity: a model that knows which teams are playing should beat one that doesn't.

    The synthetic seasons are generated from fixed, widely spread team strengths,
    so any working model must exploit them.
    """
    start = datetime(2021, 1, 1)
    elo = walk_forward(populated_store, EloModel(), start=start)
    prior = walk_forward(populated_store, HomePriorModel(), start=start)
    assert elo.metrics["rps"] < prior.metrics["rps"]


def test_market_benchmark_is_hard_to_beat(populated_store):
    """The synthetic book prices off the true probabilities, so it should win.

    This mirrors reality closely enough to be the point of the whole harness: if
    your model loses to the market here, it will lose to the market live.
    """
    start = datetime(2021, 1, 1)
    market = walk_forward(populated_store, MarketModel(), start=start)
    elo = walk_forward(populated_store, EloModel(), start=start)
    assert market.metrics["rps"] < elo.metrics["rps"]


def test_compare_ranks_models_and_reports_skill(populated_store):
    table, results = compare(
        populated_store,
        [HomePriorModel(), EloModel(), MarketModel()],
        start=datetime(2021, 1, 1),
    )
    assert list(table["model"])[0] == "market"          # best RPS first
    assert "rps_skill_vs_market" in table.columns
    assert table.loc[table["model"] == "market", "rps_skill_vs_market"].iloc[0] == pytest.approx(0.0)
    assert (table.loc[table["model"] != "market", "rps_skill_vs_market"] < 0).all()


def test_insufficient_history_is_skipped_not_guessed(populated_store):
    # Very early start: nothing to train on, so nothing should be predicted.
    result = walk_forward(populated_store, EloModel(), start=datetime(2019, 8, 1),
                          end=datetime(2019, 9, 1), min_training_matches=100)
    assert result.predictions.empty


def test_clv_is_computed_correctly():
    assert closing_line_value(2.10, 2.00) == pytest.approx(0.05)
    assert closing_line_value(1.90, 2.00) < 0


def test_clv_report_flags_a_bettor_who_beats_the_close():
    bets = pd.DataFrame({
        "decimal_odds": [2.10, 2.05, 2.20, 1.95],
        "closing_odds": [2.00, 2.00, 2.00, 2.00],
    })
    report = clv_report(bets)
    assert report["n"] == 4
    assert report["mean_clv"] > 0
    assert report["beat_close_rate"] == pytest.approx(0.75)


def test_clv_report_flags_a_bettor_who_does_not():
    bets = pd.DataFrame({
        "decimal_odds": [1.90, 1.85, 1.95],
        "closing_odds": [2.00, 2.00, 2.00],
    })
    report = clv_report(bets)
    assert report["mean_clv"] < 0
    assert report["beat_close_rate"] == 0.0
