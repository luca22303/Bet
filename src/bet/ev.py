"""Expected value and stake sizing, with taxes and margin made explicit.

The textbook formula `EV = p * odds - 1` assumes the price you are quoted is the
price you get. In Germany it is not. The Rennwett- und Lotteriegesetz levies a
5.3% tax on sports betting stakes, and licensed operators pass it on in one of
several ways. A 5.3% stake tax is larger than almost any edge a public-data
model will ever find, so it is a required input here rather than an optional
adjustment: `expected_value` will not let you omit it silently.

Confirm empirically how your operator applies the tax to your account. A
deposit and a single settled bet will tell you, and the answer changes whether
a given bet is +EV at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from bet.config import GERMAN_STAKE_TAX


class TaxMode(str, Enum):
    """How an operator applies betting tax."""

    NONE = "none"
    """Operator absorbs the tax. Verify before assuming this."""

    STAKE_DEDUCTED = "stake_deducted"
    """Tax comes off your stake; only stake*(1-t) is actually wagered."""

    STAKE_ADDED = "stake_added"
    """You pay stake*(1+t) to place a bet of `stake`."""

    WINNINGS = "winnings"
    """Tax is taken from net profit on winning bets only."""


@dataclass(frozen=True)
class BetEvaluation:
    probability: float
    decimal_odds: float
    effective_odds: float
    edge: float
    expected_value: float
    breakeven_probability: float
    kelly_fraction: float
    stake_fraction: float
    is_value: bool

    def describe(self) -> str:
        verdict = "VALUE" if self.is_value else "no bet"
        return (
            f"{verdict}: p={self.probability:.4f} odds={self.decimal_odds:.2f} "
            f"(effective {self.effective_odds:.3f}) EV={self.expected_value:+.4f} "
            f"breakeven p={self.breakeven_probability:.4f} stake={self.stake_fraction:.4%}"
        )


def effective_odds(decimal_odds: float, tax_mode: TaxMode | str = TaxMode.STAKE_DEDUCTED,
                   tax_rate: float = GERMAN_STAKE_TAX) -> float:
    """Decimal odds restated net of tax, per euro you actually part with.

    Reducing every tax treatment to a single effective price keeps the rest of
    the pipeline honest: downstream code compares like with like.
    """
    mode = TaxMode(tax_mode)
    if decimal_odds <= 1.0:
        raise ValueError(f"decimal odds must exceed 1.0, got {decimal_odds}")
    if not 0.0 <= tax_rate < 1.0:
        raise ValueError(f"tax rate must lie in [0, 1), got {tax_rate}")

    if mode is TaxMode.NONE:
        return decimal_odds
    if mode is TaxMode.STAKE_DEDUCTED:
        # Outlay 1, wagered (1-t), returns (1-t)*odds on a win.
        return decimal_odds * (1.0 - tax_rate)
    if mode is TaxMode.STAKE_ADDED:
        # Outlay (1+t) to win odds. Normalise to a unit outlay.
        return decimal_odds / (1.0 + tax_rate)
    if mode is TaxMode.WINNINGS:
        # Outlay 1; on a win keep 1 + (odds-1)*(1-t).
        return 1.0 + (decimal_odds - 1.0) * (1.0 - tax_rate)
    raise ValueError(f"unhandled tax mode {mode}")


def expected_value(probability: float, decimal_odds: float, *,
                   tax_mode: TaxMode | str = TaxMode.STAKE_DEDUCTED,
                   tax_rate: float = GERMAN_STAKE_TAX) -> float:
    """EV per unit staked, net of tax. Positive means a theoretical edge."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must lie in [0, 1], got {probability}")
    return probability * effective_odds(decimal_odds, tax_mode, tax_rate) - 1.0


def breakeven_probability(decimal_odds: float, *,
                          tax_mode: TaxMode | str = TaxMode.STAKE_DEDUCTED,
                          tax_rate: float = GERMAN_STAKE_TAX) -> float:
    """The probability at which a bet stops losing money.

    Worth printing next to every signal. At 2.00 with a 5.3% stake tax you need
    52.8% rather than 50% — and the gap is bigger than most claimed edges.
    """
    return 1.0 / effective_odds(decimal_odds, tax_mode, tax_rate)


def kelly_fraction(probability: float, decimal_odds: float, *,
                   tax_mode: TaxMode | str = TaxMode.STAKE_DEDUCTED,
                   tax_rate: float = GERMAN_STAKE_TAX) -> float:
    """Full-Kelly bankroll fraction on the tax-adjusted price.

    Full Kelly is the growth-optimal stake only if `probability` is exactly
    right. It never is. Use `size_bet`, which applies a fraction and a cap.
    """
    eff = effective_odds(decimal_odds, tax_mode, tax_rate)
    b = eff - 1.0
    if b <= 0.0:
        return 0.0
    f = (probability * eff - 1.0) / b
    return max(0.0, float(f))


def size_bet(probability: float, decimal_odds: float, *,
             tax_mode: TaxMode | str = TaxMode.STAKE_DEDUCTED,
             tax_rate: float = GERMAN_STAKE_TAX,
             kelly_multiplier: float = 0.25,
             max_fraction: float = 0.02,
             min_edge: float = 0.02) -> BetEvaluation:
    """Evaluate a bet and size it conservatively.

    Defaults are deliberately timid. Quarter-Kelly, capped at 2% of bankroll,
    and nothing staked below a 2% edge, because a model fitted to roughly 300
    matches a season carries parameter uncertainty that full Kelly ignores
    entirely. Treating a noisy point estimate as a known probability is the
    standard route to ruin.
    """
    if not 0.0 < kelly_multiplier <= 1.0:
        raise ValueError("kelly_multiplier must lie in (0, 1]")

    eff = effective_odds(decimal_odds, tax_mode, tax_rate)
    ev = probability * eff - 1.0
    full_kelly = kelly_fraction(probability, decimal_odds, tax_mode=tax_mode, tax_rate=tax_rate)

    is_value = ev >= min_edge
    stake = min(full_kelly * kelly_multiplier, max_fraction) if is_value else 0.0

    return BetEvaluation(
        probability=probability,
        decimal_odds=decimal_odds,
        effective_odds=eff,
        edge=ev,
        expected_value=ev,
        breakeven_probability=1.0 / eff,
        kelly_fraction=full_kelly,
        stake_fraction=float(stake),
        is_value=is_value,
    )


def required_edge_over_market(market_probability: float, decimal_odds: float, *,
                              tax_mode: TaxMode | str = TaxMode.STAKE_DEDUCTED,
                              tax_rate: float = GERMAN_STAKE_TAX) -> float:
    """How far your probability must exceed the market's to break even.

    This is the number that decides whether the project is viable. Against a
    fat retail 1X2 price plus stake tax it is typically several percentage
    points, which is a great deal to ask of a public-data model.
    """
    return breakeven_probability(decimal_odds, tax_mode=tax_mode, tax_rate=tax_rate) - market_probability
