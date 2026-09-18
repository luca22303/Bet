import pytest

from bet.config import GERMAN_STAKE_TAX
from bet.ev import (
    TaxMode,
    breakeven_probability,
    effective_odds,
    expected_value,
    kelly_fraction,
    required_edge_over_market,
    size_bet,
)


def test_untaxed_even_money_breaks_even_at_fifty_percent():
    assert breakeven_probability(2.0, tax_mode=TaxMode.NONE) == pytest.approx(0.5)


def test_german_stake_tax_moves_the_breakeven_point_materially():
    """The number that decides whether this project is viable.

    A 5.3% stake tax turns an even-money bet from a 50% proposition into a 52.8%
    one. That 2.8 point shift is larger than most edges a public-data model will
    ever claim to find.
    """
    breakeven = breakeven_probability(2.0, tax_mode=TaxMode.STAKE_DEDUCTED,
                                      tax_rate=GERMAN_STAKE_TAX)
    assert breakeven == pytest.approx(0.528, abs=0.001)
    assert breakeven - 0.5 > 0.025


def test_tax_strictly_reduces_expected_value():
    untaxed = expected_value(0.55, 2.0, tax_mode=TaxMode.NONE)
    for mode in (TaxMode.STAKE_DEDUCTED, TaxMode.STAKE_ADDED, TaxMode.WINNINGS):
        assert expected_value(0.55, 2.0, tax_mode=mode) < untaxed


def test_a_claimed_ten_percent_edge_shrinks_to_four():
    assert expected_value(0.55, 2.0, tax_mode=TaxMode.NONE) == pytest.approx(0.10)
    assert expected_value(0.55, 2.0, tax_mode=TaxMode.STAKE_DEDUCTED) == pytest.approx(0.0417, abs=1e-3)


def test_kelly_is_zero_without_an_edge():
    assert kelly_fraction(0.40, 2.0, tax_mode=TaxMode.NONE) == 0.0


def test_stake_is_capped_and_fractional():
    """Full Kelly on a noisy estimate is a ruin machine; the cap must bind."""
    bet = size_bet(0.95, 5.0, tax_mode=TaxMode.NONE, kelly_multiplier=0.25, max_fraction=0.02)
    assert bet.kelly_fraction > 0.5         # full Kelly would stake enormously
    assert bet.stake_fraction == pytest.approx(0.02)


def test_no_stake_below_the_minimum_edge():
    bet = size_bet(0.505, 2.0, tax_mode=TaxMode.NONE, min_edge=0.02)
    assert not bet.is_value
    assert bet.stake_fraction == 0.0


def test_required_edge_over_market_is_substantial_at_retail_prices():
    # A 6% overround book plus the stake tax: how much better than the market
    # must the model be simply to break even?
    required = required_edge_over_market(0.50, 1.90, tax_mode=TaxMode.STAKE_DEDUCTED)
    assert required > 0.04


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        effective_odds(0.95)
    with pytest.raises(ValueError):
        expected_value(1.5, 2.0)
