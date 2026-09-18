import numpy as np
import pytest

from bet.odds.devig import DevigMethod, booksum, devig, overround


@pytest.mark.parametrize("method", list(DevigMethod))
def test_all_methods_return_a_probability_distribution(method):
    probs = devig([1.55, 4.50, 6.00], method=method)
    assert probs.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(probs > 0)


@pytest.mark.parametrize("method", list(DevigMethod))
def test_fair_book_is_left_alone(method):
    fair = [3.0, 3.0, 3.0]
    assert devig(fair, method=method) == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-9)


def test_overround_is_measured_correctly():
    assert overround([2.0, 2.0]) == pytest.approx(0.0, abs=1e-12)
    assert booksum([1.55, 4.50, 6.00]) > 1.0


def test_multiplicative_overstates_longshots_relative_to_shin():
    """The reason Shin is the default.

    Books load margin disproportionately onto longshots. Multiplicative devigging
    ignores that and hands back an inflated longshot probability, which is
    exactly where a naive model will think it has found value.
    """
    odds = [1.30, 5.50, 11.0]
    mult = devig(odds, DevigMethod.MULTIPLICATIVE)
    shin = devig(odds, DevigMethod.SHIN)
    assert mult[2] > shin[2]      # longshot overstated
    assert mult[0] < shin[0]      # favourite understated


def test_methods_disagree_enough_to_matter():
    # If the spread between methods exceeds a plausible edge, method choice can
    # manufacture an edge on its own. This documents that it does.
    odds = [1.30, 5.50, 11.0]
    estimates = [devig(odds, m)[2] for m in DevigMethod]
    assert (max(estimates) - min(estimates)) / min(estimates) > 0.05


@pytest.mark.parametrize("bad", [[1.0, 3.0, 4.0], [0.5, 3.0], [np.nan, 2.0, 3.0]])
def test_invalid_odds_are_rejected(bad):
    with pytest.raises(ValueError):
        devig(bad)


def test_exchange_prices_without_margin_are_normalised_not_inflated():
    probs = devig([2.05, 2.05])   # booksum below 1
    assert probs.sum() == pytest.approx(1.0)
