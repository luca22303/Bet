"""The xG path, end to end through the store.

Goals arrive at roughly 2.8 a match, so team strength takes most of a season to
surface in scorelines. Expected goals carry the same signal at several times the
sample size, which is why the blend target exists.
"""

from datetime import datetime

import numpy as np
import pytest

from bet.models.dixon_coles import DixonColesModel


def test_shots_respect_the_point_in_time_cut(store_with_shots):
    store = store_with_shots
    kickoff = store.con.execute("SELECT MIN(kickoff_utc) FROM match").fetchone()[0]
    from datetime import timedelta
    assert store.shots_as_of(kickoff - timedelta(hours=1)).empty
    assert not store.shots_as_of(kickoff + timedelta(days=30)).empty


def test_xg_target_fits_from_stored_shots(store_with_shots):
    model = DixonColesModel(xi=0.0, target="xg").fit(store_with_shots, datetime(2023, 6, 1))
    assert model.xg_params is not None
    assert model.xg_params.n_matches > 100
    assert sum(model.xg_params.attack.values()) == pytest.approx(0.0, abs=1e-8)


def test_blend_rates_sit_between_the_two_sources(store_with_shots):
    """A geometric blend must lie between its inputs, never outside them."""
    as_of = datetime(2023, 6, 1)
    blended = DixonColesModel(xi=0.0, target="blend", blend_weight=0.5).fit(store_with_shots, as_of)
    goals_only = DixonColesModel(xi=0.0, target="goals").fit(store_with_shots, as_of)
    xg_only = DixonColesModel(xi=0.0, target="xg").fit(store_with_shots, as_of)

    home, away = "bayern_munich", "fc_augsburg"
    lam_blend, _ = blended._rates(home, away)
    lam_goals, _ = goals_only._rates(home, away)
    lam_xg, _ = xg_only._rates(home, away)

    assert min(lam_goals, lam_xg) <= lam_blend <= max(lam_goals, lam_xg)


def test_blend_weight_controls_the_mix(store_with_shots):
    as_of = datetime(2023, 6, 1)
    home, away = "bayern_munich", "fc_augsburg"

    all_goals = DixonColesModel(xi=0.0, target="blend", blend_weight=0.0).fit(store_with_shots, as_of)
    all_xg = DixonColesModel(xi=0.0, target="blend", blend_weight=1.0).fit(store_with_shots, as_of)
    goals_only = DixonColesModel(xi=0.0, target="goals").fit(store_with_shots, as_of)
    xg_only = DixonColesModel(xi=0.0, target="xg").fit(store_with_shots, as_of)

    assert all_goals._rates(home, away)[0] == pytest.approx(goals_only._rates(home, away)[0])
    assert all_xg._rates(home, away)[0] == pytest.approx(xg_only._rates(home, away)[0])


def test_xg_target_degrades_to_goals_when_no_shots_exist(populated_store):
    """The store may hold no Understat data yet; that must not crash a run."""
    model = DixonColesModel(xi=0.0, target="blend").fit(populated_store, datetime(2023, 1, 1))
    assert model.xg_params is None

    fixtures = populated_store.fixtures_between(datetime(2023, 1, 1), datetime(2023, 2, 1))
    probs = model.predict(populated_store, fixtures, datetime(2023, 1, 1))
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_all_three_targets_produce_valid_probabilities(store_with_shots):
    as_of = datetime(2023, 6, 1)
    fixtures = store_with_shots.fixtures_between(datetime(2023, 6, 1), datetime(2023, 12, 1))
    for target in ("goals", "xg", "blend"):
        model = DixonColesModel(xi=0.0, target=target).fit(store_with_shots, as_of)
        probs = model.predict(store_with_shots, fixtures, as_of)
        assert np.allclose(probs.sum(axis=1), 1.0), target
        assert np.all(probs > 0), target
