"""Spatial aggregation for the dashboard layer."""

import numpy as np
import pandas as pd
import pytest

from bet.spatial.pitch import (
    bin_heatmap,
    field_tilt,
    heatmap_to_frame,
    shot_map,
    shot_profile,
    zone_of,
)


@pytest.fixture
def shots():
    return pd.DataFrame({
        "team_id": ["a"] * 5 + ["b"] * 3,
        "x": [0.95, 0.88, 0.70, 0.60, 0.93, 0.92, 0.75, 0.55],
        "y": [0.50, 0.42, 0.55, 0.30, 0.48, 0.48, 0.60, 0.35],
        "xg": [0.45, 0.18, 0.06, 0.03, 0.40, 0.35, 0.05, 0.02],
    })


@pytest.mark.parametrize("x,zone", [
    (0.1, "defensive_third"), (0.5, "middle_third"), (0.9, "final_third"),
    (None, "unknown"), (float("nan"), "unknown"),
])
def test_zone_classification(x, zone):
    assert zone_of(x) == zone


def test_shot_geometry_is_physically_sensible(shots):
    """Closer shots must have shorter distances and wider angles."""
    mapped = shot_map(shots, "a").sort_values("x", ascending=False)
    assert mapped["distance_m"].is_monotonic_increasing
    close = mapped.iloc[0]
    far = mapped.iloc[-1]
    assert close["distance_m"] < 10
    assert far["distance_m"] > 35
    assert close["angle_deg"] > far["angle_deg"]


def test_penalty_spot_distance_is_about_eleven_metres():
    """A known reference point: the spot is 11m out, centred."""
    spot = pd.DataFrame({"team_id": ["a"], "x": [1 - 11.0 / 105.0], "y": [0.5], "xg": [0.76]})
    assert shot_map(spot)["distance_m"].iloc[0] == pytest.approx(11.0, abs=0.1)


def test_box_detection():
    inside = pd.DataFrame({"team_id": ["a"], "x": [0.90], "y": [0.5], "xg": [0.3]})
    outside = pd.DataFrame({"team_id": ["a"], "x": [0.70], "y": [0.5], "xg": [0.05]})
    assert bool(shot_map(inside)["is_box"].iloc[0])
    assert not bool(shot_map(outside)["is_box"].iloc[0])


def test_shot_profile_separates_chance_quality_from_volume(shots):
    """Two sides with equal xG are not equally good.

    One creates chances; the other shoots from distance and accumulates xG by
    volume. xg_per_shot is what tells them apart.
    """
    close_range = pd.DataFrame({"team_id": ["a"] * 3, "x": [0.93, 0.94, 0.92],
                                "y": [0.5, 0.48, 0.52], "xg": [0.4, 0.35, 0.45]})
    long_range = pd.DataFrame({"team_id": ["b"] * 12, "x": [0.62] * 12,
                               "y": [0.5] * 12, "xg": [0.1] * 12})

    close_profile = shot_profile(close_range)
    long_profile = shot_profile(long_range)

    assert close_profile["total_xg"] == pytest.approx(1.2)
    assert long_profile["total_xg"] == pytest.approx(1.2)      # identical totals
    assert close_profile["xg_per_shot"] > long_profile["xg_per_shot"] * 3
    assert close_profile["share_in_box"] > long_profile["share_in_box"]
    assert close_profile["mean_distance_m"] < long_profile["mean_distance_m"]


def test_field_tilt_measures_territorial_share(shots):
    tilt = field_tilt(shots)
    assert tilt.sum() == pytest.approx(1.0)
    assert tilt["a"] > tilt["b"]


def test_heatmap_is_normalised(shots):
    grid = bin_heatmap(shots)
    assert grid.shape == (6, 5)
    assert grid.sum() == pytest.approx(1.0)


def test_heatmap_handles_empty_and_missing_columns():
    assert bin_heatmap(pd.DataFrame()).sum() == 0.0
    assert bin_heatmap(pd.DataFrame({"a": [1]})).sum() == 0.0


def test_heatmap_frame_round_trips(shots):
    frame = heatmap_to_frame(bin_heatmap(shots))
    assert len(frame) == 30
    assert frame["share"].sum() == pytest.approx(1.0)
    assert set(frame["zone"]) <= {"defensive_third", "middle_third", "final_third"}


def test_shots_outside_the_pitch_are_clipped_not_dropped():
    odd = pd.DataFrame({"team_id": ["a", "a"], "x": [1.5, -0.2], "y": [0.5, 0.5],
                        "xg": [0.1, 0.1]})
    assert bin_heatmap(odd).sum() == pytest.approx(1.0)
