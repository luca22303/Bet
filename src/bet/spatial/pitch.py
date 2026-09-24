"""Spatial aggregation: shot maps, zone heatmaps, field tilt.

Built for looking at, not for feeding the forecast. That distinction is
deliberate and worth stating plainly, because the temptation runs the other way.

A binned heatmap is roughly fifty numbers per team per match. Bundesliga
produces 306 matches a season. Handing fifty weakly-informative spatial features
to a model trained on that little data is a fast route to overfitting, and
almost everything a touch map says about attacking quality is already inside xG,
with less noise. So these functions serve the dashboard and scouting questions --
where does this side build, has that full-back moved inside, is a team's xG
coming from good positions or from hopeful shots -- and the forecast keeps
reading xG.

The one spatial number that does earn a place as a model feature is field tilt:
a single scalar, well sampled, measuring territorial dominance in a way that
possession share does not.

Coordinates follow Understat's convention: x and y in [0, 1], x = 0 at the
defending goal line and x = 1 at the attacking one, always from the shooting
team's perspective.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Pitch thirds by x, and channels by y. Coarse on purpose: finer bins give the
# appearance of precision that this data does not support.
ZONES = {
    "defensive_third": (0.0, 1 / 3),
    "middle_third": (1 / 3, 2 / 3),
    "final_third": (2 / 3, 1.0),
}

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0


def zone_of(x: float) -> str:
    """Which third of the pitch an x coordinate falls in."""
    if x is None or not np.isfinite(x):
        return "unknown"
    if x < 1 / 3:
        return "defensive_third"
    if x < 2 / 3:
        return "middle_third"
    return "final_third"


def bin_heatmap(events: pd.DataFrame, *, x_col: str = "x", y_col: str = "y",
                x_bins: int = 6, y_bins: int = 5, normalise: bool = True) -> np.ndarray:
    """Bin events into a coarse pitch grid.

    Returns an (x_bins, y_bins) array, normalised to sum to one so that maps
    from teams with different event counts can be compared directly.
    """
    if events.empty or x_col not in events.columns or y_col not in events.columns:
        return np.zeros((x_bins, y_bins))

    valid = events[[x_col, y_col]].dropna()
    if valid.empty:
        return np.zeros((x_bins, y_bins))

    x = np.clip(valid[x_col].to_numpy(dtype=float), 0.0, 0.999999)
    y = np.clip(valid[y_col].to_numpy(dtype=float), 0.0, 0.999999)

    grid, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins], range=[[0, 1], [0, 1]])
    if normalise and grid.sum() > 0:
        grid = grid / grid.sum()
    return grid


def field_tilt(events: pd.DataFrame, team_col: str = "team_id", x_col: str = "x",
               threshold: float = 2 / 3) -> pd.Series:
    """Share of all final-third events belonging to each team.

    Territorial dominance. Unlike possession share it does not reward a side for
    passing along its own back line, which is why it survives as a feature while
    the rest of the heatmap does not.
    """
    if events.empty or x_col not in events.columns:
        return pd.Series(dtype=float)

    final_third = events[events[x_col] >= threshold]
    if final_third.empty:
        return pd.Series(dtype=float)

    counts = final_third.groupby(team_col).size()
    return counts / counts.sum()


def shot_map(shots: pd.DataFrame, team_id: str | None = None) -> pd.DataFrame:
    """Shots with distance, angle and zone attached.

    Distance and angle are what actually drive an xG estimate, so recovering
    them makes it possible to ask whether a team's xG comes from good positions
    or from a high volume of hopeful ones -- two very different teams that look
    identical in a season xG total.
    """
    if shots.empty:
        return shots

    frame = shots if team_id is None else shots[shots["team_id"] == team_id]
    frame = frame.dropna(subset=["x", "y"]).copy()
    if frame.empty:
        return frame

    # Goal centre is (1.0, 0.5) in normalised coordinates.
    dx = (1.0 - frame["x"]) * PITCH_LENGTH_M
    dy = (frame["y"] - 0.5) * PITCH_WIDTH_M
    frame["distance_m"] = np.sqrt(dx ** 2 + dy ** 2)

    # Angle subtended by the 7.32m goal mouth from the shot location. Wider is
    # better, and it collapses toward zero from tight angles regardless of distance.
    goal_half = 7.32 / 2.0
    left = np.arctan2(goal_half - dy, np.maximum(dx, 1e-6))
    right = np.arctan2(-goal_half - dy, np.maximum(dx, 1e-6))
    frame["angle_rad"] = np.abs(left - right)
    frame["angle_deg"] = np.degrees(frame["angle_rad"])
    frame["zone"] = frame["x"].map(zone_of)
    frame["is_box"] = (frame["x"] >= 1 - 16.5 / PITCH_LENGTH_M) & (frame["y"].between(0.21, 0.79))
    return frame


def shot_profile(shots: pd.DataFrame, team_id: str | None = None) -> dict[str, float]:
    """Summarise shot quality for a team.

    `xg_per_shot` is the number to read first. Two sides with equal season xG
    and very different xg_per_shot are not equally good: one is creating chances,
    the other is shooting from distance and accumulating xG by volume.
    """
    mapped = shot_map(shots, team_id)
    if mapped.empty:
        return {"shots": 0}

    total_xg = float(mapped["xg"].sum()) if "xg" in mapped.columns else float("nan")
    n = len(mapped)
    return {
        "shots": n,
        "total_xg": total_xg,
        "xg_per_shot": total_xg / n if n else float("nan"),
        "mean_distance_m": float(mapped["distance_m"].mean()),
        "median_distance_m": float(mapped["distance_m"].median()),
        "mean_angle_deg": float(mapped["angle_deg"].mean()),
        "share_in_box": float(mapped["is_box"].mean()),
        "share_beyond_20m": float((mapped["distance_m"] > 20).mean()),
    }


# The three columns FBref's possession table reports that genuinely
# partition the pitch -- every touch falls in exactly one -- so their shares
# sum to one and can be drawn as adjacent, non-overlapping bands.
PITCH_THIRD_COLUMNS = ("touches_def_third", "touches_mid_third", "touches_att_third")

# The penalty-area columns are not a fourth and fifth independent zone: a
# touch in the box is also a touch in that third, so these are subsets
# already counted inside PITCH_THIRD_COLUMNS, not disjoint from them.
PENALTY_AREA_COLUMNS = ("touches_def_pen", "touches_att_pen")


def touch_zone_shares(rows: pd.DataFrame) -> dict | None:
    """Share of touches in each pitch third, from FBref's own zone breakdown.

    Coarse by construction -- three bands, not a continuous surface -- but a
    real, ingested number rather than anything modelled or guessed. `rows` is
    one or more `player_match_stat`-shaped rows: a single player's for the
    click-through detail, a whole team's for the match-level heatmap tab.

    Def Pen and Att Pen are reported alongside as penalty-area counts, not
    folded in as a fourth and fifth share -- see `PENALTY_AREA_COLUMNS`.

    `None` when nothing here has ever gone through the possession table --
    every zone column null, not merely zero -- so a team never covered by it
    is not drawn as though it played out of exactly no space.
    """
    zone_cols = list(PITCH_THIRD_COLUMNS)
    if rows.empty or not all(c in rows.columns for c in zone_cols):
        return None
    if rows[zone_cols].isna().all(axis=None):
        return None

    totals = {c: float(rows[c].fillna(0.0).sum()) for c in zone_cols}
    grand_total = sum(totals.values())
    if grand_total <= 0:
        return None

    shares = {c.removeprefix("touches_").removesuffix("_third"): totals[c] / grand_total
             for c in zone_cols}

    penalty = {}
    for c in PENALTY_AREA_COLUMNS:
        if c in rows.columns and rows[c].notna().any():
            penalty[c.removeprefix("touches_")] = float(rows[c].fillna(0.0).sum())

    return {"thirds": shares, "penalty": penalty, "touches": grand_total}


# Fewer points than this and a density grid would just be showing a handful
# of isolated bumps as though they were a real spatial pattern; ten touches
# tell you almost nothing about where a player spent the match.
MIN_HEATMAP_POINTS = 10


def smooth_touch_grid(points: list[tuple[float, float]], *, x_bins: int = 24,
                      y_bins: int = 32, bandwidth: float = 0.08) -> np.ndarray | None:
    """A Gaussian-smoothed density grid from raw touch-location points.

    Finer and continuous-looking, unlike `touch_zone_shares`'s three coarse
    bands -- this is the "walking heatmap" a Sofascore-style view needs. Still
    honest about what it is: real recorded locations, smoothed only for
    legibility, not a claim of tracking data this project does not have.

    Returns an (x_bins, y_bins) grid normalised to peak at 1.0, or `None` for
    too few points -- see `MIN_HEATMAP_POINTS`.
    """
    if len(points) < MIN_HEATMAP_POINTS:
        return None

    xs = np.clip(np.array([p[0] for p in points], dtype=float), 0.0, 1.0)
    ys = np.clip(np.array([p[1] for p in points], dtype=float), 0.0, 1.0)

    grid_x = (np.arange(x_bins) + 0.5) / x_bins
    grid_y = (np.arange(y_bins) + 0.5) / y_bins
    gx, gy = np.meshgrid(grid_x, grid_y, indexing="ij")

    density = np.zeros_like(gx)
    for x, y in zip(xs, ys):
        density += np.exp(-((gx - x) ** 2 + (gy - y) ** 2) / (2 * bandwidth ** 2))

    peak = density.max()
    return density / peak if peak > 0 else None


def heatmap_to_frame(grid: np.ndarray) -> pd.DataFrame:
    """Long-format grid, ready to plot."""
    x_bins, y_bins = grid.shape
    rows = [
        {
            "x_bin": i,
            "y_bin": j,
            "x_centre": (i + 0.5) / x_bins,
            "y_centre": (j + 0.5) / y_bins,
            "zone": zone_of((i + 0.5) / x_bins),
            "share": float(grid[i, j]),
        }
        for i in range(x_bins)
        for j in range(y_bins)
    ]
    return pd.DataFrame(rows)
