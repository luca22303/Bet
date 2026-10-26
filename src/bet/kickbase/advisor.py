"""Buy/sell recommendations from a squad, a market, and recent points history.

Deliberately decoupled from `client.py`'s raw JSON shape. Kickbase's own
field names here are a guess (see `client.py`'s module docstring), but the
analysis below -- what makes a player worth buying, selling, or holding --
does not depend on getting that guess right. `parse_player` is the one
narrow seam between the two: fix field names there once a real response is
seen, and every test below keeps proving the analysis itself without ever
touching the network.

"Consistent" is treated as a real, separate thing from "good": a player who
nets 6 points every single week is a safer squad member than one who
averages the same 6 by alternating 15 and -3, even though a raw average
cannot tell them apart. The coefficient of variation (stdev / mean) is what
actually distinguishes them.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

# Kickbase position codes, as reverse-engineered by the community (1=GK
# through 4=FWD). Unverified like everything else about this API's shape --
# see client.py.
_POSITION_NAMES = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# A player needs at least this many recent matchdays on record before a
# consistency read means anything -- two points either side of a mean is not
# a trend, it is a coin flip. Matches the project-wide discipline (see
# `spatial.pitch.MIN_HEATMAP_POINTS`) of returning "unknown" rather than a
# number built on too little evidence.
MIN_MATCHES_FOR_CONSISTENCY = 3


@dataclass
class PlayerSnapshot:
    """A normalised view of one player, independent of Kickbase's raw shape."""

    player_id: str
    name: str
    team: str = ""
    position: str = ""                  # "GK" / "DEF" / "MID" / "FWD" / ""
    market_value: float | None = None
    value_trend: float | None = None    # recent change in market value; None if unknown
    status: str = "FIT"                 # "FIT" / "INJURED" / "DOUBTFUL" / "SUSPENDED"
    recent_points: list[float] = field(default_factory=list)  # oldest to newest


def parse_player(raw: dict) -> PlayerSnapshot:
    """Normalise one raw Kickbase player object.

    Every field is looked up under several plausible names and degrades to
    an honest default rather than raising -- the same tolerance
    `sofascore.py`'s `_first` gives a JSON shape nobody here has confirmed.
    """
    def first(*keys, default=None):
        for key in keys:
            value = raw.get(key)
            if value is not None:
                return value
        return default

    position_code = first("pos", "position")
    position = _POSITION_NAMES.get(position_code, "") if isinstance(position_code, int) else ""

    points_field = first("performance", "points", "matchPoints", default=[])
    recent_points = []
    if isinstance(points_field, list):
        for entry in points_field:
            if isinstance(entry, dict):
                value = first_in(entry, "points", "p", "pt")
            else:
                value = entry
            try:
                if value is not None:
                    recent_points.append(float(value))
            except (TypeError, ValueError):
                continue

    status_code = first("status", "st")
    status = {0: "FIT", 1: "INJURED", 2: "DOUBTFUL", 4: "SUSPENDED"}.get(
        status_code, "FIT") if isinstance(status_code, int) else "FIT"

    return PlayerSnapshot(
        player_id=str(first("id", "i", "playerId", default="")),
        name=str(first("name", "n", "fullName", default="unknown")),
        team=str(first("teamName", "tn", default="")),
        position=position,
        market_value=_to_float(first("marketValue", "mv")),
        value_trend=_to_float(first("marketValueTrend", "mvgl", "mvt")),
        status=status,
        recent_points=recent_points,
    )


def first_in(d: dict, *keys, default=None):
    for key in keys:
        value = d.get(key)
        if value is not None:
            return value
    return default


def _to_float(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def consistency(points: list[float]) -> float | None:
    """0-1: 1 is perfectly steady scoring, 0 is wildly erratic.

    `None` with fewer than `MIN_MATCHES_FOR_CONSISTENCY` points on record --
    not a 0, which would read as "known to be erratic" rather than
    "not enough evidence yet".

    A flat-zero run (never plays) is reported as `None` too: perfectly
    steady at scoring nothing is not the "consistent" this is meant to
    surface, it is just absent, and the caller should be looking at
    `status`/appearances instead.
    """
    if len(points) < MIN_MATCHES_FOR_CONSISTENCY:
        return None
    mean = statistics.mean(points)
    if mean <= 0:
        return None
    stdev = statistics.pstdev(points)
    cv = stdev / mean
    return max(0.0, 1.0 - cv)


def reliability_score(player: PlayerSnapshot) -> float | None:
    """Average points scaled down by how erratic they are.

    Rewards a player who is both good and steady over one who hits the same
    average by swinging between a huge haul and a blank. `None` when there
    is not enough history to say either -- see `consistency`.
    """
    cons = consistency(player.recent_points)
    if cons is None:
        return None
    return statistics.mean(player.recent_points) * cons


def rank_buy_candidates(market: list[PlayerSnapshot], budget: float,
                        top_n: int = 10) -> list[dict]:
    """Market players worth buying within `budget`, best value first.

    Ranked on points per unit of cost -- a 1000-point star at a 1000-point
    price is not a bargain, a 100-point squad player at a 20-point price is.
    Players with unknown points history or an unknown price are skipped
    rather than guessed into the ranking: a missing fact is not a zero.
    """
    candidates = []
    for player in market:
        if player.status != "FIT":
            continue
        if player.market_value is None or player.market_value <= 0:
            continue
        if player.market_value > budget:
            continue
        if not player.recent_points:
            continue
        avg_points = statistics.mean(player.recent_points)
        if avg_points <= 0:
            continue
        cons = consistency(player.recent_points)
        candidates.append({
            "player": player,
            "avg_points": avg_points,
            "consistency": cons,
            "value_per_cost": avg_points / player.market_value * 1_000_000,
        })
    candidates.sort(key=lambda c: c["value_per_cost"], reverse=True)
    return candidates[:top_n]


def rank_sell_candidates(squad: list[PlayerSnapshot], top_n: int = 10) -> list[dict]:
    """Squad players worth moving on, worst-value-for-their-price first.

    Three independent reasons surface a player here, any one of which is
    enough: not fit to play at all, a falling market value (selling now
    beats selling after it drops further), or a reliability score that
    would not make the cut as a buy candidate at this player's own price --
    the same yardstick used for the market, turned on your own squad.
    """
    candidates = []
    for player in squad:
        reasons = []
        if player.status != "FIT":
            reasons.append(f"{player.status.lower()}")
        if player.value_trend is not None and player.value_trend < 0:
            reasons.append("value falling")

        avg_points = statistics.mean(player.recent_points) if player.recent_points else None
        cons = consistency(player.recent_points) if player.recent_points else None
        if avg_points is not None and player.market_value and player.market_value > 0:
            value_per_cost = avg_points / player.market_value * 1_000_000
            if avg_points <= 0 or (cons is not None and cons < 0.3):
                reasons.append("inconsistent return")
            elif value_per_cost < 1.0:
                reasons.append("overpriced for its output")

        if not reasons:
            continue
        candidates.append({
            "player": player, "avg_points": avg_points, "consistency": cons,
            "reasons": reasons,
        })

    # Not fit first (the clearest, least debatable reason), then by how thin
    # the remaining justification is.
    candidates.sort(key=lambda c: (c["player"].status == "FIT", len(c["reasons"]) == 1))
    return candidates[:top_n]


def most_consistent(players: list[PlayerSnapshot], top_n: int = 10) -> list[dict]:
    """Players worth trusting as a reliable weekly starter, regardless of price."""
    scored = []
    for player in players:
        cons = consistency(player.recent_points)
        if cons is None:
            continue
        scored.append({
            "player": player, "consistency": cons,
            "avg_points": statistics.mean(player.recent_points),
        })
    scored.sort(key=lambda c: (c["consistency"], c["avg_points"]), reverse=True)
    return scored[:top_n]
