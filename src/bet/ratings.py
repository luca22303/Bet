"""Power ratings derived from results, so no feed is a single point of failure.

ClubElo went down and took the promoted-team prior with it. The general fix is
an alternative for every capability, but ratings are a special case: there is no
second free feed worth scraping — FiveThirtyEight retired SPI, and the rest are
paywalled — and there does not need to be. Elo is *computed* from results, and
results are the one thing this store already holds from two independent sources
that cross-check each other.

So the fallback is not another scraper to break. It is arithmetic over data
already on disk, which cannot 502.

Ratings from different providers are on different scales: ClubElo runs roughly
1300–2100 across Europe, while a league-local walk starting everyone at 1500
spans a much narrower band. Blending them per team would produce a table where
a number means something different from row to row, so a reader picks one
provider for the whole table. See `Store.ratings_as_of`.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

# Which provider to believe when several have ratings. ClubElo first: it is
# computed across all of European football, so it places a Bundesliga club
# against clubs this store has never seen, which is exactly what the promoted
# prior needs. The derived walk only knows the league it was given.
RATING_SOURCE_PREFERENCE = ("clubelo", "derived_elo")

DERIVED_SOURCE = "derived_elo"

# Shared with models.baselines.EloModel so the stored ratings and the baseline
# cannot drift apart — two definitions of "our Elo" would be a bug nobody sees.
DEFAULT_K = 20.0
DEFAULT_HOME_ADVANTAGE = 65.0
DEFAULT_INITIAL = 1500.0


def _expected(home: float, away: float, home_advantage: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(home + home_advantage - away) / 400.0))


def _actual(home_goals: float, away_goals: float) -> float:
    if home_goals > away_goals:
        return 1.0
    return 0.5 if home_goals == away_goals else 0.0


def elo_timeline(matches: pd.DataFrame, *, k: float = DEFAULT_K,
                 home_advantage: float = DEFAULT_HOME_ADVANTAGE,
                 initial: float = DEFAULT_INITIAL) -> list[dict]:
    """Every rating change, in order, with the moment it became knowable.

    Yields two rows per match — both sides move — carrying `known_at` from the
    result that caused the change. A derived rating is therefore never visible
    before the match behind it, and the point-in-time guard holds for anything
    built on it exactly as it does for an ingested one.
    """
    if matches.empty:
        return []

    ordered = matches.sort_values(["kickoff_utc", "match_id"])
    ratings: dict[str, float] = {}
    rows: list[dict] = []

    for row in ordered.itertuples(index=False):
        if pd.isna(row.home_goals) or pd.isna(row.away_goals):
            continue                       # not played yet; nothing to learn

        home = ratings.setdefault(row.home_team_id, initial)
        away = ratings.setdefault(row.away_team_id, initial)

        # Margin of victory: a 4-0 is more evidence than a 1-0. log1p keeps a
        # rout informative without letting one freak scoreline dominate.
        margin = abs(float(row.home_goals) - float(row.away_goals))
        multiplier = float(np.log1p(margin) + 1.0)

        delta = k * multiplier * (_actual(row.home_goals, row.away_goals)
                                 - _expected(home, away, home_advantage))
        ratings[row.home_team_id] = home + delta
        ratings[row.away_team_id] = away - delta

        known_at = getattr(row, "result_known_at", None)
        if known_at is None or pd.isna(known_at):
            # Without the result's own timestamp the safe assumption is the
            # match itself: never earlier, which is the direction that matters.
            known_at = row.kickoff_utc
        valid_from = pd.Timestamp(row.kickoff_utc).date()

        for team in (row.home_team_id, row.away_team_id):
            rows.append({
                "team_id": team,
                "source": DERIVED_SOURCE,
                "rating": float(ratings[team]),
                "valid_from": valid_from,
                "valid_to": None,
                "known_at": pd.Timestamp(known_at).to_pydatetime(),
            })

    return rows


def final_elo(matches: pd.DataFrame, *, k: float = DEFAULT_K,
              home_advantage: float = DEFAULT_HOME_ADVANTAGE,
              initial: float = DEFAULT_INITIAL) -> dict[str, float]:
    """Where the walk ends up — the rating each team carries now."""
    ratings: dict[str, float] = {}
    for row in elo_timeline(matches, k=k, home_advantage=home_advantage,
                            initial=initial):
        ratings[row["team_id"]] = row["rating"]
    return ratings


def derive_and_store(store, *, league: str | None = None,
                     as_of: datetime | None = None) -> int:
    """Recompute derived ratings from the results the store holds.

    Cheap enough to run on every refresh — it is a pass over a few thousand
    rows — and doing so unconditionally means the fallback is always current
    rather than only appearing once the primary feed has already failed.
    """
    as_of = as_of or datetime.utcnow()
    matches = store.matches_as_of(as_of, league=league)
    rows = elo_timeline(matches)
    if not rows:
        return 0

    frame = pd.DataFrame(rows)
    return store.upsert("team_rating", frame, ["team_id", "source", "valid_from"])
