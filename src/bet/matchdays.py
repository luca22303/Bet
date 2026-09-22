"""Grouping fixtures into matchdays, and finding the two that matter right now.

Nothing ingested carries a matchday number: football-data.co.uk's CSVs have no
round column, and OpenLigaDB's group field was never captured (see
bet.ingest.openligadb). Clustering by kickoff proximity works from what is
already on disk and needs no schema change or re-ingest -- a Bundesliga
weekend clusters Friday to Monday, and the gap to the next one is at least a
few days even without an international break to widen it further.

The same clustering also drives `bet.evaluation.backtest`'s refit boundaries;
this is the one definition of "a matchday" both share.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

# Wide enough to separate any two real matchdays (which cluster within a
# weekend, so well under 36h between the matches in one) yet well short of the
# shortest realistic gap between two different ones.
DEFAULT_GAP_HOURS = 36.0

# How far to search for the next/previous matchday once we know at least one
# fixture exists out there. A matchday itself never spans more than a few
# days, and even the widest realistic gap between matchdays -- an
# international break -- is well under two months.
DEFAULT_SEARCH_DAYS = 60


def group_by_matchday(fixtures: pd.DataFrame,
                      *, gap_hours: float = DEFAULT_GAP_HOURS) -> list[pd.DataFrame]:
    """Split fixtures into matchday-sized clusters, breaking on any large gap."""
    if fixtures.empty:
        return []
    ordered = fixtures.sort_values("kickoff_utc").reset_index(drop=True)
    kickoffs = pd.to_datetime(ordered["kickoff_utc"])
    breaks = kickoffs.diff() > pd.Timedelta(hours=gap_hours)
    group_ids = breaks.cumsum()
    return [group.reset_index(drop=True) for _, group in ordered.groupby(group_ids)]


def next_matchday(store, as_of: datetime, *, league: str | None = None,
                  search_days: int = DEFAULT_SEARCH_DAYS) -> pd.DataFrame:
    """The first cluster of fixtures kicking off at or after `as_of`.

    Bounded rather than truly unbounded, unlike `Store.next_fixture_after`:
    once the first future fixture is found, its matchday cannot span more than
    a handful of days, so a generous fixed window from that point is enough
    and keeps the follow-up query cheap. Empty when nothing is scheduled at
    all within reach -- callers fall back to `Store.next_fixture_after` for an
    honest "the season is loaded, just not playing yet" message in that case.
    """
    upcoming = store.next_fixture_after(as_of, league=league)
    if upcoming is None:
        return pd.DataFrame()
    kickoff = pd.Timestamp(upcoming["kickoff_utc"]).to_pydatetime()
    fixtures = store.fixtures_between(kickoff, kickoff + timedelta(days=5), league=league)
    groups = group_by_matchday(fixtures)
    return groups[0] if groups else pd.DataFrame()


def previous_matchday(store, as_of: datetime, *, league: str | None = None,
                      search_days: int = DEFAULT_SEARCH_DAYS) -> pd.DataFrame:
    """The most recently completed cluster of fixtures before `as_of`.

    Only fixtures with a known result count. A match that has kicked off but
    has no result yet -- in progress, or a slow source -- is not "the
    previous matchday" for a predicted-vs-actual comparison: there is no
    actual to compare against yet.
    """
    window = store.fixtures_between(
        as_of - timedelta(days=search_days), as_of, league=league)
    played = window[window["outcome"].notna()]
    groups = group_by_matchday(played)
    return groups[-1] if groups else pd.DataFrame()
