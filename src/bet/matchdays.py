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
    """The matchday that is next to complete -- including any of its own
    fixtures that have already kicked off, or finished, by `as_of`.

    A matchday spans a weekend: some of it can be over while the rest is
    still to come, and a round in progress must not lose its Friday fixture
    from the display just because kickoff has passed -- that fixture is
    exactly the one this exists to show, once played, coloured by whether
    the prediction was right.

    Finds the next fixture that has not yet kicked off, then looks both
    before and after it for the rest of its cluster, rather than only
    forward from it -- a forward-only window would miss anything in the same
    round that had already started. Bounded rather than truly unbounded,
    unlike `Store.next_fixture_after`: a matchday cannot span more than a
    handful of days, so a generous fixed window around the anchor is enough
    and keeps the follow-up query cheap. Empty when nothing is scheduled at
    all within reach -- callers fall back to `Store.next_fixture_after` for
    an honest "the season is loaded, just not playing yet" message in that
    case.
    """
    upcoming = store.next_fixture_after(as_of, league=league)
    if upcoming is None:
        return pd.DataFrame()
    kickoff = pd.Timestamp(upcoming["kickoff_utc"]).to_pydatetime()
    fixtures = store.fixtures_between(
        kickoff - timedelta(days=4), kickoff + timedelta(days=4), league=league)
    for group in group_by_matchday(fixtures):
        if upcoming["match_id"] in set(group["match_id"]):
            return group
    return pd.DataFrame()   # unreachable: `fixtures` always contains `upcoming`


def previous_matchday(store, as_of: datetime, *, league: str | None = None,
                      search_days: int = DEFAULT_SEARCH_DAYS) -> pd.DataFrame:
    """The most recently completed cluster of fixtures before `as_of`.

    A cluster counts only once every one of its fixtures has a result. A
    round is a weekend: Friday's match can finish while Saturday's has not
    even kicked off, and clustering only the played rows made that Friday
    match look like a complete "previous matchday" all on its own -- the
    same fixture `next_matchday` was, correctly, already showing as part of
    the round still in progress. One match rendered on both, identically.

    Recognising that requires seeing Saturday's fixture at all, which a
    window ending at `as_of` never does -- from a look-backward query,
    Saturday has not happened yet, so as far as it can tell Friday's match
    has no sibling and looks like a complete round of one. The window
    therefore also looks a few days past `as_of`: just enough to catch the
    rest of a round already under way, never enough to reach a whole other
    one. A cluster made entirely of fixtures still to come has nothing
    resolved either, so it is skipped by the same rule without needing a
    separate check for "this hasn't started yet".
    """
    window = store.fixtures_between(
        as_of - timedelta(days=search_days), as_of + timedelta(days=5), league=league)
    for group in reversed(group_by_matchday(window)):
        if group["outcome"].notna().all():
            return group
    return pd.DataFrame()
