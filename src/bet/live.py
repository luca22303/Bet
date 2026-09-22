"""One command that refreshes everything and rebuilds the dashboard.

The pieces to keep a model current already existed -- ingest, quality checks,
line-up polling, dashboard -- but running them meant four commands in the right
order, which is three too many to do reliably before every matchday. `bet live`
is that sequence.

It fetches from the public sources itself. Nothing here is pre-collected or
bundled: the adapters hit football-data.co.uk, ClubElo, Understat, OpenLigaDB
and FBref at the moment you run it.

Why this cannot live in the browser instead: none of those sources send an
`Access-Control-Allow-Origin` header, so a page fetching them client-side is
blocked by the browser's same-origin policy before the request leaves. The fetch
has to happen in a process that isn't a browser tab. Hence Python, and hence a
generated file rather than a page that refreshes itself.

The season defaults to the current one, worked out from the date, and only that
season is re-fetched. Re-scraping a decade every run would be slow and rude to
sources that owe you nothing.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd


def current_season_start(today: date | None = None) -> int:
    """The season a date falls in, as its opening year.

    European seasons straddle the new year, so anything before July belongs to
    the season that began the previous August.
    """
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


@dataclass
class RefreshReport:
    started: datetime
    seasons: list[int]
    sources: dict[str, str] = field(default_factory=dict)
    rows: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    # Sources that produced nothing at all, mapped to why. Kept apart from
    # `errors` because an unreachable source is an outage to note, not a fault
    # to fix: ClubElo answering 502 on http and refusing https entirely is
    # their server being down, and reporting it as "refresh failed" sent
    # someone looking for a bug in their own install.
    unavailable: dict[str, str] = field(default_factory=dict)
    output: Path | None = None
    duration_seconds: float = 0.0

    def summary(self) -> str:
        lines = [f"refreshed {', '.join(str(s) for s in self.seasons)} "
                 f"in {self.duration_seconds:.0f}s"]
        for source, status in self.sources.items():
            lines.append(f"  {source:<16} {status}")
        if self.rows:
            written = ", ".join(f"{t}={n}" for t, n in sorted(self.rows.items()))
            lines.append(f"  wrote {written}")
        if self.unavailable:
            for name, why in self.unavailable.items():
                lines.append(f"  {name:<16} unavailable — {why}")
        covered = tuple(f"{name}:" for name in self.unavailable)
        remaining = [e for e in self.errors if not e.startswith(covered)]
        if remaining:
            # Show them all. Truncating at five hid the sixth of six without
            # saying so, which is the failure mode this report exists to catch:
            # each source usually fails for its own reason, so a hidden one is
            # a fix that never gets made.
            lines.append(f"  {len(remaining)} error(s):")
            shown = remaining[:20]
            lines.extend(f"    ! {e}" for e in shown)
            if len(remaining) > len(shown):
                lines.append(f"    ... and {len(remaining) - len(shown)} more")
        if self.output:
            lines.append(f"  dashboard -> {self.output}")
        return "\n".join(lines)


def refresh(store, *, seasons: list[int] | None = None, league: str = "bundesliga",
            sources: tuple[str, ...] = ("football_data", "openligadb", "clubelo"),
            with_shots: bool = False, max_matches: int | None = None,
            cache: bool = False, on_progress=None) -> RefreshReport:
    """Pull the latest data from the live sources.

    `cache` defaults to False, unlike a historical backfill: the current
    season's file changes every matchday, and a cached copy of last week's is
    exactly the stale snapshot this command exists to prevent.
    """
    started = time.monotonic()
    seasons = seasons or [current_season_start()]
    report = RefreshReport(started=datetime.utcnow(), seasons=seasons)

    # A refresh behind a button takes minutes and gives no sign of life
    # otherwise, which reads exactly like a hang. `on_progress` is called with
    # the source about to run and its place in the queue.
    planned = [name for name in ("football_data", "openligadb", "clubelo",
                                 "understat", "fbref") if name in sources]

    def announce(name: str) -> None:
        if on_progress is not None:
            on_progress(f"{name} ({planned.index(name) + 1} of {len(planned)})")

    store.init_schema()

    if "football_data" in sources:
        from bet.ingest.football_data import FootballDataSource
        announce("football_data")
        report.sources["football_data"] = "results, odds, team stats"
        _run(report, FootballDataSource(store), seasons=seasons, league=league,
             cache=cache)

    if "openligadb" in sources:
        from bet.ingest.openligadb import OpenLigaDBSource
        announce("openligadb")
        report.sources["openligadb"] = "fixtures, results"
        _run(report, OpenLigaDBSource(store), seasons=seasons, league=league,
             cache=cache)

    if "clubelo" in sources:
        from bet.ingest.clubelo import ClubEloSource
        announce("clubelo")
        report.sources["clubelo"] = "power ratings"
        # Only the recent window: older snapshots never change, so re-walking
        # them every run is pure waste.
        _run(report, ClubEloSource(store),
             start=date.today() - timedelta(days=21), end=date.today(),
             step_days=7, cache=cache)

    if "understat" in sources:
        from bet.ingest.understat import UnderstatSource
        announce("understat")
        report.sources["understat"] = f"xG shots={'yes' if with_shots else 'no'}"
        _run(report, UnderstatSource(store), seasons=seasons, league=league,
             with_shots=with_shots, cache=cache)

    if "fbref" in sources:
        from bet.ingest.fbref import FBrefSource
        announce("fbref")
        report.sources["fbref"] = "player stats, line-ups"
        _run(report, FBrefSource(store), seasons=seasons, league=league,
             cache=cache, max_matches=max_matches)

    if "news" in sources:
        report.sources["news"] = "team news (extraction needs Ollama)"

    report.duration_seconds = time.monotonic() - started
    return report


# What a network failure actually was, in the words a person needs. The raw
# strings are nested urllib3 reprs several hundred characters long, and the one
# fact worth reading -- the host refused, or never answered -- is buried in the
# middle of them.
_CAUSES = (
    (r"(\d{3}) Server Error", lambda m: f"HTTP {m.group(1)}"),
    (r"(\d{3}) Client Error", lambda m: f"HTTP {m.group(1)}"),
    (r"ConnectTimeout|connect timeout", lambda m: "connection timed out"),
    (r"ReadTimeout|Read timed out", lambda m: "no response"),
    (r"Connection refused|NewConnectionError", lambda m: "connection refused"),
    (r"Name or service not known|NameResolution", lambda m: "DNS lookup failed"),
    (r"SSLError|CertificateError", lambda m: "TLS handshake failed"),
)


def _short_reason(errors: list[str], limit: int = 120) -> str:
    """One readable line out of a source's failures.

    Names the distinct causes rather than quoting the first exception: a host
    that answers 502 over http and refuses https entirely has told you two
    different things, and both matter when deciding whether it is their outage
    or your network.
    """
    blob = " ".join(" ".join(errors).split())

    found: list[str] = []
    for pattern, render in _CAUSES:
        match = re.search(pattern, blob)
        if match:
            described = render(match)
            if described not in found:
                found.append(described)

    if found:
        return ", ".join(found)
    return blob if len(blob) <= limit else blob[: limit - 1] + "…"


def _run(report: RefreshReport, source, **kwargs) -> None:
    """Run one adapter, folding its counts and errors into the report.

    A failing source is recorded and stepped over. One unreachable site must not
    stop the others: partial data plus a visible error beats no refresh at all,
    and the dashboard's provenance banner will show which feed went stale.
    """
    feeds = report.sources.get(source.name, source.name)

    try:
        result = source.ingest(**kwargs)
    except Exception as exc:
        report.sources[source.name] = f"FAILED: {exc}"
        report.unavailable[source.name] = _short_reason([str(exc)])
        report.errors.append(f"{source.name}: {exc}")
        return

    for table, count in result.rows_written.items():
        report.rows[table] = report.rows.get(table, 0) + count

    if not result.errors:
        return

    written = sum(result.rows_written.values())
    if written == 0:
        # Nothing landed at all, so every error says the same thing. Collapse
        # them, and name what is missing as a result rather than leaving the
        # reader to work out what "clubelo" feeds.
        report.sources[source.name] = f"unavailable ({feeds} not updated)"
        report.unavailable[source.name] = _short_reason(result.errors)
        report.errors.append(
            f"{source.name}: unavailable after {len(result.errors)} "
            f"attempt(s) — {feeds} not updated — {_short_reason(result.errors)}")
        return

    report.errors.extend(f"{source.name}: {e}" for e in result.errors[:3])
    report.sources[source.name] += f" ({len(result.errors)} errors)"


def refresh_and_render(store, output: Path, *, days: int = 8,
                       league: str = "bundesliga", seasons: list[int] | None = None,
                       sources: tuple[str, ...] = ("football_data", "openligadb", "clubelo"),
                       with_shots: bool = False, max_matches: int | None = None,
                       backtest_from: str | None = None) -> RefreshReport:
    """Fetch, then rebuild the dashboard from what was fetched."""
    from bet.dashboard import build

    report = refresh(store, seasons=seasons, league=league, sources=sources,
                     with_shots=with_shots, max_matches=max_matches)

    page = build(store, datetime.utcnow(), days=days, league=league,
                 backtest_from=backtest_from)
    output.write_text(page, encoding="utf-8")
    report.output = output.resolve()
    return report
