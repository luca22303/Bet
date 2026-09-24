"""Report what a scraped page actually contains.

These parsers were written against sources nobody here could open, so the first
real run will find something wrong with them. The useful question then is not
"did it fail" but "what does the page look like now" -- and that answer is worth
more than any number of guesses at selectors.

`bet diagnose` fetches a page, runs each parser step, and prints what each one
found or missed, along with enough structure (table ids, column names, embedded
JSON keys) to correct the parser without seeing the site. It saves the raw HTML
too, so a fix can be tested offline against the real thing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Finding:
    step: str
    ok: bool
    detail: str
    evidence: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        mark = "ok  " if self.ok else "FAIL"
        lines = [f"  [{mark}] {self.step}: {self.detail}"]
        lines.extend(f"         {e}" for e in self.evidence[:6])
        return "\n".join(lines)


@dataclass
class Diagnosis:
    source: str
    url: str
    fetched: bool
    findings: list[Finding] = field(default_factory=list)
    saved_to: Path | None = None
    html_bytes: int = 0

    @property
    def ok(self) -> bool:
        return self.fetched and all(f.ok for f in self.findings)

    def report(self) -> str:
        lines = [f"DIAGNOSE {self.source} — {self.url}", "=" * 72]
        if not self.fetched:
            lines.append("  could not fetch the page; nothing else could be checked")
        else:
            lines.append(f"  fetched {self.html_bytes:,} bytes")
            lines.extend(str(f) for f in self.findings)
        if self.saved_to:
            lines.append("")
            lines.append(f"  raw HTML saved to {self.saved_to}")
            lines.append("  Send that file (or this report) to fix the parser "
                         "without guessing.")
        return "\n".join(lines)


def diagnose_fbref(html_text: str) -> list[Finding]:
    """Check each step of the FBref match-page parser independently."""
    from bet.ingest.fbref import (
        COLUMN_MAP,
        extract_kickoff,
        extract_teams,
        flatten_columns,
        iter_tables,
        looks_like_player_table,
        squad_key,
        strip_html_comments,
        _read_table,
    )

    clean = strip_html_comments(html_text)
    findings = []

    commented = html_text.count("<!--")
    findings.append(Finding(
        "comment stripping", True,
        f"{commented} comment markers removed "
        f"({'FBref hides most tables in them' if commented else 'none present'})"))

    home, away = extract_teams(clean)
    findings.append(Finding(
        "teams", home is not None and away is not None,
        f"{home} vs {away}" if home else "no team names matched any pattern",
        [] if home else [
            "tried: og:title, <title>, <h1>, squad links",
            f"<title> is: {_first(clean, r'<title>(.*?)</title>')}",
        ]))

    kickoff = extract_kickoff(clean)
    findings.append(Finding(
        "kickoff", kickoff is not None,
        str(kickoff) if kickoff else "no kickoff found",
        [] if kickoff else [
            "tried: data-venue-epoch, <time datetime>, data-venue-date",
            f"time-ish attributes present: {_attributes(clean, 'venue|datetime|epoch')}",
        ]))

    tables = iter_tables(clean)
    player_tables, ids, columns_seen = [], [], set()
    for position, (table_html, table_id) in enumerate(tables):
        try:
            frame = flatten_columns(_read_table(table_html))
        except Exception:
            continue
        if looks_like_player_table(frame):
            player_tables.append((table_html, squad_key(table_id, position)))
            ids.append(table_id or f"(no id, position {position})")
            columns_seen.update(frame.columns)

    findings.append(Finding(
        "player tables", len(player_tables) >= 2,
        f"{len(player_tables)} of {len(tables)} tables look like player stats",
        ids[:6] if player_tables else [
            f"{len(tables)} tables found; none had a 'player' column plus a stat",
            f"table ids: {[i for _, i in tables[:6]]}",
        ]))

    if columns_seen:
        mapped = sorted(c for c in COLUMN_MAP if c in columns_seen)
        unmapped = sorted(columns_seen - set(COLUMN_MAP) - {"player", "pos"})
        findings.append(Finding(
            "columns", len(mapped) >= 3,
            f"{len(mapped)} of {len(COLUMN_MAP)} known columns present",
            [f"mapped: {', '.join(mapped[:12])}",
             f"unmapped: {', '.join(unmapped[:12])}"]))

    squads = {key for _, key in player_tables}
    findings.append(Finding(
        "squad grouping", len(squads) >= 2,
        f"{len(squads)} distinct squads identified",
        [] if len(squads) >= 2 else
        ["expected two; tables could not be split between the sides"]))

    return findings


def diagnose_kicker(html_text: str) -> list[Finding]:
    """Check each kicker strategy independently, not just the winner."""
    from bet.ingest.kicker import (
        _teams_from_json,
        _teams_from_markup,
        _teams_from_text,
        extract_embedded_json,
        parse_lineup_page,
        plausible_formation,
    )

    findings = []

    payloads = extract_embedded_json(html_text)
    keys = []
    for payload in payloads[:3]:
        if isinstance(payload, dict):
            keys.extend(list(payload.keys())[:6])
    findings.append(Finding(
        "embedded JSON", bool(payloads),
        f"{len(payloads)} JSON payload(s) found",
        [f"top-level keys: {', '.join(keys[:10])}"] if keys else
        ["tried: __NEXT_DATA__, __NUXT__, __INITIAL_STATE__, ld+json"]))

    formation = plausible_formation(html_text)
    findings.append(Finding(
        "formation", formation is not None,
        formation or "no digit run summing to ten outfield players",
        [] if formation else
        [f"digit runs present: {_digit_runs(html_text)}"]))

    for label, extractor in (("strategy: embedded-json", _teams_from_json),
                             ("strategy: markup", _teams_from_markup),
                             ("strategy: generic", _teams_from_text)):
        try:
            teams = extractor(html_text)
        except Exception as exc:
            findings.append(Finding(label, False, f"raised {type(exc).__name__}: {exc}"))
            continue
        findings.append(Finding(
            label, bool(teams),
            f"{len(teams)} team(s)" if teams else "found nothing",
            [f'{t["name"]} {t["formation"]} ({len(t["players"])} players): '
             f'{", ".join(t["players"][:5])}' for t in teams]))

    result = parse_lineup_page(html_text)
    findings.append(Finding(
        "overall", bool(result["teams"]),
        f'strategy={result["strategy"]}, confirmed={result["confirmed"]}',
        [] if result["teams"] else
        [f"class names on the page: {_classes(html_text)}"]))

    return findings


def diagnose_kicker_ticker(html_text: str) -> list[Finding]:
    """Check each ticker strategy independently, not just the winner."""
    from bet.ingest.kicker import (
        _events_from_json,
        _events_from_markup,
        _events_from_text,
        extract_embedded_json,
        parse_ticker_page,
    )

    findings = []

    payloads = extract_embedded_json(html_text)
    keys = []
    for payload in payloads[:3]:
        if isinstance(payload, dict):
            keys.extend(list(payload.keys())[:6])
    findings.append(Finding(
        "embedded JSON", bool(payloads),
        f"{len(payloads)} JSON payload(s) found",
        [f"top-level keys: {', '.join(keys[:10])}"] if keys else
        ["tried: __NEXT_DATA__, __NUXT__, __INITIAL_STATE__, ld+json"]))

    minutes = len(re.findall(r"\d{1,3}(?:\+\d{1,2})?\s*[’'`´]", html_text))
    findings.append(Finding(
        "minute markers", minutes > 0,
        f"{minutes} apostrophe-minute pattern(s) found on the page",
        [] if minutes else ["no \"N'\" or \"N+M'\" text anywhere on the page"]))

    for label, extractor in (("strategy: embedded-json", _events_from_json),
                             ("strategy: markup", _events_from_markup),
                             ("strategy: generic", _events_from_text)):
        try:
            events = extractor(html_text)
        except Exception as exc:
            findings.append(Finding(label, False, f"raised {type(exc).__name__}: {exc}"))
            continue
        evidence = []
        for e in events[:6]:
            stoppage = f"+{e['stoppage']}" if e["stoppage"] else ""
            evidence.append(f"{e['minute']}'{stoppage} [{e['event_type']}] "
                            f"{e['description'][:70]}")
        findings.append(Finding(
            label, bool(events),
            f"{len(events)} event(s)" if events else "found nothing", evidence))

    result = parse_ticker_page(html_text)
    findings.append(Finding(
        "overall", bool(result["events"]),
        f'strategy={result["strategy"]}, {len(result["events"])} event(s)',
        [] if result["events"] else
        [f"class names on the page: {_classes(html_text)}"]))

    return findings


def _first(text: str, pattern: str) -> str:
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return (match.group(1).strip()[:90] if match else "(not found)")


def _attributes(text: str, keyword: str) -> str:
    found = sorted(set(re.findall(rf'([a-z-]*(?:{keyword})[a-z-]*)=', text, re.IGNORECASE)))
    return ", ".join(found[:8]) or "(none)"


def _digit_runs(text: str) -> str:
    runs = sorted(set(re.findall(r"\b\d(?:-\d){1,4}\b", text)))
    return ", ".join(runs[:10]) or "(none)"


def _classes(text: str) -> str:
    classes: set[str] = set()
    for value in re.findall(r'class="([^"]{1,120})"', text)[:400]:
        classes.update(value.split())
    interesting = [c for c in sorted(classes)
                   if re.search(r"line|team|player|spieler|form|squad|name"
                                r"|ticker|event|verlauf", c, re.I)]
    return ", ".join(interesting[:12]) or "(nothing name-like)"


def run(source: str, url: str, store, *, save_dir: Path | None = None) -> Diagnosis:
    """Fetch one page and report what each parser step made of it."""
    from bet.ingest.base import Source

    class _Fetcher(Source):
        name = f"diagnose_{source}"

        def ingest(self, **kwargs):
            raise NotImplementedError

    fetcher = _Fetcher(store)
    diagnosis = Diagnosis(source=source, url=url, fetched=False)

    try:
        text, path = fetcher.fetch(url, suffix=".html", cache=False)
    except Exception as exc:
        diagnosis.findings.append(Finding("fetch", False, str(exc)))
        return diagnosis

    diagnosis.fetched = True
    diagnosis.html_bytes = len(text)

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        target = save_dir / f"{source}-{datetime.utcnow():%Y%m%d-%H%M%S}.html"
        target.write_text(text, encoding="utf-8")
        diagnosis.saved_to = target
    else:
        diagnosis.saved_to = path

    checkers = {"fbref": diagnose_fbref, "kicker": diagnose_kicker,
                "kicker_ticker": diagnose_kicker_ticker}
    checker = checkers.get(source)
    if checker is None:
        diagnosis.findings.append(
            Finding("source", False,
                    f"no diagnostics for {source!r}; expected one of {sorted(checkers)}"))
        return diagnosis

    diagnosis.findings.extend(checker(text))
    return diagnosis
