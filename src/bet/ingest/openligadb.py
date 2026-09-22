"""OpenLigaDB: free community JSON API for German football.

Included because it is the one source here with no scraping and no terms-of-use
grey area — a plain public API. Coverage is fixtures, results and matchday
structure rather than advanced metrics, which makes it the right source for the
forward fixture list a weekly prediction run needs, and a useful cross-check on
results ingested from football-data.co.uk.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from bet.config import SETTINGS
from bet.ingest.base import IngestResult, Source, make_match_id, season_label
from bet.teams import UnknownTeamError, resolve

BASE_URL = "https://api.openligadb.de"
LEAGUE_SHORTCUTS = {"bundesliga": "bl1", "bundesliga2": "bl2"}


def field(entry, name: str):
    """Read a field regardless of how the API capitalises it.

    OpenLigaDB serves the same data under two spellings: PascalCase
    (`MatchDateTimeUTC`) on the original openligadb.de/api endpoints, and
    camelCase (`matchDateTimeUTC`) on api.openligadb.de, which is what
    `BASE_URL` points at. Pinning one spelling makes the adapter fail
    completely the day the other is served -- and it was: every entry raised
    `KeyError: 'MatchDateTimeUTC'`, so a whole season parsed to nothing.

    Matching case-insensitively costs one dict comprehension per miss and
    survives the switch in either direction, which is worth more than being
    strict about a capital letter nobody promised us.
    """
    if not isinstance(entry, dict):
        raise TypeError(f"expected a JSON object, got {type(entry).__name__}")
    if name in entry:
        return entry[name]
    folded = {key.lower(): value for key, value in entry.items()}
    if name.lower() in folded:
        return folded[name.lower()]
    raise KeyError(name)


def opt(entry, name: str, default=None):
    """`field`, but a missing field is not an error."""
    try:
        return field(entry, name)
    except (KeyError, TypeError):
        return default


def parse_kickoff(entry) -> datetime:
    """Kick-off as naive UTC.

    The UTC field is the one to trust: `MatchDateTime` is German local time,
    so reading it as UTC silently shifts every fixture by an hour or two and
    moves late Saturday games across a date boundary.
    """
    raw = opt(entry, "MatchDateTimeUTC")
    if not raw:
        raw = field(entry, "MatchDateTime")     # older payloads; local time
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).replace(tzinfo=None)
    parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def team_name(entry, side: str) -> str:
    team = field(entry, side)
    return field(team, "TeamName")


def _describe(entry, exc: Exception) -> str:
    """Say what was wrong *and* what the payload actually offered.

    `openligadb entry: 'MatchDateTimeUTC'` -- a bare KeyError repr -- says a
    key was missing but not which keys exist, which is the one fact that
    identifies the fix. Listing them turns a guess into a diagnosis.
    """
    if isinstance(exc, UnknownTeamError):
        return f"unknown club: {exc}"
    if isinstance(exc, KeyError):
        keys = ", ".join(sorted(entry)[:14]) if isinstance(entry, dict) else "not an object"
        return f"no field {exc.args[0]!r} — entry has: {keys}"
    return f"{type(exc).__name__}: {exc}"


class OpenLigaDBSource(Source):
    name = "openligadb"

    def ingest(self, seasons: list[int], league: str = "bundesliga",
               cache: bool = True) -> IngestResult:
        result = IngestResult(source=self.name)
        shortcut = LEAGUE_SHORTCUTS.get(league)
        if shortcut is None:
            raise ValueError(f"unknown league {league!r}")

        matches, results = [], []
        for start_year in seasons:
            url = f"{BASE_URL}/getmatchdata/{shortcut}/{start_year}"
            try:
                text, _ = self.fetch(url, suffix=".json", cache=cache)
                payload = json.loads(text)
            except Exception as exc:
                result.errors.append(f"{url}: {exc}")
                continue
            result.documents_fetched += 1

            m, r = self._parse_season(payload, league, season_label(start_year), result)
            matches.extend(m)
            results.extend(r)

        if matches:
            result.rows_written["match"] = self.store.upsert(
                "match", pd.DataFrame(matches).drop_duplicates("match_id"), ["match_id"])
        if results:
            result.rows_written["match_result"] = self.store.upsert(
                "match_result", pd.DataFrame(results).drop_duplicates(["match_id", "source"]),
                ["match_id", "source"])
        return result

    def _parse_season(self, payload: list, league: str, season: str,
                      result: IngestResult) -> tuple[list, list]:
        matches, results = [], []
        skipped: list[str] = []

        for entry in payload:
            try:
                kickoff = parse_kickoff(entry)
                home = resolve(team_name(entry, "Team1"))
                away = resolve(team_name(entry, "Team2"))
            except (KeyError, TypeError, ValueError, UnknownTeamError) as exc:
                skipped.append(_describe(entry, exc))
                continue

            match_id = make_match_id(league, season, home, away)
            matches.append({
                "match_id": match_id, "source": self.name, "league": league, "season": season,
                "kickoff_utc": kickoff, "home_team_id": home, "away_team_id": away,
                "known_at": kickoff - timedelta(days=30),
            })

            if not opt(entry, "MatchIsFinished"):
                continue
            scores = opt(entry, "MatchResults") or []
            final = next((r for r in scores if opt(r, "ResultTypeID") == 2), None)
            half = next((r for r in scores if opt(r, "ResultTypeID") == 1), None)
            if final is None:
                continue

            try:
                home_goals = int(field(final, "PointsTeam1"))
                away_goals = int(field(final, "PointsTeam2"))
            except (KeyError, TypeError, ValueError) as exc:
                skipped.append(_describe(final, exc))
                continue

            results.append({
                "match_id": match_id, "source": self.name,
                "home_goals": home_goals, "away_goals": away_goals,
                "outcome": "H" if home_goals > away_goals else ("D" if home_goals == away_goals else "A"),
                "ht_home": int(opt(half, "PointsTeam1")) if half and opt(half, "PointsTeam1") is not None else None,
                "ht_away": int(opt(half, "PointsTeam2")) if half and opt(half, "PointsTeam2") is not None else None,
                "known_at": kickoff + timedelta(seconds=SETTINGS.result_known_after_seconds),
            })

        # One line per season rather than one per fixture. A changed field name
        # breaks all 306 entries identically, and 306 copies of the same
        # sentence bury every other source's error in the report.
        if skipped:
            counts: dict[str, int] = {}
            for reason in skipped:
                counts[reason] = counts.get(reason, 0) + 1
            worst = max(counts, key=counts.get)
            result.errors.append(
                f"openligadb {season}: skipped {len(skipped)} of {len(payload)} "
                f"entries — {worst}"
                + (f" (and {len(counts) - 1} other reason(s))" if len(counts) > 1 else ""))
        return matches, results

    def upcoming(self, league: str = "bundesliga") -> pd.DataFrame:
        """Fixtures for the current matchday — the weekly prediction input."""
        shortcut = LEAGUE_SHORTCUTS[league]
        text, _ = self.fetch(f"{BASE_URL}/getmatchdata/{shortcut}", suffix=".json", cache=False)
        rows = []
        for entry in json.loads(text):
            try:
                rows.append({
                    "kickoff_utc": parse_kickoff(entry),
                    "home_team_id": resolve(team_name(entry, "Team1")),
                    "away_team_id": resolve(team_name(entry, "Team2")),
                    "finished": bool(opt(entry, "MatchIsFinished")),
                })
            except (KeyError, TypeError, ValueError, UnknownTeamError):
                continue
        return pd.DataFrame(rows)
