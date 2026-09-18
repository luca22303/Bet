"""OpenLigaDB: free community JSON API for German football.

Included because it is the one source here with no scraping and no terms-of-use
grey area — a plain public API. Coverage is fixtures, results and matchday
structure rather than advanced metrics, which makes it the right source for the
forward fixture list a weekly prediction run needs, and a useful cross-check on
results ingested from football-data.co.uk.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pandas as pd

from bet.config import SETTINGS
from bet.ingest.base import IngestResult, Source, make_match_id, season_label
from bet.teams import UnknownTeamError, resolve

BASE_URL = "https://api.openligadb.de"
LEAGUE_SHORTCUTS = {"bundesliga": "bl1", "bundesliga2": "bl2"}


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
                "match_result", pd.DataFrame(results).drop_duplicates("match_id"), ["match_id"])
        return result

    def _parse_season(self, payload: list, league: str, season: str,
                      result: IngestResult) -> tuple[list, list]:
        matches, results = [], []
        for entry in payload:
            try:
                kickoff = datetime.fromisoformat(entry["MatchDateTimeUTC"].replace("Z", "+00:00")).replace(tzinfo=None)
                home = resolve(entry["Team1"]["TeamName"])
                away = resolve(entry["Team2"]["TeamName"])
            except (KeyError, TypeError, ValueError, UnknownTeamError) as exc:
                result.errors.append(f"openligadb entry: {exc}")
                continue

            match_id = make_match_id(league, season, home, away)
            matches.append({
                "match_id": match_id, "source": self.name, "league": league, "season": season,
                "kickoff_utc": kickoff, "home_team_id": home, "away_team_id": away,
                "known_at": kickoff - timedelta(days=30),
            })

            if not entry.get("MatchIsFinished"):
                continue
            final = next((r for r in entry.get("MatchResults", []) if r.get("ResultTypeID") == 2), None)
            half = next((r for r in entry.get("MatchResults", []) if r.get("ResultTypeID") == 1), None)
            if final is None:
                continue

            home_goals, away_goals = int(final["PointsTeam1"]), int(final["PointsTeam2"])
            results.append({
                "match_id": match_id,
                "home_goals": home_goals, "away_goals": away_goals,
                "outcome": "H" if home_goals > away_goals else ("D" if home_goals == away_goals else "A"),
                "ht_home": int(half["PointsTeam1"]) if half else None,
                "ht_away": int(half["PointsTeam2"]) if half else None,
                "known_at": kickoff + timedelta(seconds=SETTINGS.result_known_after_seconds),
            })
        return matches, results

    def upcoming(self, league: str = "bundesliga") -> pd.DataFrame:
        """Fixtures for the current matchday — the weekly prediction input."""
        shortcut = LEAGUE_SHORTCUTS[league]
        text, _ = self.fetch(f"{BASE_URL}/getmatchdata/{shortcut}", suffix=".json", cache=False)
        rows = []
        for entry in json.loads(text):
            try:
                rows.append({
                    "kickoff_utc": datetime.fromisoformat(
                        entry["MatchDateTimeUTC"].replace("Z", "+00:00")).replace(tzinfo=None),
                    "home_team_id": resolve(entry["Team1"]["TeamName"]),
                    "away_team_id": resolve(entry["Team2"]["TeamName"]),
                    "finished": bool(entry.get("MatchIsFinished")),
                })
            except (KeyError, TypeError, ValueError, UnknownTeamError):
                continue
        return pd.DataFrame(rows)
