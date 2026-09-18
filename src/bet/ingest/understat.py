"""Understat: shot-level expected goals.

The cleanest free signal of team quality. Goals are a small, noisy sample —
roughly 2.8 per match — so a team's true attacking strength takes most of a
season to show up in the scoreline. Shot-level xG converges far faster, which
matters when the whole league produces only 306 matches a year.

Understat embeds its payloads in the page as JavaScript string literals of the
form `var shotsData = JSON.parse('\\x7B...')`. Parsing is a matter of finding
the assignment, pulling the escaped literal and decoding it.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta

import pandas as pd

from bet.config import SETTINGS
from bet.ingest.base import IngestResult, Source, make_match_id, season_label
from bet.teams import UnknownTeamError, resolve

BASE_URL = "https://understat.com"


def extract_json_var(html: str, variable: str):
    """Pull `var <variable> = JSON.parse('...')` out of an Understat page."""
    pattern = rf"var\s+{re.escape(variable)}\s*=\s*JSON\.parse\('(.*?)'\)"
    match = re.search(pattern, html, re.DOTALL)
    if not match:
        raise ValueError(f"variable {variable!r} not found in page")
    # The payload is hex-escaped ASCII; decoding via unicode_escape restores it.
    return json.loads(match.group(1).encode("utf-8").decode("unicode_escape"))


class UnderstatSource(Source):
    name = "understat"

    def ingest(self, seasons: list[int], league: str = "bundesliga",
               with_shots: bool = False, cache: bool = True,
               max_matches: int | None = None) -> IngestResult:
        """Ingest season match xG, and optionally every shot.

        `with_shots` costs one request per match — roughly 306 per season. Leave
        it off until the team-level xG pipeline is working, and expect it to
        take the better part of an hour per season at the default delay.
        """
        result = IngestResult(source=self.name)
        shots: list[dict] = []

        for start_year in seasons:
            url = f"{BASE_URL}/league/Bundesliga/{start_year}"
            try:
                html, _ = self.fetch(url, suffix=".html", cache=cache)
                dates_data = extract_json_var(html, "datesData")
            except Exception as exc:
                result.errors.append(f"{url}: {exc}")
                continue
            result.documents_fetched += 1

            season = season_label(start_year)
            fixtures = self._parse_fixtures(dates_data, league, season, result)

            if with_shots:
                targets = fixtures[:max_matches] if max_matches else fixtures
                for fixture in targets:
                    if not fixture["understat_id"]:
                        continue
                    try:
                        match_html, _ = self.fetch(
                            f"{BASE_URL}/match/{fixture['understat_id']}", suffix=".html", cache=cache)
                        shots_data = extract_json_var(match_html, "shotsData")
                    except Exception as exc:
                        result.errors.append(f"match {fixture['understat_id']}: {exc}")
                        continue
                    result.documents_fetched += 1
                    shots.extend(self._parse_shots(shots_data, fixture))

        if shots:
            frame = pd.DataFrame(shots).drop_duplicates("shot_id")
            result.rows_written["shot"] = self.store.upsert("shot", frame, ["shot_id"])
        return result

    def _parse_fixtures(self, dates_data: list, league: str, season: str,
                        result: IngestResult) -> list[dict]:
        fixtures = []
        for entry in dates_data:
            if not entry.get("isResult"):
                continue
            try:
                kickoff = datetime.strptime(entry["datetime"], "%Y-%m-%d %H:%M:%S")
                home = resolve(entry["h"]["title"])
                away = resolve(entry["a"]["title"])
            except (KeyError, TypeError, ValueError, UnknownTeamError) as exc:
                result.errors.append(f"understat fixture: {exc}")
                continue
            fixtures.append({
                "understat_id": entry.get("id"),
                "match_id": make_match_id(league, season, home, away),
                "kickoff_utc": kickoff,
                "home_team_id": home,
                "away_team_id": away,
            })
        return fixtures

    def _parse_shots(self, shots_data: dict, fixture: dict) -> list[dict]:
        rows = []
        known_at = fixture["kickoff_utc"] + timedelta(seconds=SETTINGS.result_known_after_seconds)

        for side in ("h", "a"):
            team_id = fixture["home_team_id"] if side == "h" else fixture["away_team_id"]
            for shot in shots_data.get(side, []):
                try:
                    rows.append({
                        "shot_id": f"{fixture['understat_id']}:{shot['id']}",
                        "match_id": fixture["match_id"],
                        "team_id": team_id,
                        "player_name": shot.get("player"),
                        "minute": int(shot["minute"]) if shot.get("minute") else None,
                        # Understat coordinates are fractions of pitch length
                        # and width, already normalised to [0, 1].
                        "x": float(shot["X"]) if shot.get("X") else None,
                        "y": float(shot["Y"]) if shot.get("Y") else None,
                        "xg": float(shot["xG"]) if shot.get("xG") else None,
                        "body_part": shot.get("shotType"),
                        "situation": shot.get("situation"),
                        "result": shot.get("result"),
                        "known_at": known_at,
                    })
                except (KeyError, TypeError, ValueError):
                    continue
        return rows


def team_xg_table(store, as_of: datetime) -> pd.DataFrame:
    """Aggregate ingested shots into xG for and against per team.

    Respects the point-in-time cut, so this is safe to call from inside a model.
    """
    shots = store.shots_as_of(as_of)
    if shots.empty:
        return pd.DataFrame(columns=["team_id", "matches", "xg_for", "xg_against"])

    matches = store.matches_as_of(as_of)[["match_id", "home_team_id", "away_team_id"]]
    merged = shots.merge(matches, on="match_id", how="inner")
    merged["opponent_id"] = merged.apply(
        lambda r: r["away_team_id"] if r["team_id"] == r["home_team_id"] else r["home_team_id"], axis=1)

    xg_for = merged.groupby("team_id")["xg"].sum().rename("xg_for")
    xg_against = merged.groupby("opponent_id")["xg"].sum().rename("xg_against")
    played = merged.groupby("team_id")["match_id"].nunique().rename("matches")

    return pd.concat([played, xg_for, xg_against], axis=1).fillna(0.0).reset_index().rename(
        columns={"index": "team_id"})
