"""Sofascore: post-match player ratings and touch-location heatmaps.

Sofascore exposes a JSON API (api.sofascore.com) rather than server-rendered
HTML, so unlike kicker/fbref this needs no multi-strategy fallback across
markup guesses -- just one defensive parse of the JSON shape, tolerant of
several plausible field names since the exact response has never been seen
from this sandbox either. Outbound access to sofascore.com was blocked in the
environment this was written in, the same as fbref.com and kicker.de; this
adapter carries the same caution and needs the same verify-on-a-real-machine
step before it is trusted.

Two purely additive facts, neither available from any other source here:

    Rating -- Sofascore's own post-match score out of 10, published by the
    site itself. Stored under its own source tag in `player_match_rating`
    rather than folded into FBref's box score: it is a subjective number the
    site computed, not a stat this project stands behind the same way it
    stands behind a shot count.

    Heatmap points -- individual touch locations, fine enough to draw a
    continuous-looking density (`spatial.pitch.smooth_touch_grid`) rather
    than FBref's three coarse zones. Real recorded locations, not tracking
    data invented here, and not the same thing as the pitch-zone breakdown
    FBref's possession table already supplies -- this is finer, that one is
    more certain to be real, and the dashboard shows both.

Both are purely for the dashboard. Nothing in the forecast model reads
either -- see `spatial/pitch.py`'s own module docstring on why spatial detail
does not belong in a model this data-constrained.

Never creates a player record. A rating or a heatmap only attaches to a
player already known from FBref, resolved the same way a kicker line-up
listing is: `resolve_within_squad` against the match's own squad, never
guessed. An unresolved name costs one rating or one player's heatmap, not a
polluted history, so an unmatched entry is skipped rather than merged.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pandas as pd

from bet.config import SETTINGS
from bet.ingest.base import IngestResult, Source
from bet.players import resolve_within_squad

BASE_URL = "https://api.sofascore.com/api/v1"

# Sofascore's own coordinate scale is 0-100 (percentage of pitch length and
# width); this project's convention (spatial/pitch.py) is 0-1, matching the
# scale Understat's shot coordinates already use.
COORDINATE_SCALE = 100.0


def _first(entry: dict, *keys, default=None):
    for key in keys:
        value = entry.get(key)
        if value is not None:
            return value
    return default


def _to_float(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _player_name(entry: dict) -> str | None:
    player = entry.get("player") if isinstance(entry.get("player"), dict) else entry
    name = _first(player, "name", "shortName", "fullName")
    return str(name).strip() or None if name else None


def _player_sofascore_id(entry: dict):
    player = entry.get("player") if isinstance(entry.get("player"), dict) else entry
    return _first(player, "id", "playerId")


def parse_lineup_ratings(payload: dict) -> dict:
    """Player identities and ratings from a Sofascore event-lineups payload.

    Returns `{"home": [...], "away": [...]}`, each entry a dict with `name`,
    `sofascore_id`, `rating` (or `None`), `position` and `is_starter`.
    Missing or oddly-shaped fields degrade to `None` rather than raising, so
    one player's unexpected shape does not lose the rest of the XI.
    """
    result: dict[str, list[dict]] = {"home": [], "away": []}
    if not isinstance(payload, dict):
        return result

    for side in ("home", "away"):
        block = payload.get(side)
        if not isinstance(block, dict):
            continue
        for entry in block.get("players") or []:
            if not isinstance(entry, dict):
                continue
            name = _player_name(entry)
            if not name:
                continue
            stats = entry.get("statistics")
            stats = stats if isinstance(stats, dict) else {}
            rating = _to_float(_first(stats, "rating", "sofascoreRating",
                                      default=_first(entry, "rating", "sofascoreRating")))
            result[side].append({
                "name": name,
                "sofascore_id": _player_sofascore_id(entry),
                "rating": rating,
                "position": entry.get("position"),
                "is_starter": not bool(entry.get("substitute", False)),
            })
    return result


def parse_heatmap_points(payload: dict) -> list[tuple[float, float]]:
    """Touch-location points from a Sofascore player-heatmap payload.

    Converts Sofascore's 0-100 coordinate scale to this project's 0-1
    convention on the way in, rather than carrying mixed units further into
    the store. A point missing either coordinate, or non-numeric, is dropped
    rather than defaulting it to some placeholder location.
    """
    if not isinstance(payload, dict):
        return []
    raw_points = payload.get("points")
    if raw_points is None:
        raw_points = payload.get("heatmap")

    points = []
    for point in raw_points or []:
        if not isinstance(point, dict):
            continue
        x, y = _to_float(point.get("x")), _to_float(point.get("y"))
        if x is None or y is None:
            continue
        points.append((x / COORDINATE_SCALE, y / COORDINATE_SCALE))
    return points


class SofascoreSource(Source):
    name = "sofascore"

    def ingest(self, event_ids: list[str], match_ids: list[str] | None = None,
              home_teams: list[str] | None = None, away_teams: list[str] | None = None,
              kickoffs: list[datetime] | None = None, cache: bool = True) -> IngestResult:
        """Fetch each match's lineups payload and store its player ratings.

        Requires the match's own home/away team ids rather than resolving a
        team name from the payload: Sofascore's lineups response only ever
        labels a side "home"/"away", so the caller's own fixture knowledge is
        the more reliable route -- the same shortcut FBref's own team
        assignment takes from page order rather than a name match.

        The four fixture-knowledge lists are optional and may be shorter
        than `event_ids` -- a missing entry costs one event, recorded as an
        error, not a crash -- the same tolerance kicker's own `ingest`/
        `ingest_ticker` give a caller that does not have every fixture's
        details in hand yet.

        Also accumulates, on `self.resolved_players`, a
        (event_id, Sofascore player id) -> (player_id, team_id, match_id,
        event_id) map for every player this call resolved, which
        `ingest_heatmaps` uses so it does not have to repeat the squad match
        and risk disagreeing with which player a rating just went to. Keyed
        on the pair rather than the player id alone: the same real player
        carries the same Sofascore id across every match they play, and
        keying on id alone would let a later match silently replace an
        earlier one still waiting on its heatmap. Accumulates rather than
        replacing on each call, so a caller batching many matches across
        several `ingest()` calls can still fetch every one of their heatmaps
        with a single later `ingest_heatmaps()`, rather than only the last
        batch's.
        """
        result = IngestResult(source=self.name)
        rating_rows: list[dict] = []
        if not hasattr(self, "resolved_players"):
            self.resolved_players: dict[tuple[str, str], dict] = {}

        for i, event_id in enumerate(event_ids):
            match_id = match_ids[i] if match_ids and i < len(match_ids) else None
            home_team = home_teams[i] if home_teams and i < len(home_teams) else None
            away_team = away_teams[i] if away_teams and i < len(away_teams) else None
            kickoff = kickoffs[i] if kickoffs and i < len(kickoffs) else None
            if None in (match_id, home_team, away_team, kickoff):
                result.errors.append(f"event {event_id}: no fixture supplied to attach it to")
                continue

            url = f"{BASE_URL}/event/{event_id}/lineups"
            try:
                text, _ = self.fetch(url, suffix=".json", cache=cache)
                payload = json.loads(text)
            except Exception as exc:
                result.errors.append(f"{url}: {exc}")
                continue
            result.documents_fetched += 1

            parsed = parse_lineup_ratings(payload)
            if not parsed["home"] and not parsed["away"]:
                result.errors.append(f"{url}: no lineup ratings found on page")
                continue

            rows, resolved = self._rating_rows_for_match(
                parsed, match_id, event_id, home_team, away_team, kickoff, result)
            rating_rows.extend(rows)
            self.resolved_players.update(resolved)

        if rating_rows:
            frame = pd.DataFrame(rating_rows).drop_duplicates(["match_id", "player_id", "source"])
            result.rows_written["player_match_rating"] = self.store.upsert(
                "player_match_rating", frame, ["match_id", "player_id", "source"])
        return result

    def _rating_rows_for_match(self, parsed: dict, match_id: str, event_id: str,
                               home_team: str, away_team: str, kickoff: datetime,
                               result: IngestResult
                               ) -> tuple[list[dict], dict[tuple[str, str], dict]]:
        # A rating, like an FBref box score, is knowable once the match is
        # over -- never before, and Sofascore gives no earlier timestamp to
        # prefer over that floor.
        known_at = kickoff + timedelta(seconds=SETTINGS.result_known_after_seconds)

        rows = []
        resolved: dict[tuple[str, str], dict] = {}
        for side, team_id in (("home", home_team), ("away", away_team)):
            squad = self.store.squad_as_of(kickoff, team_id)
            for entry in parsed[side]:
                player_id = resolve_within_squad(entry["name"], squad)
                if player_id is None:
                    # Unmatched is recorded, never guessed: a wrong match
                    # pollutes two players' histories at once.
                    result.errors.append(f"unmatched player {entry['name']!r} for {team_id}")
                    continue

                if entry.get("sofascore_id") is not None:
                    # Keyed on (event_id, sofascore_id), not the bare
                    # sofascore_id: the same real player carries the same
                    # Sofascore id across every match they play, so keying on
                    # id alone would let a later match's ref silently replace
                    # an earlier one still waiting on `ingest_heatmaps`,
                    # losing that earlier match's heatmap with no error to
                    # say why.
                    resolved[(event_id, str(entry["sofascore_id"]))] = {
                        "player_id": player_id, "team_id": team_id,
                        "match_id": match_id, "event_id": event_id, "known_at": known_at,
                    }

                if entry["rating"] is None:
                    continue
                rows.append({
                    "match_id": match_id, "player_id": player_id, "team_id": team_id,
                    "source": self.name, "rating": entry["rating"], "known_at": known_at,
                })
        return rows, resolved

    def ingest_heatmaps(self, cache: bool = True) -> IngestResult:
        """Fetch a heatmap for each player `ingest` most recently resolved.

        Built on that call's own resolution deliberately, rather than
        matching names again here -- see `ingest`'s docstring. Call `ingest`
        first; this raises if it never ran.
        """
        if not hasattr(self, "resolved_players"):
            raise RuntimeError("ingest_heatmaps needs ingest() run first, to know which "
                              "Sofascore player ids belong to which of our own players")

        result = IngestResult(source=self.name)
        point_rows: list[dict] = []

        for (event_id, sofascore_id), ref in self.resolved_players.items():
            url = f"{BASE_URL}/event/{event_id}/player/{sofascore_id}/heatmap"
            try:
                text, _ = self.fetch(url, suffix=".json", cache=cache)
                payload = json.loads(text)
            except Exception as exc:
                result.errors.append(f"{url}: {exc}")
                continue
            result.documents_fetched += 1

            points = parse_heatmap_points(payload)
            if not points:
                result.errors.append(f"{url}: no heatmap points found")
                continue

            point_rows.extend(self._heatmap_rows(points, ref))

        if point_rows:
            frame = pd.DataFrame(point_rows).drop_duplicates(
                ["match_id", "player_id", "source", "sequence"])
            result.rows_written["player_heatmap_point"] = self.store.upsert(
                "player_heatmap_point", frame, ["match_id", "player_id", "source", "sequence"])
        return result

    def _heatmap_rows(self, points: list[tuple[float, float]], ref: dict) -> list[dict]:
        # Knowable once the match is over, the same floor as the rating
        # fetched for this same player in `ingest` -- not merely "the moment
        # this second request happened to run", which would make a
        # historical match's heatmap look newly knowable today.
        return [{
            "match_id": ref["match_id"], "player_id": ref["player_id"],
            "team_id": ref["team_id"], "source": self.name, "sequence": sequence,
            "x": x, "y": y, "known_at": ref["known_at"],
        } for sequence, (x, y) in enumerate(points)]
