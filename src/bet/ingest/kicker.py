"""kicker.de: predicted and confirmed line-ups for upcoming fixtures.

The only genuinely time-sensitive signal in this system. Everything else here --
xG, form, ratings -- is public days in advance and fully priced by the market
long before kickoff. The confirmed XI lands roughly an hour out and is the one
piece of information that moves a price while you can still act on it.

Note what that does and does not buy you. Retail books suspend or shade markets
the moment team news breaks, so the window is minutes, not hours, and it closes
fastest on exactly the fixtures everyone is watching. The realistic use is not
beating the market to the news; it is making sure your own model is not pricing
a fixture around a striker who is on the bench.

Historical XIs come from FBref for free, since a match page records who actually
played. This source exists for the forward-looking case.

The page structure here is written against kicker's public line-up pages and
has NOT been verified against live HTML -- outbound access to kicker.de was
blocked in the environment where this was written. `parse_lineup_page` is
isolated and covered by fixture-based tests so it can be corrected in one place
once a real page is available.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

import pandas as pd

from bet.ingest.base import IngestResult, Source
from bet.players import resolve_within_squad
from bet.teams import UnknownTeamError, resolve

BASE_URL = "https://www.kicker.de"


def parse_lineup_page(html: str) -> dict:
    """Pull team names, formations and named players out of a line-up page.

    Returns `{"teams": [{"name", "formation", "players": [...]}, ...],
              "confirmed": bool}`.

    `confirmed` distinguishes a published XI from a journalist's prediction. It
    matters more than it looks: a predicted XI is a guess with an error rate,
    and treating one as confirmed will quietly corrupt every downstream minutes
    estimate.
    """
    confirmed = bool(re.search(r"(?i)(aufstellung best[aä]tigt|offizielle aufstellung)", html))

    teams = []
    for block in re.findall(r'<div[^>]*class="[^"]*kick__lineup__team[^"]*"[^>]*>(.*?)</div>\s*</div>',
                            html, re.DOTALL):
        name_match = re.search(r'<[^>]*class="[^"]*team-name[^"]*"[^>]*>([^<]+)<', block)
        formation_match = re.search(r'(\d(?:-\d){2,4})', block)
        players = [p.strip() for p in
                   re.findall(r'<[^>]*class="[^"]*player-name[^"]*"[^>]*>([^<]+)<', block)
                   if p.strip()]
        if name_match and players:
            teams.append({
                "name": name_match.group(1).strip(),
                "formation": formation_match.group(1) if formation_match else None,
                "players": players,
            })

    return {"teams": teams, "confirmed": confirmed}


class KickerSource(Source):
    name = "kicker"

    def ingest(self, urls: list[str], match_ids: list[str] | None = None,
               kickoffs: list[datetime] | None = None, cache: bool = False,
               lead_minutes: int = 60) -> IngestResult:
        """Fetch line-up pages and store them against known fixtures.

        `cache` defaults to False here, unlike every other adapter: a line-up
        page changes as team news firms up, and a cached copy of Thursday's
        guess is worse than no data at all.
        """
        result = IngestResult(source=self.name)
        rows: list[dict] = []

        for i, url in enumerate(urls):
            try:
                html, _ = self.fetch(url, suffix=".html", cache=cache)
            except Exception as exc:
                result.errors.append(f"{url}: {exc}")
                continue
            result.documents_fetched += 1

            parsed = parse_lineup_page(html)
            if not parsed["teams"]:
                result.errors.append(f"{url}: no line-ups found on page")
                continue

            match_id = match_ids[i] if match_ids and i < len(match_ids) else None
            kickoff = kickoffs[i] if kickoffs and i < len(kickoffs) else None
            if match_id is None or kickoff is None:
                result.errors.append(f"{url}: no fixture supplied to attach the line-up to")
                continue

            # A confirmed XI is knowable now; a prediction is stamped at the
            # moment it was read, never at the kickoff it refers to.
            known_at = (kickoff - timedelta(minutes=lead_minutes)
                        if parsed["confirmed"] else datetime.utcnow())
            rows.extend(self._rows_for_match(parsed, match_id, known_at, result))

        if rows:
            frame = pd.DataFrame(rows).drop_duplicates(
                ["match_id", "player_id", "source", "known_at"])
            result.rows_written["lineup"] = self.store.upsert(
                "lineup", frame, ["match_id", "player_id", "source", "known_at"])
        return result

    def _rows_for_match(self, parsed: dict, match_id: str, known_at: datetime,
                        result: IngestResult) -> list[dict]:
        rows = []
        for team_block in parsed["teams"]:
            try:
                team_id = resolve(team_block["name"])
            except UnknownTeamError as exc:
                result.errors.append(str(exc))
                continue

            squad = self.store.squad_as_of(known_at, team_id)
            for position, name in enumerate(team_block["players"]):
                player_id = resolve_within_squad(name, squad)
                if player_id is None:
                    # Unmatched is recorded, never guessed: a wrong match
                    # pollutes two players' histories at once.
                    result.errors.append(f"unmatched player {name!r} for {team_id}")
                    continue
                rows.append({
                    "match_id": match_id, "player_id": player_id, "team_id": team_id,
                    "source": self.name,
                    "is_starter": position < 11,
                    "is_confirmed": parsed["confirmed"],
                    "shirt_number": None,
                    "formation": team_block["formation"],
                    "known_at": known_at,
                })
        return rows
