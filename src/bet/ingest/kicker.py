"""kicker.de: predicted and confirmed line-ups, and the live match ticker.

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

The ticker (`parse_ticker_page`) is a pure display feature: goals, cards,
substitutions, kickoff/half/full-time markers, shown in the dashboard's Ticker
tab and never read by a model. That changes the right level of caution --
an unresolved player or a misclassified event type costs nothing beyond a
slightly less complete tab, unlike a line-up mismatch, which would corrupt
two players' histories -- so it is built to degrade gracefully rather than to
drop a whole event over one field it could not place.

The page structure here is written against kicker's public line-up and
match-ticker pages and has NOT been verified against live HTML -- outbound
access to kicker.de was blocked in the environment where this was written.
Both parsers are isolated and covered by fixture-based tests so they can be
corrected in one place once a real page is available.
"""

from __future__ import annotations

import html
import json
import re
from datetime import datetime, timedelta

import pandas as pd

from bet.ingest.base import IngestResult, Source
from bet.players import resolve_within_squad
from bet.teams import UnknownTeamError, resolve

BASE_URL = "https://www.kicker.de"


# A formation is two to five numbers joined by dashes, summing to ten outfield
# players. The sum check is what separates "4-2-3-1" from a date or a score.
FORMATION_PATTERN = re.compile(r"\b(\d(?:-\d){1,4})\b")

# Words that look like names in the markup but are not players.
_NAME_STOPWORDS = {
    "aufstellung", "startelf", "bank", "ersatzbank", "trainer", "coach",
    "auswechslungen", "tore", "spielinfo", "live", "ticker", "mehr",
    "anzeige", "werbung", "lineup", "substitutes", "manager",
}


def plausible_formation(text: str) -> str | None:
    """The first formation-shaped string whose numbers add up to ten.

    The arithmetic matters. A page is full of digit-dash-digit runs -- dates,
    scores, aggregate results -- and without the sum check a parser happily
    reads "2-1" as a formation.
    """
    for match in FORMATION_PATTERN.finditer(text):
        candidate = match.group(1)
        try:
            if sum(int(part) for part in candidate.split("-")) == 10:
                return candidate
        except ValueError:
            continue
    return None


def _clean_name(raw: str) -> str:
    text = html.unescape(raw).strip()
    text = re.sub(r"\s+", " ", text)
    # Shirt numbers lead many line-up entries.
    text = re.sub(r"^\d{1,2}[.\s]+", "", text)
    return text.strip()


def _is_name_like(text: str) -> bool:
    """Whether a string could be a player's name.

    Deliberately loose on shape and strict on the stopword list: German pages
    carry headings and section labels that pass any length or capitalisation
    test, and those are what pollute a line-up.
    """
    if not (2 <= len(text) <= 40):
        return False
    if text.lower() in _NAME_STOPWORDS:
        return False
    if any(char.isdigit() for char in text):
        return False
    return bool(re.search(r"[A-Za-zÀ-ÿ]", text))


def extract_embedded_json(html_text: str) -> list[dict]:
    """Any JSON payloads the page embeds.

    Tried first because a site that ships its data as JSON gives a far more
    stable target than its markup: a redesign changes class names constantly
    and the data shape rarely.
    """
    payloads = []
    patterns = [
        r'<script[^>]+id="__NEXT_DATA__"[^>]*>(.*?)</script>',
        r"window\.__NUXT__\s*=\s*(\{.*?\});",
        r"window\.__INITIAL_STATE__\s*=\s*(\{.*?\});",
        r'<script[^>]+type="application/ld\+json"[^>]*>(.*?)</script>',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, html_text, re.DOTALL):
            try:
                payloads.append(json.loads(match.group(1)))
            except (json.JSONDecodeError, ValueError):
                continue
    return payloads


def _walk_for_lineups(node, found: list[dict], depth: int = 0) -> None:
    """Search a decoded JSON tree for anything shaped like a line-up.

    The key names differ between sites and change between releases, so the
    match is on shape -- a list of objects each carrying a name-ish field --
    rather than on a path someone wrote down once.
    """
    if depth > 8:
        return
    if isinstance(node, dict):
        for key, value in node.items():
            if (isinstance(value, list) and len(value) >= 10
                    and all(isinstance(v, dict) for v in value[:3])):
                names = [_name_from(v) for v in value]
                if sum(1 for n in names if n) >= 10:
                    # The club name usually sits beside the squad rather than
                    # inside it, so the containing object is where to look.
                    found.append({
                        "key": key,
                        "team": _team_from(node),
                        "players": [n for n in names if n],
                    })
            _walk_for_lineups(value, found, depth + 1)
    elif isinstance(node, list):
        for item in node[:40]:
            _walk_for_lineups(item, found, depth + 1)


def _team_from(container: dict) -> str | None:
    """A club name from the object that holds a squad list."""
    for key in ("name", "teamName", "clubName", "shortName", "title", "longName"):
        value = container.get(key)
        if isinstance(value, str) and 2 <= len(value.strip()) <= 40:
            return value.strip()
    return None


def _name_from(entry: dict) -> str | None:
    for key in ("name", "shortName", "fullName", "playerName", "displayName",
                "lastName", "surname"):
        value = entry.get(key)
        if isinstance(value, str) and _is_name_like(value.strip()):
            return value.strip()
    return None


def parse_lineup_page(html_text: str) -> dict:
    """Pull team names, formations and named players from a line-up page.

    Returns `{"teams": [...], "confirmed": bool, "strategy": str}`.

    Three strategies, tried in order of how well each survives a redesign:
    embedded JSON, then class-attribute markup, then a generic scan for
    formation strings and name-like text. `strategy` names the one that worked,
    so a page that starts parsing differently is visible rather than silent.

    kicker.de is unreachable from the environment this was written in, so none
    of these has met the real page. They are ordered so that the most
    structure-independent one is the last line of defence rather than the only
    one.
    """
    confirmed = bool(re.search(
        r"(?i)(aufstellung\s+best[aä]tigt|offizielle\s+aufstellung|confirmed\s+line)",
        html_text))

    for strategy, extractor in (("embedded-json", _teams_from_json),
                                ("markup", _teams_from_markup),
                                ("generic", _teams_from_text)):
        teams = extractor(html_text)
        if teams:
            return {"teams": teams, "confirmed": confirmed, "strategy": strategy}

    return {"teams": [], "confirmed": confirmed, "strategy": "none"}


def _teams_from_json(html_text: str) -> list[dict]:
    found: list[dict] = []
    for payload in extract_embedded_json(html_text):
        _walk_for_lineups(payload, found)
    if len(found) < 2:
        return []

    formation = plausible_formation(html_text)
    teams = []
    for block in found[:2]:
        teams.append({"name": block.get("team") or block["key"],
                      "formation": formation,
                      "players": [_clean_name(n) for n in block["players"][:18]]})
    return teams


def _teams_from_markup(html_text: str) -> list[dict]:
    """Class-attribute markup, matched loosely.

    Matches any class *containing* the words rather than equalling them, since
    build tooling routinely suffixes class names with content hashes.
    """
    blocks = re.findall(
        r'<(?:div|section|ul)[^>]*class="[^"]*(?:lineup|aufstellung|formation)[^"]*"[^>]*>(.*?)(?=<(?:div|section|ul)[^>]*class="[^"]*(?:lineup|aufstellung|formation)|\Z)',
        html_text, re.DOTALL | re.IGNORECASE)

    teams = []
    for block in blocks:
        name_match = re.search(
            r'class="[^"]*(?:team-?name|club|verein)[^"]*"[^>]*>\s*([^<]{3,40})<', block,
            re.IGNORECASE)
        # The negative lookbehind matters: a bare `name` pattern also matches
        # `team-name`, which quietly adds the club to its own line-up as a
        # twelfth player.
        players = [
            _clean_name(p) for p in re.findall(
                r'class="[^"]*(?<!team-)(?<!team_)(?:player-?name|spieler|name)[^"]*"'
                r'[^>]*>\s*([^<]{2,40})<',
                block, re.IGNORECASE)]
        players = [p for p in players if _is_name_like(p)]
        if name_match and len(players) >= 10:
            teams.append({"name": name_match.group(1).strip(),
                          "formation": plausible_formation(block),
                          "players": players[:18]})
    return teams[:2]


def _teams_from_text(html_text: str) -> list[dict]:
    """Last resort: formation strings plus the names that follow them.

    Assumes almost nothing about the markup -- only that a formation appears
    near its eleven players, which is true of any layout a reader could
    understand.
    """
    formations = [(m.start(), m.group(1))
                  for m in FORMATION_PATTERN.finditer(html_text)
                  if sum(int(p) for p in m.group(1).split("-")) == 10]
    if len(formations) < 2:
        return []

    teams = []
    for index, (position, formation) in enumerate(formations[:2]):
        end = formations[index + 1][0] if index + 1 < len(formations) else len(html_text)
        segment = html_text[position:end]
        names = [_clean_name(n) for n in re.findall(r">\s*([^<>{}]{2,40}?)\s*<", segment)]
        names = [n for n in names if _is_name_like(n)]
        if len(names) >= 10:
            teams.append({"name": names[0], "formation": formation,
                          "players": names[1:19]})
    return teams if len(teams) == 2 else []


# ------------------------------------------------------------- live ticker


# Checked in order: an own goal and a missed or saved penalty both contain
# the word "Tor"/"Elfmeter" that a plain goal or scored penalty also uses, so
# the more specific pattern has to be tried first or it never gets a turn.
EVENT_KEYWORDS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    ("own_goal", re.compile(r"(?i)eigentor")),
    ("penalty_goal", re.compile(r"(?i)elfmeter.{0,25}(verwandelt|nutzt|versenkt)")),
    ("penalty_missed", re.compile(r"(?i)elfmeter.{0,25}(verschossen|pariert|kl[aä]rt)")),
    ("goal", re.compile(r"(?i)\btor\b")),
    ("second_yellow", re.compile(r"(?i)gelb-?rote karte")),
    ("red_card", re.compile(r"(?i)rote karte")),
    ("yellow_card", re.compile(r"(?i)gelbe karte")),
    ("substitution", re.compile(r"(?i)wechsel|ausgewechselt|eingewechselt")),
    ("var_review", re.compile(r"(?i)video-?beweis|\bvar\b")),
    ("kickoff", re.compile(r"(?i)anpfiff")),
    ("half_time", re.compile(r"(?i)halbzeit")),
    ("full_time", re.compile(r"(?i)abpfiff|spielende")),
)

# "45'", "90+2'" -- the near-universal minute marker any football ticker
# uses, in any language, which is what makes the generic text fallback below
# possible without knowing anything about the page's markup.
MINUTE_PATTERN = re.compile(r"(\d{1,3})(?:\+(\d{1,2}))?\s*[’'`´]")


def classify_event(text: str) -> str:
    """The event type a ticker line describes, from its own wording alone.

    Falls back to "note" for anything unrecognised -- a substitution
    announcement worded differently, a VAR delay, commentary -- so an
    unfamiliar line is still shown with its original text rather than
    dropped or misfiled.
    """
    for event_type, pattern in EVENT_KEYWORDS:
        if pattern.search(text):
            return event_type
    return "note"


def _parse_minute(text: str) -> tuple[int | None, int | None]:
    match = MINUTE_PATTERN.search(text)
    if not match:
        return None, None
    return int(match.group(1)), (int(match.group(2)) if match.group(2) else None)


def _clean_ticker_text(raw: str) -> str:
    text = html.unescape(raw)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _team_name_from_event(entry: dict) -> str | None:
    team = entry.get("team")
    if isinstance(team, dict):
        return _team_from(team)
    if isinstance(team, str) and team.strip():
        return team.strip()
    return _team_from(entry)


def _player_name_from_event(entry: dict) -> str | None:
    player = entry.get("player") or entry.get("scorer")
    if isinstance(player, dict):
        return _name_from(player)
    if isinstance(player, str) and _is_name_like(player.strip()):
        return player.strip()
    return _name_from(entry)


def _looks_like_ticker_entry(entry: dict) -> bool:
    has_minute = any(k in entry for k in ("minute", "min", "time", "gameTime"))
    has_text = any(isinstance(entry.get(k), str) and entry.get(k).strip()
                   for k in ("text", "description", "message", "event", "title"))
    return has_minute and has_text


def _event_from_dict(entry: dict) -> dict | None:
    minute_raw = next((entry[k] for k in ("minute", "min", "time", "gameTime")
                       if k in entry), None)
    text = next((entry[k].strip() for k in ("text", "description", "message", "event", "title")
                if isinstance(entry.get(k), str) and entry.get(k).strip()), None)
    if minute_raw is None or not text:
        return None

    if isinstance(minute_raw, str):
        minute, stoppage = _parse_minute(minute_raw + "'")
        if minute is None:
            digits = re.match(r"\s*(\d{1,3})", minute_raw)
            minute = int(digits.group(1)) if digits else None
    else:
        try:
            minute, stoppage = int(minute_raw), None
        except (TypeError, ValueError):
            minute, stoppage = None, None
    if minute is None:
        return None

    description = _clean_ticker_text(text)
    return {
        "minute": minute, "stoppage": stoppage, "description": description,
        "event_type": classify_event(description),
        "team": _team_name_from_event(entry),
        "player": _player_name_from_event(entry),
    }


def _walk_for_events(node, found: list[dict], depth: int = 0) -> None:
    """Search a decoded JSON tree for anything shaped like a ticker.

    Same shape-not-path approach as `_walk_for_lineups`: matched on "a list of
    several dict entries, each carrying a minute and some text", since the key
    names holding that list differ between sites and between releases.
    """
    if depth > 8:
        return
    if isinstance(node, dict):
        for key, value in node.items():
            if (isinstance(value, list) and len(value) >= 2
                    and all(isinstance(v, dict) for v in value[:3])
                    and sum(1 for v in value[:5] if _looks_like_ticker_entry(v)) >= 2):
                for entry in value:
                    event = _event_from_dict(entry)
                    if event:
                        found.append(event)
                continue        # this list is the ticker; nothing more in it
            _walk_for_events(value, found, depth + 1)
    elif isinstance(node, list):
        for item in node[:200]:
            _walk_for_events(item, found, depth + 1)


def _events_from_json(html_text: str) -> list[dict]:
    events: list[dict] = []
    for payload in extract_embedded_json(html_text):
        _walk_for_events(payload, events)
    return events


def _events_from_markup(html_text: str) -> list[dict]:
    """Class-attribute markup, matched loosely by content the same way the
    line-up markup tier is: any block whose class merely contains a plausible
    ticker word, since build tooling routinely suffixes class names."""
    blocks = re.findall(
        r'<(?:div|li|article)[^>]*class="[^"]*(?:ticker|matchevent|spielverlauf)'
        r'[^"]*"[^>]*>(.*?)</(?:div|li|article)>',
        html_text, re.DOTALL | re.IGNORECASE)

    events = []
    for block in blocks:
        text = _clean_ticker_text(block)
        minute, stoppage = _parse_minute(text)
        if minute is None:
            continue
        description = MINUTE_PATTERN.sub("", text, count=1).strip(" -:.")
        if len(description) < 4:
            continue
        events.append({
            "minute": minute, "stoppage": stoppage, "description": description,
            "event_type": classify_event(description), "team": None, "player": None,
        })
    return events


def _events_from_text(html_text: str) -> list[dict]:
    """Last resort: every minute marker on the page, paired with the text
    that follows it up to the next one.

    Assumes almost nothing about the markup -- only the near-universal
    "<minute>' <description>" convention any football ticker uses, which is
    also why this needs no German-specific pattern to find the events
    themselves, only to classify what each one is afterwards.
    """
    text = _clean_ticker_text(html_text)
    markers = list(MINUTE_PATTERN.finditer(text))

    events = []
    for i, marker in enumerate(markers):
        start = marker.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else min(start + 200, len(text))
        description = text[start:end].strip(" -:.")
        if len(description) < 8 or not re.search(r"[A-Za-zÀ-ÿ]{3,}", description):
            continue
        events.append({
            "minute": int(marker.group(1)),
            "stoppage": int(marker.group(2)) if marker.group(2) else None,
            "description": description,
            "event_type": classify_event(description), "team": None, "player": None,
        })
    return events[:200]


def parse_ticker_page(html_text: str) -> dict:
    """Pull a match's event log from a live-ticker page.

    Returns `{"events": [...], "strategy": str}`, each event carrying
    `sequence` (chronological order, since a minute alone does not
    disambiguate two events sharing it), `minute`, `stoppage`, `event_type`,
    `team`, `player` and `description`.

    Three strategies, same ordering logic as `parse_lineup_page`: embedded
    JSON survives a redesign best, then class-attribute markup, then a
    generic minute-marker scan as the strategy independent of markup
    entirely. `strategy` names the one that worked, so a page that starts
    parsing differently is visible rather than silently empty.
    """
    for strategy, extractor in (("embedded-json", _events_from_json),
                                ("markup", _events_from_markup),
                                ("generic", _events_from_text)):
        events = extractor(html_text)
        if events:
            events.sort(key=lambda e: (e["minute"], e["stoppage"] or 0))
            for sequence, event in enumerate(events):
                event["sequence"] = sequence
            return {"events": events, "strategy": strategy}
    return {"events": [], "strategy": "none"}


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

    def ingest_ticker(self, urls: list[str], match_ids: list[str] | None = None,
                      kickoffs: list[datetime] | None = None,
                      cache: bool = False) -> IngestResult:
        """Fetch a match's live-ticker page and store its event log.

        Purely a display feature -- see the module docstring -- so an event
        whose team or player could not be resolved is still kept, unlike an
        unmatched line-up entry. `cache` defaults to False for the same
        reason as the line-up fetch: a ticker changes throughout the match.
        """
        result = IngestResult(source=self.name)
        rows: list[dict] = []

        for i, url in enumerate(urls):
            try:
                html_text, _ = self.fetch(url, suffix=".html", cache=cache)
            except Exception as exc:
                result.errors.append(f"{url}: {exc}")
                continue
            result.documents_fetched += 1

            parsed = parse_ticker_page(html_text)
            if not parsed["events"]:
                result.errors.append(f"{url}: no ticker events found on page")
                continue

            match_id = match_ids[i] if match_ids and i < len(match_ids) else None
            kickoff = kickoffs[i] if kickoffs and i < len(kickoffs) else None
            if match_id is None or kickoff is None:
                result.errors.append(f"{url}: no fixture supplied to attach the ticker to")
                continue

            rows.extend(self._event_rows(parsed["events"], match_id, kickoff))

        if rows:
            frame = pd.DataFrame(rows).drop_duplicates(["match_id", "source", "sequence"])
            result.rows_written["match_event"] = self.store.upsert(
                "match_event", frame, ["match_id", "source", "sequence"])
        return result

    def _event_rows(self, events: list[dict], match_id: str,
                    kickoff: datetime) -> list[dict]:
        rows = []
        for event in events:
            team_id = None
            if event.get("team"):
                try:
                    team_id = resolve(event["team"])
                except UnknownTeamError:
                    team_id = None        # kept as description text either way

            player_id = None
            if event.get("player") and team_id:
                squad = self.store.squad_as_of(kickoff, team_id)
                player_id = resolve_within_squad(event["player"], squad)

            minute = event.get("minute") or 0
            # The page gives a minute, not a wall-clock time, so kickoff plus
            # that minute is the best available floor for when this became
            # knowable -- always at or after kickoff, never before it.
            known_at = kickoff + timedelta(minutes=minute)
            rows.append({
                "match_id": match_id, "source": self.name, "sequence": event["sequence"],
                "minute": event.get("minute"), "stoppage": event.get("stoppage"),
                "event_type": event["event_type"], "team_id": team_id,
                "player_id": player_id, "player_name": event.get("player"),
                "detail": None, "description": event["description"],
                "known_at": known_at,
            })
        return rows
