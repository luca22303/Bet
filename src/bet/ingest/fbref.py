"""FBref: per-match player lines.

Deliberately scrapes match pages rather than season totals. FBref's season
tables are cumulative and updated live, so a scrape taken today contains
matches that had not been played on the date you want to backtest, and there is
no way to subtract them back out. Per-match rows carry the final whistle as
their known_at, so any past Saturday's per-90 rates can be rebuilt exactly.

One match page carries both squads across several stat tables, so a season
costs ~306 requests rather than ~500 player pages.

Two quirks break most FBref parsers and are handled here:

    Commented tables. All but the first table on a match page are wrapped in
    HTML comments to speed up rendering. BeautifulSoup will not find them until
    the comment markers are stripped.

    Multi-level headers. Tables have a two-row header ("Performance | Gls") that
    pandas returns as a MultiIndex, with the top level repeating across groups.
    Flattening needs the group prefix or 'Att' from passing collides with 'Att'
    from take-ons.

Sports Reference's terms restrict scraping. This is rate-limited to one request
every three seconds and every page is archived so it is fetched once. Fine for
personal analysis; do not build a product on it.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import pandas as pd

from bet.config import SETTINGS
from bet.ingest.base import IngestResult, Source, make_match_id, season_label
from bet.players import make_player_id, position_group
from bet.teams import UnknownTeamError, resolve

BASE_URL = "https://fbref.com"
BUNDESLIGA_COMP_ID = 20

# Flattened FBref column -> our schema column.
COLUMN_MAP = {
    "min": "minutes",
    "performance_gls": "goals",
    "performance_ast": "assists",
    "performance_sh": "shots",
    "performance_sot": "shots_on_target",
    "performance_crdy": "yellow_cards",
    "performance_crdr": "red_cards",
    "performance_touches": "touches",
    "performance_tkl": "tackles",
    "performance_int": "interceptions",
    "performance_blocks": "blocks",
    "performance_fls": "fouls",
    "expected_xg": "xg",
    "expected_npxg": "npxg",
    "expected_xag": "xa",
    "passes_cmp": "passes_completed",
    "passes_att": "passes_attempted",
    "passes_prgp": "progressive_passes",
    "carries_carries": "carries",
    # Tables other than the summary use bare headings.
    "tackles_tkl": "tackles",
    "int": "interceptions",
    "blocks_blocks": "blocks",
    "fls": "fouls",
    "sh": "shots",
    "sot": "shots_on_target",
}

NUMERIC_COLUMNS = [
    "minutes", "goals", "assists", "shots", "shots_on_target", "xg", "npxg", "xa",
    "passes_completed", "passes_attempted", "progressive_passes", "touches",
    "carries", "tackles", "interceptions", "blocks", "fouls",
    "yellow_cards", "red_cards",
]


def strip_html_comments(html: str) -> str:
    """Unwrap FBref's commented-out tables so a parser can see them."""
    return html.replace("<!--", "").replace("-->", "")


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse a MultiIndex header into single lowercase names.

    FBref repeats a group name across its columns and uses 'Unnamed: N_level_0'
    for ungrouped ones, so the prefix is dropped when it is noise and kept when
    it disambiguates.
    """
    if not isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame.columns]
        return frame

    names = []
    for upper, lower in frame.columns:
        upper = str(upper).strip()
        lower = str(lower).strip()
        if upper.startswith("Unnamed") or not upper:
            names.append(lower.lower().replace(" ", "_"))
        else:
            names.append(f"{upper}_{lower}".lower().replace(" ", "_").replace("-", "_"))
    frame.columns = names
    return frame


def parse_minutes(value) -> float:
    """FBref writes minutes as '90' or, for two-legged totals, '45+45'."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    text = str(value).strip()
    if not text or text == "0":
        return 0.0
    try:
        return float(sum(int(part) for part in text.split("+") if part.strip()))
    except ValueError:
        return 0.0


def extract_player_id(row_html: str | None, name: str) -> str:
    """Prefer FBref's stable player id from the row's link over a name slug."""
    if row_html:
        match = re.search(r"/en/players/([a-f0-9]{8})/", row_html)
        if match:
            return make_player_id("fbref", match.group(1))
    return make_player_id(name=name)


class FBrefSource(Source):
    name = "fbref"

    def ingest(self, seasons: list[int], league: str = "bundesliga",
               cache: bool = True, max_matches: int | None = None) -> IngestResult:
        """Walk each season's schedule and parse every match page."""
        result = IngestResult(source=self.name)
        players: list[dict] = []
        stats: list[dict] = []
        lineups: list[dict] = []

        for start_year in seasons:
            season = season_label(start_year)
            try:
                links = self._season_match_links(start_year, cache=cache)
            except Exception as exc:
                result.errors.append(f"schedule {season}: {exc}")
                continue
            result.documents_fetched += 1

            if max_matches:
                links = links[:max_matches]

            for url in links:
                try:
                    html, _ = self.fetch(url, suffix=".html", cache=cache)
                except Exception as exc:
                    result.errors.append(f"{url}: {exc}")
                    continue
                result.documents_fetched += 1

                try:
                    p, s = self._parse_match_page(html, league, season, result)
                    players.extend(p)
                    stats.extend(s)
                    lineups.extend(self._lineup_rows(s))
                except Exception as exc:
                    result.errors.append(f"parse {url}: {exc}")

        if players:
            frame = pd.DataFrame(players).drop_duplicates("player_id")
            result.rows_written["player"] = self.store.upsert("player", frame, ["player_id"])
        if stats:
            frame = pd.DataFrame(stats).drop_duplicates(["match_id", "player_id", "source"])
            result.rows_written["player_match_stat"] = self.store.upsert(
                "player_match_stat", frame, ["match_id", "player_id", "source"])
        if lineups:
            frame = pd.DataFrame(lineups).drop_duplicates(
                ["match_id", "player_id", "source", "known_at"])
            result.rows_written["lineup"] = self.store.upsert(
                "lineup", frame, ["match_id", "player_id", "source", "known_at"])
        return result

    @staticmethod
    def _lineup_rows(stats: list[dict]) -> list[dict]:
        """Historical XIs, derived from who actually played.

        These are knowable at kickoff and not before, so they are stamped at the
        kickoff itself. That makes them useful for backtesting a model that
        assumes the confirmed XI, and useless for one pretending to know it
        earlier -- which is the correct distinction.
        """
        rows = []
        for record in stats:
            if not record.get("started") and not (record.get("minutes") or 0) > 0:
                continue
            rows.append({
                "match_id": record["match_id"],
                "player_id": record["player_id"],
                "team_id": record["team_id"],
                "source": "fbref",
                "is_starter": bool(record.get("started")),
                "is_confirmed": True,
                "shirt_number": None,
                "formation": None,
                # The result known_at sits after the match; an XI is known at
                # kickoff, so it is walked back to the kickoff itself.
                "known_at": record["known_at"] - timedelta(
                    seconds=SETTINGS.result_known_after_seconds),
            })
        return rows

    def _season_match_links(self, start_year: int, *, cache: bool = True) -> list[str]:
        season = f"{start_year}-{start_year + 1}"
        url = (f"{BASE_URL}/en/comps/{BUNDESLIGA_COMP_ID}/{season}/schedule/"
               f"{season}-Bundesliga-Scores-and-Fixtures")
        html, _ = self.fetch(url, suffix=".html", cache=cache)
        links = re.findall(r'href="(/en/matches/[a-f0-9]{8}/[^"]+)"', strip_html_comments(html))
        # Preserve document order while removing the duplicate links FBref emits.
        return list(dict.fromkeys(f"{BASE_URL}{href}" for href in links))

    def _parse_match_page(self, html: str, league: str, season: str,
                          result: IngestResult) -> tuple[list[dict], list[dict]]:
        clean = strip_html_comments(html)

        kickoff = self._extract_kickoff(clean)
        home_name, away_name = self._extract_teams(clean)
        if kickoff is None or home_name is None or away_name is None:
            raise ValueError("could not identify fixture from page")

        try:
            home_team = resolve(home_name)
            away_team = resolve(away_name)
        except UnknownTeamError as exc:
            result.errors.append(str(exc))
            return [], []

        match_id = make_match_id(league, season, home_team, away_team)
        # A player line is public once the match is over, never before.
        known_at = kickoff + timedelta(seconds=SETTINGS.result_known_after_seconds)

        players: list[dict] = []
        stats: dict[str, dict] = {}

        for table_html, team_id in self._player_tables(clean, home_team, away_team):
            try:
                frame = pd.read_html(StringIO(table_html))[0]
            except (ValueError, ImportError):
                continue
            frame = flatten_columns(frame)
            if "player" not in frame.columns:
                continue

            row_links = re.findall(r'<tr[^>]*>.*?</tr>', table_html, re.DOTALL)
            for i, row in frame.iterrows():
                name = str(row.get("player", "")).strip()
                # FBref appends a totals row to every table.
                if not name or name.lower().startswith(("total", "nan", "player")):
                    continue

                row_html = row_links[i + 1] if i + 1 < len(row_links) else None
                player_id = extract_player_id(row_html, name)

                players.append({
                    "player_id": player_id, "full_name": name,
                    "source": self.name,
                    "source_id": player_id.split(":", 1)[1] if ":" in player_id else None,
                    "known_at": known_at,
                })

                record = stats.setdefault(player_id, {
                    "match_id": match_id, "player_id": player_id, "team_id": team_id,
                    "source": self.name, "position": None, "started": None,
                    "known_at": known_at,
                    **{c: None for c in NUMERIC_COLUMNS},
                })

                if record["position"] is None and row.get("pos") is not None:
                    record["position"] = str(row.get("pos"))

                for source_col, target in COLUMN_MAP.items():
                    if source_col not in frame.columns:
                        continue
                    value = row.get(source_col)
                    if target == "minutes":
                        parsed = parse_minutes(value)
                    else:
                        parsed = pd.to_numeric(value, errors="coerce")
                        parsed = None if pd.isna(parsed) else float(parsed)
                    # Later tables repeat columns; keep the first non-null value.
                    if parsed is not None and record.get(target) is None:
                        record[target] = parsed

        # FBref lists starters before substitutes within each team's table.
        self._mark_starters(stats)
        return players, list(stats.values())

    @staticmethod
    def _mark_starters(stats: dict[str, dict]) -> None:
        by_team: dict[str, list[dict]] = {}
        for record in stats.values():
            by_team.setdefault(record["team_id"], []).append(record)
        for records in by_team.values():
            for i, record in enumerate(records):
                record["started"] = i < 11

    @staticmethod
    def _extract_kickoff(html: str) -> datetime | None:
        match = re.search(r'<span class="venuetime"[^>]*data-venue-epoch="(\d+)"', html)
        if match:
            return datetime.utcfromtimestamp(int(match.group(1)))
        match = re.search(r'<meta name="Description" content="[^"]*?(\w+ \d{1,2}, \d{4})', html)
        if match:
            try:
                return datetime.strptime(match.group(1), "%B %d, %Y").replace(hour=15, minute=30)
            except ValueError:
                return None
        return None

    @staticmethod
    def _extract_teams(html: str) -> tuple[str | None, str | None]:
        names = re.findall(r'<h1>\s*<span>([^<]+?)\s+vs\.?\s+([^<]+?)\s+Match Report', html)
        if names:
            return names[0][0].strip(), names[0][1].strip()
        squads = re.findall(r'id="a_[^"]*"[^>]*>\s*<a[^>]*/squads/[^"]*"[^>]*>([^<]+)</a>', html)
        if len(squads) >= 2:
            return squads[0].strip(), squads[1].strip()
        return None, None

    @staticmethod
    def _player_tables(html: str, home_team: str, away_team: str):
        """Yield each per-team player stats table with the team it belongs to.

        FBref ids them stats_{squad_hash}_{kind}; the first squad hash seen is
        the home side, matching the page's ordering.
        """
        blocks = re.findall(
            r'(<table[^>]*id="stats_([a-f0-9]{8})_(?:summary|passing|defense|misc|possession)"[^>]*>.*?</table>)',
            html, re.DOTALL)
        squad_order: list[str] = []
        for _, squad_hash in blocks:
            if squad_hash not in squad_order:
                squad_order.append(squad_hash)
        mapping = {}
        if squad_order:
            mapping[squad_order[0]] = home_team
        if len(squad_order) > 1:
            mapping[squad_order[1]] = away_team
        for table_html, squad_hash in blocks:
            team = mapping.get(squad_hash)
            if team:
                yield table_html, team
