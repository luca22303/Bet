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


def drop_footer(table_html: str) -> str:
    """Remove a table's ``<tfoot>``.

    FBref closes every per-team table with a totals row reading "17 Players".
    It parses as an ordinary row, and its name matches no obvious skip rule, so
    without this each squad gains a phantom player carrying the team's summed
    minutes and shots -- which then distorts every per-90 rate built from them.
    """
    return re.sub(r"<tfoot\b.*?</tfoot>", "", table_html, flags=re.DOTALL | re.IGNORECASE)


def _read_table(table_html: str) -> pd.DataFrame:
    """Parse one HTML table.

    `pandas.read_html` needs a parser backend (lxml, or bs4 + html5lib) and
    raises ImportError without one. That must not be caught alongside a
    malformed-table ValueError: swallowing it turns a missing dependency into
    an ingest that quietly returns nothing, which is the exact silent-scraper
    failure this project spends a whole module trying to detect.
    """
    try:
        return pd.read_html(StringIO(drop_footer(table_html)))[0]
    except ImportError as exc:
        raise RuntimeError(
            "pandas.read_html needs a parser backend. Install the project's "
            "dependencies (`pip install -e .`) or `pip install lxml`."
        ) from exc


def strip_html_comments(html: str) -> str:
    """Unwrap FBref's commented-out tables so a parser can see them."""
    return html.replace("<!--", "").replace("-->", "")


def iter_tables(html: str) -> list[tuple[str, str | None]]:
    """Every ``<table>`` in the document, with its id when it has one.

    Deliberately does not filter on the id. The previous version matched
    ``id="stats_<hash>_summary"`` and would have returned nothing the moment
    FBref changed that convention -- and since no one here can open the live
    page, an id format is a guess. Content decides which tables matter;
    the id is only a grouping hint when present.
    """
    tables = []
    for match in re.finditer(r"<table\b([^>]*)>(.*?)</table>", html, re.DOTALL | re.IGNORECASE):
        attributes, body = match.group(1), match.group(0)
        id_match = re.search(r'id="([^"]+)"', attributes)
        tables.append((body, id_match.group(1) if id_match else None))
    return tables


def looks_like_player_table(frame: pd.DataFrame) -> bool:
    """Whether a parsed table is a per-player stat line.

    The test is content: a player column plus at least one recognisable stat.
    Every per-team table on a match page has both; the schedule, the shot log
    and the officials table do not.
    """
    columns = set(frame.columns)
    if "player" not in columns:
        return False
    return bool(columns & {"min", "minutes", "performance_gls", "performance_sh",
                           "expected_xg", "passes_cmp", "tackles_tkl", "sh", "gls"})


# Rows that are summaries rather than players. The numeric form ("17 Players")
# is FBref's own totals row and matches no keyword, so it needs its own pattern.
_NOT_A_PLAYER = re.compile(r"^(total|nan|player|squad|\d+\s+players?)\b", re.IGNORECASE)


def _is_not_a_player(name: str) -> bool:
    return bool(_NOT_A_PLAYER.match(name.strip()))


def squad_key(table_id: str | None, position: int) -> str:
    """Group tables belonging to the same team.

    FBref ids carry a squad hash, which groups a team's summary, passing and
    defensive tables together. Without one, tables are grouped by the order
    they appear -- the page lists one team's tables then the other's, so
    alternating by position is the right fallback.
    """
    if table_id:
        hash_match = re.search(r"([a-f0-9]{8})", table_id)
        if hash_match:
            return hash_match.group(1)
    return f"position:{position}"


def extract_teams(html: str) -> tuple[str | None, str | None]:
    """Home and away side, tried several ways.

    Any one of these can break on a redesign, so all of them are attempted
    before giving up. The order runs most-structured to least.
    """
    patterns = [
        # og:title and <title> are the most stable things on a page.
        r'<meta[^>]+property="og:title"[^>]+content="([^"]+?)\s+vs\.?\s+([^"]+?)\s+Match Report',
        r"<title>\s*([^<|]+?)\s+vs\.?\s+([^<|]+?)\s+Match Report",
        r"<h1>\s*<span>([^<]+?)\s+vs\.?\s+([^<]+?)\s+Match Report",
        r"<h1>[^<]*?([A-Z][^<]*?)\s+vs\.?\s+([^<]+?)\s+Match Report",
    ]
    for pattern in patterns:
        match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip(), match.group(2).strip()

    # Last resort: the two squad links in the scorebox.
    squads = re.findall(r'/squads/[a-f0-9]{8}/[^"]*"[^>]*>([^<]{3,40})</a>', html)
    if len(squads) >= 2:
        return squads[0].strip(), squads[1].strip()
    return None, None


def extract_kickoff(html: str) -> datetime | None:
    """Kickoff time, tried several ways."""
    epoch = re.search(r'data-venue-epoch="(\d+)"', html)
    if epoch:
        return datetime.utcfromtimestamp(int(epoch.group(1)))

    iso = re.search(r'<time[^>]+datetime="([^"]+)"', html)
    if iso:
        try:
            return datetime.fromisoformat(
                iso.group(1).replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            pass

    # A date with no time: default to a typical afternoon slot. The prediction
    # cut-off is an hour out and matchdays cluster, so the imprecision is safe.
    for pattern, fmt in ((r'data-venue-date="(\d{4}-\d{2}-\d{2})"', "%Y-%m-%d"),
                         (r'<span class="venuetime"[^>]*>\s*\(?([\d:]+)', None),
                         (r"(\w+ \d{1,2}, \d{4})", "%B %d, %Y")):
        match = re.search(pattern, html)
        if match and fmt:
            try:
                return datetime.strptime(match.group(1), fmt).replace(hour=15, minute=30)
            except ValueError:
                continue
    return None


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse a MultiIndex header into single lowercase names.

    FBref repeats a group name across its columns and uses
    ``Unnamed: N_level_0`` for ungrouped ones, so the prefix is dropped when it
    is noise and kept when it disambiguates -- without it, ``Att`` from passing
    collides with ``Att`` from take-ons.
    """
    if not isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [_normalise_column(str(c)) for c in frame.columns]
        return frame

    names = []
    for upper, lower in frame.columns:
        upper, lower = str(upper).strip(), str(lower).strip()
        if upper.startswith("Unnamed") or not upper or upper == lower:
            names.append(_normalise_column(lower))
        else:
            names.append(_normalise_column(f"{upper}_{lower}"))
    frame.columns = names
    return frame


def _normalise_column(name: str) -> str:
    text = name.strip().lower().replace(" ", "_").replace("-", "_")
    return re.sub(r"[^a-z0-9_%+]", "", text)


def parse_minutes(value) -> float:
    """FBref writes minutes as '90', or '45+45' for a two-legged total."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    text = str(value).strip().replace(",", "")
    if not text or text in {"0", "nan"}:
        return 0.0
    try:
        return float(sum(int(part) for part in text.split("+") if part.strip()))
    except ValueError:
        return 0.0


def extract_player_id(row_html: str | None, name: str) -> str:
    """Prefer FBref's stable player id from the row's link over a name slug."""
    if row_html:
        match = re.search(r"/(?:en/)?players/([a-f0-9]{8})/", row_html)
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

        kickoff = extract_kickoff(clean)
        home_name, away_name = extract_teams(clean)
        if kickoff is None or home_name is None or away_name is None:
            raise ValueError(
                f"could not identify the fixture (kickoff={kickoff is not None}, "
                f"teams={home_name is not None and away_name is not None}); "
                "run `bet diagnose --source fbref --url <page>` to see what the "
                "page actually contains")

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
                frame = flatten_columns(_read_table(table_html))
            except ValueError:
                continue

            row_blocks = re.findall(r"<tr\b[^>]*>.*?</tr>", table_html, re.DOTALL)
            # Header rows come first, so data rows are the tail of the list.
            data_rows = row_blocks[-len(frame):] if len(row_blocks) >= len(frame) else row_blocks

            for i, row in frame.iterrows():
                name = str(row.get("player", "")).strip()
                # FBref appends a totals row, and repeats the header mid-table.
                if not name or _is_not_a_player(name):
                    continue

                row_html = data_rows[i] if i < len(data_rows) else None
                player_id = extract_player_id(row_html, name)

                players.append({
                    "player_id": player_id, "full_name": name, "source": self.name,
                    "source_id": player_id.split(":", 1)[1] if ":" in player_id else None,
                    "known_at": known_at,
                })

                record = stats.setdefault(player_id, {
                    "match_id": match_id, "player_id": player_id, "team_id": team_id,
                    "source": self.name, "position": None, "started": None,
                    "known_at": known_at, **{c: None for c in NUMERIC_COLUMNS},
                })
                if record["position"] is None and row.get("pos") is not None:
                    record["position"] = str(row.get("pos"))

                self._merge_stats(record, frame.columns, row)

        if not stats:
            raise ValueError(
                "no player tables found on the page; run "
                "`bet diagnose --source fbref --url <page>` to see its structure")

        self._mark_starters(stats)
        return players, list(stats.values())

    @staticmethod
    def _merge_stats(record: dict, columns, row) -> None:
        """Copy recognised stats into the record, first non-null wins.

        Later tables repeat columns the summary already supplied, so the first
        value seen is kept rather than being overwritten by a narrower table.
        """
        for source_column, target in COLUMN_MAP.items():
            if source_column not in columns:
                continue
            value = row.get(source_column)
            if target == "minutes":
                parsed = parse_minutes(value)
            else:
                number = pd.to_numeric(value, errors="coerce")
                parsed = None if pd.isna(number) else float(number)
            if parsed is not None and record.get(target) is None:
                record[target] = parsed

    @staticmethod
    def _mark_starters(stats: dict[str, dict]) -> None:
        """Flag the first eleven of each side.

        FBref lists starters before substitutes within a team's table, so
        document order carries the distinction that no column does.
        """
        by_team: dict[str, list[dict]] = {}
        for record in stats.values():
            by_team.setdefault(record["team_id"], []).append(record)
        for records in by_team.values():
            for i, record in enumerate(records):
                record["started"] = i < 11

    def _player_tables(self, html: str, home_team: str, away_team: str):
        """Yield each per-player table with the team it belongs to.

        Tables are picked by content -- a player column plus a recognisable
        stat -- rather than by id, so a change to FBref's id convention cannot
        silently empty the ingest. Ids still group a team's several tables when
        present; otherwise the page's own ordering does, since it lists one
        side's tables before the other's.
        """
        candidates = []
        for position, (table_html, table_id) in enumerate(iter_tables(html)):
            try:
                frame = flatten_columns(_read_table(table_html))
            except ValueError:
                continue
            if looks_like_player_table(frame):
                candidates.append((table_html, squad_key(table_id, position)))

        # First squad seen is the home side, matching the page's ordering.
        order: list[str] = []
        for _, key in candidates:
            if key not in order:
                order.append(key)
        mapping = {}
        if order:
            mapping[order[0]] = home_team
        if len(order) > 1:
            mapping[order[1]] = away_team

        for table_html, key in candidates:
            team = mapping.get(key)
            if team:
                yield table_html, team



