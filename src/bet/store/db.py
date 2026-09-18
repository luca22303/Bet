"""Store access. All reads that feed a model go through an `as_of` cut."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import duckdb
import pandas as pd

from bet.config import SETTINGS
from bet.store.schema import DDL, PIT_TABLES

# Which source to believe when several report the same match. football-data.co.uk
# is first because it is the longest-established and carries the odds the rest of
# the pipeline is scored against, so results and prices stay consistent.
RESULT_SOURCE_PREFERENCE = ("football_data", "openligadb", "understat", "fbref")


@contextmanager
def connect(db_path: Path | str | None = None, *, read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    path = Path(db_path) if db_path is not None else SETTINGS.db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path), read_only=read_only)
    try:
        yield con
    finally:
        con.close()


class Store:
    """Thin wrapper over DuckDB that makes the point-in-time cut hard to skip."""

    def __init__(self, con: duckdb.DuckDBPyConnection):
        self.con = con

    @classmethod
    def open(cls, db_path: Path | str | None = None, *, read_only: bool = False) -> "Store":
        path = Path(db_path) if db_path is not None else SETTINGS.db_path
        path.parent.mkdir(parents=True, exist_ok=True)
        return cls(duckdb.connect(str(path), read_only=read_only))

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------------------------------------------------------- schema

    def init_schema(self) -> None:
        self.con.execute(DDL)

    # ----------------------------------------------------------- write paths

    def upsert(self, table: str, frame: pd.DataFrame, key_columns: list[str]) -> int:
        """Insert rows, replacing any that collide on `key_columns`.

        Ingestion is re-run constantly during development; it has to be
        idempotent or the store fills with duplicates that quietly double the
        weight of whichever matches happened to be re-fetched.
        """
        if frame.empty:
            return 0

        if "known_at" in frame.columns and frame["known_at"].isna().any():
            raise ValueError(f"{table}: known_at must be set on every row")

        self.con.register("_incoming", frame)
        try:
            cols = ", ".join(f'"{c}"' for c in frame.columns)
            predicate = " AND ".join(f't."{k}" = i."{k}"' for k in key_columns)
            self.con.execute(
                f"DELETE FROM {table} t WHERE EXISTS "
                f"(SELECT 1 FROM _incoming i WHERE {predicate})"
            )
            self.con.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM _incoming")
        finally:
            self.con.unregister("_incoming")
        return len(frame)

    # ------------------------------------------------------ point-in-time reads

    def matches_as_of(self, as_of: datetime, *, league: str | None = None, played_only: bool = True) -> pd.DataFrame:
        """Matches whose result was public by `as_of` — a model's training set."""
        where = ["m.known_at <= ?"]
        params: list = [as_of]
        if played_only:
            where.append("r.known_at <= ?")
            params.append(as_of)
        if league:
            where.append("m.league = ?")
            params.append(league)

        join = "JOIN" if played_only else "LEFT JOIN"
        sql = f"""
            SELECT m.match_id, m.league, m.season, m.kickoff_utc,
                   m.home_team_id, m.away_team_id,
                   r.home_goals, r.away_goals, r.outcome
            FROM match m
            {join} ({self._preferred_results_sql()}) r ON r.match_id = m.match_id
            WHERE {' AND '.join(where)}
            ORDER BY m.kickoff_utc
        """
        return self.con.execute(sql, params).df()

    @staticmethod
    def _preferred_results_sql() -> str:
        """One result row per match, choosing between sources deterministically.

        Several sources may report the same match. Rather than letting whichever
        ingest ran last win, a fixed preference decides, so a backtest is
        reproducible regardless of the order the ingests happened to run in.
        """
        cases = "\n                    ".join(
            f"WHEN source = '{name}' THEN {i}"
            for i, name in enumerate(RESULT_SOURCE_PREFERENCE))
        return f"""
            SELECT * FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY match_id
                    ORDER BY CASE
                    {cases}
                    ELSE 99 END
                ) AS source_rank
                FROM match_result
            ) WHERE source_rank = 1
        """

    def fixtures_between(self, start: datetime, end: datetime, *, league: str | None = None) -> pd.DataFrame:
        """Matches kicking off in a window, with results attached for scoring."""
        where = ["m.kickoff_utc >= ?", "m.kickoff_utc < ?"]
        params: list = [start, end]
        if league:
            where.append("m.league = ?")
            params.append(league)
        sql = f"""
            SELECT m.match_id, m.league, m.season, m.kickoff_utc,
                   m.home_team_id, m.away_team_id,
                   r.home_goals, r.away_goals, r.outcome
            FROM match m
            LEFT JOIN ({self._preferred_results_sql()}) r ON r.match_id = m.match_id
            WHERE {' AND '.join(where)}
            ORDER BY m.kickoff_utc
        """
        return self.con.execute(sql, params).df()

    def odds_as_of(self, as_of: datetime, match_ids: list[str] | None = None,
                   *, market: str = "1x2", book: str | None = None) -> pd.DataFrame:
        """Latest price per (match, book, selection) that was quoted by `as_of`."""
        where = ["known_at <= ?", "market = ?"]
        params: list = [as_of, market]
        if book:
            where.append("book = ?")
            params.append(book)
        if match_ids is not None:
            if not match_ids:
                return pd.DataFrame(columns=["match_id", "book", "market", "selection", "decimal_odds", "quoted_at"])
            placeholders = ", ".join("?" for _ in match_ids)
            where.append(f"match_id IN ({placeholders})")
            params.extend(match_ids)

        sql = f"""
            SELECT match_id, book, market, selection, decimal_odds, quoted_at
            FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY match_id, book, market, selection
                    ORDER BY quoted_at DESC
                ) AS rn
                FROM odds_quote
                WHERE {' AND '.join(where)}
            )
            WHERE rn = 1
        """
        return self.con.execute(sql, params).df()

    def closing_odds(self, match_ids: list[str] | None = None, *, market: str = "1x2",
                     book: str | None = None) -> pd.DataFrame:
        """Closing prices. Used for scoring only — never as a model input."""
        where = ["is_closing", "market = ?"]
        params: list = [market]
        if book:
            where.append("book = ?")
            params.append(book)
        if match_ids is not None:
            if not match_ids:
                return pd.DataFrame(columns=["match_id", "book", "selection", "decimal_odds"])
            placeholders = ", ".join("?" for _ in match_ids)
            where.append(f"match_id IN ({placeholders})")
            params.extend(match_ids)
        sql = f"""
            SELECT match_id, book, market, selection, decimal_odds, quoted_at
            FROM odds_quote
            WHERE {' AND '.join(where)}
        """
        return self.con.execute(sql, params).df()

    def ratings_as_of(self, as_of: datetime, *, source: str = "clubelo") -> pd.DataFrame:
        """Most recent rating per team that was published by `as_of`."""
        sql = """
            SELECT team_id, rating, valid_from
            FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY team_id ORDER BY valid_from DESC
                ) AS rn
                FROM team_rating
                WHERE known_at <= ? AND source = ?
            )
            WHERE rn = 1
        """
        return self.con.execute(sql, [as_of, source]).df()

    def shots_as_of(self, as_of: datetime) -> pd.DataFrame:
        return self.con.execute(
            "SELECT * FROM shot WHERE known_at <= ? ORDER BY match_id, minute", [as_of]
        ).df()

    # ------------------------------------------------- point-in-time: players

    def player_stats_as_of(self, as_of: datetime, *, player_ids: list[str] | None = None,
                           since: datetime | None = None) -> pd.DataFrame:
        """Per-match player lines that were public by `as_of`.

        `since` restricts to a recent window, which is what form-sensitive rates
        want: a striker's shot rate two seasons ago says less than his rate over
        the last ten matches.
        """
        where = ["s.known_at <= ?"]
        params: list = [as_of]
        if since is not None:
            where.append("s.known_at >= ?")
            params.append(since)
        if player_ids is not None:
            if not player_ids:
                return pd.DataFrame()
            placeholders = ", ".join("?" for _ in player_ids)
            where.append(f"s.player_id IN ({placeholders})")
            params.extend(player_ids)

        sql = f"""
            SELECT s.*, m.kickoff_utc, m.season,
                   CASE WHEN s.team_id = m.home_team_id
                        THEN m.away_team_id ELSE m.home_team_id END AS opponent_id,
                   s.team_id = m.home_team_id AS at_home
            FROM player_match_stat s
            JOIN match m ON m.match_id = s.match_id
            WHERE {' AND '.join(where)}
            ORDER BY m.kickoff_utc
        """
        return self.con.execute(sql, params).df()

    def player_rates_as_of(self, as_of: datetime, *, lookback_days: int = 540,
                           min_minutes: float = 180.0) -> pd.DataFrame:
        """Per-90 rates per player, rebuilt from the matches knowable at `as_of`.

        Computed from per-match rows rather than read from a stored season total,
        which is the only way the number can be correct for a past date.
        """
        since = as_of - timedelta(days=lookback_days)
        stats = self.player_stats_as_of(as_of, since=since)
        if stats.empty:
            return pd.DataFrame()

        counting = ["goals", "assists", "shots", "shots_on_target", "xg", "npxg", "xa",
                    "tackles", "interceptions", "blocks", "fouls",
                    "yellow_cards", "red_cards", "progressive_passes", "touches"]
        present = [c for c in counting if c in stats.columns]

        grouped = stats.groupby("player_id").agg(
            team_id=("team_id", "last"),
            position=("position", "last"),
            matches=("match_id", "nunique"),
            starts=("started", "sum"),
            minutes=("minutes", "sum"),
            **{c: (c, "sum") for c in present},
        ).reset_index()

        grouped = grouped[grouped["minutes"] >= min_minutes].copy()
        if grouped.empty:
            return grouped

        per90 = grouped["minutes"] / 90.0
        for column in present:
            grouped[f"{column}_p90"] = grouped[column] / per90
        grouped["minutes_per_match"] = grouped["minutes"] / grouped["matches"]
        grouped["start_rate"] = grouped["starts"] / grouped["matches"]
        return grouped

    def team_concessions_as_of(self, as_of: datetime, *, lookback_days: int = 540,
                               columns: tuple[str, ...] = ("shots", "shots_on_target",
                                                           "fouls", "yellow_cards")) -> pd.DataFrame:
        """What each team allows opponents, per match.

        The opponent adjustment for a player prop: a striker facing a defence
        that concedes 16 shots a game is in a different market from the same
        striker facing one that concedes 8.
        """
        since = as_of - timedelta(days=lookback_days)
        stats = self.player_stats_as_of(as_of, since=since)
        if stats.empty:
            return pd.DataFrame()

        present = [c for c in columns if c in stats.columns]
        if not present:
            return pd.DataFrame()

        by_match = stats.groupby(["match_id", "opponent_id"])[present].sum().reset_index()
        conceded = by_match.groupby("opponent_id").agg(
            matches=("match_id", "nunique"),
            **{c: (c, "sum") for c in present},
        ).reset_index().rename(columns={"opponent_id": "team_id"})

        for column in present:
            conceded[f"{column}_conceded_per_match"] = conceded[column] / conceded["matches"]
        return conceded

    def lineup_as_of(self, as_of: datetime, match_ids: list[str] | None = None,
                     *, confirmed_only: bool = False) -> pd.DataFrame:
        """Most recent line-up report per match and player, knowable at `as_of`.

        A predicted XI published on Thursday and the confirmed XI an hour before
        kickoff are both stored; this returns whichever was latest at `as_of`.
        """
        where = ["known_at <= ?"]
        params: list = [as_of]
        if confirmed_only:
            where.append("is_confirmed")
        if match_ids is not None:
            if not match_ids:
                return pd.DataFrame()
            placeholders = ", ".join("?" for _ in match_ids)
            where.append(f"match_id IN ({placeholders})")
            params.extend(match_ids)

        sql = f"""
            SELECT match_id, player_id, team_id, source, is_starter,
                   is_confirmed, shirt_number, formation, known_at
            FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY match_id, player_id
                    ORDER BY is_confirmed DESC, known_at DESC
                ) AS rn
                FROM lineup
                WHERE {' AND '.join(where)}
            )
            WHERE rn = 1
            ORDER BY known_at DESC, match_id, player_id
        """
        return self.con.execute(sql, params).df()

    def availability_as_of(self, as_of: datetime, *, team_id: str | None = None) -> pd.DataFrame:
        """Latest known status per player at `as_of`.

        Reports supersede each other, so only the most recent per player counts;
        a Tuesday "doubtful" is irrelevant once Friday says "fit".
        """
        where = ["known_at <= ?"]
        params: list = [as_of]
        if team_id:
            where.append("team_id = ?")
            params.append(team_id)
        sql = f"""
            SELECT player_id, team_id, source, status, reason,
                   expected_return, confidence, known_at
            FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY player_id ORDER BY known_at DESC
                ) AS rn
                FROM player_availability
                WHERE {' AND '.join(where)}
            )
            WHERE rn = 1
        """
        return self.con.execute(sql, params).df()

    def squad_as_of(self, as_of: datetime, team_id: str, *,
                    lookback_days: int = 400) -> dict[str, str]:
        """player_id -> full name for a team's recent squad.

        The lookup `resolve_within_squad` needs to turn a surname in a line-up
        listing into a known player.
        """
        since = as_of - timedelta(days=lookback_days)
        rows = self.con.execute(
            """
            SELECT DISTINCT s.player_id, p.full_name
            FROM player_match_stat s
            JOIN match m ON m.match_id = s.match_id
            LEFT JOIN player p ON p.player_id = s.player_id
            WHERE s.team_id = ? AND s.known_at <= ? AND s.known_at >= ?
            """,
            [team_id, as_of, since],
        ).df()
        return {r.player_id: (r.full_name or r.player_id) for r in rows.itertuples()}

    # ------------------------------------------------------------ diagnostics

    def leakage_report(self) -> pd.DataFrame:
        """Rows whose `known_at` precedes the kickoff they describe.

        A result or closing price that claims to have been knowable before the
        match started is a bug in an ingest adapter. Run this after every
        ingest; a backtest on top of a leaking store is meaningless.
        """
        checks = [
            ("match_result", """
                SELECT COUNT(*) FROM match_result r JOIN match m USING (match_id)
                WHERE r.known_at < m.kickoff_utc
            """),
            ("odds_quote", """
                SELECT COUNT(*) FROM odds_quote o JOIN match m USING (match_id)
                WHERE o.known_at > m.kickoff_utc AND o.is_closing
            """),
            ("shot", """
                SELECT COUNT(*) FROM shot s JOIN match m USING (match_id)
                WHERE s.known_at < m.kickoff_utc
            """),
            ("player_match_stat", """
                SELECT COUNT(*) FROM player_match_stat s JOIN match m USING (match_id)
                WHERE s.known_at < m.kickoff_utc
            """),
            ("lineup", """
                SELECT COUNT(*) FROM lineup l JOIN match m USING (match_id)
                WHERE l.known_at > m.kickoff_utc
            """),
        ]
        rows = [{"table": name, "violations": self.con.execute(sql).fetchone()[0]} for name, sql in checks]
        return pd.DataFrame(rows)

    def summary(self) -> pd.DataFrame:
        rows = []
        for table in ("raw_document",) + PIT_TABLES + ("prediction",):
            try:
                n = self.con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except duckdb.Error:
                n = 0
            rows.append({"table": table, "rows": n})
        return pd.DataFrame(rows)
