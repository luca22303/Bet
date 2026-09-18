"""Store access. All reads that feed a model go through an `as_of` cut."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

import duckdb
import pandas as pd

from bet.config import SETTINGS
from bet.store.schema import DDL, PIT_TABLES


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
            {join} match_result r ON r.match_id = m.match_id
            WHERE {' AND '.join(where)}
            ORDER BY m.kickoff_utc
        """
        return self.con.execute(sql, params).df()

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
            LEFT JOIN match_result r ON r.match_id = m.match_id
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
