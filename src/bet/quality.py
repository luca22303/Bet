"""Data quality: reconciliation, coverage, and staleness.

Scrapers do not fail loudly. A site changes its markup and the parser starts
returning empty tables; a column is renamed and a stat silently becomes null; a
club is spelled differently after a sponsorship change and half a season lands
under a team that does not exist. Every one of those produces a model that keeps
running and quietly gets worse, which is the worst failure mode available.

The defence is redundancy plus checking. Results arrive from two independent
sources (football-data.co.uk and OpenLigaDB), so they can be compared; where a
field has only one source, the checks here are coverage and staleness instead --
weaker, and labelled as such rather than dressed up.

`bet quality` should be run after every ingest. A clean report is not proof the
data is right, but a dirty one is proof something is wrong, and that is the more
useful of the two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


@dataclass
class QualityIssue:
    severity: str          # error / warning / info
    check: str
    detail: str
    count: int = 0

    def __str__(self) -> str:
        marker = {"error": "ERROR", "warning": "WARN ", "info": "INFO "}.get(self.severity, "?")
        suffix = f" ({self.count})" if self.count else ""
        return f"{marker} {self.check}: {self.detail}{suffix}"


@dataclass
class QualityReport:
    as_of: datetime
    issues: list[QualityIssue] = field(default_factory=list)
    coverage: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def errors(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_text(self) -> str:
        lines = [f"DATA QUALITY - {self.as_of:%Y-%m-%d %H:%M} UTC", "=" * 64]
        if not self.coverage.empty:
            lines.append("")
            lines.append("coverage")
            lines.append(self.coverage.to_string(index=False))
        lines.append("")
        if self.issues:
            lines.extend(str(issue) for issue in sorted(
                self.issues, key=lambda i: {"error": 0, "warning": 1, "info": 2}[i.severity]))
        else:
            lines.append("no issues found")
        lines.append("")
        lines.append(f"{len(self.errors)} errors, {len(self.warnings)} warnings")
        return "\n".join(lines)


def reconcile_results(store) -> list[QualityIssue]:
    """Compare results ingested from different sources for the same fixture.

    The strongest check available, because it needs no assumption about what a
    correct value looks like: two independent scrapes of the same match either
    agree or one of them is wrong.
    """
    issues = []
    rows = store.con.execute(
        """
        SELECT r.match_id, r.source, r.home_goals, r.away_goals
        FROM match_result r
        """
    ).df()
    if rows.empty:
        return [QualityIssue("warning", "reconcile_results", "no results in the store")]

    grouped = rows.groupby("match_id").agg(
        sources=("source", "nunique"),
        home_variants=("home_goals", "nunique"),
        away_variants=("away_goals", "nunique"),
    ).reset_index()

    multi = grouped[grouped["sources"] > 1]
    if multi.empty:
        issues.append(QualityIssue(
            "info", "reconcile_results",
            "only one source per match, so results cannot be cross-checked; "
            "ingest openligadb alongside football_data to enable this"))
        return issues

    conflicts = multi[(multi["home_variants"] > 1) | (multi["away_variants"] > 1)]
    if not conflicts.empty:
        issues.append(QualityIssue(
            "error", "reconcile_results",
            "sources disagree on the scoreline", len(conflicts)))
    else:
        issues.append(QualityIssue(
            "info", "reconcile_results",
            f"{len(multi)} matches cross-checked, all sources agree"))
    return issues


def check_odds_sanity(store) -> list[QualityIssue]:
    """Prices that cannot be real, and books whose margin looks wrong."""
    issues = []
    impossible = store.con.execute(
        "SELECT COUNT(*) FROM odds_quote WHERE decimal_odds <= 1.0 OR decimal_odds > 1000"
    ).fetchone()[0]
    if impossible:
        issues.append(QualityIssue("error", "odds_sanity",
                                   "odds outside a plausible range", impossible))

    booksums = store.con.execute(
        """
        SELECT match_id, book, SUM(1.0 / decimal_odds) AS booksum
        FROM odds_quote
        WHERE market = '1x2' AND is_closing
        GROUP BY match_id, book
        HAVING COUNT(*) = 3
        """
    ).df()
    if booksums.empty:
        issues.append(QualityIssue("warning", "odds_sanity", "no complete 1X2 closing books"))
        return issues

    # A book summing below 1.0 is an arbitrage, which is far more likely to be a
    # parsing error than a real price.
    arb = booksums[booksums["booksum"] < 0.995]
    if not arb.empty:
        issues.append(QualityIssue("error", "odds_sanity",
                                   "closing book sums below 1.0 (likely mis-parsed)", len(arb)))

    fat = booksums[booksums["booksum"] > 1.20]
    if not fat.empty:
        issues.append(QualityIssue("warning", "odds_sanity",
                                   "closing book margin above 20%", len(fat)))

    issues.append(QualityIssue(
        "info", "odds_sanity",
        f"median closing overround {booksums['booksum'].median() - 1:.2%} "
        f"across {len(booksums)} books"))
    return issues


def check_coverage(store, *, seasons_expected: int | None = None) -> tuple[pd.DataFrame, list[QualityIssue]]:
    """Rows per season per table, to make a gap visible.

    A season with 300 matches and zero player rows is a broken FBref run, and it
    is invisible in any aggregate count.
    """
    issues = []
    frame = store.con.execute(
        """
        SELECT m.season,
               COUNT(DISTINCT m.match_id) AS matches,
               COUNT(DISTINCT r.match_id) AS results,
               COUNT(DISTINCT s.match_id) AS with_shots,
               COUNT(DISTINCT p.match_id) AS with_player_stats,
               COUNT(DISTINCT t.match_id) AS with_team_stats,
               COUNT(DISTINCT o.match_id) AS with_odds
        FROM match m
        LEFT JOIN match_result r ON r.match_id = m.match_id
        LEFT JOIN shot s ON s.match_id = m.match_id
        LEFT JOIN player_match_stat p ON p.match_id = m.match_id
        LEFT JOIN team_match_stat t ON t.match_id = m.match_id
        LEFT JOIN odds_quote o ON o.match_id = m.match_id
        GROUP BY m.season ORDER BY m.season
        """
    ).df()

    if frame.empty:
        return frame, [QualityIssue("error", "coverage", "the store is empty")]

    for row in frame.itertuples():
        if row.matches and row.results < row.matches * 0.95:
            issues.append(QualityIssue(
                "warning", "coverage",
                f"{row.season}: {row.results}/{row.matches} matches have results"))
        # A Bundesliga season is 306 matches; anything far below that mid-season
        # is a partial ingest rather than a real gap.
        if row.matches and row.matches < 200:
            issues.append(QualityIssue(
                "info", "coverage",
                f"{row.season}: only {row.matches} matches (partial season?)"))

    if seasons_expected and len(frame) < seasons_expected:
        issues.append(QualityIssue(
            "warning", "coverage",
            f"{len(frame)} seasons present, {seasons_expected} expected"))
    return frame, issues


def check_staleness(store, as_of: datetime | None = None, *,
                    warn_after_days: int = 10) -> list[QualityIssue]:
    """How long ago each fact type was last updated.

    Catches the failure where ingestion silently stopped working weeks ago and
    the model has been running on an ageing snapshot ever since.
    """
    as_of = as_of or datetime.utcnow()
    issues = []
    tables = {
        "match_result": "known_at",
        "odds_quote": "known_at",
        "shot": "known_at",
        "player_match_stat": "known_at",
        "team_rating": "known_at",
        "player_availability": "known_at",
    }
    for table, column in tables.items():
        try:
            latest = store.con.execute(f"SELECT MAX({column}) FROM {table}").fetchone()[0]
        except Exception:
            continue
        if latest is None:
            issues.append(QualityIssue("info", "staleness", f"{table} is empty"))
            continue
        age = (pd.Timestamp(as_of) - pd.Timestamp(latest)).days
        if age > warn_after_days:
            issues.append(QualityIssue(
                "warning", "staleness",
                f"{table} newest row is {age} days old"))
    return issues


def check_team_identity(store) -> list[QualityIssue]:
    """Teams that appear too rarely to be real.

    The signature of a name-resolution failure: a club splits into a canonical
    id with most of its matches and a near-duplicate with a handful.
    """
    issues = []
    counts = store.con.execute(
        """
        SELECT team_id, COUNT(*) AS appearances FROM (
            SELECT home_team_id AS team_id FROM match
            UNION ALL SELECT away_team_id FROM match
        ) GROUP BY team_id ORDER BY appearances
        """
    ).df()
    if counts.empty:
        return issues

    # A club playing a full season appears 34 times. Single digits across the
    # whole store means something was mapped wrong.
    rare = counts[counts["appearances"] < 10]
    if not rare.empty:
        issues.append(QualityIssue(
            "warning", "team_identity",
            f"teams with very few appearances ({', '.join(rare['team_id'].head(5))})",
            len(rare)))
    return issues


def run_quality_checks(store, as_of: datetime | None = None,
                       *, seasons_expected: int | None = None) -> QualityReport:
    """Every check, in one report."""
    as_of = as_of or datetime.utcnow()
    report = QualityReport(as_of=as_of)

    coverage, coverage_issues = check_coverage(store, seasons_expected=seasons_expected)
    report.coverage = coverage
    report.issues.extend(coverage_issues)
    report.issues.extend(reconcile_results(store))
    report.issues.extend(check_odds_sanity(store))
    report.issues.extend(check_staleness(store, as_of))
    report.issues.extend(check_team_identity(store))

    # The point-in-time guard is the one check whose failure invalidates
    # everything else, so it is reported as an error rather than a warning.
    leakage = store.leakage_report()
    violations = int(leakage["violations"].sum())
    if violations:
        report.issues.append(QualityIssue(
            "error", "point_in_time",
            "facts are visible before they occurred; backtests are not trustworthy",
            violations))

    return report
