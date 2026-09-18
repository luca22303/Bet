"""Command line entry point.

    bet init                        create the database schema
    bet ingest --source all         fetch and load free public data
    bet backtest --from 2018-08-01  walk models forward and score them
    bet status                      row counts and the leakage check
    bet check                       point-in-time integrity only
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime

import pandas as pd

from bet.config import SETTINGS
from bet.evaluation.backtest import compare
from bet.models.baselines import EloModel, HomePriorModel, MarketModel
from bet.store import Store


def _parse_season_range(text: str) -> list[int]:
    """'2015-2026' -> [2015 ... 2025]; '2019,2021' -> [2019, 2021]."""
    if "-" in text:
        lo, hi = text.split("-", 1)
        return list(range(int(lo), int(hi)))
    return [int(part) for part in text.split(",") if part.strip()]


def cmd_init(args) -> int:
    SETTINGS.ensure_dirs()
    with Store.open(args.db) as store:
        store.init_schema()
        print(f"schema ready at {args.db or SETTINGS.db_path}")
        print(store.summary().to_string(index=False))
    return 0


def cmd_ingest(args) -> int:
    from bet.ingest.clubelo import ClubEloSource
    from bet.ingest.football_data import FootballDataSource
    from bet.ingest.openligadb import OpenLigaDBSource
    from bet.ingest.understat import UnderstatSource

    SETTINGS.ensure_dirs()
    seasons = _parse_season_range(args.seasons)
    wanted = {"football_data", "clubelo", "understat", "openligadb"} if args.source == "all" else {args.source}

    with Store.open(args.db) as store:
        store.init_schema()
        results = []

        if "football_data" in wanted:
            print(f"football-data.co.uk: {len(seasons)} seasons...")
            results.append(FootballDataSource(store).ingest(seasons, league=args.league))
        if "openligadb" in wanted:
            print("openligadb...")
            results.append(OpenLigaDBSource(store).ingest(seasons, league=args.league))
        if "clubelo" in wanted:
            print("clubelo (weekly snapshots, this is the slow one)...")
            results.append(ClubEloSource(store).ingest(
                start=date(min(seasons), 7, 1), end=date.today(), step_days=args.elo_step))
        if "understat" in wanted:
            print(f"understat (shots={'yes' if args.shots else 'no'})...")
            results.append(UnderstatSource(store).ingest(
                seasons, league=args.league, with_shots=args.shots))

        print()
        for result in results:
            print(f"  {result}")
            for error in result.errors[:5]:
                print(f"    ! {error}")
            if len(result.errors) > 5:
                print(f"    ! ... and {len(result.errors) - 5} more")

        print()
        print(store.summary().to_string(index=False))
        print()
        _print_leakage(store)
    return 0


def cmd_backtest(args) -> int:
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end) if args.end else None

    with Store.open(args.db, read_only=True) as store:
        models = [HomePriorModel(), EloModel(), MarketModel()]
        table, results = compare(store, models, start=start, end=end,
                                 league=args.league, verbose=args.verbose)

        if table.empty:
            print("no predictions produced - is the store populated? try: bet ingest --source all")
            return 1

        print()
        print("=" * 78)
        print(f"WALK-FORWARD BACKTEST  {start:%Y-%m-%d} to {end or 'now':%Y-%m-%d}"
              if end else f"WALK-FORWARD BACKTEST  from {start:%Y-%m-%d}")
        print("=" * 78)
        print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
        print()
        print("RPS is the headline metric (lower is better).")
        print("A model that cannot beat 'market' out of sample has no betting edge.")

        if args.calibration:
            for name, result in results.items():
                if result.calibration.empty:
                    continue
                print(f"\ncalibration - {name}")
                print(result.calibration.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

        if args.out:
            frames = [r.predictions for r in results.values() if not r.predictions.empty]
            pd.concat(frames, ignore_index=True).to_parquet(args.out, index=False)
            print(f"\npredictions written to {args.out}")
    return 0


def _print_leakage(store) -> None:
    report = store.leakage_report()
    total = int(report["violations"].sum())
    print("point-in-time integrity:")
    print(report.to_string(index=False))
    if total:
        print(f"\n  FAIL: {total} rows are knowable before they should be. "
              "Backtests on this store are not trustworthy.")
    else:
        print("\n  OK: no facts are visible before they occurred.")


def cmd_status(args) -> int:
    with Store.open(args.db, read_only=True) as store:
        print(store.summary().to_string(index=False))
        print()
        _print_leakage(store)
    return 0


def cmd_check(args) -> int:
    with Store.open(args.db, read_only=True) as store:
        report = store.leakage_report()
        _print_leakage(store)
        return 1 if int(report["violations"].sum()) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bet", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=None, help="path to the DuckDB file")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="create the database schema")
    p_init.set_defaults(func=cmd_init)

    p_ingest = sub.add_parser("ingest", help="fetch and load public data")
    p_ingest.add_argument("--source", default="all",
                          choices=["all", "football_data", "clubelo", "understat", "openligadb"])
    p_ingest.add_argument("--seasons", default="2015-2026", help="e.g. 2015-2026 or 2019,2021")
    p_ingest.add_argument("--league", default="bundesliga")
    p_ingest.add_argument("--shots", action="store_true",
                          help="also fetch per-match shots from Understat (slow: ~306 requests/season)")
    p_ingest.add_argument("--elo-step", type=int, default=7, help="days between ClubElo snapshots")
    p_ingest.set_defaults(func=cmd_ingest)

    p_back = sub.add_parser("backtest", help="walk models forward and score them")
    p_back.add_argument("--from", dest="start", default="2018-08-01")
    p_back.add_argument("--to", dest="end", default=None)
    p_back.add_argument("--league", default="bundesliga")
    p_back.add_argument("--calibration", action="store_true", help="print calibration tables")
    p_back.add_argument("--out", default=None, help="write predictions to a parquet file")
    p_back.add_argument("--verbose", action="store_true")
    p_back.set_defaults(func=cmd_backtest)

    p_status = sub.add_parser("status", help="row counts and integrity check")
    p_status.set_defaults(func=cmd_status)

    p_check = sub.add_parser("check", help="point-in-time integrity check only")
    p_check.set_defaults(func=cmd_check)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
