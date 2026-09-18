"""Command line entry point.

    bet init                        create the database schema
    bet ingest --source all         fetch and load free public data
    bet backtest --from 2018-08-01  walk models forward and score them
    bet tune --from 2018-08-01      choose the time-decay rate by out-of-sample score
    bet predict                     probabilities and fair odds for upcoming fixtures
    bet status                      row counts and the leakage check
    bet check                       point-in-time integrity only
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta

import pandas as pd

from bet.config import SETTINGS
from bet.evaluation.backtest import compare
from bet.models.baselines import EloModel, HomePriorModel, MarketModel
from bet.models.dixon_coles import DixonColesModel, tune_decay
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
        models = [
            HomePriorModel(),
            EloModel(),
            DixonColesModel(xi=args.xi, target=args.target),
            MarketModel(),
        ]
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


def cmd_tune(args) -> int:
    """Sweep the time-decay rate and rank by out-of-sample score."""
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end) if args.end else None

    with Store.open(args.db, read_only=True) as store:
        candidates = tuple(float(x) for x in args.candidates.split(","))
        print(f"sweeping xi over {candidates}")
        print("(each candidate is a full walk-forward, so this takes a while)\n")
        table = tune_decay(store, start=start, end=end, candidates=candidates,
                           target=args.target, league=args.league, verbose=True)
        if table.empty:
            print("no results - is the store populated?")
            return 1

        print()
        print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
        best = table.iloc[0]
        print(f"\nbest xi = {best['xi']} (half-life {best['half_life_days']:.0f} days)")
        print("Tuning and reporting on the same period flatters the result;")
        print("for a clean number, tune on earlier seasons and report on later ones.")
    return 0


def cmd_predict(args) -> int:
    """Fit on everything known and price the next fixtures."""
    from bet.ev import TaxMode, size_bet

    as_of = datetime.fromisoformat(args.as_of) if args.as_of else datetime.utcnow()

    with Store.open(args.db, read_only=True) as store:
        model = DixonColesModel(xi=args.xi, target=args.target)
        model.fit(store, as_of)
        if model.params is None:
            print("not enough training data - run: bet ingest --source football_data")
            return 1

        fixtures = store.fixtures_between(as_of, as_of + timedelta(days=args.days),
                                          league=args.league)
        if fixtures.empty:
            print(f"no fixtures in the next {args.days} days")
            print("fixture lists come from openligadb: bet ingest --source openligadb")
            return 1

        probs = model.predict(store, fixtures, as_of)
        rows = []
        for (_, fixture), p in zip(fixtures.iterrows(), probs):
            lam, mu = model._rates(fixture["home_team_id"], fixture["away_team_id"])
            rows.append({
                "kickoff": pd.Timestamp(fixture["kickoff_utc"]).strftime("%Y-%m-%d %H:%M"),
                "home": fixture["home_team_id"],
                "away": fixture["away_team_id"],
                "xg_h": round(lam, 2), "xg_a": round(mu, 2),
                "p_H": round(p[0], 4), "p_D": round(p[1], 4), "p_A": round(p[2], 4),
                # Fair odds: the price at which the bet is break-even before any
                # margin or tax. Every real price you see will be worse.
                "fair_H": round(1 / p[0], 2), "fair_D": round(1 / p[1], 2),
                "fair_A": round(1 / p[2], 2),
            })

        print(f"model: dixon_coles_{args.target} (xi={args.xi}, "
              f"fitted on {model.params.n_matches} matches, rho={model.params.rho:.4f})")
        print(f"as of: {as_of:%Y-%m-%d %H:%M} UTC\n")
        print(pd.DataFrame(rows).to_string(index=False))
        print("\nFair odds exclude bookmaker margin and betting tax.")
        print("A real price must beat these by several percent before a bet is +EV;")
        print("see bet.ev.required_edge_over_market.")
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
    p_back.add_argument("--xi", type=float, default=0.0018, help="Dixon-Coles time-decay rate")
    p_back.add_argument("--target", default="goals", choices=["goals", "xg", "blend"])
    p_back.add_argument("--calibration", action="store_true", help="print calibration tables")
    p_back.add_argument("--out", default=None, help="write predictions to a parquet file")
    p_back.add_argument("--verbose", action="store_true")
    p_back.set_defaults(func=cmd_backtest)

    p_tune = sub.add_parser("tune", help="choose the time-decay rate by out-of-sample score")
    p_tune.add_argument("--from", dest="start", default="2018-08-01")
    p_tune.add_argument("--to", dest="end", default=None)
    p_tune.add_argument("--league", default="bundesliga")
    p_tune.add_argument("--target", default="goals", choices=["goals", "xg", "blend"])
    p_tune.add_argument("--candidates", default="0,0.0005,0.001,0.0015,0.002,0.003,0.005")
    p_tune.set_defaults(func=cmd_tune)

    p_predict = sub.add_parser("predict", help="price upcoming fixtures")
    p_predict.add_argument("--as-of", dest="as_of", default=None)
    p_predict.add_argument("--days", type=int, default=10)
    p_predict.add_argument("--league", default="bundesliga")
    p_predict.add_argument("--xi", type=float, default=0.0018)
    p_predict.add_argument("--target", default="goals", choices=["goals", "xg", "blend"])
    p_predict.set_defaults(func=cmd_predict)

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
