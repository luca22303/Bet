"""Command line entry point.

    bet init                        create the database schema
    bet ingest --source all         fetch and load free public data
    bet backtest --from 2018-08-01  walk models forward and score them
    bet tune --from 2018-08-01      choose the time-decay rate by out-of-sample score
    bet predict                     probabilities and fair odds for upcoming fixtures
    bet props --stat shots          player prop prices for a fixture
    bet scout --team bayern_munich  shot profile and spatial summary
    bet news --team bayern_munich   extract availability from a news file (local LLM)
    bet brief                       compile the matchday recommendations
    bet quality                     data quality and cross-source reconciliation
    bet status                      row counts and the leakage check
    bet check                       point-in-time integrity only
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

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
    from bet.ingest.fbref import FBrefSource
    from bet.ingest.understat import UnderstatSource

    SETTINGS.ensure_dirs()
    seasons = _parse_season_range(args.seasons)
    all_sources = {"football_data", "clubelo", "understat", "openligadb"}
    wanted = all_sources if args.source == "all" else {args.source}

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
        if "fbref" in wanted:
            # One request per match page; a season is ~306 of them.
            print("fbref player stats (slow: ~306 requests per season)...")
            results.append(FBrefSource(store).ingest(
                seasons, league=args.league, max_matches=args.max_matches))

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


def cmd_props(args) -> int:
    """Price player props for a fixture."""
    from bet.models.props import SUPPORTED_STATS, PlayerPropModel

    as_of = datetime.fromisoformat(args.as_of) if args.as_of else datetime.utcnow()

    with Store.open(args.db, read_only=True) as store:
        model = PlayerPropModel(stat=args.stat).fit(store, as_of)
        if model.rates.empty:
            print("no player data - run: bet ingest --source fbref --seasons 2022-2026")
            return 1

        fixtures = store.fixtures_between(as_of, as_of + timedelta(days=args.days),
                                          league=args.league)
        if fixtures.empty:
            print(f"no fixtures in the next {args.days} days")
            return 1

        # Expected goals come from the match model, so the prop and the 1X2
        # price cannot imply different things about the same fixture.
        match_model = DixonColesModel(xi=args.xi).fit(store, as_of)

        line = args.line if args.line is not None else SUPPORTED_STATS[args.stat]
        print(f"{args.stat} over {line}  (as of {as_of:%Y-%m-%d %H:%M} UTC)\n")

        for fixture in fixtures.itertuples(index=False):
            home_xg = away_xg = None
            if match_model.params is not None:
                home_xg, away_xg = match_model._rates(fixture.home_team_id, fixture.away_team_id)

            frame = model.predict_match(
                store, fixture.match_id, fixture.home_team_id, fixture.away_team_id,
                as_of, home_expected_goals=home_xg, away_expected_goals=away_xg, line=line)
            if frame.empty:
                continue

            print(f"{fixture.home_team_id} vs {fixture.away_team_id}"
                  f"  ({pd.Timestamp(fixture.kickoff_utc):%Y-%m-%d %H:%M})")
            shown = frame.drop(columns=["match_id", "stat"], errors="ignore")
            print(shown.head(args.top).to_string(index=False))
            print()

        print("Fair odds exclude margin and betting tax. Props are the softer market,")
        print("but a thin sample_90s means the rate is mostly prior, not evidence.")
    return 0


def cmd_scout(args) -> int:
    """Shot profile and spatial summary for a team."""
    from bet.spatial.pitch import bin_heatmap, heatmap_to_frame, shot_profile

    as_of = datetime.fromisoformat(args.as_of) if args.as_of else datetime.utcnow()

    with Store.open(args.db, read_only=True) as store:
        shots = store.shots_as_of(as_of)
        if shots.empty:
            print("no shot data - run: bet ingest --source understat --shots")
            return 1

        team_shots = shots[shots["team_id"] == args.team]
        if team_shots.empty:
            print(f"no shots recorded for {args.team}")
            return 1

        print(f"shot profile - {args.team} (as of {as_of:%Y-%m-%d})\n")
        for key, value in shot_profile(team_shots).items():
            print(f"  {key:22s} {value:.3f}" if isinstance(value, float)
                  else f"  {key:22s} {value}")

        print("\nshot density by pitch zone (share of shots)")
        frame = heatmap_to_frame(bin_heatmap(team_shots))
        by_zone = frame.groupby("zone")["share"].sum().sort_values(ascending=False)
        print(by_zone.to_string(float_format=lambda v: f"{v:.3f}"))

        print("\nxg_per_shot is the number to read first: two sides with equal")
        print("season xG are not equally good if one gets there by volume.")
    return 0


def cmd_news(args) -> int:
    """Extract player availability from a news file using a local LLM."""
    from bet.extract import OllamaBackend, extract_and_verify, to_availability_rows

    text = Path(args.file).read_text(encoding="utf-8", errors="replace")
    published = datetime.fromisoformat(args.published) if args.published else datetime.utcnow()

    backend = OllamaBackend(model=args.model, host=args.host)
    healthy, message = backend.health()
    if not healthy:
        print(f"LLM backend unavailable: {message}")
        print("\nStart Ollama and pull a model:")
        print("  ollama serve")
        print(f"  ollama pull {args.model}")
        return 1
    print(f"backend: {message}")

    with Store.open(args.db) as store:
        store.init_schema()
        squad = store.squad_as_of(published, args.team)
        if not squad:
            print(f"no squad on record for {args.team} - run: bet ingest --source fbref")
            return 1

        result = extract_and_verify(
            backend, text, args.team, published, list(squad.values()),
            min_confidence=args.min_confidence, judge=not args.no_judge)

        print(f"\n{result.summary()}")
        for entry in result.accepted:
            print(f"  ACCEPT {entry.player_name:<24} {entry.status.value:<10} "
                  f"conf={entry.source_confidence:.2f}  {entry.reason or ''}")
        for rejection in result.rejected:
            name = rejection["entry"].get("player_name", "?")
            print(f"  reject {name:<24} {rejection['reason']}")

        rows = to_availability_rows(result, store, args.team, published)
        if rows.empty:
            print("\nnothing written")
            return 0

        written = store.upsert("player_availability", rows,
                               ["player_id", "source", "known_at"])
        print(f"\n{written} availability rows written")
    return 0


def cmd_brief(args) -> int:
    """Compile the matchday brief."""
    from bet.recommend import build_brief, narrate

    as_of = datetime.fromisoformat(args.as_of) if args.as_of else datetime.utcnow()

    with Store.open(args.db, read_only=True) as store:
        brief = build_brief(store, as_of, days=args.days, league=args.league,
                            xi=args.xi, use_availability=not args.no_availability,
                            prop_stat=args.prop_stat, min_edge=args.min_edge)
        print(brief.to_text())

        if args.narrate:
            from bet.extract import OllamaBackend
            backend = OllamaBackend(model=args.model, host=args.host)
            healthy, message = backend.health()
            if not healthy:
                print(f"\n(narration skipped: {message})")
            else:
                print("\n" + "=" * 72)
                print("SUMMARY")
                print("=" * 72)
                print(narrate(brief, backend))
    return 0


def cmd_quality(args) -> int:
    """Data quality checks and cross-source reconciliation."""
    from bet.quality import run_quality_checks

    with Store.open(args.db, read_only=True) as store:
        report = run_quality_checks(store, seasons_expected=args.seasons_expected)
        print(report.to_text())
        return 0 if report.passed else 1


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
                          choices=["all", "football_data", "clubelo", "understat",
                                   "openligadb", "fbref"])
    p_ingest.add_argument("--seasons", default="2015-2026", help="e.g. 2015-2026 or 2019,2021")
    p_ingest.add_argument("--league", default="bundesliga")
    p_ingest.add_argument("--shots", action="store_true",
                          help="also fetch per-match shots from Understat (slow: ~306 requests/season)")
    p_ingest.add_argument("--elo-step", type=int, default=7, help="days between ClubElo snapshots")
    p_ingest.add_argument("--max-matches", type=int, default=None,
                          help="cap match pages fetched per season (fbref)")
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

    p_props = sub.add_parser("props", help="price player props for a fixture")
    p_props.add_argument("--stat", default="shots")
    p_props.add_argument("--line", type=float, default=None)
    p_props.add_argument("--as-of", dest="as_of", default=None)
    p_props.add_argument("--days", type=int, default=10)
    p_props.add_argument("--top", type=int, default=12, help="players shown per fixture")
    p_props.add_argument("--league", default="bundesliga")
    p_props.add_argument("--xi", type=float, default=0.0018)
    p_props.set_defaults(func=cmd_props)

    p_scout = sub.add_parser("scout", help="shot profile and spatial summary")
    p_scout.add_argument("--team", required=True)
    p_scout.add_argument("--as-of", dest="as_of", default=None)
    p_scout.set_defaults(func=cmd_scout)

    p_news = sub.add_parser("news", help="extract availability from a news file")
    p_news.add_argument("--file", required=True, help="path to a text file of team news")
    p_news.add_argument("--team", required=True, help="canonical team id")
    p_news.add_argument("--published", default=None, help="publication timestamp (ISO)")
    p_news.add_argument("--model", default="qwen2.5:7b-instruct")
    p_news.add_argument("--host", default="http://localhost:11434")
    p_news.add_argument("--min-confidence", type=float, default=0.5)
    p_news.add_argument("--no-judge", action="store_true",
                        help="skip verification (faster, less reliable)")
    p_news.set_defaults(func=cmd_news)

    p_brief = sub.add_parser("brief", help="compile the matchday recommendations")
    p_brief.add_argument("--as-of", dest="as_of", default=None)
    p_brief.add_argument("--days", type=int, default=8)
    p_brief.add_argument("--league", default="bundesliga")
    p_brief.add_argument("--xi", type=float, default=0.0018)
    p_brief.add_argument("--prop-stat", default="shots")
    p_brief.add_argument("--min-edge", type=float, default=0.02)
    p_brief.add_argument("--no-availability", action="store_true")
    p_brief.add_argument("--narrate", action="store_true",
                         help="add a written summary via the local LLM")
    p_brief.add_argument("--model", default="qwen2.5:7b-instruct")
    p_brief.add_argument("--host", default="http://localhost:11434")
    p_brief.set_defaults(func=cmd_brief)

    p_quality = sub.add_parser("quality", help="data quality and reconciliation")
    p_quality.add_argument("--seasons-expected", type=int, default=None)
    p_quality.set_defaults(func=cmd_quality)

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
