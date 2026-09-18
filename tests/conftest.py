"""Synthetic Bundesliga seasons for testing.

The data sites are unreachable from CI, and more importantly a test that
depends on a live website is not a test. These fixtures generate seasons from
known team strengths, which has a second benefit: because the true strengths
are known, a model can be checked for recovering them rather than merely for
running without raising.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from bet.config import SETTINGS
from bet.store import Store

TEAMS = [
    "bayern_munich", "borussia_dortmund", "rb_leipzig", "bayer_leverkusen",
    "borussia_monchengladbach", "eintracht_frankfurt", "vfl_wolfsburg", "sc_freiburg",
    "tsg_hoffenheim", "fc_union_berlin", "vfb_stuttgart", "sv_werder_bremen",
    "fsv_mainz_05", "fc_augsburg", "vfl_bochum", "1_fc_koln",
    "fc_st_pauli", "1_fc_heidenheim",
]

# Attack and defence strengths on the log-rate scale. Bayern strongest.
TRUE_ATTACK = {team: 0.45 - 0.05 * i for i, team in enumerate(TEAMS)}
TRUE_DEFENCE = {team: -0.35 + 0.045 * i for i, team in enumerate(TEAMS)}
TRUE_HOME_ADVANTAGE = 0.26


def generate_season(start: datetime, season: str, rng: np.random.Generator,
                    league: str = "bundesliga") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Double round robin with Poisson goals drawn from the true strengths."""
    matches, results, quotes = [], [], []
    kickoff = start

    pairings = [(h, a) for h in TEAMS for a in TEAMS if h != a]
    rng.shuffle(pairings)

    for round_index in range(0, len(pairings), 9):
        block = pairings[round_index:round_index + 9]
        for home, away in block:
            lam = np.exp(TRUE_ATTACK[home] + TRUE_DEFENCE[away] + TRUE_HOME_ADVANTAGE)
            mu = np.exp(TRUE_ATTACK[away] + TRUE_DEFENCE[home])
            home_goals, away_goals = int(rng.poisson(lam)), int(rng.poisson(mu))
            outcome = "H" if home_goals > away_goals else ("D" if home_goals == away_goals else "A")
            match_id = f"{league}:{season}:{home}:{away}"

            matches.append({
                "match_id": match_id, "source": "synthetic", "league": league, "season": season,
                "kickoff_utc": kickoff, "home_team_id": home, "away_team_id": away,
                "known_at": kickoff - timedelta(days=30),
            })
            results.append({
                "match_id": match_id, "source": "synthetic",
                "home_goals": home_goals, "away_goals": away_goals,
                "outcome": outcome, "ht_home": None, "ht_away": None,
                "known_at": kickoff + timedelta(seconds=SETTINGS.result_known_after_seconds),
            })

            # A book that knows the true rates, with a 6% margin applied.
            true_probs = _score_matrix_probs(lam, mu)
            margin_probs = true_probs * 1.06
            for selection, prob in zip(("H", "D", "A"), margin_probs):
                quotes.append({
                    "match_id": match_id, "book": "synthetic_book", "market": "1x2",
                    "selection": selection, "decimal_odds": float(1.0 / prob),
                    "quoted_at": kickoff, "is_closing": True, "known_at": kickoff,
                })
        kickoff += timedelta(days=7)

    return pd.DataFrame(matches), pd.DataFrame(results), pd.DataFrame(quotes)


def _score_matrix_probs(lam: float, mu: float, max_goals: int = 12) -> np.ndarray:
    from scipy.stats import poisson
    home = poisson.pmf(np.arange(max_goals + 1), lam)
    away = poisson.pmf(np.arange(max_goals + 1), mu)
    matrix = np.outer(home, away)
    return np.array([np.tril(matrix, -1).sum(), np.trace(matrix), np.triu(matrix, 1).sum()])


@pytest.fixture
def store(tmp_path):
    with Store.open(tmp_path / "test.duckdb") as s:
        s.init_schema()
        yield s


@pytest.fixture
def populated_store(store):
    """Five synthetic seasons, enough for a meaningful walk-forward."""
    rng = np.random.default_rng(20260918)
    for offset in range(5):
        season_start = datetime(2019 + offset, 8, 10, 15, 30)
        season = f"{2019 + offset}-{str(2020 + offset)[-2:]}"
        matches, results, quotes = generate_season(season_start, season, rng)
        store.upsert("match", matches, ["match_id"])
        store.upsert("match_result", results, ["match_id", "source"])
        store.upsert("odds_quote", quotes,
                     ["match_id", "book", "market", "selection", "quoted_at"])
    return store


def generate_shots(matches: pd.DataFrame, results: pd.DataFrame,
                   rng: np.random.Generator) -> pd.DataFrame:
    """Synthetic shot-level xG consistent with the scorelines.

    Real xG is a noisy estimate of the same underlying rate that produced the
    goals, so the generator draws shots around the realised score rather than
    independently of it.
    """
    merged = matches.merge(results[["match_id", "home_goals", "away_goals"]], on="match_id")
    rows = []
    for row in merged.itertuples(index=False):
        for side, team, goals in (("h", row.home_team_id, row.home_goals),
                                  ("a", row.away_team_id, row.away_goals)):
            n_shots = max(1, int(rng.poisson(9)))
            # Total xG hovers around the realised goals, with the usual spread.
            total = max(0.15, goals + rng.normal(0.0, 0.55))
            weights = rng.dirichlet(np.ones(n_shots))
            for i, share in enumerate(weights):
                rows.append({
                    "shot_id": f"{row.match_id}:{side}:{i}",
                    "match_id": row.match_id,
                    "team_id": team,
                    "player_name": None,
                    "minute": int(rng.integers(1, 91)),
                    "x": float(rng.uniform(0.6, 1.0)),
                    "y": float(rng.uniform(0.2, 0.8)),
                    "xg": float(min(0.95, total * share)),
                    "body_part": "RightFoot",
                    "situation": "OpenPlay",
                    "result": "Goal" if i == 0 and goals > 0 else "MissedShots",
                    "known_at": row.kickoff_utc + timedelta(
                        seconds=SETTINGS.result_known_after_seconds),
                })
    return pd.DataFrame(rows)


@pytest.fixture
def store_with_shots(store):
    """Three synthetic seasons with shot-level xG attached."""
    rng = np.random.default_rng(4242)
    for offset in range(3):
        season_start = datetime(2021 + offset, 8, 10, 15, 30)
        season = f"{2021 + offset}-{str(2022 + offset)[-2:]}"
        matches, results, quotes = generate_season(season_start, season, rng)
        store.upsert("match", matches, ["match_id"])
        store.upsert("match_result", results, ["match_id", "source"])
        store.upsert("odds_quote", quotes,
                     ["match_id", "book", "market", "selection", "quoted_at"])
        store.upsert("shot", generate_shots(matches, results, rng), ["shot_id"])
    return store


# Squad archetypes: (suffix, position, shots per 90, expected assists per 90,
# start probability). Creative output is a separate rate rather than noise, so a
# playmaker and a poacher are distinguishable -- with a flat random xa the whole
# squad ends up with near-identical attacking contributions and nothing that
# depends on player quality can be tested.
SQUAD_TEMPLATE = [
    ("gk", "GK", 0.05, 0.01, 1.00),
    ("cb1", "DF", 0.35, 0.04, 0.95), ("cb2", "DF", 0.30, 0.03, 0.90),
    ("lb", "DF", 0.45, 0.12, 0.85), ("rb", "DF", 0.50, 0.14, 0.85),
    ("dm", "MF", 0.70, 0.08, 0.90), ("cm1", "MF", 1.10, 0.18, 0.80),
    ("cm2", "MF", 1.30, 0.22, 0.70),
    ("am", "MF,FW", 2.10, 0.42, 0.75), ("lw", "FW", 2.40, 0.30, 0.70),
    ("rw", "FW", 2.30, 0.28, 0.65),
    ("st", "FW", 3.40, 0.18, 0.85),
    ("sub1", "MF", 1.20, 0.15, 0.15), ("sub2", "FW", 2.00, 0.16, 0.12),
    ("sub3", "DF", 0.30, 0.03, 0.10),
]


def generate_player_stats(matches: pd.DataFrame, rng: np.random.Generator,
                          source: str = "synthetic") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-match player lines drawn from fixed per-90 rates.

    Rates are known, so a prop model can be checked for recovering them rather
    than merely for producing a number.
    """
    players, stats = [], []
    seen: dict[str, str] = {}

    for row in matches.itertuples(index=False):
        for team_id in (row.home_team_id, row.away_team_id):
            known_at = row.kickoff_utc + timedelta(
                seconds=SETTINGS.result_known_after_seconds)

            for suffix, position, shot_rate, assist_rate, start_prob in SQUAD_TEMPLATE:
                player_id = f"{team_id}_{suffix}"
                full_name = f"{suffix.upper()} {team_id.replace('_', ' ').title()}"
                if player_id not in seen:
                    seen[player_id] = full_name
                    players.append({
                        "player_id": player_id, "full_name": full_name,
                        "source": source, "source_id": None, "known_at": known_at,
                    })

                started = rng.random() < start_prob
                if started:
                    minutes = float(rng.integers(60, 91))
                elif rng.random() < 0.45:
                    minutes = float(rng.integers(5, 35))
                else:
                    continue  # unused substitute

                nineties = minutes / 90.0
                shots = int(rng.poisson(shot_rate * nineties))
                # xG per shot varies by role: a striker shoots from better
                # positions than a centre-back lashing at a corner.
                xg_per_shot = 0.13 if position.startswith("FW") else 0.07
                stats.append({
                    "match_id": row.match_id, "player_id": player_id, "team_id": team_id,
                    "source": source, "position": position, "started": started,
                    "minutes": minutes,
                    "goals": float(rng.binomial(shots, 0.11)) if shots else 0.0,
                    "assists": float(rng.poisson(assist_rate * nineties * 0.8)),
                    "shots": float(shots),
                    "shots_on_target": float(rng.binomial(shots, 0.35)) if shots else 0.0,
                    "xg": float(shots * xg_per_shot * rng.uniform(0.8, 1.2)),
                    "npxg": float(shots * xg_per_shot * rng.uniform(0.7, 1.0)),
                    "xa": float(rng.gamma(2.0, assist_rate * nineties / 2.0))
                    if assist_rate > 0 else 0.0,
                    "passes_completed": float(rng.integers(10, 70)),
                    "passes_attempted": float(rng.integers(15, 85)),
                    "progressive_passes": float(rng.integers(0, 9)),
                    "touches": float(rng.integers(20, 100)),
                    "carries": float(rng.integers(10, 60)),
                    "tackles": float(rng.poisson(1.4 * nineties)),
                    "interceptions": float(rng.poisson(0.9 * nineties)),
                    "blocks": float(rng.poisson(0.6 * nineties)),
                    "fouls": float(rng.poisson(1.0 * nineties)),
                    "yellow_cards": float(rng.binomial(1, 0.12)),
                    "red_cards": 0.0,
                    "known_at": known_at,
                })

    return pd.DataFrame(players), pd.DataFrame(stats)


def lineup_rows_from_stats(stats: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Line-ups stamped at kickoff, as a confirmed XI would be."""
    kickoffs = dict(zip(matches["match_id"], matches["kickoff_utc"]))
    return pd.DataFrame([{
        "match_id": r.match_id, "player_id": r.player_id, "team_id": r.team_id,
        "source": "synthetic", "is_starter": bool(r.started), "is_confirmed": True,
        "shirt_number": None, "formation": "4-2-3-1",
        "known_at": kickoffs[r.match_id],
    } for r in stats.itertuples()])


@pytest.fixture
def store_with_players(store):
    """Three synthetic seasons with per-match player lines and line-ups."""
    rng = np.random.default_rng(777)
    for offset in range(3):
        season_start = datetime(2021 + offset, 8, 10, 15, 30)
        season = f"{2021 + offset}-{str(2022 + offset)[-2:]}"
        matches, results, quotes = generate_season(season_start, season, rng)
        store.upsert("match", matches, ["match_id"])
        store.upsert("match_result", results, ["match_id", "source"])
        store.upsert("odds_quote", quotes,
                     ["match_id", "book", "market", "selection", "quoted_at"])

        players, stats = generate_player_stats(matches, rng)
        store.upsert("player", players.drop_duplicates("player_id"), ["player_id"])
        store.upsert("player_match_stat", stats, ["match_id", "player_id", "source"])
        store.upsert("lineup", lineup_rows_from_stats(stats, matches),
                     ["match_id", "player_id", "source", "known_at"])
    return store
