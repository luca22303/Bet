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
                "match_id": match_id, "home_goals": home_goals, "away_goals": away_goals,
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
        store.upsert("match_result", results, ["match_id"])
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
        store.upsert("match_result", results, ["match_id"])
        store.upsert("odds_quote", quotes,
                     ["match_id", "book", "market", "selection", "quoted_at"])
        store.upsert("shot", generate_shots(matches, results, rng), ["shot_id"])
    return store
