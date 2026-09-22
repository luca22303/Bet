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


def round_robin(teams: list[str]) -> list[list[tuple[str, str]]]:
    """A double round robin where every team plays exactly once per matchday.

    The circle method: fix one team and rotate the rest, which yields n-1
    matchdays covering every pairing once, then mirror it with home and away
    swapped for the return fixtures.

    Shuffling pairings into fixed-size rounds instead -- the obvious shortcut --
    lets a team appear twice on one matchday and not at all for a month. Nothing
    downstream survives that: "days since last start" becomes meaningless,
    rotation looks like squad turnover, and every player in a line-up reads as
    six weeks stale.
    """
    rotation = list(teams)
    if len(rotation) % 2:
        rotation.append(None)          # bye, for an odd league size

    half = len(rotation) // 2
    first_leg = []
    for matchday in range(len(rotation) - 1):
        pairs = []
        for i in range(half):
            home, away = rotation[i], rotation[-(i + 1)]
            if home is None or away is None:
                continue
            # Alternate which side is at home so no team is always the host.
            pairs.append((home, away) if (matchday + i) % 2 == 0 else (away, home))
        first_leg.append(pairs)
        rotation = [rotation[0]] + [rotation[-1]] + rotation[1:-1]

    return first_leg + [[(a, h) for h, a in pairs] for pairs in first_leg]


def generate_season(start: datetime, season: str, rng: np.random.Generator,
                    league: str = "bundesliga") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """A full season with Poisson goals drawn from the true strengths."""
    matches, results, quotes = [], [], []
    kickoff = start

    for matchday in round_robin(TEAMS):
        for home, away in matchday:
            lam = np.exp(TRUE_ATTACK[home] + TRUE_DEFENCE[away] + TRUE_HOME_ADVANTAGE)
            mu = np.exp(TRUE_ATTACK[away] + TRUE_DEFENCE[home])
            home_goals, away_goals = int(rng.poisson(lam)), int(rng.poisson(mu))
            outcome = "H" if home_goals > away_goals else ("D" if home_goals == away_goals else "A")
            match_id = f"{league}:{season}:{home}:{away}"

            matches.append({
                "match_id": match_id, "source": "synthetic", "league": league,
                "season": season, "kickoff_utc": kickoff,
                "home_team_id": home, "away_team_id": away,
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


# Squad archetypes: (suffix, position, shots/90, xA/90, defensive actions/90,
# start probability).
#
# Every rate varies by role rather than being drawn flat. That matters more than
# it looks: with a flat random xA the whole squad ends up with near-identical
# attacking contributions, and with a flat tackle rate a centre-back defends no
# better than a striker. Either one makes it impossible to test anything that
# depends on who is actually on the pitch.
SQUAD_TEMPLATE = [
    ("gk", "GK", 0.05, 0.01, 0.30, 0.94),
    # A backup keeper, because every real squad has one. Without him, ruling
    # out the first choice leaves no eligible goalkeeper and the selector has to
    # fall back to an ineligible player -- correct behaviour for a squad that
    # genuinely has no cover, but not a situation any real team is in.
    ("gk2", "GK", 0.02, 0.01, 0.25, 0.06),
    ("cb1", "DF", 0.35, 0.04, 4.20, 0.95), ("cb2", "DF", 0.30, 0.03, 3.90, 0.90),
    ("lb", "DF", 0.45, 0.12, 3.60, 0.85), ("rb", "DF", 0.50, 0.14, 3.50, 0.85),
    ("dm", "MF", 0.70, 0.08, 4.00, 0.90), ("cm1", "MF", 1.10, 0.18, 2.60, 0.80),
    ("cm2", "MF", 1.30, 0.22, 2.30, 0.70),
    ("am", "MF,FW", 2.10, 0.42, 1.20, 0.75), ("lw", "FW", 2.40, 0.30, 0.90, 0.70),
    ("rw", "FW", 2.30, 0.28, 0.85, 0.65),
    ("st", "FW", 3.40, 0.18, 0.50, 0.85),
    ("sub1", "MF", 1.20, 0.15, 2.40, 0.15), ("sub2", "FW", 2.00, 0.16, 0.80, 0.12),
    ("sub3", "DF", 0.30, 0.03, 3.40, 0.10),
]


# Plausible surnames, so anything that renders a player reads like a team sheet
# rather than a debug dump. Chosen deterministically from the id, so a given
# player keeps the same name across seasons and runs.
_SURNAMES = [
    "Baumann", "Becker", "Brandt", "Demirovic", "Engels", "Fischer", "Gosens",
    "Grifo", "Hartmann", "Hofmann", "Jakobs", "Kehrer", "Klostermann", "Koch",
    "Kramer", "Lienhart", "Maier", "Neuhaus", "Pavlovic", "Raum", "Reus",
    "Sabitzer", "Schlotterbeck", "Stach", "Tillman", "Undav", "Vogt", "Wirtz",
    "Wolf", "Zimmermann",
]


_SQUAD_SLOTS = {entry[0]: i for i, entry in enumerate(SQUAD_TEMPLATE)}


def _synthetic_name(team_id: str, suffix: str) -> str:
    """A surname that is unique within its own squad.

    Uniqueness is not cosmetic. Name resolution deliberately refuses an
    ambiguous match -- two players a source could equally mean -- so a squad
    with two Brandts makes every lookup against it fail, and the fixture would
    be testing the guard rather than the pipeline.
    """
    offset = sum(ord(c) for c in team_id)
    slot = _SQUAD_SLOTS.get(suffix, 0)
    return _SURNAMES[(offset + slot) % len(_SURNAMES)]


def _choose_starting_xi(rng: np.random.Generator) -> set[str]:
    """Eleven starters, one keeper, sampled by each player's start propensity.

    Uses the exponential-race trick for weighted sampling without replacement:
    the player with the smallest `Exponential(1) / weight` is picked first, which
    draws without replacement in proportion to the weights.
    """
    keepers = [(s, p) for s, _, _, _, _, p in SQUAD_TEMPLATE if s.startswith("gk")]
    outfield = [(s, p) for s, pos, _, _, _, p in SQUAD_TEMPLATE if not s.startswith("gk")]

    def draw(pool, count):
        keys = [(rng.exponential() / max(weight, 1e-6), suffix) for suffix, weight in pool]
        keys.sort()
        return [suffix for _, suffix in keys[:count]]

    return set(draw(keepers, 1) + draw(outfield, 10))


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

            # Exactly eleven start, one of them a goalkeeper. Drawing each
            # player's start independently lets a match field twelve or thirteen
            # starters, which makes every inferred formation nonsense ("4-4-3")
            # and quietly corrupts anything built on line-up shape.
            starting_xi = _choose_starting_xi(rng)

            for suffix, position, shot_rate, assist_rate, defence_rate, start_prob in SQUAD_TEMPLATE:
                player_id = f"{team_id}_{suffix}"
                full_name = _synthetic_name(team_id, suffix)
                if player_id not in seen:
                    seen[player_id] = full_name
                    players.append({
                        "player_id": player_id, "full_name": full_name,
                        "source": source, "source_id": None, "known_at": known_at,
                    })

                started = suffix in starting_xi
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
                    # Split the player's defensive rate across the three
                    # actions, so a centre-back and a striker are not
                    # interchangeable on this side of the ball.
                    "tackles": float(rng.poisson(defence_rate * 0.5 * nineties)),
                    "interceptions": float(rng.poisson(defence_rate * 0.33 * nineties)),
                    "blocks": float(rng.poisson(defence_rate * 0.17 * nineties)),
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
