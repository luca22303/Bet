"""Dixon-Coles (1997), with the corrections the original paper leaves implicit.

Four things this implementation does that a naive transcription of the paper
does not, each of which matters:

1.  Identifiability. The rates are
        lambda = exp(c + attack_home + defence_away + home_advantage)
        mu     = exp(c + attack_away + defence_home)
    Constraining only the attack parameters to sum to zero is not enough:
    shifting every attack by +k and every defence by -k leaves both rates
    unchanged, so the likelihood has a flat ridge and the optimiser wanders
    along it, returning different parameters on every refit. Both vectors are
    constrained here, with an explicit intercept carrying the overall level.

2.  Bounded rho. The low-score correction can drive joint probabilities
    negative -- the (0,0) cell is 1 - lambda*mu*rho -- so an unconstrained fit
    can return "probabilities" that are not probabilities. The likelihood
    rejects parameter vectors that violate positivity, and prediction clips rho
    to the valid range for each fixture's own rates.

3.  Time weighting tuned, not assumed. The decay rate xi controls how fast the
    past is forgotten, and it is a real parameter: too slow and the model
    carries last season's squad, too fast and it fits noise. `tune_decay`
    selects it by out-of-sample log loss rather than by inheriting whatever
    constant the last blog post used.

4.  Promoted teams. A model fitted on Bundesliga results alone knows nothing
    about a side that spent last season in the 2. Bundesliga, and will produce
    confident nonsense for it into October. `PromotedTeamPrior` maps ClubElo
    ratings onto fitted strengths, so a promoted side starts from evidence
    instead of from the league average.

Dixon-Coles is a baseline, not an edge. Everyone who bets football has this
model. It earns its place as the thing a better model has to beat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from bet.config import OUTCOMES
from bet.models.base import Model, Prediction

MAX_GOALS = 10
_RHO_BOUND = 0.35          # safely inside the positivity region for football rates
_MIN_TAU = 1e-9


# --------------------------------------------------------------------- params


@dataclass
class DixonColesParams:
    teams: tuple[str, ...]
    intercept: float
    attack: dict[str, float]
    defence: dict[str, float]
    home_advantage: float
    rho: float
    xi: float
    n_matches: int
    converged: bool = True
    message: str = ""

    def rates(self, home: str, away: str, *, default_attack: float = 0.0,
              default_defence: float = 0.0) -> tuple[float, float]:
        """Expected goals for a fixture. Unknown teams fall back to league average."""
        ah = self.attack.get(home, default_attack)
        aa = self.attack.get(away, default_attack)
        dh = self.defence.get(home, default_defence)
        da = self.defence.get(away, default_defence)
        lam = np.exp(self.intercept + ah + da + self.home_advantage)
        mu = np.exp(self.intercept + aa + dh)
        return float(lam), float(mu)

    def strength_table(self) -> pd.DataFrame:
        """Attack and defence by team. Negative defence means a stronger defence."""
        rows = [
            {
                "team_id": team,
                "attack": self.attack[team],
                "defence": self.defence[team],
                "net": self.attack[team] - self.defence[team],
            }
            for team in self.teams
        ]
        return pd.DataFrame(rows).sort_values("net", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ mechanics


def rho_bounds(lam: float | np.ndarray, mu: float | np.ndarray) -> tuple[float, float]:
    """Range of rho keeping all four corrected cells positive.

        1 - lambda*mu*rho > 0   ->  rho < 1/(lambda*mu)
        1 + lambda*rho    > 0   ->  rho > -1/lambda
        1 + mu*rho        > 0   ->  rho > -1/mu
        1 - rho           > 0   ->  rho < 1
    """
    lam = np.asarray(lam, dtype=float)
    mu = np.asarray(mu, dtype=float)
    lower = np.max(-1.0 / np.maximum(lam, mu))
    upper = min(float(np.min(1.0 / (lam * mu))), 1.0)
    return float(lower), float(upper)


def tau(home_goals, away_goals, lam, mu, rho):
    """Dixon-Coles low-score correction.

    Independent Poissons underpredict 0-0 and 1-1 and overpredict 1-0 and 0-1,
    because a goal changes how both sides play. This reweights exactly those
    four cells and leaves everything else alone.
    """
    x = np.asarray(home_goals)
    y = np.asarray(away_goals)
    lam = np.asarray(lam, dtype=float)
    mu = np.asarray(mu, dtype=float)

    out = np.ones(np.broadcast(x, y, lam, mu).shape, dtype=float)
    out = np.where((x == 0) & (y == 0), 1.0 - lam * mu * rho, out)
    out = np.where((x == 0) & (y == 1), 1.0 + lam * rho, out)
    out = np.where((x == 1) & (y == 0), 1.0 + mu * rho, out)
    out = np.where((x == 1) & (y == 1), 1.0 - rho, out)
    return out


def time_weights(kickoffs: pd.Series, as_of: datetime, xi: float) -> np.ndarray:
    """phi(t) = exp(-xi * days before as_of). xi = 0 weights all history equally."""
    age_days = (pd.Timestamp(as_of) - pd.to_datetime(kickoffs)).dt.total_seconds() / 86400.0
    return np.exp(-xi * np.clip(age_days.to_numpy(), 0.0, None))


def _unpack(params: np.ndarray, n_teams: int) -> tuple:
    """Expand the free parameters, restoring the sum-to-zero constraints.

    Only n-1 attack and n-1 defence values are free; the last of each is minus
    the sum of the others. This is what makes the fit identifiable.
    """
    intercept = params[0]
    attack_free = params[1:n_teams]
    defence_free = params[n_teams:2 * n_teams - 1]
    home_advantage = params[2 * n_teams - 1]
    rho = params[2 * n_teams]

    attack = np.append(attack_free, -attack_free.sum())
    defence = np.append(defence_free, -defence_free.sum())
    return intercept, attack, defence, home_advantage, rho


def _negative_log_likelihood(params, home_idx, away_idx, home_goals, away_goals,
                            weights, lgamma_home, lgamma_away, n_teams) -> float:
    intercept, attack, defence, home_advantage, rho = _unpack(params, n_teams)

    log_lam = intercept + attack[home_idx] + defence[away_idx] + home_advantage
    log_mu = intercept + attack[away_idx] + defence[home_idx]
    # Guard against the optimiser exploring extreme rates.
    log_lam = np.clip(log_lam, -8.0, 3.0)
    log_mu = np.clip(log_mu, -8.0, 3.0)
    lam = np.exp(log_lam)
    mu = np.exp(log_mu)

    correction = tau(home_goals, away_goals, lam, mu, rho)
    if np.any(correction <= _MIN_TAU):
        # This parameter vector implies a negative probability somewhere.
        return 1e10

    log_poisson = (home_goals * log_lam - lam - lgamma_home
                   + away_goals * log_mu - mu - lgamma_away)
    return float(-np.sum(weights * (np.log(correction) + log_poisson)))


def fit_dixon_coles(matches: pd.DataFrame, *, as_of: datetime, xi: float = 0.0018,
                    max_iterations: int = 400) -> DixonColesParams:
    """Time-weighted maximum likelihood fit on observed goals."""
    required = {"home_team_id", "away_team_id", "home_goals", "away_goals", "kickoff_utc"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"matches is missing columns {sorted(missing)}")
    if matches.empty:
        raise ValueError("cannot fit on an empty match set")

    teams = tuple(sorted(set(matches["home_team_id"]) | set(matches["away_team_id"])))
    n_teams = len(teams)
    if n_teams < 2:
        raise ValueError("need at least two teams")

    index = {team: i for i, team in enumerate(teams)}
    home_idx = matches["home_team_id"].map(index).to_numpy()
    away_idx = matches["away_team_id"].map(index).to_numpy()
    home_goals = matches["home_goals"].to_numpy(dtype=float)
    away_goals = matches["away_goals"].to_numpy(dtype=float)
    weights = time_weights(matches["kickoff_utc"], as_of, xi)

    # Constant across the optimisation, so computed once.
    lgamma_home = gammaln(home_goals + 1.0)
    lgamma_away = gammaln(away_goals + 1.0)

    start = np.zeros(2 * n_teams + 1)
    start[0] = np.log(max(np.average(home_goals, weights=weights), 0.2))
    start[2 * n_teams - 1] = 0.25      # home advantage, roughly its usual size
    start[2 * n_teams] = -0.05         # rho is reliably negative in football

    bounds = (
        [(-3.0, 3.0)]                                   # intercept
        + [(-2.5, 2.5)] * (n_teams - 1)                 # attack
        + [(-2.5, 2.5)] * (n_teams - 1)                 # defence
        + [(-1.0, 1.5)]                                 # home advantage
        + [(-_RHO_BOUND, _RHO_BOUND)]                   # rho, kept in the safe region
    )

    result = minimize(
        _negative_log_likelihood,
        start,
        args=(home_idx, away_idx, home_goals, away_goals, weights,
              lgamma_home, lgamma_away, n_teams),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iterations, "ftol": 1e-10},
    )

    intercept, attack, defence, home_advantage, rho = _unpack(result.x, n_teams)
    return DixonColesParams(
        teams=teams,
        intercept=float(intercept),
        attack={team: float(attack[i]) for team, i in index.items()},
        defence={team: float(defence[i]) for team, i in index.items()},
        home_advantage=float(home_advantage),
        rho=float(rho),
        xi=xi,
        n_matches=len(matches),
        converged=bool(result.success),
        message=str(result.message),
    )


def fit_log_rates(matches: pd.DataFrame, *, as_of: datetime, xi: float = 0.0018,
                  home_value: str = "home_xg", away_value: str = "away_xg",
                  floor: float = 0.05) -> DixonColesParams:
    """Fit the same attack/defence structure to a continuous target such as xG.

    Goals are a small, noisy sample -- roughly 2.8 a match -- so a team's true
    strength takes most of a season to surface in scorelines. Expected goals
    converge far faster, which matters a great deal when the league produces
    only 306 matches a year.

    Poisson likelihood needs counts, so this fits log-rates by weighted least
    squares on the same design matrix instead. Rho is not identified here and is
    left at zero; the caller supplies it from the goals fit.
    """
    teams = tuple(sorted(set(matches["home_team_id"]) | set(matches["away_team_id"])))
    n_teams = len(teams)
    index = {team: i for i, team in enumerate(teams)}

    home_vals = np.maximum(matches[home_value].to_numpy(dtype=float), floor)
    away_vals = np.maximum(matches[away_value].to_numpy(dtype=float), floor)
    weights = time_weights(matches["kickoff_utc"], as_of, xi)

    n = len(matches)
    # Two observations per match (home rate and away rate) stacked into one
    # design matrix. Columns: intercept | attack (n-1) | defence (n-1) | home.
    n_free = 1 + (n_teams - 1) + (n_teams - 1) + 1
    design = np.zeros((2 * n, n_free))
    target = np.concatenate([np.log(home_vals), np.log(away_vals)])
    sample_weights = np.concatenate([weights, weights])

    design[:, 0] = 1.0
    home_idx = matches["home_team_id"].map(index).to_numpy()
    away_idx = matches["away_team_id"].map(index).to_numpy()

    def set_effect(rows, team_idx, offset, size):
        """Encode a sum-to-zero effect: the last team is minus the sum of the rest."""
        for row, t in zip(rows, team_idx):
            if t < size:
                design[row, offset + t] = 1.0
            else:
                design[row, offset:offset + size] = -1.0

    attack_offset, defence_offset = 1, 1 + (n_teams - 1)
    size = n_teams - 1
    set_effect(range(n), home_idx, attack_offset, size)
    set_effect(range(n, 2 * n), away_idx, attack_offset, size)
    set_effect(range(n), away_idx, defence_offset, size)
    set_effect(range(n, 2 * n), home_idx, defence_offset, size)
    design[:n, -1] = 1.0   # home advantage applies to the home rate only

    sqrt_w = np.sqrt(sample_weights)[:, None]
    solution, *_ = np.linalg.lstsq(design * sqrt_w, target * sqrt_w.ravel(), rcond=None)

    attack_free = solution[attack_offset:attack_offset + size]
    defence_free = solution[defence_offset:defence_offset + size]
    attack = np.append(attack_free, -attack_free.sum())
    defence = np.append(defence_free, -defence_free.sum())

    return DixonColesParams(
        teams=teams,
        intercept=float(solution[0]),
        attack={team: float(attack[i]) for team, i in index.items()},
        defence={team: float(defence[i]) for team, i in index.items()},
        home_advantage=float(solution[-1]),
        rho=0.0,
        xi=xi,
        n_matches=len(matches),
    )


def score_matrix(lam: float, mu: float, rho: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """Joint distribution over scorelines, corrected and renormalised.

    Rho is clipped to this fixture's own valid range: the correction is an
    approximation and a rho that is fine for one pair of rates can be invalid
    for another.
    """
    lower, upper = rho_bounds(lam, mu)
    safe_rho = float(np.clip(rho, lower + 1e-9, upper - 1e-9))

    goals = np.arange(max_goals + 1)
    home_pmf = np.exp(goals * np.log(lam) - lam - gammaln(goals + 1.0))
    away_pmf = np.exp(goals * np.log(mu) - mu - gammaln(goals + 1.0))
    matrix = np.outer(home_pmf, away_pmf)

    x = goals[:, None]
    y = goals[None, :]
    matrix = matrix * tau(np.broadcast_to(x, matrix.shape),
                          np.broadcast_to(y, matrix.shape), lam, mu, safe_rho)

    matrix = np.clip(matrix, 0.0, None)
    total = matrix.sum()
    # Truncation at max_goals and the tau correction both cost a little mass.
    return matrix / total if total > 0 else matrix


def outcome_probabilities(matrix: np.ndarray) -> np.ndarray:
    """Collapse a scoreline matrix into (home win, draw, away win)."""
    return np.array([
        float(np.tril(matrix, -1).sum()),
        float(np.trace(matrix)),
        float(np.triu(matrix, 1).sum()),
    ])


def match_probabilities(lam: float, mu: float, rho: float,
                        max_goals: int = MAX_GOALS) -> np.ndarray:
    return outcome_probabilities(score_matrix(lam, mu, rho, max_goals))


def over_under(matrix: np.ndarray, line: float = 2.5) -> tuple[float, float]:
    """Totals market probabilities from the same scoreline matrix."""
    goals = np.arange(matrix.shape[0])
    totals = goals[:, None] + goals[None, :]
    over = float(matrix[totals > line].sum())
    return over, 1.0 - over


def both_teams_to_score(matrix: np.ndarray) -> float:
    return float(matrix[1:, 1:].sum())


# ------------------------------------------------------------------- the model


class DixonColesModel(Model):
    """Time-weighted Dixon-Coles, optionally blended with xG-derived rates.

    `target`:
        "goals" -- the classical fit.
        "xg"    -- attack and defence estimated from expected goals, which
                   converge far faster than scorelines on a short season.
        "blend" -- a geometric blend of the two rate sets, which is usually the
                   best of the three because xG is a cleaner signal while goals
                   are what actually gets paid out.

    Rho always comes from the goals fit: the low-score correction describes how
    scorelines behave, and xG has no notion of a 1-1.
    """

    def __init__(self, xi: float = 0.0018, target: str = "goals",
                 blend_weight: float = 0.5, use_promoted_prior: bool = True,
                 use_availability: bool = False, min_training_matches: int = 100,
                 name: str | None = None) -> None:
        if target not in {"goals", "xg", "blend"}:
            raise ValueError(f"target must be goals, xg or blend, got {target!r}")
        if not 0.0 <= blend_weight <= 1.0:
            raise ValueError("blend_weight must lie in [0, 1]")

        self.xi = xi
        self.target = target
        self.blend_weight = blend_weight
        self.use_promoted_prior = use_promoted_prior
        self.use_availability = use_availability
        self.min_training_matches = min_training_matches
        self.name = name or f"dixon_coles_{target}"

        self.params: DixonColesParams | None = None
        self.xg_params: DixonColesParams | None = None
        self.prior = None
        self.player_rates: pd.DataFrame = pd.DataFrame()
        self.absence_impacts: dict[str, object] = {}
        self._fallback = np.array([0.45, 0.25, 0.30])

    # ------------------------------------------------------------------ fit

    def fit(self, store, as_of: datetime) -> "DixonColesModel":
        from bet.models.promoted import apply_prior, count_appearances, fit_promoted_prior

        matches = store.matches_as_of(as_of)
        if len(matches) < self.min_training_matches:
            self.params = None
            return self

        self.params = fit_dixon_coles(matches, as_of=as_of, xi=self.xi)

        if self.target in {"xg", "blend"}:
            xg_matches = self._xg_frame(store, as_of, matches)
            self.xg_params = (
                fit_log_rates(xg_matches, as_of=as_of, xi=self.xi)
                if xg_matches is not None and len(xg_matches) >= self.min_training_matches
                else None
            )
        else:
            self.xg_params = None

        if self.use_availability:
            # Per-90 rates as they stood at as_of, so the absence adjustment is
            # point-in-time correct like everything else.
            self.player_rates = store.player_rates_as_of(as_of, min_minutes=0.0)

        if self.use_promoted_prior:
            ratings = store.ratings_as_of(as_of)
            appearances = count_appearances(matches)
            self.prior = fit_promoted_prior(self.params, ratings, appearances)
            adjusted = apply_prior(self.params, self.prior, ratings, appearances)
            self.params.attack = adjusted["attack"]
            self.params.defence = adjusted["defence"]

        return self

    def _xg_frame(self, store, as_of: datetime, matches: pd.DataFrame) -> pd.DataFrame | None:
        """Aggregate ingested shots into per-match xG for and against."""
        shots = store.shots_as_of(as_of)
        if shots.empty:
            return None

        totals = shots.groupby(["match_id", "team_id"])["xg"].sum().reset_index()
        merged = matches.merge(totals, on="match_id", how="inner")
        if merged.empty:
            return None

        home_xg = merged[merged["team_id"] == merged["home_team_id"]][["match_id", "xg"]]
        away_xg = merged[merged["team_id"] == merged["away_team_id"]][["match_id", "xg"]]

        frame = (matches
                 .merge(home_xg.rename(columns={"xg": "home_xg"}), on="match_id", how="inner")
                 .merge(away_xg.rename(columns={"xg": "away_xg"}), on="match_id", how="inner"))
        return frame if not frame.empty else None

    # -------------------------------------------------------------- predict

    def predict(self, store, fixtures: pd.DataFrame, as_of: datetime) -> Prediction:
        if self.params is None:
            return self._validate(np.tile(self._fallback, (len(fixtures), 1)), len(fixtures))

        out = np.zeros((len(fixtures), 3))
        for i, row in enumerate(fixtures.itertuples(index=False)):
            lam, mu = self._rates(row.home_team_id, row.away_team_id)
            if self.use_availability:
                lam, mu = self._apply_availability(
                    store, as_of, row.home_team_id, row.away_team_id, lam, mu)
            out[i] = match_probabilities(lam, mu, self.params.rho)
        return self._validate(out, len(fixtures))

    def _absence_impact(self, store, as_of: datetime, team_id: str):
        """Cached per-team absence impact for this as_of."""
        from bet.availability import absences_from_store, estimate_absence_impact

        key = f"{team_id}:{as_of.isoformat()}"
        if key not in self.absence_impacts:
            absences = absences_from_store(store, as_of, team_id)
            self.absence_impacts[key] = estimate_absence_impact(
                self.player_rates, team_id, absences)
        return self.absence_impacts[key]

    def _apply_availability(self, store, as_of: datetime, home: str, away: str,
                            lam: float, mu: float) -> tuple[float, float]:
        """Shift both rates for who is missing on each side.

        Rates are exponential in the parameters, so an absence is an additive
        shift on the log scale. A team losing attackers scores less; a team
        losing defenders concedes more, which raises the *opponent's* rate --
        hence the opposite sign on the defence term.
        """
        home_impact = self._absence_impact(store, as_of, home)
        away_impact = self._absence_impact(store, as_of, away)

        log_lam = np.log(lam) + home_impact.attack_log_shift - away_impact.defence_log_shift
        log_mu = np.log(mu) + away_impact.attack_log_shift - home_impact.defence_log_shift
        # Clipped for the same reason the rates are clipped during fitting: a
        # long injury list must not be able to produce an absurd scoreline.
        return float(np.exp(np.clip(log_lam, -3.0, 2.5))), float(np.exp(np.clip(log_mu, -3.0, 2.5)))

    def _rates(self, home: str, away: str) -> tuple[float, float]:
        lam, mu = self.params.rates(home, away)

        if self.target == "goals" or self.xg_params is None:
            return lam, mu

        lam_xg, mu_xg = self.xg_params.rates(home, away)
        if self.target == "xg":
            return lam_xg, mu_xg

        # Geometric blend: averaging on the log-rate scale keeps rates positive
        # and treats a proportional difference the same way at any level.
        w = self.blend_weight
        return (float(np.exp(w * np.log(lam_xg) + (1 - w) * np.log(lam))),
                float(np.exp(w * np.log(mu_xg) + (1 - w) * np.log(mu))))

    # --------------------------------------------------------------- extras

    def predict_scorelines(self, home: str, away: str, max_goals: int = MAX_GOALS) -> pd.DataFrame:
        """Full scoreline distribution for one fixture."""
        if self.params is None:
            raise RuntimeError("model is not fitted")
        lam, mu = self._rates(home, away)
        matrix = score_matrix(lam, mu, self.params.rho, max_goals)
        return pd.DataFrame(
            matrix,
            index=pd.Index(range(max_goals + 1), name="home_goals"),
            columns=pd.Index(range(max_goals + 1), name="away_goals"),
        )

    def predict_markets(self, home: str, away: str) -> dict[str, float]:
        """1X2, totals and both-teams-to-score from a single scoreline matrix.

        Deriving every market from one matrix keeps them mutually consistent --
        the totals price and the 1X2 price can never imply different scorelines.
        """
        if self.params is None:
            raise RuntimeError("model is not fitted")
        lam, mu = self._rates(home, away)
        matrix = score_matrix(lam, mu, self.params.rho)
        probs = outcome_probabilities(matrix)
        over25, under25 = over_under(matrix, 2.5)
        return {
            "expected_home_goals": lam,
            "expected_away_goals": mu,
            **{f"p_{o.lower()}": float(p) for o, p in zip(OUTCOMES, probs)},
            "p_over_2_5": over25,
            "p_under_2_5": under25,
            "p_btts": both_teams_to_score(matrix),
        }


def tune_decay(store, *, start: datetime, end: datetime | None = None,
               candidates: tuple[float, ...] = (0.0, 0.0005, 0.001, 0.0015,
                                                0.002, 0.003, 0.005),
               target: str = "goals", league: str | None = None,
               verbose: bool = False) -> pd.DataFrame:
    """Select the time-decay rate by out-of-sample score, not by assumption.

    Xi is a genuine tuning parameter and the right value is not obvious: too
    little decay and the model still believes in a squad that has since been
    sold, too much and it fits the last three weeks of noise. Picking it by
    backtest is the only defensible approach, and the sweep costs one
    walk-forward per candidate.

    Note that this tunes on the same period you then report on, which mildly
    flatters the result. For a clean number, tune on earlier seasons and report
    on later ones.
    """
    from bet.evaluation.backtest import walk_forward

    rows = []
    for xi in candidates:
        model = DixonColesModel(xi=xi, target=target, name=f"dc_xi_{xi}")
        result = walk_forward(store, model, start=start, end=end, league=league)
        if result.predictions.empty:
            continue
        rows.append({"xi": xi, **result.metrics,
                     "half_life_days": float(np.log(2) / xi) if xi > 0 else float("inf")})
        if verbose:
            print(f"  xi={xi:<8} rps={result.metrics['rps']:.5f} "
                  f"logloss={result.metrics['log_loss']:.5f}")

    table = pd.DataFrame(rows)
    return table.sort_values("log_loss").reset_index(drop=True) if not table.empty else table
