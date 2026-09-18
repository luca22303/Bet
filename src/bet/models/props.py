"""Player prop modelling: shots, shots on target, tackles, cards.

The strategic argument for this module: 1X2 is the most efficient market in
football, priced by every sharp in the world, and a public-data model is
unlikely to beat it. Player props are priced far more loosely, partly because
there are hundreds of them per matchday and partly because they need exactly the
player-level data that most bettors do not assemble. That is where a player
pipeline pays for itself, and it is not where a team-level Dixon-Coles model can
help at all.

A prop count is built from four multiplicative pieces, each estimated:

    expected count = per-90 rate x (minutes / 90) x opponent factor x volume factor

Per-90 rate is shrunk toward the position-group mean by empirical Bayes, because
a striker with two starts has a rate that is mostly noise, and taking it at face
value is how prop models lose money on squad players.

Opponent factor is what a defence concedes relative to league average, shrunk
the same way.

Volume factor scales with how many goals the team is expected to score, taken
from the Dixon-Coles engine so the prop and the match model cannot disagree.
Shots scale sub-linearly with expected goals, so the elasticity is below one.

The count distribution is negative binomial, not Poisson. Player shot counts are
reliably overdispersed: a striker takes one shot in a quiet match and seven in an
open one. Overdispersion moves probability out of the middle and into both tails,
so which way it moves a price depends on where the line sits. Near or below the
mean it lowers the overs; in the upper tail it raises them sharply. At an
expected 2.0 shots, Poisson prices over 5.5 at 1.7% while a negative binomial
with r=3 prices it at 5.0% -- a factor of three, on exactly the alternative lines
a book is least careful about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from bet.players import position_group

# Stats that can be modelled as counts, with a sensible market line each.
SUPPORTED_STATS = {
    "shots": 1.5,
    "shots_on_target": 0.5,
    "goals": 0.5,
    "assists": 0.5,
    "tackles": 1.5,
    "interceptions": 0.5,
    "fouls": 0.5,
    "yellow_cards": 0.5,
}

# Shots respond to a team's attacking volume, but less than proportionally: a
# team expected to score twice as much does not take twice as many shots.
DEFAULT_ELASTICITY = 0.7


@dataclass
class PropPrediction:
    player_id: str
    stat: str
    line: float
    expected_minutes: float
    rate_per90: float
    expected_count: float
    dispersion: float
    p_over: float
    p_under: float
    fair_odds_over: float
    fair_odds_under: float
    opponent_factor: float = 1.0
    volume_factor: float = 1.0
    sample_90s: float = 0.0

    def describe(self) -> str:
        return (f"{self.player_id} {self.stat} o{self.line}: "
                f"{self.expected_count:.2f} expected, p={self.p_over:.3f}, "
                f"fair {self.fair_odds_over:.2f} "
                f"({self.sample_90s:.1f} nineties of evidence)")


@dataclass
class StatPriors:
    """Shrinkage priors for one stat, estimated from the league cross-section."""

    stat: str
    group_means: dict[str, float] = field(default_factory=dict)
    league_mean: float = 0.0
    prior_strength: float = 5.0          # in units of 90 minutes
    dispersion: float = 5.0              # negative binomial r; higher is nearer Poisson
    league_conceded_per_match: float = 0.0

    def prior_for(self, position: str | None) -> float:
        return self.group_means.get(position_group(position), self.league_mean)


def estimate_priors(rates: pd.DataFrame, stat: str, *,
                    min_prior_strength: float = 1.0,
                    max_prior_strength: float = 40.0) -> StatPriors:
    """Method-of-moments empirical Bayes priors for a stat.

    For a Gamma-Poisson mixture the between-player variance in true rates is the
    observed variance minus the part explained by sampling noise. What is left
    sets how much to trust a small sample: if players genuinely differ a lot,
    shrink less; if the spread is mostly noise, shrink hard.
    """
    column = f"{stat}_p90"
    if column not in rates.columns or rates.empty:
        return StatPriors(stat=stat)

    usable = rates[(rates["minutes"] > 0) & rates[column].notna()].copy()
    if usable.empty:
        return StatPriors(stat=stat)

    usable["nineties"] = usable["minutes"] / 90.0
    weights = usable["nineties"].to_numpy()
    values = usable[column].to_numpy()

    league_mean = float(np.average(values, weights=weights))
    if league_mean <= 0:
        return StatPriors(stat=stat, league_mean=league_mean)

    observed_variance = float(np.average((values - league_mean) ** 2, weights=weights))
    # A pure Poisson process would produce this much variance on its own.
    sampling_variance = league_mean / max(float(np.mean(weights)), 1e-9)
    between_variance = max(observed_variance - sampling_variance, 1e-6)

    # Gamma prior with mean league_mean and variance between_variance.
    prior_strength = float(np.clip(league_mean / between_variance,
                                   min_prior_strength, max_prior_strength))

    group_means: dict[str, float] = {}
    if "position" in usable.columns:
        usable["group"] = usable["position"].map(position_group)
        for group, block in usable.groupby("group"):
            if block["nineties"].sum() > 5:
                group_means[group] = float(
                    np.average(block[column], weights=block["nineties"]))

    # Negative binomial dispersion from the same excess variance. Larger r means
    # closer to Poisson; the floor stops a degenerate estimate.
    dispersion = float(np.clip(league_mean ** 2 / between_variance, 0.5, 60.0))

    return StatPriors(
        stat=stat,
        group_means=group_means,
        league_mean=league_mean,
        prior_strength=prior_strength,
        dispersion=dispersion,
    )


def shrink_rate(observed_count: float, nineties: float, prior_rate: float,
                prior_strength: float) -> float:
    """Posterior mean rate under a Gamma-Poisson model.

    With no evidence this returns the prior; with a lot it returns the player's
    own rate. The interesting region is in between, which is where most squad
    players live all season.
    """
    if nineties <= 0:
        return prior_rate
    alpha = prior_rate * prior_strength
    return float((observed_count + alpha) / (nineties + prior_strength))


def negative_binomial_over(expected: float, dispersion: float, line: float) -> float:
    """P(count > line) for an overdispersed count.

    A prop line of 1.5 means the bet wins on 2 or more, so the threshold is the
    floor of the line.
    """
    if expected <= 0:
        return 0.0
    threshold = int(np.floor(line))
    r = max(dispersion, 1e-6)
    p = r / (r + expected)
    return float(1.0 - nbinom.cdf(threshold, r, p))


def poisson_over(expected: float, line: float) -> float:
    """Poisson equivalent, kept for comparison.

    Differs from the negative binomial in opposite directions either side of the
    mean, so it is worth printing both when a prop price looks surprising.
    """
    if expected <= 0:
        return 0.0
    return float(1.0 - poisson.cdf(int(np.floor(line)), expected))


class PlayerPropModel:
    """Counts for a single stat, fitted across the league."""

    def __init__(self, stat: str = "shots", lookback_days: int = 540,
                 min_minutes: float = 180.0, elasticity: float = DEFAULT_ELASTICITY,
                 opponent_shrinkage: float = 6.0) -> None:
        if stat not in SUPPORTED_STATS:
            raise ValueError(f"unsupported stat {stat!r}; expected one of {sorted(SUPPORTED_STATS)}")
        self.stat = stat
        self.lookback_days = lookback_days
        self.min_minutes = min_minutes
        self.elasticity = elasticity
        self.opponent_shrinkage = opponent_shrinkage

        self.priors = StatPriors(stat=stat)
        self.rates: pd.DataFrame = pd.DataFrame()
        self.opponent_factors: dict[str, float] = {}
        self.league_goal_rate: float = 1.5
        self.as_of: datetime | None = None

    # ------------------------------------------------------------------ fit

    def fit(self, store, as_of: datetime) -> "PlayerPropModel":
        self.as_of = as_of
        self.rates = store.player_rates_as_of(
            as_of, lookback_days=self.lookback_days, min_minutes=0.0)
        if self.rates.empty:
            return self

        self.priors = estimate_priors(self.rates, self.stat)
        self._fit_opponent_factors(store, as_of)

        history = store.matches_as_of(as_of)
        if not history.empty:
            recent = history[pd.to_datetime(history["kickoff_utc"])
                             >= pd.Timestamp(as_of) - pd.Timedelta(days=self.lookback_days)]
            source = recent if not recent.empty else history
            self.league_goal_rate = float(
                (source["home_goals"].sum() + source["away_goals"].sum()) / (2 * len(source)))
        return self

    def _fit_opponent_factors(self, store, as_of: datetime) -> None:
        """How much of this stat each defence concedes, relative to average."""
        conceded = store.team_concessions_as_of(
            as_of, lookback_days=self.lookback_days, columns=(self.stat,))
        column = f"{self.stat}_conceded_per_match"
        if conceded.empty or column not in conceded.columns:
            return

        league_mean = float(conceded[column].mean())
        if league_mean <= 0:
            return
        self.priors.league_conceded_per_match = league_mean

        # Shrunk toward 1.0: ten matches of defensive data is a small sample and
        # an unshrunk factor swings prop prices far more than the evidence warrants.
        for row in conceded.itertuples():
            observed = getattr(row, column)
            matches = float(row.matches)
            weight = matches / (matches + self.opponent_shrinkage)
            self.opponent_factors[row.team_id] = float(
                weight * (observed / league_mean) + (1 - weight) * 1.0)

    # -------------------------------------------------------------- predict

    def expected_minutes(self, player_id: str, *, is_starter: bool | None = None,
                         starter_minutes: float = 78.0,
                         substitute_minutes: float = 18.0) -> float:
        """Minutes a player is expected to play.

        The single largest driver of a prop price, and the one most often got
        wrong. A confirmed XI collapses the uncertainty entirely, which is why
        the confirmed line-up matters more than any amount of form data.
        """
        row = self._player_row(player_id)
        if is_starter is True:
            return float(row["minutes_when_starting"]) if row is not None else starter_minutes
        if is_starter is False:
            return substitute_minutes
        if row is None:
            return 0.5 * starter_minutes
        # Unknown: weight by how often this player starts.
        start_rate = float(row["start_rate"])
        return start_rate * float(row["minutes_when_starting"]) + (1 - start_rate) * substitute_minutes

    def _player_row(self, player_id: str):
        if self.rates.empty:
            return None
        match = self.rates[self.rates["player_id"] == player_id]
        if match.empty:
            return None
        row = match.iloc[0].to_dict()
        matches = max(float(row.get("matches") or 0), 1.0)
        starts = float(row.get("starts") or 0)
        minutes = float(row.get("minutes") or 0)
        row["start_rate"] = np.clip(starts / matches, 0.0, 1.0)
        # Back out typical minutes in a start from the totals available.
        row["minutes_when_starting"] = float(
            np.clip(minutes / max(starts, 1.0) if starts > 0 else minutes / matches, 10.0, 90.0))
        return row

    def predict(self, player_id: str, opponent_id: str | None = None, *,
                line: float | None = None, is_starter: bool | None = None,
                team_expected_goals: float | None = None,
                minutes_override: float | None = None) -> PropPrediction:
        """Price one prop."""
        line = SUPPORTED_STATS[self.stat] if line is None else line
        row = self._player_row(player_id)

        nineties = float(row["minutes"]) / 90.0 if row is not None else 0.0
        observed = float(row.get(self.stat) or 0.0) if row is not None else 0.0
        position = row.get("position") if row is not None else None

        rate = shrink_rate(observed, nineties,
                           self.priors.prior_for(position), self.priors.prior_strength)

        minutes = (minutes_override if minutes_override is not None
                   else self.expected_minutes(player_id, is_starter=is_starter))

        opponent_factor = self.opponent_factors.get(opponent_id, 1.0) if opponent_id else 1.0

        volume_factor = 1.0
        if team_expected_goals is not None and self.league_goal_rate > 0:
            volume_factor = float(
                (team_expected_goals / self.league_goal_rate) ** self.elasticity)

        expected = rate * (minutes / 90.0) * opponent_factor * volume_factor
        p_over = negative_binomial_over(expected, self.priors.dispersion, line)
        p_under = 1.0 - p_over

        return PropPrediction(
            player_id=player_id, stat=self.stat, line=line,
            expected_minutes=minutes, rate_per90=rate, expected_count=expected,
            dispersion=self.priors.dispersion,
            p_over=p_over, p_under=p_under,
            fair_odds_over=float(1.0 / p_over) if p_over > 0 else float("inf"),
            fair_odds_under=float(1.0 / p_under) if p_under > 0 else float("inf"),
            opponent_factor=opponent_factor, volume_factor=volume_factor,
            sample_90s=nineties,
        )

    def predict_match(self, store, match_id: str, home_team: str, away_team: str,
                      as_of: datetime, *, home_expected_goals: float | None = None,
                      away_expected_goals: float | None = None,
                      line: float | None = None,
                      confirmed_only: bool = False) -> pd.DataFrame:
        """Price this stat for everyone in a fixture's line-up.

        Falls back to recent starters when no line-up has been published, which
        is the normal case more than an hour before kickoff.
        """
        lineup = store.lineup_as_of(as_of, [match_id], confirmed_only=confirmed_only)
        rows = []

        for team_id, opponent_id, expected_goals in (
            (home_team, away_team, home_expected_goals),
            (away_team, home_team, away_expected_goals),
        ):
            if not lineup.empty:
                squad = lineup[lineup["team_id"] == team_id]
                entries = [(r.player_id, bool(r.is_starter)) for r in squad.itertuples()]
            else:
                entries = [(pid, None) for pid in self._likely_starters(team_id)]

            for player_id, is_starter in entries:
                prediction = self.predict(
                    player_id, opponent_id, line=line, is_starter=is_starter,
                    team_expected_goals=expected_goals)
                rows.append({
                    "match_id": match_id, "team_id": team_id, "opponent_id": opponent_id,
                    "player_id": player_id, "is_starter": is_starter,
                    "stat": self.stat, "line": prediction.line,
                    "expected_minutes": round(prediction.expected_minutes, 1),
                    "rate_p90": round(prediction.rate_per90, 3),
                    "expected": round(prediction.expected_count, 3),
                    "p_over": round(prediction.p_over, 4),
                    "fair_over": round(prediction.fair_odds_over, 2),
                    "sample_90s": round(prediction.sample_90s, 1),
                })

        frame = pd.DataFrame(rows)
        return frame.sort_values("expected", ascending=False).reset_index(drop=True) if not frame.empty else frame

    def _likely_starters(self, team_id: str, n: int = 14) -> list[str]:
        """Squad members most likely to feature, by recent minutes."""
        if self.rates.empty:
            return []
        squad = self.rates[self.rates["team_id"] == team_id]
        if squad.empty:
            return []
        return squad.nlargest(n, "minutes")["player_id"].tolist()
