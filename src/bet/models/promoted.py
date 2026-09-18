"""Priors for teams with too little top-flight history.

Every August, three sides arrive from the 2. Bundesliga with no Bundesliga
record at all. A model fitted on Bundesliga results alone has nothing to say
about them, so it assigns them the league average -- which is badly wrong in
both directions, since promoted teams are usually weaker than average but
occasionally are not. The error persists into October, and because the market
prices these teams using information the model lacks, the model will believe it
has found enormous value on exactly the fixtures it understands least.

ClubElo tracks clubs across divisions, so it holds precisely the missing
information. The mapping from rating to fitted strength is learned from the
teams that do have history, not assumed: fit a line through
(Elo, attack) and (Elo, defence) for established teams, then read off the
promoted side's rating. No hand-tuned constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass
class PromotedTeamPrior:
    """Linear maps from an Elo rating onto attack and defence strength."""

    attack_slope: float = 0.0
    attack_intercept: float = 0.0
    defence_slope: float = 0.0
    defence_intercept: float = 0.0
    rating_mean: float = 1500.0
    fitted: bool = False
    n_reference_teams: int = 0

    def attack_for(self, rating: float) -> float:
        if not self.fitted:
            return 0.0
        return self.attack_slope * (rating - self.rating_mean) + self.attack_intercept

    def defence_for(self, rating: float) -> float:
        if not self.fitted:
            return 0.0
        return self.defence_slope * (rating - self.rating_mean) + self.defence_intercept


def fit_promoted_prior(params, ratings: pd.DataFrame, appearances: pd.Series,
                       *, min_matches: int = 30) -> PromotedTeamPrior:
    """Learn the rating-to-strength relationship from established teams.

    Only teams with enough matches are used as reference points; including the
    sparse teams would be circular, since their strengths are the ones being
    estimated.
    """
    if ratings.empty:
        return PromotedTeamPrior()

    rating_by_team = dict(zip(ratings["team_id"], ratings["rating"]))
    rows = [
        (rating_by_team[team], params.attack[team], params.defence[team])
        for team in params.teams
        if team in rating_by_team and appearances.get(team, 0) >= min_matches
    ]
    # Two points define a line but tell you nothing; require a real sample.
    if len(rows) < 6:
        return PromotedTeamPrior()

    values = np.array(rows, dtype=float)
    centred = values[:, 0] - values[:, 0].mean()
    if np.allclose(centred, 0.0):
        return PromotedTeamPrior()

    attack_slope, attack_intercept = np.polyfit(centred, values[:, 1], 1)
    defence_slope, defence_intercept = np.polyfit(centred, values[:, 2], 1)

    return PromotedTeamPrior(
        attack_slope=float(attack_slope),
        attack_intercept=float(attack_intercept),
        defence_slope=float(defence_slope),
        defence_intercept=float(defence_intercept),
        rating_mean=float(values[:, 0].mean()),
        fitted=True,
        n_reference_teams=len(rows),
    )


def apply_prior(params, prior: PromotedTeamPrior, ratings: pd.DataFrame,
                appearances: pd.Series, *, min_matches: int = 30,
                full_confidence_matches: int = 60) -> dict[str, dict[str, float]]:
    """Blend fitted strengths toward the Elo prior for thin-data teams.

    The weight on the fitted value ramps linearly with matches played, so a
    promoted side starts on its prior in August and has moved fully onto its own
    record by midwinter. A hard switch at a threshold would put a visible
    discontinuity in the middle of the season.
    """
    if not prior.fitted or ratings.empty:
        return {"attack": dict(params.attack), "defence": dict(params.defence)}

    rating_by_team = dict(zip(ratings["team_id"], ratings["rating"]))
    attack = dict(params.attack)
    defence = dict(params.defence)

    for team in params.teams:
        played = float(appearances.get(team, 0))
        if played >= full_confidence_matches or team not in rating_by_team:
            continue

        rating = rating_by_team[team]
        weight = np.clip(played / full_confidence_matches, 0.0, 1.0)
        attack[team] = weight * attack[team] + (1 - weight) * prior.attack_for(rating)
        defence[team] = weight * defence[team] + (1 - weight) * prior.defence_for(rating)

    return {"attack": attack, "defence": defence}


def count_appearances(matches: pd.DataFrame) -> pd.Series:
    """Matches played per team in the training window."""
    if matches.empty:
        return pd.Series(dtype=float)
    return pd.concat([matches["home_team_id"], matches["away_team_id"]]).value_counts()
