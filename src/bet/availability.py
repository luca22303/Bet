"""Player availability and its effect on team strength.

The hard part is not knowing a player is injured. It is knowing what that is
worth, and this is where most injury-adjusted models quietly fall apart: someone
picks a number ("drop attack 15% if the first-choice striker is out"), the number
is never checked against anything, and it silently decalibrates every probability
the model produces. A miscalibrated model is worse than no adjustment at all,
because it produces confident wrong prices rather than honest uncertain ones.

So nothing here is hand-tuned. The impact of an absence is estimated from the
data already in the store:

    1.  What the absent player contributes per 90, measured in whatever the
        relevant output is (xG for attack, defensive actions for defence).
    2.  What his replacement contributes -- the best available player in the same
        position group, not a league-average abstraction.
    3.  The difference, as a share of the team's total output.

That share converts directly into a multiplicative change in the team's expected
goals, and since Dixon-Coles works on log rates, into an additive shift in the
attack parameter. A 6% loss of attacking output becomes attack + log(0.94).

Two honest caveats, both worth stating before anyone bets on this:

The market prices known injuries faster than any scraper. Team news moves lines
within minutes and retail books suspend markets while it does. The realistic
value here is not beating the market to the news; it is stopping your own model
pricing a fixture around a striker who is on the bench.

Absence impact is bounded by what a squad-level model can see. A defensive
midfielder holding a shape contributes in ways no per-90 counting stat captures,
and this will understate him. The cap on total adjustment exists partly for that
reason: when the estimate is uncertain, a smaller adjustment is the safer error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd

from bet.players import position_group


class AvailabilityStatus(str, Enum):
    """How likely a player is to feature."""

    FIT = "FIT"
    DOUBTFUL = "DOUBTFUL"
    OUT = "OUT"
    SUSPENDED = "SUSPENDED"
    UNKNOWN = "UNKNOWN"

    @property
    def availability_weight(self) -> float:
        """Probability the player features, used to scale the adjustment.

        A doubtful player is genuinely uncertain, so he counts as a partial
        absence rather than being forced to one side of a binary.
        """
        return {
            AvailabilityStatus.FIT: 1.0,
            AvailabilityStatus.DOUBTFUL: 0.5,
            AvailabilityStatus.OUT: 0.0,
            AvailabilityStatus.SUSPENDED: 0.0,
            AvailabilityStatus.UNKNOWN: 1.0,
        }[self]


# Which per-90 statistic stands in for a player's contribution on each side of
# the ball. Attack uses xG plus xA, because a winger who creates but rarely
# shoots is not replaceable by a striker with the same xG.
ATTACK_COLUMNS = ("xg_p90", "xa_p90")
DEFENCE_COLUMNS = ("tackles_p90", "interceptions_p90", "blocks_p90")

# Ceiling on how far one team's strength may be moved by absences, on the log
# rate scale. exp(-0.35) is about a 30% cut in expected goals -- already an
# extreme claim for a squad-level estimate, and a deliberate refusal to let a
# long injury list produce an absurd price.
MAX_LOG_ADJUSTMENT = 0.35

# Minutes of evidence at which a player's own per-90 rates are trusted as much
# as the squad average. A fringe player with 300 minutes can easily post a
# freak per-90; taken at face value it makes him look like a better replacement
# than the starter he is meant to be covering for, which zeroes out the very
# absence being measured. Ten matches is the halfway point.
CONTRIBUTION_PRIOR_MINUTES = 900.0


@dataclass
class AbsenceImpact:
    team_id: str
    absent_players: list[str] = field(default_factory=list)
    attack_multiplier: float = 1.0
    defence_multiplier: float = 1.0
    attack_log_shift: float = 0.0
    defence_log_shift: float = 0.0
    capped: bool = False
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        if not self.absent_players:
            return f"{self.team_id}: full squad available"
        cap = " (capped)" if self.capped else ""
        return (f"{self.team_id}: {len(self.absent_players)} absent, "
                f"attack x{self.attack_multiplier:.3f}, "
                f"defence x{self.defence_multiplier:.3f}{cap}")


def _contribution(row: pd.Series, columns: tuple[str, ...]) -> float:
    return float(sum(float(row.get(c) or 0.0) for c in columns))


def _shrunk_contribution(squad: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    """Per-90 contribution, shrunk toward the squad mean by minutes played.

    The same empirical-Bayes idea the prop model uses, and needed here for the
    same reason: a per-90 rate from a few hundred minutes is mostly noise. Left
    unshrunk it produces a specific, silent failure -- a fringe forward with one
    good spell outranks the first-choice striker, so the striker's "replacement"
    looks better than he is and his absence costs the model nothing.
    """
    raw = squad.apply(lambda r: _contribution(r, columns), axis=1)
    minutes = squad["minutes"].astype(float)
    if minutes.sum() <= 0:
        return raw

    squad_mean = float(np.average(raw, weights=minutes))
    weight = minutes / (minutes + CONTRIBUTION_PRIOR_MINUTES)
    return weight * raw + (1.0 - weight) * squad_mean


def estimate_absence_impact(rates: pd.DataFrame, team_id: str,
                            absences: dict[str, AvailabilityStatus],
                            *, min_minutes: float = 270.0,
                            squad_size: int = 11) -> AbsenceImpact:
    """Estimate how much a team's output falls when specific players are absent.

    `rates` is the per-90 table from `Store.player_rates_as_of`, so this is
    automatically point-in-time correct -- the impact is computed from what was
    knowable, not from the player's eventual season.
    """
    impact = AbsenceImpact(team_id=team_id)
    if rates.empty or not absences:
        return impact

    squad = rates[(rates["team_id"] == team_id) & (rates["minutes"] >= min_minutes)].copy()
    if squad.empty:
        impact.notes.append("no squad data with enough minutes; no adjustment made")
        return impact

    squad["group"] = squad["position"].map(position_group)
    squad["attack_contribution"] = _shrunk_contribution(squad, ATTACK_COLUMNS)
    squad["defence_contribution"] = _shrunk_contribution(squad, DEFENCE_COLUMNS)

    # The baseline is the strongest available eleven, not the whole squad: a
    # team's output comes from who plays, so fringe players must not dilute the
    # denominator and shrink every absence toward nothing.
    starters = squad.nlargest(min(squad_size, len(squad)), "minutes")
    attack_total = float(starters["attack_contribution"].sum())
    defence_total = float(starters["defence_contribution"].sum())

    attack_loss = 0.0
    defence_loss = 0.0

    # How much a player actually plays, as a proxy for whether he would have
    # been in the XI. Without this, a reserve keeper being ruled out changes the
    # forecast even though the same eleven take the field either way.
    max_minutes = float(squad["minutes"].max()) or 1.0

    for player_id, status in absences.items():
        weight = 1.0 - status.availability_weight
        if weight <= 0:
            continue

        player = squad[squad["player_id"] == player_id]
        if player.empty:
            impact.notes.append(f"{player_id}: no rate data, treated as no impact")
            continue

        row = player.iloc[0]
        impact.absent_players.append(player_id)

        # A fringe player's absence barely moves the XI, so it barely moves the
        # forecast. A first-choice player's absence moves both.
        start_share = float(np.clip(float(row["minutes"]) / max_minutes, 0.0, 1.0))
        replacement = _best_replacement(squad, row, absences)
        for column, total, accumulator in (
            ("attack_contribution", attack_total, "attack"),
            ("defence_contribution", defence_total, "defence"),
        ):
            if total <= 0:
                continue
            absent_value = float(row[column])
            replacement_value = float(replacement[column]) if replacement is not None else 0.0
            # Floored at zero: an absence may cost a team nothing, but it can
            # never make the team better. A negative delta here is always an
            # artifact of replacement matching rather than a real effect -- most
            # obviously for a goalkeeper, where a squad with one keeper has no
            # same-position cover and the fallback is an outfielder whose
            # attacking contribution is far higher. Without the floor, ruling
            # out the first-choice keeper raises the team's expected goals.
            delta = max(0.0, start_share * (absent_value - replacement_value) / total)
            if accumulator == "attack":
                attack_loss += weight * delta
            else:
                defence_loss += weight * delta

    impact.attack_multiplier, impact.attack_log_shift, attack_capped = _to_multiplier(attack_loss)
    # A defensive absence makes the team concede more, so it raises the
    # opponent's rate. The sign is handled where the shift is applied.
    impact.defence_multiplier, impact.defence_log_shift, defence_capped = _to_multiplier(defence_loss)
    impact.capped = attack_capped or defence_capped
    return impact


def _best_replacement(squad: pd.DataFrame, absent: pd.Series,
                      absences: dict[str, AvailabilityStatus]) -> pd.Series | None:
    """The best available player likely to replace the absentee.

    Replacement level is squad-specific on purpose. A club with a strong bench
    loses far less to an injury than one without, and a league-average
    replacement erases exactly that difference.

    When no same-position cover exists, the fallback is the weakest available
    outfielder rather than nothing. A team always fields eleven players, so
    somebody plays; treating an uncovered position as a total loss of that
    player's output overstates the absence badly enough to rank a squad
    midfielder as a worse loss than a first-choice striker.
    """
    unavailable = {pid for pid, status in absences.items() if status.availability_weight < 1.0}
    available = squad[
        (~squad["player_id"].isin(unavailable))
        & (squad["player_id"] != absent["player_id"])
    ]
    if available.empty:
        return None

    same_position = available[available["group"] == absent["group"]]
    if not same_position.empty:
        # Ranked by minutes: the player the manager already trusts most.
        return same_position.nlargest(1, "minutes").iloc[0]

    # No cover in that role: whoever is furthest down the rotation steps in.
    return available.nsmallest(1, "minutes").iloc[0]


def _to_multiplier(loss_share: float) -> tuple[float, float, bool]:
    """Convert a fractional output loss into a capped multiplier and log shift."""
    multiplier = float(np.clip(1.0 - loss_share, 0.05, 2.0))
    log_shift = float(np.log(multiplier))
    capped = abs(log_shift) > MAX_LOG_ADJUSTMENT
    if capped:
        log_shift = float(np.clip(log_shift, -MAX_LOG_ADJUSTMENT, MAX_LOG_ADJUSTMENT))
        multiplier = float(np.exp(log_shift))
    return multiplier, log_shift, capped


def absences_from_store(store, as_of: datetime, team_id: str,
                        *, stale_after_days: int = 21) -> dict[str, AvailabilityStatus]:
    """Current absences for a team, from the latest report per player.

    Reports go stale. An injury note from six weeks ago with no update is not
    evidence the player is still out, and treating it as such would sideline
    half a squad by March.
    """
    availability = store.availability_as_of(as_of, team_id=team_id)
    if availability.empty:
        return {}

    absences: dict[str, AvailabilityStatus] = {}
    for row in availability.itertuples():
        age_days = (pd.Timestamp(as_of) - pd.Timestamp(row.known_at)).days
        if age_days > stale_after_days:
            continue

        # An expected return date that has passed supersedes the status.
        if row.expected_return is not None and not pd.isna(row.expected_return):
            if pd.Timestamp(row.expected_return) <= pd.Timestamp(as_of):
                continue

        try:
            status = AvailabilityStatus(str(row.status).upper())
        except ValueError:
            continue
        if status.availability_weight < 1.0:
            absences[row.player_id] = status

    return absences
