"""Predicted line-ups: who will actually be on the pitch.

The problem this fixes. A team's strength was being computed from squad-wide
per-90 rates weighted by cumulative minutes, which quietly answers the wrong
question. A striker who played every week until September and has not appeared
since still carries a large share of the season's minutes, so the model keeps
pricing the team as though he plays. Meanwhile the squad player who has started
the last six matches barely registers.

So strength is computed from the eleven expected to start, not from the squad.
Three pieces:

    Start propensity. An exponentially recency-weighted start rate. A player who
    started the last five matches scores near one; a player who has not featured
    in six weeks scores near zero, however many minutes he banked in August.

    Formation. Managers are creatures of habit, and the recent formation
    distribution predicts the next one well. It also constrains the eleven: a
    side playing 3-4-3 fields three centre-backs, so the XI cannot simply be the
    eleven highest-propensity players regardless of shape.

    Rotation. Some managers rotate heavily and some name the same team every
    week. The consistency of recent line-ups says how much to trust the
    prediction, and that confidence is reported rather than hidden.

Once a confirmed XI is published the prediction is discarded entirely. A
confirmed line-up is not evidence to be blended with a guess; it is the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from bet.availability import (
    ATTACK_COLUMNS,
    CONTRIBUTION_PRIOR_MINUTES,
    DEFENCE_COLUMNS,
    AvailabilityStatus,
)
from bet.players import position_group

# How fast a past appearance stops predicting the next one. Three weeks is
# roughly three matchdays: recent enough to track a change of first choice,
# slow enough not to over-read a single rested weekend.
DEFAULT_HALF_LIFE_DAYS = 21.0

# Formation shapes as (defenders, midfielders, forwards). Goalkeeper is implied.
FORMATIONS = {
    "4-2-3-1": (4, 5, 1),
    "4-3-3": (4, 3, 3),
    "4-4-2": (4, 4, 2),
    "3-4-3": (3, 4, 3),
    "3-5-2": (3, 5, 2),
    "5-3-2": (5, 3, 2),
    "5-4-1": (5, 4, 1),
    "4-1-4-1": (4, 5, 1),
    "3-4-2-1": (3, 6, 1),
}
DEFAULT_FORMATION = "4-2-3-1"


@dataclass
class PredictedLineup:
    team_id: str
    as_of: datetime
    formation: str
    formation_confidence: float
    starters: list[str] = field(default_factory=list)
    bench: list[str] = field(default_factory=list)
    propensities: dict[str, float] = field(default_factory=dict)
    is_confirmed: bool = False
    rotation_score: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        """How much to trust this XI.

        A confirmed line-up is certain. Otherwise confidence is the mean
        propensity of the chosen eleven: a settled side scores high, a heavily
        rotated one low, and the difference should be visible downstream rather
        than buried.
        """
        if self.is_confirmed:
            return 1.0
        if not self.starters:
            return 0.0
        return float(np.mean([self.propensities.get(p, 0.0) for p in self.starters]))

    def describe(self) -> str:
        source = "confirmed" if self.is_confirmed else f"predicted ({self.confidence:.0%})"
        return f"{self.team_id} {self.formation} [{source}]"


def recency_weights(kickoffs, as_of: datetime,
                    half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> np.ndarray:
    """Exponential decay on match age, expressed as a half-life.

    A half-life is easier to reason about than a decay constant: at 21 days, a
    match three weeks ago counts half as much as last weekend.
    """
    age_days = (pd.Timestamp(as_of) - pd.to_datetime(pd.Series(kickoffs))).dt.total_seconds() / 86400.0
    age_days = np.clip(age_days.to_numpy(), 0.0, None)
    return np.exp(-np.log(2.0) * age_days / half_life_days)


def _team_history(store, team_id: str, as_of: datetime, *,
                  lookback_days: int = 180) -> tuple[pd.DataFrame, bool]:
    """This team's own per-match lines, widening the lookback when the normal
    window has nothing for this team at all.

    Recency weighting already fades an old match on its own -- a hard cutoff
    discarding everything older than `lookback_days` should not have to do
    that job again. Applied unconditionally, though, it turns "this team's
    last recorded match was 200 days ago" into "no history at all", which
    produces a shapeless, 0%-confidence guess and an empty predicted XI when
    the honest worst case is "assume they set up the same way they last did".

    Returns the squad's rows (possibly from beyond the normal window) and
    whether the fallback had to be used at all.
    """
    since = as_of - timedelta(days=lookback_days)
    stats = store.player_stats_as_of(as_of, since=since)
    squad = stats[stats["team_id"] == team_id].copy() if not stats.empty else stats
    if not squad.empty:
        return squad, False

    stats = store.player_stats_as_of(as_of)          # unbounded
    squad = stats[stats["team_id"] == team_id].copy() if not stats.empty else stats
    return squad, not squad.empty


def start_propensity(store, team_id: str, as_of: datetime, *,
                     lookback_days: int = 180,
                     half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> pd.DataFrame:
    """Recency-weighted probability each player starts the next match.

    This is the number that stops a long-absent star being priced as a starter.
    A player who has not appeared in months has near-zero weight on every recent
    match, so his propensity collapses regardless of how much he played earlier
    in the season.
    """
    squad, _ = _team_history(store, team_id, as_of, lookback_days=lookback_days)
    if squad.empty:
        return pd.DataFrame(columns=["player_id", "propensity", "position",
                                     "days_since_start", "appearances"])

    # Every match the team played in the window is a chance to start, so a
    # player who was absent is scored against those matches too rather than
    # being silently excluded from his own denominator.
    team_matches = squad[["match_id", "kickoff_utc"]].drop_duplicates()
    match_weights = dict(zip(team_matches["match_id"],
                             recency_weights(team_matches["kickoff_utc"], as_of, half_life_days)))
    total_weight = float(sum(match_weights.values()))
    if total_weight <= 0:
        return pd.DataFrame(columns=["player_id", "propensity", "position",
                                     "days_since_start", "appearances"])

    squad["weight"] = squad["match_id"].map(match_weights)
    squad["started_flag"] = squad["started"].fillna(False).astype(bool)

    rows = []
    for player_id, block in squad.groupby("player_id"):
        started = block[block["started_flag"]]
        weighted_starts = float(started["weight"].sum())

        last_start = pd.to_datetime(started["kickoff_utc"]).max() if not started.empty else None
        days_since = ((pd.Timestamp(as_of) - last_start).days
                      if last_start is not None else None)

        rows.append({
            "player_id": player_id,
            "position": block["position"].dropna().iloc[-1] if block["position"].notna().any() else None,
            "propensity": weighted_starts / total_weight,
            "days_since_start": days_since,
            "appearances": int(len(block)),
            "starts": int(len(started)),
        })

    frame = pd.DataFrame(rows)
    frame["group"] = frame["position"].map(position_group)
    return frame.sort_values("propensity", ascending=False).reset_index(drop=True)


def infer_formation(groups: list[str]) -> str:
    """Name a shape from the position groups of an eleven.

    Line-up sources are inconsistent about formation strings and FBref supplies
    none at all, so it is derived from who is on the pitch. An unrecognised
    count is returned verbatim rather than forced onto the nearest known shape.
    """
    defenders = sum(1 for g in groups if g == "defender")
    midfielders = sum(1 for g in groups if g == "midfielder")
    forwards = sum(1 for g in groups if g == "forward")

    for name, (d, m, f) in FORMATIONS.items():
        if (d, m, f) == (defenders, midfielders, forwards):
            return name
    return f"{defenders}-{midfielders}-{forwards}"


def predict_formation(store, team_id: str, as_of: datetime, *,
                      lookback_days: int = 180,
                      half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> tuple[str, float, float]:
    """Most likely formation, its confidence, and how much the side rotates.

    Returns `(formation, confidence, rotation_score)`. Rotation is the mean
    churn in personnel between consecutive line-ups: near zero for a manager who
    names the same team every week, high for one who does not. It is the honest
    measure of how far to trust any predicted XI.
    """
    squad, _ = _team_history(store, team_id, as_of, lookback_days=lookback_days)
    squad = squad[squad["started"].fillna(False)].copy() if not squad.empty else squad
    if squad.empty:
        return DEFAULT_FORMATION, 0.0, 0.0

    squad["group"] = squad["position"].map(position_group)

    shapes: dict[str, float] = {}
    elevens: list[tuple[datetime, set]] = []

    for match_id, block in squad.groupby("match_id"):
        kickoff = pd.to_datetime(block["kickoff_utc"].iloc[0])
        weight = float(recency_weights([kickoff], as_of, half_life_days)[0])
        outfield = [g for g in block["group"] if g != "goalkeeper"]
        if len(outfield) < 9:
            continue      # incomplete line-up record
        shapes[infer_formation(outfield)] = shapes.get(infer_formation(outfield), 0.0) + weight
        elevens.append((kickoff, set(block["player_id"])))

    if not shapes:
        return DEFAULT_FORMATION, 0.0, 0.0

    total = sum(shapes.values())
    formation = max(shapes, key=shapes.get)
    confidence = shapes[formation] / total if total else 0.0

    # Churn between consecutive line-ups, most recent first.
    elevens.sort(key=lambda item: item[0], reverse=True)
    changes = [
        len(elevens[i][1] ^ elevens[i + 1][1]) / 22.0
        for i in range(min(len(elevens) - 1, 8))
    ]
    rotation = float(np.mean(changes)) if changes else 0.0

    return formation, float(confidence), rotation


def predict_lineup(store, team_id: str, as_of: datetime, *,
                   lookback_days: int = 180,
                   half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
                   match_id: str | None = None,
                   use_confirmed: bool = True,
                   absences: dict[str, AvailabilityStatus] | None = None) -> PredictedLineup:
    """The eleven expected to start, or the confirmed eleven if it is published.

    A confirmed line-up replaces the prediction outright. It is not evidence to
    be blended with a guess -- it is the answer, and it is the single most
    valuable input this system can have, which is why it is checked first.
    """
    if use_confirmed and match_id:
        confirmed = store.lineup_as_of(as_of, [match_id], confirmed_only=True)
        confirmed = confirmed[confirmed["team_id"] == team_id] if not confirmed.empty else confirmed
        if not confirmed.empty and confirmed["is_starter"].sum() >= 10:
            starters = confirmed[confirmed["is_starter"]]["player_id"].tolist()
            bench = confirmed[~confirmed["is_starter"]]["player_id"].tolist()
            propensity = start_propensity(store, team_id, as_of,
                                          lookback_days=lookback_days,
                                          half_life_days=half_life_days)
            groups = _groups_for(propensity, starters)
            return PredictedLineup(
                team_id=team_id, as_of=as_of,
                formation=infer_formation([g for g in groups if g != "goalkeeper"]),
                formation_confidence=1.0,
                starters=starters, bench=bench,
                propensities={p: 1.0 for p in starters},
                is_confirmed=True,
                notes=["confirmed line-up"],
            )

    propensity = start_propensity(store, team_id, as_of,
                                  lookback_days=lookback_days,
                                  half_life_days=half_life_days)
    formation, formation_confidence, rotation = predict_formation(
        store, team_id, as_of, lookback_days=lookback_days, half_life_days=half_life_days)

    lineup = PredictedLineup(
        team_id=team_id, as_of=as_of, formation=formation,
        formation_confidence=formation_confidence, rotation_score=rotation)

    if propensity.empty:
        lineup.notes.append("no recent appearance data; no XI predicted")
        return lineup

    # `start_propensity`/`predict_formation` already widened their own search
    # when the normal window had nothing for this team; this just checks
    # whether that happened, so the prediction says plainly that it is
    # standing on an old match rather than reporting a number with no context.
    _, used_fallback = _team_history(store, team_id, as_of, lookback_days=lookback_days)
    if used_fallback:
        stale_days = int(propensity["days_since_start"].max()) \
            if propensity["days_since_start"].notna().any() else None
        lineup.notes.append(
            f"no appearance data in the last {lookback_days} days -- predicted from "
            + (f"a match {stale_days}d ago" if stale_days is not None else "older history")
            + "; treat this XI as a weak guess")

    available = propensity.copy()
    if absences:
        # An unavailable player cannot start, whatever his recent record. This
        # is how the availability layer now reaches the model: it edits who is
        # eligible rather than applying a separate correction afterwards.
        for player_id, status in absences.items():
            mask = available["player_id"] == player_id
            available.loc[mask, "propensity"] *= status.availability_weight

    lineup.propensities = dict(zip(available["player_id"], available["propensity"]))
    lineup.starters, lineup.bench = _select_eleven(available, formation)

    if rotation > 0.25:
        lineup.notes.append(
            f"heavy rotation ({rotation:.0%} churn between line-ups); XI is uncertain")
    return lineup


def _groups_for(propensity: pd.DataFrame, player_ids: list[str]) -> list[str]:
    lookup = dict(zip(propensity["player_id"], propensity.get("group", [])))
    return [lookup.get(p, "unknown") for p in player_ids]


def _select_eleven(available: pd.DataFrame, formation: str) -> tuple[list[str], list[str]]:
    """Fill each formation slot with the highest-propensity eligible player.

    Selecting the eleven highest propensities outright would field shapes no
    manager uses -- six forwards, no left back. The formation constrains the
    choice, which is the whole reason it is predicted first.
    """
    shape = FORMATIONS.get(formation)
    if shape is None:
        parts = formation.split("-")
        try:
            numbers = [int(p) for p in parts]
            shape = (numbers[0], sum(numbers[1:-1]) + numbers[-1] - numbers[-1], numbers[-1])
            shape = (numbers[0], sum(numbers[1:-1]), numbers[-1])
        except (ValueError, IndexError):
            shape = FORMATIONS[DEFAULT_FORMATION]

    needed = {"goalkeeper": 1, "defender": shape[0],
              "midfielder": shape[1], "forward": shape[2]}

    # A player scaled to zero propensity is unavailable, not merely unlikely, and
    # must leave the pool entirely. Sorting alone does not remove him: when every
    # forward in a squad is ruled out, taking the top three still returns three
    # ruled-out forwards, and the absence silently has no effect.
    eligible = available[available["propensity"] > 1e-9]
    if eligible.empty:
        eligible = available
    ordered = eligible.sort_values("propensity", ascending=False)

    starters: list[str] = []
    for group, count in needed.items():
        pool = ordered[ordered["group"] == group]
        starters.extend(pool.head(count)["player_id"].tolist())

    # Positions the squad cannot fill (a thin bench, or unmapped positions) are
    # topped up by propensity so an XI is still produced. Ineligible players are
    # only reached if there is genuinely nobody else, which means the squad data
    # is too thin to predict from at all.
    if len(starters) < 11:
        remaining = ordered[~ordered["player_id"].isin(starters)]
        starters.extend(remaining.head(11 - len(starters))["player_id"].tolist())
    if len(starters) < 11:
        fallback = available[~available["player_id"].isin(starters)].sort_values(
            "propensity", ascending=False)
        starters.extend(fallback.head(11 - len(starters))["player_id"].tolist())

    starters = starters[:11]
    bench = ordered[~ordered["player_id"].isin(starters)].head(9)["player_id"].tolist()
    return starters, bench


def lineup_strength(rates: pd.DataFrame, player_ids: list[str],
                    *, weights: dict[str, float] | None = None) -> dict[str, float]:
    """Attack and defence output of a specific eleven.

    Contributions are shrunk toward the squad mean by minutes for the same
    reason as everywhere else: a per-90 from a few hundred minutes is mostly
    noise, and an unshrunk fringe player can outrank the starter he replaces.
    """
    if rates.empty or not player_ids:
        return {"attack": 0.0, "defence": 0.0, "players": 0}

    squad = rates[rates["player_id"].isin(player_ids)].copy()
    if squad.empty:
        return {"attack": 0.0, "defence": 0.0, "players": 0}

    minutes = squad["minutes"].astype(float)
    shrink = minutes / (minutes + CONTRIBUTION_PRIOR_MINUTES)

    totals = {}
    for label, columns in (("attack", ATTACK_COLUMNS), ("defence", DEFENCE_COLUMNS)):
        raw = squad.apply(
            lambda r: float(sum(float(r.get(c) or 0.0) for c in columns)), axis=1)
        mean = float(np.average(raw, weights=minutes)) if minutes.sum() > 0 else 0.0
        shrunk = shrink * raw + (1.0 - shrink) * mean
        if weights:
            player_weights = squad["player_id"].map(lambda p: weights.get(p, 1.0))
            shrunk = shrunk * player_weights
        totals[label] = float(shrunk.sum())

    totals["players"] = int(len(squad))
    return totals


def lineup_strength_shift(store, team_id: str, as_of: datetime,
                          rates: pd.DataFrame, lineup: PredictedLineup,
                          *, baseline_matches: int = 10,
                          max_shift: float = 0.35) -> dict[str, float]:
    """How far this XI departs from the team's typical XI, on the log scale.

    The fitted Dixon-Coles parameters already encode a team's usual output, so
    what matters is the deviation: this eleven against the elevens that produced
    the fit. Comparing against an absolute scale would double-count the team's
    quality and shift every fixture.
    """
    empty = {"attack_shift": 0.0, "defence_shift": 0.0,
             "attack_ratio": 1.0, "defence_ratio": 1.0, "baseline_matches": 0}
    if rates.empty or not lineup.starters:
        return empty

    recent = store.lineup_as_of(as_of)
    if recent.empty:
        return empty

    recent = recent[(recent["team_id"] == team_id) & (recent["is_starter"])]
    if recent.empty:
        return empty

    # The most recent completed line-ups define "typical" for this side.
    #
    # `match_id` is a tiebreaker, not decoration: every fixture on a matchday
    # shares a kickoff, so `known_at` alone leaves large groups of ties and the
    # baseline set is then decided by whatever row order the database happens to
    # return. That made the shift -- and so every price built on it -- differ
    # between runs on identical data, which is fatal for a backtest.
    match_order = (recent.sort_values(["known_at", "match_id"], ascending=[False, True])
                   ["match_id"].drop_duplicates().head(baseline_matches).tolist())
    if not match_order:
        return empty

    baselines = [
        lineup_strength(rates, recent[recent["match_id"] == m]["player_id"].tolist())
        for m in match_order
    ]
    baselines = [b for b in baselines if b["players"] >= 8]
    if not baselines:
        return empty

    predicted = lineup_strength(rates, lineup.starters)
    result = {"baseline_matches": len(baselines)}

    for label in ("attack", "defence"):
        typical = float(np.mean([b[label] for b in baselines]))
        if typical <= 0 or predicted[label] <= 0:
            result[f"{label}_shift"] = 0.0
            result[f"{label}_ratio"] = 1.0
            continue
        ratio = predicted[label] / typical
        shift = float(np.clip(np.log(ratio), -max_shift, max_shift))
        result[f"{label}_shift"] = shift
        result[f"{label}_ratio"] = float(np.exp(shift))

    return result
