"""Matchday briefs: everything the system knows about one round, in one place.

This is the layer the rest of the project exists to feed. It pulls together the
match model, the absence adjustments, player props, and market prices where they
exist, and produces a brief for a human to act on.

The design commitment is that a brief states its own uncertainty. Every
recommendation carries the evidence behind it -- how much data the rate is built
on, how far the model diverges from the market, what the adjustment for absences
actually was -- because a recommendation with a probability and nothing else is
indistinguishable from a guess, and the whole point of the layers underneath is
to be able to tell the difference.

Expected value is always reported net of margin and betting tax. A positive
number before tax is not an edge; showing it as one would undo the honesty the
rest of the project is built on.

The narration step is optional and cosmetic. A local model turns the table into
prose; it is never allowed to compute anything, change a number, or decide what
is recommended. Those come from the statistical layers, and a language model's
job here is presentation only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from bet.availability import absences_from_store, estimate_absence_impact
from bet.config import GERMAN_STAKE_TAX, OUTCOMES
from bet.ev import TaxMode, size_bet
from bet.models.dixon_coles import DixonColesModel
from bet.models.props import PlayerPropModel
from bet.odds.devig import DevigMethod, devig


@dataclass
class MatchRecommendation:
    match_id: str
    kickoff: datetime
    home_team: str
    away_team: str
    expected_home_goals: float
    expected_away_goals: float
    probabilities: dict[str, float]
    fair_odds: dict[str, float]
    market_odds: dict[str, float] = field(default_factory=dict)
    market_probabilities: dict[str, float] = field(default_factory=dict)
    value_bets: list[dict] = field(default_factory=list)
    absences: dict[str, list[str]] = field(default_factory=dict)
    absence_effect: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def headline(self) -> str:
        best = max(self.probabilities, key=self.probabilities.get)
        label = {"H": self.home_team, "D": "draw", "A": self.away_team}[best]
        return f"{label} {self.probabilities[best]:.0%}"


@dataclass
class MatchdayBrief:
    as_of: datetime
    window_days: int
    matches: list[MatchRecommendation] = field(default_factory=list)
    prop_picks: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list[str] = field(default_factory=list)
    narrative: str | None = None

    @property
    def value_bet_count(self) -> int:
        return sum(len(m.value_bets) for m in self.matches)

    def to_text(self) -> str:
        """The brief as plain text, with no model in the loop."""
        lines = [
            f"MATCHDAY BRIEF - {self.as_of:%Y-%m-%d %H:%M} UTC",
            f"{len(self.matches)} fixtures in the next {self.window_days} days",
            "=" * 72,
        ]

        for match in self.matches:
            lines.append("")
            lines.append(f"{match.home_team} vs {match.away_team}  "
                         f"({match.kickoff:%a %d %b %H:%M})")
            lines.append(f"  expected goals  {match.expected_home_goals:.2f} - "
                         f"{match.expected_away_goals:.2f}")
            lines.append("  model           " + "  ".join(
                f"{o} {match.probabilities[o]:.1%} (fair {match.fair_odds[o]:.2f})"
                for o in OUTCOMES))

            if match.market_probabilities:
                lines.append("  market          " + "  ".join(
                    f"{o} {match.market_probabilities[o]:.1%}" for o in OUTCOMES))

            for side, players in match.absences.items():
                if players:
                    effect = match.absence_effect.get(side, 1.0)
                    lines.append(f"  absent ({side:<4})  {', '.join(players)} "
                                 f"[attack x{effect:.3f}]")

            if match.value_bets:
                for bet in match.value_bets:
                    lines.append(f"  VALUE  {bet['selection']} @ {bet['odds']:.2f}  "
                                 f"EV {bet['ev']:+.2%}  stake {bet['stake']:.2%} of bankroll")
            elif match.market_odds:
                lines.append("  no value at current prices")

            for note in match.notes:
                lines.append(f"  note: {note}")

        if not self.prop_picks.empty:
            lines.append("")
            lines.append("-" * 72)
            lines.append("PLAYER PROPS (model prices; no market comparison available)")
            lines.append(self.prop_picks.to_string(index=False))

        if self.warnings:
            lines.append("")
            lines.append("-" * 72)
            for warning in self.warnings:
                lines.append(f"! {warning}")

        lines.append("")
        lines.append("Fair odds and EV are net of betting tax where configured.")
        lines.append("A model edge is not a proven edge: check `bet backtest` first.")
        return "\n".join(lines)


def build_brief(store, as_of: datetime | None = None, *, days: int = 8,
                league: str = "bundesliga", xi: float = 0.0018,
                use_availability: bool = True, prop_stat: str = "shots",
                max_props: int = 12, tax_mode: TaxMode | str = TaxMode.STAKE_DEDUCTED,
                tax_rate: float = GERMAN_STAKE_TAX, min_edge: float = 0.02,
                devig_method: DevigMethod | str = DevigMethod.SHIN) -> MatchdayBrief:
    """Compile everything the system knows about the coming fixtures."""
    as_of = as_of or datetime.utcnow()
    brief = MatchdayBrief(as_of=as_of, window_days=days)

    fixtures = store.fixtures_between(as_of, as_of + timedelta(days=days), league=league)
    if fixtures.empty:
        brief.warnings.append(
            f"no fixtures found in the next {days} days - "
            "run: bet ingest --source openligadb")
        return brief

    match_model = DixonColesModel(xi=xi, use_availability=use_availability).fit(store, as_of)
    if match_model.params is None:
        brief.warnings.append(
            "not enough match history to fit the model - "
            "run: bet ingest --source football_data")
        return brief

    prop_model = PlayerPropModel(stat=prop_stat).fit(store, as_of)
    if prop_model.rates.empty:
        brief.warnings.append(
            "no player data, so props are unavailable - "
            "run: bet ingest --source fbref")

    odds = store.odds_as_of(as_of, fixtures["match_id"].tolist(), market="1x2")
    if odds.empty:
        brief.warnings.append(
            "no market prices knowable at this time, so no value can be assessed; "
            "the model prices below are fair odds only")

    prop_rows = []
    for fixture in fixtures.itertuples(index=False):
        recommendation = _recommend_match(
            store, match_model, fixture, as_of, odds,
            tax_mode=tax_mode, tax_rate=tax_rate, min_edge=min_edge,
            devig_method=devig_method, use_availability=use_availability)
        brief.matches.append(recommendation)

        if not prop_model.rates.empty:
            frame = prop_model.predict_match(
                store, fixture.match_id, fixture.home_team_id, fixture.away_team_id,
                as_of,
                home_expected_goals=recommendation.expected_home_goals,
                away_expected_goals=recommendation.expected_away_goals)
            if not frame.empty:
                prop_rows.append(frame)

    if prop_rows:
        combined = pd.concat(prop_rows, ignore_index=True)
        # Thin samples are mostly prior rather than evidence, so they are not
        # surfaced as picks even when the expected count looks high.
        confident = combined[combined["sample_90s"] >= 5.0]
        source = confident if not confident.empty else combined
        brief.prop_picks = source.nlargest(max_props, "expected").reset_index(drop=True)

    return brief


def _recommend_match(store, model: DixonColesModel, fixture, as_of: datetime,
                     odds: pd.DataFrame, *, tax_mode, tax_rate: float,
                     min_edge: float, devig_method, use_availability: bool) -> MatchRecommendation:
    lam, mu = model._rates(fixture.home_team_id, fixture.away_team_id)

    absences: dict[str, list[str]] = {}
    effects: dict[str, float] = {}
    if use_availability:
        lam, mu = model._apply_availability(
            store, as_of, fixture.home_team_id, fixture.away_team_id, lam, mu)
        for side, team_id in (("home", fixture.home_team_id), ("away", fixture.away_team_id)):
            impact = model._absence_impact(store, as_of, team_id)
            absences[side] = list(impact.absent_players)
            effects[side] = impact.attack_multiplier

    from bet.models.dixon_coles import match_probabilities
    probs = match_probabilities(lam, mu, model.params.rho)
    probabilities = {o: float(p) for o, p in zip(OUTCOMES, probs)}

    recommendation = MatchRecommendation(
        match_id=fixture.match_id,
        kickoff=pd.Timestamp(fixture.kickoff_utc).to_pydatetime(),
        home_team=fixture.home_team_id,
        away_team=fixture.away_team_id,
        expected_home_goals=lam,
        expected_away_goals=mu,
        probabilities=probabilities,
        fair_odds={o: (1.0 / p if p > 0 else float("inf"))
                   for o, p in probabilities.items()},
        absences=absences,
        absence_effect=effects,
    )

    match_odds = odds[odds["match_id"] == fixture.match_id] if not odds.empty else pd.DataFrame()
    if match_odds.empty:
        return recommendation

    # Best available price per selection, which is what a bettor would actually take.
    best = match_odds.groupby("selection")["decimal_odds"].max().to_dict()
    if set(best) != set(OUTCOMES):
        recommendation.notes.append("incomplete market prices; no value assessed")
        return recommendation

    recommendation.market_odds = {o: float(best[o]) for o in OUTCOMES}
    market_probs = devig([best[o] for o in OUTCOMES], method=devig_method)
    recommendation.market_probabilities = {o: float(p) for o, p in zip(OUTCOMES, market_probs)}

    for outcome in OUTCOMES:
        evaluation = size_bet(
            probabilities[outcome], float(best[outcome]),
            tax_mode=tax_mode, tax_rate=tax_rate, min_edge=min_edge)
        if evaluation.is_value:
            divergence = probabilities[outcome] - recommendation.market_probabilities[outcome]
            recommendation.value_bets.append({
                "selection": outcome,
                "odds": float(best[outcome]),
                "model_probability": probabilities[outcome],
                "market_probability": recommendation.market_probabilities[outcome],
                "divergence": divergence,
                "ev": evaluation.expected_value,
                "stake": evaluation.stake_fraction,
                "breakeven": evaluation.breakeven_probability,
            })
            # A large divergence from the market is usually the model being
            # wrong, not the market. Saying so is the point.
            if divergence > 0.10:
                recommendation.notes.append(
                    f"{outcome}: model is {divergence:.1%} above the market - "
                    "large divergences are usually model error, not value")

    return recommendation


NARRATION_SYSTEM = """\
You write short, factual football betting briefs from structured model output.

You are given numbers that have already been computed. Report them; never \
recalculate, never add a number that is not in the input, and never change a \
recommendation.

Be plain and brief. Two or three sentences per fixture. No hype, no filler, no \
"exciting clash" language.

Where the model diverges sharply from the market, say so and note that this \
usually means model error rather than value.

If there are no value bets, say that clearly. A quiet matchday is a normal \
result and should not be dressed up."""


def narrate(brief: MatchdayBrief, backend, *, max_tokens: int = 2000) -> str:
    """Turn a brief into prose with a local model.

    Presentation only. The model is handed finished numbers and asked to read
    them out; it never computes anything and never decides what is recommended.
    Everything it could get wrong is already on the table above it.
    """
    payload = {
        "as_of": brief.as_of.isoformat(),
        "fixtures": [
            {
                "home": m.home_team, "away": m.away_team,
                "kickoff": m.kickoff.isoformat(),
                "expected_goals": [round(m.expected_home_goals, 2),
                                   round(m.expected_away_goals, 2)],
                "probabilities": {k: round(v, 4) for k, v in m.probabilities.items()},
                "market_probabilities": {k: round(v, 4)
                                         for k, v in m.market_probabilities.items()},
                "absences": m.absences,
                "value_bets": m.value_bets,
                "notes": m.notes,
            }
            for m in brief.matches
        ],
        "warnings": brief.warnings,
    }

    import json
    prompt = (
        "Write a matchday brief from this model output.\n\n"
        + json.dumps(payload, indent=2, default=str)
    )
    return backend.generate_text(NARRATION_SYSTEM, prompt, max_tokens=max_tokens)
