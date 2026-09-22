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
    lineups: dict[str, object] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    # Set only for a played match shown retrospectively (the previous
    # matchday). None for anything still in the future -- there is nothing to
    # compare a prediction against yet, and `predicted_correct` below reflects
    # that by staying None too rather than guessing.
    actual_home_goals: float | None = None
    actual_away_goals: float | None = None
    actual_outcome: str | None = None

    @property
    def headline(self) -> str:
        best = max(self.probabilities, key=self.probabilities.get)
        label = {"H": self.home_team, "D": "draw", "A": self.away_team}[best]
        return f"{label} {self.probabilities[best]:.0%}"

    @property
    def predicted_outcome(self) -> str:
        """The model's most likely result, as an outcome letter."""
        return max(self.probabilities, key=self.probabilities.get)

    @property
    def predicted_correct(self) -> bool | None:
        """Whether the model's top pick matches what happened.

        None when there is no actual result to check against -- a future
        fixture is neither right nor wrong yet.
        """
        if self.actual_outcome is None:
            return None
        return self.predicted_outcome == self.actual_outcome


@dataclass
class MatchdayBrief:
    as_of: datetime
    window_days: int
    matches: list[MatchRecommendation] = field(default_factory=list)
    prop_picks: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list[str] = field(default_factory=list)
    narrative: str | None = None
    # Free-text description of what "the coming fixtures" meant here -- "the
    # next matchday" or "the next 8 days" -- so display code does not need to
    # reverse-engineer it from window_days. Falls back to the day-window
    # phrasing when unset, which keeps every existing caller's text unchanged.
    label: str | None = None

    @property
    def scope_label(self) -> str:
        return self.label or f"in the next {self.window_days} days"

    @property
    def value_bet_count(self) -> int:
        return sum(len(m.value_bets) for m in self.matches)

    def to_text(self) -> str:
        """The brief as plain text, with no model in the loop."""
        lines = [
            f"MATCHDAY BRIEF - {self.as_of:%Y-%m-%d %H:%M} UTC",
            f"{len(self.matches)} fixtures {self.scope_label}",
            "=" * 72,
        ]

        for match in self.matches:
            lines.append("")
            lines.append(f"{match.home_team} vs {match.away_team}  "
                         f"({match.kickoff:%a %d %b %H:%M})")
            if match.actual_outcome is not None:
                verdict = "correct" if match.predicted_correct else "missed"
                lines.append(f"  full time       {match.actual_home_goals:.0f} - "
                             f"{match.actual_away_goals:.0f}  ({verdict})")
            lines.append(f"  expected goals  {match.expected_home_goals:.2f} - "
                         f"{match.expected_away_goals:.2f}")
            lines.append("  model           " + "  ".join(
                f"{o} {match.probabilities[o]:.1%} (fair {match.fair_odds[o]:.2f})"
                for o in OUTCOMES))

            if match.market_probabilities:
                lines.append("  market          " + "  ".join(
                    f"{o} {match.market_probabilities[o]:.1%}" for o in OUTCOMES))

            for side in ("home", "away"):
                lineup = match.lineups.get(side)
                if lineup is not None and lineup.starters:
                    effect = match.absence_effect.get(side, 1.0)
                    source = "CONFIRMED" if lineup.is_confirmed else f"predicted {lineup.confidence:.0%}"
                    lines.append(f"  XI ({side:<4})      {lineup.formation} [{source}]  "
                                 f"attack x{effect:.3f}")
                    for note in lineup.notes:
                        lines.append(f"                  {note}")
                players = match.absences.get(side, [])
                if players:
                    lines.append(f"  absent ({side:<4})  {', '.join(players)}")

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


def no_fixtures_message(store, as_of: datetime, days: int, league: str) -> str:
    """Tell a genuine schedule gap from a real data problem.

    An empty lookahead window is not evidence that ingestion is broken: the
    Bundesliga does not play every week, and a fixed window regularly lands
    between matchdays (international breaks, the winter pause). Telling
    someone to re-run an ingest that already succeeded, when the season is
    fully loaded and simply not playing this week, sends them chasing a
    problem that does not exist.
    """
    upcoming = store.next_fixture_after(as_of, league=league)
    if upcoming is None:
        return (f"no fixtures found for {league} at all - "
                "run: bet ingest --source openligadb")

    gap_days = (pd.Timestamp(upcoming["kickoff_utc"]) - pd.Timestamp(as_of)).days
    return (f"no fixtures in the next {days} days, but the season is loaded - "
            f"the next {league} match is {upcoming['home_team_id']} vs "
            f"{upcoming['away_team_id']} on {pd.Timestamp(upcoming['kickoff_utc']):%d %b} "
            f"({gap_days} day(s) away)")


def build_brief(store, as_of: datetime | None = None, *, days: int = 8,
                league: str = "bundesliga", xi: float = 0.0018,
                use_availability: bool = True, prop_stat: str = "shots",
                max_props: int = 12, tax_mode: TaxMode | str = TaxMode.STAKE_DEDUCTED,
                tax_rate: float = GERMAN_STAKE_TAX, min_edge: float = 0.02,
                devig_method: DevigMethod | str = DevigMethod.SHIN) -> MatchdayBrief:
    """Compile everything the system knows about the coming fixtures.

    The day-window entry point used by `bet predict`/`bet brief` and the CLI's
    `--days`. The dashboard uses `next_matchday_brief`/`previous_matchday_brief`
    instead, which share the same core (`build_matchday_brief`) but pick their
    fixtures by matchday rather than by a fixed window.
    """
    as_of = as_of or datetime.utcnow()
    fixtures = store.fixtures_between(as_of, as_of + timedelta(days=days), league=league)
    if fixtures.empty:
        brief = MatchdayBrief(as_of=as_of, window_days=days)
        brief.warnings.append(no_fixtures_message(store, as_of, days, league))
        return brief

    return build_matchday_brief(
        store, fixtures, as_of, xi=xi, use_availability=use_availability,
        prop_stat=prop_stat, max_props=max_props, tax_mode=tax_mode,
        tax_rate=tax_rate, min_edge=min_edge, devig_method=devig_method,
        window_days=days)


def build_matchday_brief(store, fixtures: pd.DataFrame, as_of: datetime, *,
                         xi: float = 0.0018, use_availability: bool = True,
                         prop_stat: str = "shots", max_props: int = 12,
                         tax_mode: TaxMode | str = TaxMode.STAKE_DEDUCTED,
                         tax_rate: float = GERMAN_STAKE_TAX, min_edge: float = 0.02,
                         devig_method: DevigMethod | str = DevigMethod.SHIN,
                         window_days: int = 0, label: str | None = None) -> MatchdayBrief:
    """Recommendations for an explicit set of fixtures, fit as of `as_of`.

    The core shared by every brief this project builds: `build_brief` (a day
    window, fit on everything known right now) and the matchday-scoped ones in
    this module (fit on everything known right now for the next matchday, or
    on only what was known before kickoff for the previous one). All three are
    "what does the model say about this exact list of matches, using only
    what was knowable by `as_of`" -- a question that does not care whether
    `as_of` is in the past or fixtures were chosen by a window or a cluster.

    If `fixtures` carries real `home_goals`/`away_goals`/`outcome` values --
    which `fixtures_between` attaches for anything already played, regardless
    of `as_of` -- each `MatchRecommendation` is stamped with the actual result
    alongside the prediction. A future fixture has none of those, so nothing
    is stamped and `predicted_correct` stays `None`; this needs no separate
    flag to tell the two cases apart.
    """
    brief = MatchdayBrief(as_of=as_of, window_days=window_days, label=label)
    if fixtures.empty:
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

        if pd.notna(getattr(fixture, "outcome", None)):
            recommendation.actual_home_goals = float(fixture.home_goals)
            recommendation.actual_away_goals = float(fixture.away_goals)
            recommendation.actual_outcome = fixture.outcome

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


def next_matchday_brief(store, as_of: datetime | None = None, *,
                        league: str = "bundesliga", **kwargs) -> MatchdayBrief:
    """The next matchday, fit on everything known right now.

    Unlike `build_brief`, this is not a fixed lookahead: it is exactly the
    fixtures in the next cluster of matches, however many days out that turns
    out to be, so it does not go blank during an international break the way
    a fixed window would.
    """
    from bet.matchdays import next_matchday

    as_of = as_of or datetime.utcnow()
    fixtures = next_matchday(store, as_of, league=league)
    if fixtures.empty:
        brief = MatchdayBrief(as_of=as_of, window_days=0, label="in the next matchday")
        brief.warnings.append(no_fixtures_message(store, as_of, 0, league))
        return brief

    return build_matchday_brief(store, fixtures, as_of, label="in the next matchday", **kwargs)


def previous_matchday_brief(store, as_of: datetime | None = None, *,
                            league: str = "bundesliga",
                            lead_seconds: int | None = None,
                            search_days: int | None = None,
                            **kwargs) -> MatchdayBrief:
    """The most recently completed matchday, predicted and then checked.

    Fit as of just before it kicked off -- `earliest kickoff - lead time`, the
    same convention `bet.evaluation.backtest.walk_forward` uses -- so this is
    a genuine retrodiction: what the model would actually have said, using
    only what was known at the time, not what it says with the benefit of
    hindsight. The actual scoreline is attached afterwards for comparison,
    never fed to the model itself.
    """
    from bet.config import SETTINGS
    from bet.matchdays import previous_matchday

    as_of = as_of or datetime.utcnow()
    search_kwargs = {} if search_days is None else {"search_days": search_days}
    fixtures = previous_matchday(store, as_of, league=league, **search_kwargs)
    if fixtures.empty:
        brief = MatchdayBrief(as_of=as_of, window_days=0, label="the previous matchday")
        brief.warnings.append("no completed matchday found yet")
        return brief

    lead = timedelta(seconds=lead_seconds if lead_seconds is not None
                     else SETTINGS.prediction_lead_seconds)
    fit_as_of = pd.Timestamp(fixtures["kickoff_utc"].min()).to_pydatetime() - lead

    return build_matchday_brief(store, fixtures, fit_as_of,
                                label="in the previous matchday", **kwargs)


def _recommend_match(store, model: DixonColesModel, fixture, as_of: datetime,
                     odds: pd.DataFrame, *, tax_mode, tax_rate: float,
                     min_edge: float, devig_method, use_availability: bool) -> MatchRecommendation:
    lam, mu = model._rates(fixture.home_team_id, fixture.away_team_id)

    absences: dict[str, list[str]] = {}
    effects: dict[str, float] = {}
    lineups: dict[str, object] = {}
    if use_availability:
        lam, mu = model._apply_availability(
            store, as_of, fixture.home_team_id, fixture.away_team_id, lam, mu,
            match_id=fixture.match_id)
        from bet.availability import absences_from_store
        for side, team_id in (("home", fixture.home_team_id), ("away", fixture.away_team_id)):
            lineup, shift = model._lineup_for(store, as_of, team_id, fixture.match_id)
            lineups[side] = lineup
            absences[side] = sorted(absences_from_store(store, as_of, team_id))
            effects[side] = shift["attack_ratio"]

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
        lineups=lineups,
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
