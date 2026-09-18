"""The T-60 workflow: confirmed line-ups, then re-price.

Everything else this system knows is public days ahead and long since priced.
The confirmed eleven, published around an hour before kickoff, is the one input
that arrives while a bet can still be placed -- and the one that collapses the
largest source of uncertainty in the model, since a predicted XI is a guess and
a confirmed XI is not.

The loop is deliberately small:

    1.  Find fixtures entering the window.
    2.  Fetch and store the confirmed line-up.
    3.  Re-price with the eleven that will actually play.
    4.  Diff against the pre-line-up price and report what moved.

Step 4 is the part worth having. A price that barely moves on team news means
the model had already priced the XI correctly and there is nothing to act on. A
large move means the model was pricing a striker who is on the bench, and the
number to trust is the new one.

Two things this does not do. It does not place bets. And it does not assume you
will beat the market to the news: books move within seconds of an XI dropping
and suspend markets while they do. The realistic value is making sure your own
number is right, not racing a bookmaker to a wire.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from bet.config import OUTCOMES
from bet.lineups import predict_lineup
from bet.models.dixon_coles import DixonColesModel, match_probabilities


@dataclass
class PriceMove:
    match_id: str
    home_team: str
    away_team: str
    kickoff: datetime
    before: dict[str, float]
    after: dict[str, float]
    lineup_confirmed: dict[str, bool] = field(default_factory=dict)
    formations: dict[str, str] = field(default_factory=dict)
    missing_from_xi: dict[str, list[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def largest_move(self) -> float:
        return max(abs(self.after[o] - self.before[o]) for o in OUTCOMES)

    @property
    def is_material(self) -> bool:
        """Whether the news actually changed the forecast.

        Two points is the threshold: below that the model had already priced the
        eleven correctly and the update is noise, not information.
        """
        return self.largest_move >= 0.02

    def describe(self) -> str:
        arrow = "MOVED" if self.is_material else "steady"
        parts = "  ".join(
            f"{o} {self.before[o]:.1%}->{self.after[o]:.1%}" for o in OUTCOMES)
        return f"{arrow} {self.home_team} vs {self.away_team}: {parts}"


def fixtures_entering_window(store, now: datetime, *, lead_minutes: int = 60,
                             tolerance_minutes: int = 20,
                             league: str = "bundesliga") -> pd.DataFrame:
    """Fixtures whose line-ups should be published about now.

    A window rather than an instant, because polling is periodic and an exact
    T-60 would be missed by every run that does not land on the minute.
    """
    centre = now + timedelta(minutes=lead_minutes)
    return store.fixtures_between(
        centre - timedelta(minutes=tolerance_minutes),
        centre + timedelta(minutes=tolerance_minutes),
        league=league)


def reprice_on_lineup(store, match_id: str, home_team: str, away_team: str,
                      kickoff: datetime, *, now: datetime | None = None,
                      xi: float = 0.0018, lead_minutes: int = 60) -> PriceMove:
    """Price a fixture before and after the line-up, and report the difference.

    'Before' is deliberately not a cached number from an earlier run: it is a
    fresh prediction from the same model with confirmed line-ups switched off,
    so the only thing that differs between the two prices is the team news.
    Anything else -- a refit, newer results, a changed parameter -- would
    contaminate the comparison.
    """
    now = now or datetime.utcnow()

    model = DixonColesModel(xi=xi, use_availability=True).fit(store, now)
    if model.params is None:
        raise RuntimeError("not enough history to price this fixture")

    fixtures = pd.DataFrame([{
        "match_id": match_id, "home_team_id": home_team,
        "away_team_id": away_team, "kickoff_utc": kickoff,
    }])

    before = _price(model, store, fixtures, now, use_confirmed=False)
    after = _price(model, store, fixtures, now, use_confirmed=True)

    move = PriceMove(
        match_id=match_id, home_team=home_team, away_team=away_team,
        kickoff=kickoff,
        before={o: float(p) for o, p in zip(OUTCOMES, before)},
        after={o: float(p) for o, p in zip(OUTCOMES, after)},
    )

    from bet.availability import absences_from_store
    for side, team_id in (("home", home_team), ("away", away_team)):
        absences = absences_from_store(store, now, team_id)
        predicted = predict_lineup(store, team_id, now, match_id=match_id,
                                   use_confirmed=False, absences=absences)
        confirmed = predict_lineup(store, team_id, now, match_id=match_id,
                                   use_confirmed=True, absences=absences)

        move.lineup_confirmed[side] = confirmed.is_confirmed
        move.formations[side] = confirmed.formation

        if confirmed.is_confirmed:
            # Players the model expected to start who are not in the XI. This is
            # the concrete reason a price moved, and the thing worth reading.
            dropped = [p for p in predicted.starters if p not in confirmed.starters]
            move.missing_from_xi[side] = dropped
            if predicted.formation != confirmed.formation:
                move.notes.append(
                    f"{side}: formation differs from predicted "
                    f"({predicted.formation} -> {confirmed.formation})")
        else:
            move.notes.append(f"{side}: no confirmed line-up yet; price is unchanged")

    return move


def _price(model: DixonColesModel, store, fixtures: pd.DataFrame,
           as_of: datetime, *, use_confirmed: bool) -> np.ndarray:
    """One fixture's probabilities, with confirmed line-ups on or off."""
    row = fixtures.iloc[0]
    lam, mu = model._rates(row["home_team_id"], row["away_team_id"])

    from bet.availability import absences_from_store
    from bet.lineups import lineup_strength_shift

    shifts = {}
    for side, team_id in (("home", row["home_team_id"]), ("away", row["away_team_id"])):
        absences = absences_from_store(store, as_of, team_id)
        lineup = predict_lineup(store, team_id, as_of,
                                match_id=row["match_id"] if use_confirmed else None,
                                use_confirmed=use_confirmed, absences=absences)
        shifts[side] = lineup_strength_shift(store, team_id, as_of,
                                             model.player_rates, lineup)

    log_lam = np.log(lam) + shifts["home"]["attack_shift"] - shifts["away"]["defence_shift"]
    log_mu = np.log(mu) + shifts["away"]["attack_shift"] - shifts["home"]["defence_shift"]
    return match_probabilities(
        float(np.exp(np.clip(log_lam, -3.0, 2.5))),
        float(np.exp(np.clip(log_mu, -3.0, 2.5))),
        model.params.rho)


def run_matchday_check(store, now: datetime | None = None, *,
                       lead_minutes: int = 60, tolerance_minutes: int = 20,
                       league: str = "bundesliga",
                       fetch_lineups=None) -> list[PriceMove]:
    """One polling pass: fetch line-ups for fixtures in the window, then re-price.

    `fetch_lineups` is injected rather than imported so the workflow can be
    tested without a network, and so the line-up source can be swapped without
    touching the pricing logic.
    """
    now = now or datetime.utcnow()
    fixtures = fixtures_entering_window(
        store, now, lead_minutes=lead_minutes,
        tolerance_minutes=tolerance_minutes, league=league)

    moves: list[PriceMove] = []
    for fixture in fixtures.itertuples(index=False):
        if fetch_lineups is not None:
            try:
                fetch_lineups(store, fixture)
            except Exception as exc:
                # A failed fetch is reported, never fatal: the other fixtures in
                # the window still need pricing.
                moves.append(PriceMove(
                    match_id=fixture.match_id, home_team=fixture.home_team_id,
                    away_team=fixture.away_team_id,
                    kickoff=pd.Timestamp(fixture.kickoff_utc).to_pydatetime(),
                    before={o: 1 / 3 for o in OUTCOMES},
                    after={o: 1 / 3 for o in OUTCOMES},
                    notes=[f"line-up fetch failed: {exc}"]))
                continue

        try:
            moves.append(reprice_on_lineup(
                store, fixture.match_id, fixture.home_team_id, fixture.away_team_id,
                pd.Timestamp(fixture.kickoff_utc).to_pydatetime(),
                now=now, lead_minutes=lead_minutes))
        except RuntimeError as exc:
            moves.append(PriceMove(
                match_id=fixture.match_id, home_team=fixture.home_team_id,
                away_team=fixture.away_team_id,
                kickoff=pd.Timestamp(fixture.kickoff_utc).to_pydatetime(),
                before={o: 1 / 3 for o in OUTCOMES},
                after={o: 1 / 3 for o in OUTCOMES},
                notes=[str(exc)]))

    return moves


def format_moves(moves: list[PriceMove]) -> str:
    """Render a polling pass for a terminal."""
    if not moves:
        return "no fixtures in the line-up window"

    lines = [f"LINE-UP CHECK - {len(moves)} fixture(s) in window", "=" * 72]
    for move in moves:
        lines.append("")
        lines.append(move.describe())
        lines.append(f"  kickoff {move.kickoff:%a %d %b %H:%M}")
        for side in ("home", "away"):
            if side in move.formations:
                status = "confirmed" if move.lineup_confirmed.get(side) else "predicted"
                lines.append(f"  {side:<5} {move.formations[side]} ({status})")
            dropped = move.missing_from_xi.get(side, [])
            if dropped:
                lines.append(f"        expected to start but benched: {', '.join(dropped)}")
        for note in move.notes:
            lines.append(f"  note: {note}")

    material = [m for m in moves if m.is_material]
    lines.append("")
    lines.append(f"{len(material)} of {len(moves)} fixtures moved materially (>=2 points).")
    lines.append("A steady price means the model had already priced the XI correctly.")
    return "\n".join(lines)
