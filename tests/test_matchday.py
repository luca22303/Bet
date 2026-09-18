"""The T-60 line-up workflow."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from bet.config import OUTCOMES
from bet.matchday import (
    PriceMove,
    fixtures_entering_window,
    format_moves,
    reprice_on_lineup,
    run_matchday_check,
)


@pytest.fixture
def upcoming(store_with_players):
    """A fixture positioned an hour ahead of 'now'."""
    store = store_with_players
    as_of = datetime(2024, 1, 5)
    row = store.fixtures_between(as_of, as_of + timedelta(days=30)).iloc[0]
    now = pd.Timestamp(row["kickoff_utc"]).to_pydatetime() - timedelta(minutes=60)
    return store, row, now


def test_window_finds_fixtures_about_to_start(upcoming):
    store, row, now = upcoming
    found = fixtures_entering_window(store, now, lead_minutes=60)
    assert row["match_id"] in set(found["match_id"])


def test_window_excludes_distant_fixtures(upcoming):
    """A window, not an instant: polling is periodic."""
    store, row, now = upcoming
    found = fixtures_entering_window(store, now - timedelta(days=2), lead_minutes=60)
    assert row["match_id"] not in set(found["match_id"])


def test_price_is_unchanged_without_a_confirmed_lineup(upcoming):
    store, row, now = upcoming
    move = reprice_on_lineup(store, row["match_id"], row["home_team_id"],
                             row["away_team_id"],
                             pd.Timestamp(row["kickoff_utc"]).to_pydatetime(), now=now)
    assert move.before == move.after
    assert not move.is_material
    assert any("no confirmed line-up" in n for n in move.notes)


def test_a_weakened_confirmed_lineup_moves_the_price(upcoming):
    """The whole point of the T-60 check."""
    store, row, now = upcoming

    rates = store.player_rates_as_of(now, min_minutes=0.0)
    squad = rates[rates["team_id"] == row["home_team_id"]]
    # Field the eleven with the least attacking output the squad can offer.
    weakest = squad.nsmallest(11, "xg_p90")["player_id"].tolist()

    store.upsert("lineup", pd.DataFrame([{
        "match_id": row["match_id"], "player_id": p, "team_id": row["home_team_id"],
        "source": "confirmed_test", "is_starter": True, "is_confirmed": True,
        "shirt_number": None, "formation": None, "known_at": now,
    } for p in weakest]), ["match_id", "player_id", "source", "known_at"])

    move = reprice_on_lineup(store, row["match_id"], row["home_team_id"],
                             row["away_team_id"],
                             pd.Timestamp(row["kickoff_utc"]).to_pydatetime(), now=now)

    assert move.lineup_confirmed["home"]
    assert move.after["H"] < move.before["H"]      # weaker home side, shorter price
    assert sum(move.after.values()) == pytest.approx(1.0)
    assert move.missing_from_xi["home"]            # names who was expected but benched


def test_material_threshold_separates_news_from_noise():
    """A steady price means the model had already priced the XI correctly."""
    steady = PriceMove("m", "a", "b", datetime.utcnow(),
                       {"H": 0.50, "D": 0.25, "A": 0.25},
                       {"H": 0.505, "D": 0.25, "A": 0.245})
    moved = PriceMove("m", "a", "b", datetime.utcnow(),
                      {"H": 0.50, "D": 0.25, "A": 0.25},
                      {"H": 0.42, "D": 0.27, "A": 0.31})
    assert not steady.is_material
    assert moved.is_material
    assert moved.largest_move == pytest.approx(0.08)


def test_polling_pass_covers_every_fixture_in_the_window(upcoming):
    store, row, now = upcoming
    moves = run_matchday_check(store, now, lead_minutes=60)
    assert moves
    assert row["match_id"] in {m.match_id for m in moves}


def test_a_failed_fetch_does_not_stop_the_other_fixtures(upcoming):
    """One broken source must not block the rest of the window."""
    store, row, now = upcoming

    def failing_fetch(store, fixture):
        raise RuntimeError("source unreachable")

    moves = run_matchday_check(store, now, lead_minutes=60, fetch_lineups=failing_fetch)
    assert moves
    assert all(any("fetch failed" in n for n in m.notes) for m in moves)


def test_output_renders(upcoming):
    store, row, now = upcoming
    text = format_moves(run_matchday_check(store, now, lead_minutes=60))
    assert "LINE-UP CHECK" in text
    assert "kickoff" in text


def test_empty_window_is_reported():
    assert "no fixtures" in format_moves([])
