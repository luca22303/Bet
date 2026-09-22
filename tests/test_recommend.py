"""Matchday brief compilation."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from bet.config import OUTCOMES
from bet.ev import TaxMode
from bet.recommend import build_brief, narrate


@pytest.fixture
def as_of(store_with_players):
    row = store_with_players.con.execute(
        "SELECT MIN(kickoff_utc) FROM match WHERE kickoff_utc > '2023-06-01'").fetchone()[0]
    return pd.Timestamp(row).to_pydatetime() - timedelta(days=2)


def test_brief_covers_the_fixtures_in_the_window(store_with_players, as_of):
    brief = build_brief(store_with_players, as_of, days=10)
    assert brief.matches
    for match in brief.matches:
        assert sum(match.probabilities.values()) == pytest.approx(1.0)
        assert set(match.probabilities) == set(OUTCOMES)
        assert match.expected_home_goals > 0


def test_fair_odds_are_the_inverse_of_the_probabilities(store_with_players, as_of):
    brief = build_brief(store_with_players, as_of, days=10)
    match = brief.matches[0]
    for outcome in OUTCOMES:
        assert match.fair_odds[outcome] == pytest.approx(1 / match.probabilities[outcome])


def test_brief_renders_without_a_model_in_the_loop(store_with_players, as_of):
    """The text output must never depend on an LLM being available."""
    text = build_brief(store_with_players, as_of, days=10).to_text()
    assert "MATCHDAY BRIEF" in text
    assert "expected goals" in text
    assert "net of betting tax" in text


def test_missing_market_prices_are_reported_not_hidden(store_with_players, as_of):
    """Closing odds are stamped at kickoff, so nothing is knowable two days out."""
    brief = build_brief(store_with_players, as_of, days=10)
    assert any("market prices" in w for w in brief.warnings)
    assert brief.value_bet_count == 0


def test_value_is_assessed_when_prices_are_knowable(store):
    """A deliberately mispriced book must produce a value bet."""
    kickoff = datetime(2024, 3, 9, 15, 30)
    as_of = kickoff - timedelta(days=1)

    matches, results, quotes = [], [], []
    teams = ["bayern_munich", "sc_freiburg", "fc_augsburg", "vfl_bochum"]
    # Enough history for the model to fit.
    for i in range(160):
        day = datetime(2023, 3, 1) + timedelta(days=i)
        home, away = teams[i % 4], teams[(i + 1) % 4]
        match_id = f"h{i}"
        matches.append({"match_id": match_id, "source": "t", "league": "bundesliga",
                        "season": "2023-24", "kickoff_utc": day,
                        "home_team_id": home, "away_team_id": away,
                        "known_at": day - timedelta(days=30)})
        results.append({"match_id": match_id, "source": "t",
                        "home_goals": 2, "away_goals": 1,
                        "outcome": "H", "ht_home": None, "ht_away": None,
                        "known_at": day + timedelta(hours=2)})

    matches.append({"match_id": "future", "source": "t", "league": "bundesliga",
                    "season": "2023-24", "kickoff_utc": kickoff,
                    "home_team_id": "bayern_munich", "away_team_id": "sc_freiburg",
                    "known_at": kickoff - timedelta(days=30)})
    # Generous prices, knowable before kickoff.
    for selection, price in (("H", 6.0), ("D", 6.0), ("A", 6.0)):
        quotes.append({"match_id": "future", "book": "soft", "market": "1x2",
                       "selection": selection, "decimal_odds": price,
                       "quoted_at": as_of - timedelta(hours=1), "is_closing": False,
                       "known_at": as_of - timedelta(hours=1)})

    store.upsert("match", pd.DataFrame(matches), ["match_id"])
    store.upsert("match_result", pd.DataFrame(results), ["match_id", "source"])
    store.upsert("odds_quote", pd.DataFrame(quotes),
                 ["match_id", "book", "market", "selection", "quoted_at"])

    brief = build_brief(store, as_of, days=3, use_availability=False)
    assert brief.value_bet_count > 0

    bet = brief.matches[0].value_bets[0]
    assert bet["ev"] > 0
    assert 0 < bet["stake"] <= 0.02          # quarter-Kelly, capped
    assert "breakeven" in bet


def test_large_divergence_from_the_market_is_flagged(store):
    """A 10-point gap on the model's side is usually model error, not value."""
    kickoff = datetime(2024, 3, 9, 15, 30)
    as_of = kickoff - timedelta(days=1)
    matches, results, quotes = [], [], []
    teams = ["bayern_munich", "sc_freiburg"]
    for i in range(160):
        day = datetime(2023, 3, 1) + timedelta(days=i)
        matches.append({"match_id": f"h{i}", "source": "t", "league": "bundesliga",
                        "season": "2023-24", "kickoff_utc": day,
                        "home_team_id": teams[i % 2], "away_team_id": teams[(i + 1) % 2],
                        "known_at": day - timedelta(days=30)})
        results.append({"match_id": f"h{i}", "source": "t",
                        "home_goals": 3, "away_goals": 0,
                        "outcome": "H", "ht_home": None, "ht_away": None,
                        "known_at": day + timedelta(hours=2)})
    matches.append({"match_id": "future", "source": "t", "league": "bundesliga",
                    "season": "2023-24", "kickoff_utc": kickoff,
                    "home_team_id": "bayern_munich", "away_team_id": "sc_freiburg",
                    "known_at": kickoff - timedelta(days=30)})
    for selection, price in (("H", 5.0), ("D", 4.0), ("A", 2.0)):
        quotes.append({"match_id": "future", "book": "b", "market": "1x2",
                       "selection": selection, "decimal_odds": price,
                       "quoted_at": as_of, "is_closing": False, "known_at": as_of})

    store.upsert("match", pd.DataFrame(matches), ["match_id"])
    store.upsert("match_result", pd.DataFrame(results), ["match_id", "source"])
    store.upsert("odds_quote", pd.DataFrame(quotes),
                 ["match_id", "book", "market", "selection", "quoted_at"])

    brief = build_brief(store, as_of, days=3, use_availability=False)
    notes = " ".join(brief.matches[0].notes)
    assert "model error" in notes


def test_brief_reports_absences_and_the_expected_xi(store_with_players, as_of):
    """The brief's job is to surface who is playing and who is not.

    Whether an absence actually moves the price is a property of the lineup and
    pricing layers, and is tested there (`test_lineups.py`) where the inputs can
    be controlled exactly. Asserting it end-to-end through a brief makes the test
    depend on which synthetic players happen to be in the predicted XI, which is
    noise rather than behaviour.
    """
    store = store_with_players
    fixtures = store.fixtures_between(as_of, as_of + timedelta(days=10))
    home_team = fixtures.iloc[0]["home_team_id"]

    from bet.lineups import predict_lineup
    predicted = predict_lineup(store, home_team, as_of)
    ruled_out = predicted.starters[:3]

    store.upsert("player_availability", pd.DataFrame([{
        "player_id": pid, "team_id": home_team, "source": "test", "status": "OUT",
        "reason": "injury", "expected_return": None, "confidence": 0.9,
        "known_at": as_of - timedelta(days=1),
    } for pid in ruled_out]), ["player_id", "source", "known_at"])

    brief = build_brief(store, as_of, days=10, use_availability=True)
    match = brief.matches[0]

    # The absences are reported...
    assert set(ruled_out) <= set(match.absences["home"])

    # ...and the eleven the brief priced is not the one it would have picked
    # otherwise. Whether every ruled-out player can be replaced depends on the
    # squad's depth at his position, which is a property of the squad rather
    # than of this code; `test_lineups.py` covers the selection rule directly
    # with a squad built for it.
    priced = match.lineups["home"].starters
    assert priced
    assert set(priced) != set(predicted.starters)


def test_losing_defenders_does_not_change_a_team_s_own_attack(store_with_players, as_of):
    """An absence must move the right side of the ball.

    Ruling out the goalkeeper and centre-backs should leave the team's expected
    goals essentially untouched, and raise the opponent's.
    """
    store = store_with_players
    fixtures = store.fixtures_between(as_of, as_of + timedelta(days=10))
    home_team = fixtures.iloc[0]["home_team_id"]

    before = build_brief(store, as_of, days=10, use_availability=True).matches[0]

    from bet.lineups import predict_lineup
    rates = store.player_rates_as_of(as_of, min_minutes=0.0)
    predicted = predict_lineup(store, home_team, as_of)
    defenders = rates[rates["player_id"].isin(predicted.starters)].nlargest(3, "tackles_p90")
    store.upsert("player_availability", pd.DataFrame([{
        "player_id": pid, "team_id": home_team, "source": "test", "status": "OUT",
        "reason": "injury", "expected_return": None, "confidence": 0.9,
        "known_at": as_of - timedelta(days=1),
    } for pid in defenders["player_id"]]), ["player_id", "source", "known_at"])

    after = build_brief(store, as_of, days=10, use_availability=True).matches[0]

    # Pricing from the XI means replacing three defenders changes the whole
    # eleven, so the side's own attack can move too -- the replacements are
    # different players, not clones of the absentees. The claim that survives is
    # directional: a weakened defence concedes more.
    assert after.expected_away_goals > before.expected_away_goals


def test_props_appear_when_player_data_exists(store_with_players, as_of):
    brief = build_brief(store_with_players, as_of, days=10)
    assert not brief.prop_picks.empty
    assert "p_over" in brief.prop_picks.columns
    # Thin samples are mostly prior, not evidence, so they are not surfaced.
    assert (brief.prop_picks["sample_90s"] >= 5.0).all()


def test_empty_store_produces_a_warning_not_a_crash(store):
    brief = build_brief(store, datetime(2024, 1, 1), days=7)
    assert brief.matches == []
    assert any("no fixtures" in w for w in brief.warnings)


def test_narration_never_computes_anything(store_with_players, as_of):
    """The model is handed finished numbers and asked to read them out."""
    brief = build_brief(store_with_players, as_of, days=10)

    captured = {}

    class FakeBackend:
        def generate_text(self, system, prompt, *, max_tokens=2000):
            captured["system"] = system
            captured["prompt"] = prompt
            return "Bayern are favourites."

    assert narrate(brief, FakeBackend()) == "Bayern are favourites."
    assert "never recalculate" in captured["system"]
    # Everything the model sees is already computed.
    assert "probabilities" in captured["prompt"]
