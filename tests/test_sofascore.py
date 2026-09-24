"""Sofascore ratings and heatmap-point parsing.

Sofascore is unreachable from the environment this was written in, same as
FBref and kicker.de -- see `bet.ingest.sofascore`'s module docstring. Rather
than several markup strategies (there is no HTML here, just one JSON
response), the defence here is tolerance for several plausible field names
within that one shape, and graceful degradation -- one oddly-shaped player
or point is dropped, not the whole payload.
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from bet.ingest.sofascore import (
    COORDINATE_SCALE,
    SofascoreSource,
    parse_heatmap_points,
    parse_lineup_ratings,
)


# --------------------------------------------------------------- ratings


def test_parse_lineup_ratings_reads_both_sides():
    payload = {
        "home": {"players": [
            {"player": {"name": "Harry Kane", "id": 111}, "position": "F",
             "substitute": False, "statistics": {"rating": 7.8}},
        ]},
        "away": {"players": [
            {"player": {"name": "Marco Reus", "id": 222}, "position": "M",
             "substitute": True, "statistics": {"rating": 6.4}},
        ]},
    }
    result = parse_lineup_ratings(payload)
    assert result["home"] == [{"name": "Harry Kane", "sofascore_id": 111,
                               "rating": 7.8, "position": "F", "is_starter": True}]
    assert result["away"][0]["is_starter"] is False
    assert result["away"][0]["rating"] == 6.4


@pytest.mark.parametrize("entry", [
    {"player": {"name": "Kane", "id": 1}, "rating": 7.1},              # bare, no "statistics"
    {"player": {"name": "Kane", "id": 1}, "statistics": {"sofascoreRating": 7.1}},
    {"name": "Kane", "id": 1, "statistics": {"rating": 7.1}},          # flat, no "player" wrapper
])
def test_parse_lineup_ratings_tries_several_plausible_shapes(entry):
    """The exact response shape has never been seen from this sandbox, so
    several plausible field names are tried rather than one guess."""
    payload = {"home": {"players": [entry]}, "away": {"players": []}}
    result = parse_lineup_ratings(payload)
    assert result["home"][0]["rating"] == pytest.approx(7.1)


def test_parse_lineup_ratings_degrades_gracefully_not_all_or_nothing():
    """One player's odd shape must not lose the rest of the XI."""
    payload = {"home": {"players": [
        {"player": {"name": "Kane", "id": 1}, "statistics": {"rating": 7.5}},
        "not even a dict",
        {"player": {"id": 2}},                       # no name at all
        {"player": {"name": "Musiala", "id": 3}},     # no rating published
    ]}, "away": {"players": []}}
    result = parse_lineup_ratings(payload)
    names = [p["name"] for p in result["home"]]
    assert names == ["Kane", "Musiala"]
    assert result["home"][1]["rating"] is None


def test_parse_lineup_ratings_handles_a_missing_or_malformed_payload():
    assert parse_lineup_ratings({}) == {"home": [], "away": []}
    assert parse_lineup_ratings({"home": "not a dict"}) == {"home": [], "away": []}
    assert parse_lineup_ratings(None) == {"home": [], "away": []}


# ------------------------------------------------------------- heatmaps


def test_parse_heatmap_points_converts_the_0_100_scale_to_0_1():
    payload = {"points": [{"x": 75.0, "y": 40.0}, {"x": 10, "y": 90}]}
    points = parse_heatmap_points(payload)
    assert points == [(0.75, 0.4), (0.1, 0.9)]


def test_parse_heatmap_points_accepts_the_heatmap_key_too():
    payload = {"heatmap": [{"x": 50, "y": 50}]}
    assert parse_heatmap_points(payload) == [(0.5, 0.5)]


def test_parse_heatmap_points_drops_points_missing_a_coordinate():
    payload = {"points": [{"x": 50, "y": 50}, {"x": 50}, {"y": 50}, {}, "garbage"]}
    assert parse_heatmap_points(payload) == [(0.5, 0.5)]


def test_parse_heatmap_points_handles_a_missing_or_malformed_payload():
    assert parse_heatmap_points({}) == []
    assert parse_heatmap_points(None) == []
    assert parse_heatmap_points({"points": "not a list"}) == []


def test_coordinate_scale_is_documented_and_used():
    assert COORDINATE_SCALE == 100.0


# --------------------------------------------------------- row-building


@pytest.fixture
def seeded_store(store):
    """A minimal squad for two teams, so `resolve_within_squad` has
    something real to match against -- the same store shape
    `Store.squad_as_of` reads from `player_match_stat` and `player`.

    The squad has to come from a match that finished *before* the one being
    rated -- `squad_as_of(kickoff, ...)` only sees appearances already
    `known_at <= kickoff`, the same point-in-time gate every other read
    here respects, so seeding it via the match being rated itself would
    make its own kickoff always too early to see it.
    """
    earlier_kickoff = datetime(2024, 8, 10, 15, 30)
    kickoff = datetime(2024, 8, 24, 15, 30)
    store.upsert("match", pd.DataFrame([
        {"match_id": "earlier", "source": "test", "league": "bundesliga", "season": "2024-25",
         "kickoff_utc": earlier_kickoff, "home_team_id": "bayern_munich",
         "away_team_id": "borussia_dortmund", "known_at": earlier_kickoff - timedelta(days=30)},
        {"match_id": "seed", "source": "test", "league": "bundesliga", "season": "2024-25",
         "kickoff_utc": kickoff, "home_team_id": "bayern_munich",
         "away_team_id": "borussia_dortmund", "known_at": kickoff - timedelta(days=30)},
    ]), ["match_id"])
    store.upsert("player", pd.DataFrame([
        {"player_id": "kane_h", "full_name": "Harry Kane", "source": "fbref",
         "source_id": None, "known_at": earlier_kickoff},
        {"player_id": "reus_m", "full_name": "Marco Reus", "source": "fbref",
         "source_id": None, "known_at": earlier_kickoff},
    ]), ["player_id"])
    store.upsert("player_match_stat", pd.DataFrame([
        {"match_id": "earlier", "player_id": "kane_h", "team_id": "bayern_munich",
         "source": "fbref", "known_at": earlier_kickoff + timedelta(hours=2)},
        {"match_id": "earlier", "player_id": "reus_m", "team_id": "borussia_dortmund",
         "source": "fbref", "known_at": earlier_kickoff + timedelta(hours=2)},
    ]), ["match_id", "player_id", "source"])
    return store


def test_rating_rows_resolve_known_players_and_skip_unknown_ones(seeded_store):
    source = SofascoreSource(seeded_store, raw_dir="/tmp", delay=0)
    kickoff = datetime(2024, 8, 24, 15, 30)
    parsed = {
        "home": [{"name": "Harry Kane", "sofascore_id": 111, "rating": 7.8,
                 "position": "F", "is_starter": True}],
        "away": [{"name": "Marco Reus", "sofascore_id": 222, "rating": 6.9,
                 "position": "M", "is_starter": True},
                {"name": "Someone Unheard Of", "sofascore_id": 333, "rating": 5.0,
                 "position": "M", "is_starter": False}],
    }
    from bet.ingest.base import IngestResult
    result = IngestResult(source="sofascore")
    rows, resolved = source._rating_rows_for_match(
        parsed, "m1", "event42", "bayern_munich", "borussia_dortmund", kickoff, result)

    assert {r["player_id"]: r["rating"] for r in rows} == {"kane_h": 7.8, "reus_m": 6.9}
    # The unresolved player never gets into `resolved` either -- there is no
    # known player_id to key its heatmap fetch against. Keyed on
    # (event_id, sofascore_id), not the bare id -- see `ingest`'s docstring.
    assert set(resolved) == {("event42", "111"), ("event42", "222")}
    assert resolved[("event42", "111")]["player_id"] == "kane_h"
    assert any("Someone Unheard Of" in e for e in result.errors)
    # known_at is at or after kickoff, the point-in-time floor the leakage
    # guard checks -- Sofascore gives no earlier timestamp to prefer.
    assert all(r["known_at"] >= kickoff for r in rows)


def test_heatmap_rows_carry_the_players_own_known_at():
    source = SofascoreSource.__new__(SofascoreSource)
    source.name = "sofascore"
    ref = {"match_id": "m1", "player_id": "kane_h", "team_id": "bayern_munich",
           "known_at": datetime(2024, 8, 24, 17, 45)}
    rows = source._heatmap_rows([(0.7, 0.5), (0.3, 0.4)], ref)
    assert len(rows) == 2
    assert [r["sequence"] for r in rows] == [0, 1]
    assert all(r["known_at"] == ref["known_at"] for r in rows)
    assert rows[0]["x"] == 0.7 and rows[0]["y"] == 0.5


def test_ingest_writes_ratings_and_populates_resolved_players_for_the_heatmap_step(seeded_store):
    """End to end through `ingest` (with `.fetch` stubbed -- no network
    layer to mock against for a site nobody here can reach), proving the
    handoff to `ingest_heatmaps` via `resolved_players` actually works."""
    import json

    source = SofascoreSource(seeded_store, raw_dir="/tmp", delay=0)
    payload = {
        "home": {"players": [
            {"player": {"name": "Harry Kane", "id": 111}, "statistics": {"rating": 7.8}},
        ]},
        "away": {"players": [
            {"player": {"name": "Marco Reus", "id": 222}, "statistics": {"rating": 6.9}},
        ]},
    }
    source.fetch = lambda url, **kw: (json.dumps(payload), None)

    kickoff = datetime(2024, 8, 24, 15, 30)
    result = source.ingest(["event42"], ["seed"], ["bayern_munich"],
                           ["borussia_dortmund"], [kickoff])
    assert result.rows_written.get("player_match_rating") == 2
    assert set(source.resolved_players) == {("event42", "111"), ("event42", "222")}

    ratings = seeded_store.player_ratings_as_of(kickoff + timedelta(days=1), "seed")
    assert len(ratings) == 2
    assert set(ratings["source"]) == {"sofascore"}

    heatmap_payload = {"points": [{"x": 70, "y": 50}] * 12}
    source.fetch = lambda url, **kw: (json.dumps(heatmap_payload), None)
    heatmap_result = source.ingest_heatmaps()
    assert heatmap_result.rows_written.get("player_heatmap_point") == 24   # 12 points x 2 players

    points = seeded_store.heatmap_points_as_of(kickoff + timedelta(days=1), "seed",
                                               player_id="kane_h")
    assert len(points) == 12


def test_ingest_heatmaps_without_a_prior_ingest_call_raises(store):
    source = SofascoreSource(store, raw_dir="/tmp", delay=0)
    with pytest.raises(RuntimeError):
        source.ingest_heatmaps()


def test_resolved_players_accumulate_across_several_ingest_calls(seeded_store):
    """A caller batching many matches across several `ingest()` calls must
    still be able to fetch every one of their heatmaps with a single later
    `ingest_heatmaps()` -- a second `ingest()` call must not silently erase
    the first batch's resolved players before their heatmaps were fetched."""
    import json

    source = SofascoreSource(seeded_store, raw_dir="/tmp", delay=0)
    kickoff = datetime(2024, 8, 24, 15, 30)

    kane_payload = {"home": {"players": [
        {"player": {"name": "Harry Kane", "id": 111}, "statistics": {"rating": 7.8}},
    ]}, "away": {"players": []}}
    source.fetch = lambda url, **kw: (json.dumps(kane_payload), None)
    source.ingest(["event42"], ["seed"], ["bayern_munich"], ["borussia_dortmund"], [kickoff])
    assert set(source.resolved_players) == {("event42", "111")}

    reus_payload = {"home": {"players": []}, "away": {"players": [
        {"player": {"name": "Marco Reus", "id": 222}, "statistics": {"rating": 6.9}},
    ]}}
    source.fetch = lambda url, **kw: (json.dumps(reus_payload), None)
    source.ingest(["event43"], ["seed"], ["bayern_munich"], ["borussia_dortmund"], [kickoff])
    assert set(source.resolved_players) == {("event42", "111"), ("event43", "222")}


def test_the_same_player_across_two_matches_keeps_both_refs(seeded_store):
    """The real bug this guards: Harry Kane carries the same Sofascore id
    (111) in every match he plays. Keying `resolved_players` on that bare id
    would let the second match's ref silently replace the first's, so
    `ingest_heatmaps` would only ever fetch his most recent match -- his
    earlier one lost with no error to say why."""
    import json

    source = SofascoreSource(seeded_store, raw_dir="/tmp", delay=0)
    kickoff = datetime(2024, 8, 24, 15, 30)
    kane_payload = {"home": {"players": [
        {"player": {"name": "Harry Kane", "id": 111}, "statistics": {"rating": 7.8}},
    ]}, "away": {"players": []}}
    source.fetch = lambda url, **kw: (json.dumps(kane_payload), None)

    source.ingest(["eventA"], ["seed"], ["bayern_munich"], ["borussia_dortmund"], [kickoff])
    source.ingest(["eventB"], ["seed2"], ["bayern_munich"], ["borussia_dortmund"], [kickoff])

    assert set(source.resolved_players) == {("eventA", "111"), ("eventB", "111")}
    assert source.resolved_players[("eventA", "111")]["match_id"] == "seed"
    assert source.resolved_players[("eventB", "111")]["match_id"] == "seed2"


def test_ingest_tolerates_a_shorter_fixture_list_than_event_ids(seeded_store):
    """A missing entry costs one event, recorded as an error -- not an
    IndexError -- the same tolerance kicker's own `ingest`/`ingest_ticker`
    give a caller that does not have every fixture's details in hand yet."""
    import json

    source = SofascoreSource(seeded_store, raw_dir="/tmp", delay=0)
    payload = {"home": {"players": [
        {"player": {"name": "Harry Kane", "id": 111}, "statistics": {"rating": 7.8}},
    ]}, "away": {"players": []}}
    source.fetch = lambda url, **kw: (json.dumps(payload), None)

    kickoff = datetime(2024, 8, 24, 15, 30)
    # Two events, but fixture-knowledge lists for only the first.
    result = source.ingest(["event42", "event43"], ["seed"],
                           ["bayern_munich"], ["borussia_dortmund"], [kickoff])
    assert result.rows_written.get("player_match_rating") == 1
    assert result.documents_fetched == 1          # the second was never even fetched
    assert any("event43" in e for e in result.errors)
