"""Kickbase advisor and client.

The client is a guess at an unverified, unreachable API (see
`bet.kickbase.client`'s module docstring) so its tests mock `requests`
entirely and only check the client's own behaviour: credential handling,
single-attempt failure, and response-shape tolerance. The advisor tests
never touch the network at all -- they build `PlayerSnapshot`s directly and
check the recommendation logic, which is the part meant to survive a wrong
guess about Kickbase's JSON shape.
"""

import pytest

from bet.kickbase.advisor import (
    PlayerSnapshot,
    consistency,
    most_consistent,
    parse_player,
    rank_buy_candidates,
    rank_sell_candidates,
    reliability_score,
)
from bet.kickbase.client import (
    KickbaseAPIError,
    KickbaseAuthError,
    KickbaseClient,
)


def _player(name, points, market_value=1_000_000, status="FIT", value_trend=0.0,
            position="MID") -> PlayerSnapshot:
    return PlayerSnapshot(
        player_id=name, name=name, team="FC Test", position=position,
        market_value=market_value, value_trend=value_trend, status=status,
        recent_points=points,
    )


# --------------------------------------------------------------- consistency


def test_consistency_is_none_with_too_little_history():
    assert consistency([5.0, 6.0]) is None


def test_consistency_is_high_for_a_steady_scorer():
    steady = consistency([6.0, 6.0, 6.0, 6.0])
    erratic = consistency([15.0, -3.0, 15.0, -3.0])
    assert steady == pytest.approx(1.0)
    assert erratic < steady


def test_consistency_is_none_for_a_player_who_never_scores():
    assert consistency([0.0, 0.0, 0.0, 0.0]) is None


def test_reliability_score_rewards_steady_over_erratic_at_the_same_average():
    steady = reliability_score(_player("Steady", [6.0, 6.0, 6.0, 6.0]))
    erratic = reliability_score(_player("Erratic", [15.0, -3.0, 15.0, -3.0]))
    assert steady > erratic


def test_reliability_score_is_none_without_enough_history():
    assert reliability_score(_player("Rookie", [5.0, 4.0])) is None


# --------------------------------------------------------------- parse_player


def test_parse_player_reads_plausible_field_names():
    raw = {
        "id": 42, "name": "Harry Kane", "teamName": "Bayern", "pos": 4,
        "marketValue": 50_000_000, "marketValueTrend": -2.0, "status": 0,
        "performance": [{"points": 120}, {"points": 95}, {"p": 140}],
    }
    player = parse_player(raw)
    assert player.player_id == "42"
    assert player.name == "Harry Kane"
    assert player.team == "Bayern"
    assert player.position == "FWD"
    assert player.market_value == 50_000_000
    assert player.value_trend == -2.0
    assert player.status == "FIT"
    assert player.recent_points == [120.0, 95.0, 140.0]


def test_parse_player_degrades_gracefully_on_an_empty_payload():
    player = parse_player({})
    assert player.name == "unknown"
    assert player.position == ""
    assert player.market_value is None
    assert player.recent_points == []


def test_parse_player_maps_injured_status():
    player = parse_player({"status": 1})
    assert player.status == "INJURED"


def test_parse_player_skips_unparseable_point_entries():
    raw = {"performance": [{"points": "not-a-number"}, {"points": 10}, None]}
    player = parse_player(raw)
    assert player.recent_points == [10.0]


# --------------------------------------------------------------- rank_buy_candidates


def test_rank_buy_candidates_prefers_better_value_per_cost():
    cheap_good = _player("Bargain", [10.0, 10.0, 10.0], market_value=1_000_000)
    expensive_mediocre = _player("Overpriced", [10.0, 10.0, 10.0], market_value=10_000_000)
    result = rank_buy_candidates([cheap_good, expensive_mediocre], budget=20_000_000)
    assert [c["player"].name for c in result] == ["Bargain", "Overpriced"]


def test_rank_buy_candidates_excludes_players_over_budget():
    affordable = _player("Affordable", [8.0, 8.0, 8.0], market_value=1_000_000)
    too_expensive = _player("TooRich", [8.0, 8.0, 8.0], market_value=5_000_000)
    result = rank_buy_candidates([affordable, too_expensive], budget=2_000_000)
    assert [c["player"].name for c in result] == ["Affordable"]


def test_rank_buy_candidates_excludes_injured_players():
    injured = _player("Hurt", [10.0, 10.0, 10.0], status="INJURED")
    result = rank_buy_candidates([injured], budget=10_000_000)
    assert result == []


def test_rank_buy_candidates_excludes_players_without_points_history():
    no_history = _player("Unknown", [])
    result = rank_buy_candidates([no_history], budget=10_000_000)
    assert result == []


def test_rank_buy_candidates_excludes_players_with_no_market_value():
    unpriced = PlayerSnapshot(player_id="x", name="Unpriced", recent_points=[5.0, 5.0, 5.0])
    result = rank_buy_candidates([unpriced], budget=10_000_000)
    assert result == []


def test_rank_buy_candidates_respects_top_n():
    players = [_player(f"P{i}", [10.0, 10.0, 10.0], market_value=1_000_000) for i in range(5)]
    result = rank_buy_candidates(players, budget=10_000_000, top_n=2)
    assert len(result) == 2


# --------------------------------------------------------------- rank_sell_candidates


def test_rank_sell_candidates_flags_an_injured_player():
    injured = _player("Hurt", [10.0, 10.0, 10.0], status="INJURED")
    result = rank_sell_candidates([injured])
    assert len(result) == 1
    assert "injured" in result[0]["reasons"]


def test_rank_sell_candidates_flags_falling_value():
    falling = _player("Falling", [10.0, 10.0, 10.0], value_trend=-500_000)
    result = rank_sell_candidates([falling])
    assert "value falling" in result[0]["reasons"]


def test_rank_sell_candidates_flags_inconsistent_return():
    erratic = _player("Erratic", [20.0, -5.0, 20.0, -5.0])
    result = rank_sell_candidates([erratic])
    assert any("inconsistent" in r for r in result[0]["reasons"])


def test_rank_sell_candidates_ignores_a_fine_fit_steady_player():
    good = _player("Good", [10.0, 10.0, 10.0], market_value=100, value_trend=1.0)
    result = rank_sell_candidates([good])
    assert result == []


# --------------------------------------------------------------- most_consistent


def test_most_consistent_ranks_steadiest_first():
    steady = _player("Steady", [6.0, 6.0, 6.0])
    erratic = _player("Erratic", [15.0, -3.0, 15.0])
    result = most_consistent([steady, erratic])
    assert [c["player"].name for c in result] == ["Steady", "Erratic"]


def test_most_consistent_skips_players_without_enough_history():
    result = most_consistent([_player("Rookie", [5.0])])
    assert result == []


# --------------------------------------------------------------- KickbaseClient


def test_client_requires_credentials(monkeypatch):
    monkeypatch.delenv("KICKBASE_EMAIL", raising=False)
    monkeypatch.delenv("KICKBASE_PASSWORD", raising=False)
    with pytest.raises(KickbaseAuthError):
        KickbaseClient()


def test_client_reads_credentials_from_env(monkeypatch):
    monkeypatch.setenv("KICKBASE_EMAIL", "me@example.com")
    monkeypatch.setenv("KICKBASE_PASSWORD", "hunter2")
    client = KickbaseClient()
    assert client.email == "me@example.com"
    assert client.password == "hunter2"


def test_client_prefers_explicit_credentials_over_env(monkeypatch):
    monkeypatch.setenv("KICKBASE_EMAIL", "env@example.com")
    monkeypatch.setenv("KICKBASE_PASSWORD", "env-pass")
    client = KickbaseClient(email="explicit@example.com", password="explicit-pass")
    assert client.email == "explicit@example.com"
    assert client.password == "explicit-pass"


class _FakeResponse:
    def __init__(self, status_code=200, json_body=None, text=""):
        self.status_code = status_code
        self._json_body = json_body
        self.text = text if text else (str(json_body) if json_body is not None else "")

    def json(self):
        if self._json_body is None:
            raise ValueError("no json")
        return self._json_body


class _FakeSession:
    def __init__(self, response):
        self.headers = {}
        self._response = response
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append(("POST", url, json))
        return self._response

    def get(self, url, timeout=None):
        self.calls.append(("GET", url, None))
        return self._response


def _client(response) -> KickbaseClient:
    return KickbaseClient(email="a@b.com", password="secret", session=_FakeSession(response))


def test_login_sets_token_and_auth_header_on_success():
    response = _FakeResponse(200, {"token": "abc123", "user": {"id": "u1"}})
    client = _client(response)
    payload = client.login()
    assert payload["token"] == "abc123"
    assert client.token == "abc123"
    assert client.user_id == "u1"
    assert client.session.headers["Authorization"] == "Bearer abc123"


def test_login_accepts_alternate_token_field_names():
    client = _client(_FakeResponse(200, {"accessToken": "xyz"}))
    client.login()
    assert client.token == "xyz"


def test_login_raises_loudly_on_non_200():
    client = _client(_FakeResponse(403, text="forbidden"))
    with pytest.raises(KickbaseAuthError):
        client.login()


def test_login_raises_when_no_token_field_is_recognised():
    client = _client(_FakeResponse(200, {"unexpected": "shape"}))
    with pytest.raises(KickbaseAuthError):
        client.login()


def test_login_raises_on_non_json_response():
    client = _client(_FakeResponse(200, json_body=None, text="<html>not json</html>"))
    with pytest.raises(KickbaseAuthError):
        client.login()


def test_get_requires_login_first():
    client = _client(_FakeResponse(200, {}))
    with pytest.raises(KickbaseAuthError):
        client.leagues()


def test_squad_unwraps_known_field_names():
    client = _client(_FakeResponse(200, {"token": "t"}))
    client.login()
    client.session._response = _FakeResponse(200, {"players": [{"id": 1}]})
    assert client.squad("1") == [{"id": 1}]


def test_api_call_raises_with_body_on_failure():
    client = _client(_FakeResponse(200, {"token": "t"}))
    client.login()
    client.session._response = _FakeResponse(500, text="server error")
    with pytest.raises(KickbaseAPIError) as exc_info:
        client.market("1")
    assert exc_info.value.status_code == 500
