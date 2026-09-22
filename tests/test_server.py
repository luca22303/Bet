"""The local web server.

Exercised over real HTTP against a real socket on a free port, because the
things worth testing here -- routing, the concurrency guard, what a failed
refresh reports -- only exist at that boundary.
"""

import json
import socket
import threading
import urllib.error
import urllib.request
from datetime import datetime

import pytest

from bet.server import ServerState, make_server, run_refresh, status_payload


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def running(store_with_players, tmp_path):
    """A server bound to a free port, serving a populated store."""
    # The whole schema has to exist in the target, not only the tables being
    # copied: the dashboard reads every fact table, and a missing one produces
    # the error page rather than a dashboard.
    from bet.store import Store

    db = tmp_path / "served.duckdb"
    with Store.open(db) as target:
        target.init_schema()

    store_with_players.con.execute(f"ATTACH '{db}' AS out")
    for table in ("match", "match_result", "odds_quote", "player",
                  "player_match_stat", "lineup"):
        store_with_players.con.execute(
            f"INSERT INTO out.{table} SELECT * FROM {table}")
    store_with_players.con.execute("DETACH out")

    state = ServerState(db_path=db, days=10)
    httpd = make_server(state, port=_free_port())
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        yield base, state
    finally:
        httpd.shutdown()
        httpd.server_close()


def _get(url: str):
    with urllib.request.urlopen(url, timeout=20) as response:
        return response.status, response.read().decode("utf-8")


def _post(url: str):
    request = urllib.request.Request(url, method="POST", data=b"")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_health_endpoint(running):
    base, _ = running
    status, body = _get(f"{base}/healthz")
    assert status == 200
    assert json.loads(body) == {"ok": True}


def test_root_serves_the_dashboard(running):
    base, _ = running
    status, body = _get(f"{base}/")
    assert status == 200
    assert body.startswith("<!DOCTYPE html>")
    assert "Bundesliga model" in body


def test_served_page_carries_the_refresh_control(running):
    """The control only appears when a server is there to answer it."""
    base, _ = running
    _, body = _get(f"{base}/")
    assert 'id="refresh-btn"' in body
    assert "/api/refresh" in body


def test_written_file_has_no_refresh_control(store_with_players):
    """A file opened from disk would post to a server that is not running."""
    from bet.dashboard import build
    page = build(store_with_players, datetime(2024, 1, 5), days=10)
    assert 'id="refresh-btn"' not in page


def test_page_is_not_cached(running):
    """A cached copy would show stale numbers after a refresh."""
    base, _ = running
    with urllib.request.urlopen(f"{base}/", timeout=20) as response:
        assert response.headers["Cache-Control"] == "no-store"


def test_status_reports_provenance(running):
    base, _ = running
    _, body = _get(f"{base}/api/status")
    payload = json.loads(body)
    assert payload["refreshing"] is False
    assert payload["provenance"]["synthetic"] is True
    assert payload["provenance"]["sources"] == ["synthetic"]


def test_unknown_routes_404(running):
    base, _ = running
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(f"{base}/nope")
    assert exc.value.code == 404


def test_a_second_refresh_is_refused_while_one_runs(running):
    """Two clicks must not start two fetches against the same database."""
    base, state = running
    assert state.begin_refresh()          # simulate one in flight
    try:
        status, body = _post(f"{base}/api/refresh")
        assert status == 409
        assert body["started"] is False
        assert "already running" in body["reason"]
    finally:
        state.end_refresh()


def test_refresh_state_round_trips():
    state = ServerState(db_path=None)
    assert state.begin_refresh()
    assert not state.begin_refresh()      # second caller is refused
    state.end_refresh(report="done")
    assert not state.refreshing
    assert state.last_refresh is not None
    assert state.begin_refresh()          # free again


def test_a_refresh_that_fetches_nothing_is_reported_as_a_failure(tmp_path, monkeypatch):
    """`refresh` collects per-source errors rather than raising.

    A run where every source was unreachable therefore returns normally, and
    reporting that as success would tell someone their data is fresh when
    nothing was fetched.
    """
    import bet.live as live
    from bet.live import RefreshReport

    def _all_sources_down(store, **kwargs):
        return RefreshReport(started=datetime.utcnow(), seasons=[2026],
                             rows={}, errors=["football_data: unreachable"])

    monkeypatch.setattr(live, "refresh", _all_sources_down)

    from bet.store import Store
    db = tmp_path / "empty.duckdb"
    with Store.open(db) as store:
        store.init_schema()

    state = ServerState(db_path=db)
    state.begin_refresh()
    run_refresh(state)

    assert not state.refreshing
    assert "nothing was fetched" in state.last_error
    assert "unreachable" in state.last_error


def test_partial_failures_are_reported_alongside_the_rows_written(tmp_path, monkeypatch):
    import bet.live as live
    from bet.live import RefreshReport

    def _partial(store, **kwargs):
        return RefreshReport(started=datetime.utcnow(), seasons=[2026],
                             rows={"match": 306}, errors=["clubelo: 403"])

    monkeypatch.setattr(live, "refresh", _partial)

    from bet.store import Store
    db = tmp_path / "partial.duckdb"
    with Store.open(db) as store:
        store.init_schema()

    state = ServerState(db_path=db)
    state.begin_refresh()
    run_refresh(state)

    assert "306 rows written" in state.last_error
    assert "clubelo" in state.last_error


def test_a_crashing_refresh_leaves_the_server_usable(tmp_path, monkeypatch):
    import bet.live as live

    def _boom(store, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(live, "refresh", _boom)

    state = ServerState(db_path=tmp_path / "x.duckdb")
    state.begin_refresh()
    run_refresh(state)

    assert not state.refreshing           # the lock is released
    assert "disk on fire" in state.last_error
    assert state.begin_refresh()          # and another attempt is allowed


def test_status_survives_a_missing_database(tmp_path):
    payload = status_payload(ServerState(db_path=tmp_path / "does-not-exist.duckdb"))
    assert "provenance" in payload
