"""A local web server, so the whole thing is one command and a bookmark.

`bet serve` renders the dashboard on request rather than writing a file, and
exposes a refresh button that re-fetches the sources without touching a
terminal. It is the difference between a tool you run and a tool you open.

Built on the standard library. A web framework would be a dependency, a build
step and a version to keep current, in exchange for routing four endpoints.

**It binds to 127.0.0.1 and must stay there.** The refresh endpoint makes
outbound requests and writes to the database, so anything that can reach it can
drive both. On a shared or untrusted network that is a remote-controlled
scraper; bound to loopback it is a local tool. There is no authentication here,
by design, because loopback-only is the assumption the design rests on -- put it
behind a reverse proxy with auth before exposing it anywhere.
"""

from __future__ import annotations

import json
import threading
import traceback
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


@dataclass
class ServerState:
    """What the server knows between requests.

    A refresh runs on a background thread so the request returns immediately and
    the page can poll; the lock keeps two clicks from starting two fetches
    against the same database.
    """

    db_path: Path | None
    days: int = 8
    league: str = "bundesliga"
    sources: tuple[str, ...] = ("football_data", "openligadb", "clubelo")
    refreshing: bool = False
    last_refresh: datetime | None = None
    last_report: str = ""
    last_error: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock)

    def begin_refresh(self) -> bool:
        """Claim the right to refresh; False if one is already running."""
        with self.lock:
            if self.refreshing:
                return False
            self.refreshing = True
            self.last_error = ""
            return True

    def end_refresh(self, report: str = "", error: str = "") -> None:
        with self.lock:
            self.refreshing = False
            self.last_refresh = datetime.utcnow()
            if report:
                self.last_report = report
            if error:
                self.last_error = error


def _open_store(state: ServerState, *, read_only: bool = False):
    from bet.store import Store
    return Store.open(state.db_path, read_only=read_only)


def render_page(state: ServerState) -> str:
    """Build the dashboard from the current contents of the store."""
    from bet.dashboard import build

    with _open_store(state, read_only=True) as store:
        return build(store, datetime.utcnow(), days=state.days,
                     league=state.league, served=True)


def run_refresh(state: ServerState) -> None:
    """Fetch from the sources. Runs on a worker thread."""
    from bet.live import refresh

    try:
        with _open_store(state) as store:
            report = refresh(store, league=state.league, sources=state.sources)

        # `refresh` collects per-source failures rather than raising, so a run
        # where every source was unreachable returns normally. Reporting that as
        # success would tell someone their data is fresh when nothing was
        # fetched -- the exact silent failure this project exists to avoid.
        rows = sum(report.rows.values())
        if rows == 0 and report.errors:
            state.end_refresh(
                report=report.summary(),
                error=f"nothing was fetched — {report.errors[0][:160]}")
        elif report.errors:
            state.end_refresh(
                report=report.summary(),
                error=f"{rows} rows written, but {len(report.errors)} source "
                      f"error(s) — {report.errors[0][:120]}")
        else:
            state.end_refresh(report=report.summary())
    except Exception as exc:
        # A failed refresh must leave the server running and say what happened;
        # the page keeps serving whatever the store already held.
        state.end_refresh(error=f"{type(exc).__name__}: {exc}",
                          report=traceback.format_exc(limit=3))


def status_payload(state: ServerState) -> dict:
    from bet.dashboard import data_provenance

    payload = {
        "refreshing": state.refreshing,
        "last_refresh": state.last_refresh.isoformat() if state.last_refresh else None,
        "last_report": state.last_report,
        "last_error": state.last_error,
    }
    try:
        with _open_store(state, read_only=True) as store:
            provenance = data_provenance(store, datetime.utcnow())
        payload["provenance"] = {
            "sources": provenance["sources"],
            "synthetic": provenance["synthetic"],
            "stale": provenance["stale"],
            "empty": provenance["empty"],
            "worst_age_days": provenance["worst_age_days"],
        }
    except Exception as exc:
        payload["provenance"] = {"error": str(exc)}
    return payload


class Handler(BaseHTTPRequestHandler):
    """Four routes: the page, a refresh trigger, status, and a health check."""

    state: ServerState = None            # set by make_server
    server_version = "RueBet"
    sys_version = ""                      # no Python version in the banner

    def log_message(self, fmt, *args):
        # The default logger writes every asset request to stderr, which buries
        # the one line that matters: the URL to open.
        pass

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # The page is generated per request; a cached copy would show stale
        # numbers after a refresh, which is the one thing it must not do.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, status: int = 200) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/") or "/"

        if path == "/":
            try:
                page = render_page(self.state)
            except Exception as exc:
                page = _error_page(exc)
            self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/status":
            self._json(status_payload(self.state))
        elif path == "/healthz":
            self._json({"ok": True})
        else:
            self._json({"error": "not found"}, status=404)

    def do_POST(self):
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path != "/api/refresh":
            self._json({"error": "not found"}, status=404)
            return

        if not self.state.begin_refresh():
            self._json({"started": False, "reason": "a refresh is already running"},
                       status=409)
            return

        threading.Thread(target=run_refresh, args=(self.state,), daemon=True).start()
        self._json({"started": True})


def _error_page(exc: Exception) -> str:
    """Show the failure in the browser rather than an empty response."""
    from bet.viz import esc
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>RueBet — error</title></head>"
        "<body style=\"font:15px/1.6 system-ui,sans-serif;max-width:720px;"
        "margin:48px auto;padding:0 16px\">"
        "<h1 style='font-size:20px'>Could not build the dashboard</h1>"
        f"<pre style=\"background:#f4f4f2;padding:12px;border-radius:8px;"
        f"overflow-x:auto\">{esc(traceback.format_exc(limit=4))}</pre>"
        "<p>The server is still running. If the store is empty, press "
        "<b>Refresh data</b> once it loads, or run "
        "<code>bet ingest --source football_data --seasons 2015-2026</code>.</p>"
        "</body></html>")


def make_server(state: ServerState, port: int = 8765,
                host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """Bind the server. Loopback by default, and it should stay that way."""
    handler = type("BoundHandler", (Handler,), {"state": state})
    return ThreadingHTTPServer((host, port), handler)


def serve(db_path: Path | None, *, port: int = 8765, host: str = "127.0.0.1",
          days: int = 8, league: str = "bundesliga",
          sources: tuple[str, ...] = ("football_data", "openligadb", "clubelo"),
          open_browser: bool = True, refresh_on_start: bool = False) -> None:
    """Run until interrupted."""
    state = ServerState(db_path=db_path, days=days, league=league, sources=sources)
    httpd = make_server(state, port=port, host=host)
    url = f"http://{host}:{port}/"

    print(f"RueBet is running at {url}")
    if host not in {"127.0.0.1", "localhost", "::1"}:
        print("  WARNING: not bound to loopback. The refresh endpoint fetches "
              "from the internet and writes to the database, and there is no "
              "authentication. Put it behind an authenticating proxy.")
    print("  Ctrl-C to stop\n")

    if refresh_on_start and state.begin_refresh():
        print("  fetching current data in the background...")
        threading.Thread(target=run_refresh, args=(state,), daemon=True).start()

    if open_browser:
        # Opening a browser must never take the server down with it -- there may
        # not be one (a headless box, a container, an SSH session).
        try:
            threading.Timer(0.4, lambda: webbrowser.open(url)).start()
        except Exception:
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
