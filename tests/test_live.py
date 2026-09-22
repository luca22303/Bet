"""The live refresh command.

No network. Adapters are stubbed, so what is tested is the orchestration: that a
failing source is stepped over rather than stopping the run, that counts and
errors are aggregated, and that the current season is worked out correctly.
"""

from datetime import date, datetime
from pathlib import Path

import pytest

from bet.ingest.base import IngestResult
from bet.live import RefreshReport, current_season_start, refresh, refresh_and_render


@pytest.mark.parametrize("today,expected", [
    (date(2026, 9, 22), 2026),     # mid-season
    (date(2026, 3, 1), 2025),      # spring belongs to the season that began in August
    (date(2026, 7, 1), 2026),      # July starts the new one
    (date(2026, 6, 30), 2025),     # June is still the old one
])
def test_current_season_handles_the_year_boundary(today, expected):
    """European seasons straddle the new year."""
    assert current_season_start(today) == expected


class _Stub:
    def __init__(self, name, rows=None, errors=None, raises=None):
        self.name = name
        self._rows = rows or {}
        self._errors = errors or []
        self._raises = raises
        self.calls = []

    def ingest(self, **kwargs):
        self.calls.append(kwargs)
        if self._raises:
            raise self._raises
        return IngestResult(source=self.name, rows_written=dict(self._rows),
                            errors=list(self._errors))


def test_a_failing_source_does_not_stop_the_others():
    """Partial data plus a visible error beats no refresh at all."""
    report = RefreshReport(started=datetime.utcnow(), seasons=[2026])
    from bet.live import _run

    broken = _Stub("broken", raises=RuntimeError("unreachable"))
    working = _Stub("working", rows={"match": 306})

    _run(report, broken, seasons=[2026])
    _run(report, working, seasons=[2026])

    assert "FAILED" in report.sources["broken"]
    assert report.rows["match"] == 306
    assert any("unreachable" in e for e in report.errors)


def test_counts_accumulate_across_sources():
    report = RefreshReport(started=datetime.utcnow(), seasons=[2026])
    from bet.live import _run
    _run(report, _Stub("a", rows={"match": 10, "odds_quote": 30}))
    _run(report, _Stub("b", rows={"match": 5}))
    assert report.rows == {"match": 15, "odds_quote": 30}


def test_source_errors_are_surfaced_in_the_status_line():
    report = RefreshReport(started=datetime.utcnow(), seasons=[2026])
    report.sources["a"] = "results"
    from bet.live import _run
    _run(report, _Stub("a", rows={"match": 1}, errors=["bad row", "bad row 2"]))
    assert "2 errors" in report.sources["a"]


def test_refresh_defaults_to_the_current_season(store, monkeypatch):
    """Re-scraping a decade every run would be slow and rude to the sources."""
    calls = {}

    class _Source:
        name = "football_data"

        def __init__(self, store):
            pass

        def ingest(self, **kwargs):
            calls.update(kwargs)
            return IngestResult(source="football_data")

    import bet.ingest.football_data as module
    monkeypatch.setattr(module, "FootballDataSource", _Source)

    refresh(store, sources=("football_data",))
    assert calls["seasons"] == [current_season_start()]
    # A cached copy of last week's file is the stale snapshot this prevents.
    assert calls["cache"] is False


def test_report_summary_names_what_happened():
    report = RefreshReport(started=datetime.utcnow(), seasons=[2026],
                           sources={"football_data": "results"},
                           rows={"match": 306}, errors=["boom"],
                           duration_seconds=12.0)
    text = report.summary()
    assert "2026" in text
    assert "football_data" in text
    assert "match=306" in text
    assert "boom" in text


def test_refresh_and_render_writes_a_page(store, tmp_path):
    """Even when every source fails, the page is written and says so."""
    output = tmp_path / "board.html"
    report = refresh_and_render(store, output, sources=())
    assert output.exists()
    assert report.output == output.resolve()

    page = output.read_text()
    assert "<!DOCTYPE html>" in page
    assert "No data" in page


def test_the_summary_hides_no_errors():
    """Truncation dropped the sixth of six failures without saying so."""
    from datetime import datetime as _dt

    from bet.live import RefreshReport

    report = RefreshReport(started=_dt.utcnow(), seasons=[2026],
                           errors=[f"source{i}: broke" for i in range(6)])
    text = report.summary()
    assert "6 error(s)" in text
    for i in range(6):
        assert f"source{i}" in text


def test_a_very_long_error_list_says_how_many_are_hidden():
    from datetime import datetime as _dt

    from bet.live import RefreshReport

    report = RefreshReport(started=_dt.utcnow(), seasons=[2026],
                           errors=[f"e{i}" for i in range(25)])
    text = report.summary()
    assert "25 error(s)" in text
    assert "and 5 more" in text


def test_refresh_reports_which_source_it_is_on(store, monkeypatch):
    """A run with no sign of life reads exactly like a hang."""
    from bet.live import refresh

    seen = []
    store.init_schema()

    import bet.ingest.football_data as fd
    import bet.ingest.openligadb as olg
    import bet.ingest.clubelo as ce

    class _Nothing:
        def __init__(self, name):
            self.name = name

        def ingest(self, **kwargs):
            from bet.ingest.base import IngestResult
            return IngestResult(source=self.name)

    monkeypatch.setattr(fd, "FootballDataSource", lambda store: _Nothing("football_data"))
    monkeypatch.setattr(olg, "OpenLigaDBSource", lambda store: _Nothing("openligadb"))
    monkeypatch.setattr(ce, "ClubEloSource", lambda store: _Nothing("clubelo"))

    refresh(store, seasons=[2026],
            sources=("football_data", "openligadb", "clubelo"),
            on_progress=seen.append)

    assert seen == ["football_data (1 of 3)",
                    "openligadb (2 of 3)",
                    "clubelo (3 of 3)"]


def test_progress_is_optional():
    """`bet live` and the tests call refresh without a callback."""
    import inspect

    from bet.live import refresh

    assert inspect.signature(refresh).parameters["on_progress"].default is None


def _stub_source(name, *, rows=None, errors=()):
    from bet.ingest.base import IngestResult

    class _Stub:
        def __init__(self):
            self.name = name

        def ingest(self, **kwargs):
            return IngestResult(source=name, rows_written=dict(rows or {}),
                                errors=list(errors))

    return _Stub


def _refresh_with(store, monkeypatch, stubs):
    import bet.ingest.clubelo as ce
    import bet.ingest.football_data as fd
    import bet.ingest.openligadb as olg
    from bet.live import refresh

    monkeypatch.setattr(fd, "FootballDataSource", lambda store: stubs["football_data"]())
    monkeypatch.setattr(olg, "OpenLigaDBSource", lambda store: stubs["openligadb"]())
    monkeypatch.setattr(ce, "ClubEloSource", lambda store: stubs["clubelo"]())
    store.init_schema()
    return refresh(store, seasons=[2026],
                   sources=("football_data", "openligadb", "clubelo"))


def test_a_source_that_fetched_nothing_is_reported_once_with_its_consequence(
        store, monkeypatch):
    """Three snapshots of a down host produced three 900-character errors.

    Repeating nested urllib3 detail buries the one fact worth reading, and
    "clubelo" alone does not tell anyone what they are now missing.
    """
    report = _refresh_with(store, monkeypatch, {
        "football_data": _stub_source("football_data", rows={"match": 1026}),
        "openligadb": _stub_source("openligadb", rows={"match": 306}),
        "clubelo": _stub_source("clubelo", errors=[
            "clubelo 2026-09-01: no scheme answered — " + "x" * 600,
            "clubelo 2026-09-08: no scheme answered — " + "x" * 600,
            "clubelo 2026-09-15: no scheme answered — " + "x" * 600,
        ]),
    })

    assert list(report.unavailable) == ["clubelo"]
    clubelo_errors = [e for e in report.errors if e.startswith("clubelo")]
    assert len(clubelo_errors) == 1
    assert "power ratings not updated" in clubelo_errors[0]
    assert len(clubelo_errors[0]) < 320, "the whole urllib3 chain came through"


def test_a_source_that_wrote_some_rows_is_not_called_unavailable(store, monkeypatch):
    """Partial success is a real error worth reading in full."""
    report = _refresh_with(store, monkeypatch, {
        "football_data": _stub_source("football_data", rows={"match": 306},
                                      errors=["2019 season: 404"]),
        "openligadb": _stub_source("openligadb", rows={"match": 306}),
        "clubelo": _stub_source("clubelo", rows={"team_rating": 18}),
    })

    assert report.unavailable == {}
    assert any("2019 season" in e for e in report.errors)


@pytest.mark.parametrize("raw,expected", [
    ("502 Server Error: Bad Gateway for url: http://x", "HTTP 502"),
    ("404 Client Error: Not Found", "HTTP 404"),
    ("ConnectTimeoutError('Connection to x timed out. (connect timeout=10)')",
     "connection timed out"),
    ("NewConnectionError: [Errno 111] Connection refused", "connection refused"),
    ("Failed to resolve 'x' ([Errno -2] Name or service not known)", "DNS lookup failed"),
])
def test_causes_are_named_not_quoted(raw, expected):
    from bet.live import _short_reason
    assert _short_reason([raw]) == expected


def test_both_causes_are_named_when_a_host_fails_two_ways():
    """http answering 502 while https refuses are two different facts.

    Both matter when deciding whether it is their outage or your network.
    """
    from bet.live import _short_reason

    reason = _short_reason([
        "http -> 502 Server Error: Bad Gateway for url: http://api.clubelo.com/2026-09-01",
        "https -> ConnectTimeoutError('Connection to api.clubelo.com timed out. "
        "(connect timeout=10)')",
    ])
    assert reason == "HTTP 502, connection timed out"
    assert len(reason) < 60


def test_an_unrecognised_failure_still_says_something():
    from bet.live import _short_reason
    assert _short_reason(["the parser found no table"]) == "the parser found no table"


def test_the_summary_does_not_print_an_outage_twice():
    from datetime import datetime as _dt

    from bet.live import RefreshReport

    report = RefreshReport(
        started=_dt.utcnow(), seasons=[2026], rows={"match": 306},
        sources={"clubelo": "unavailable (power ratings not updated)"},
        errors=["clubelo: unavailable after 3 attempt(s) — HTTP 502"],
        unavailable={"clubelo": "HTTP 502, connection timed out"})

    text = report.summary()
    assert text.count("unavailable") == 2        # the source line and its reason
    assert "error(s):" not in text               # already covered above
