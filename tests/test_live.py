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
