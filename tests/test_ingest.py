"""Ingest adapter parsing, tested against archived-shaped payloads.

No network. These check the parsing and — more importantly — that each adapter
stamps `known_at` correctly, which is the one thing an adapter can get wrong in
a way that silently invalidates every backtest built on top of it.
"""

import json
from datetime import datetime, timedelta

import pandas as pd
import pytest

from bet.ingest.base import make_match_id, season_code, season_label
from bet.ingest.football_data import FootballDataSource, _parse_kickoff
from bet.ingest.understat import extract_json_var

CSV_SAMPLE = """Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,PSH,PSD,PSA,PSCH,PSCD,PSCA,AvgCH,AvgCD,AvgCA
D1,24/08/2024,14:30,Bayern Munich,Wolfsburg,3,2,H,1,1,D,1.25,6.50,11.0,1.22,6.80,12.0,1.24,6.60,11.5
D1,24/08/2024,14:30,Dortmund,M'gladbach,1,1,D,0,1,A,1.80,3.90,4.20,1.85,3.85,4.10,1.83,3.88,4.15
"""


def test_season_helpers():
    assert season_code(2024) == "2425"
    assert season_label(2024) == "2024-25"


def test_kickoff_parsing_handles_both_date_formats():
    assert _parse_kickoff("24/08/2024", "15:30") == datetime(2024, 8, 24, 15, 30)
    assert _parse_kickoff("24/08/24", "15:30") == datetime(2024, 8, 24, 15, 30)


def test_kickoff_defaults_when_old_files_omit_the_time():
    # Files before roughly 2019 carry no Time column.
    assert _parse_kickoff("24/08/2015", None) == datetime(2015, 8, 24, 15, 0)


def test_football_data_parsing_loads_matches_results_and_odds(store, tmp_path):
    path = tmp_path / "D1.csv"
    path.write_text(CSV_SAMPLE)

    source = FootballDataSource(store, raw_dir=tmp_path, delay=0)
    frame = pd.read_csv(path)
    from bet.ingest.base import IngestResult
    result = IngestResult(source="football_data")
    matches, results, quotes = source._parse_season(frame, "bundesliga", "2024-25", result)

    assert len(matches) == 2
    assert len(results) == 2
    assert not result.errors

    # Team names resolved through the canonical registry.
    assert matches[0]["home_team_id"] == "bayern_munich"
    assert matches[1]["away_team_id"] == "borussia_monchengladbach"

    assert results[0]["outcome"] == "H"
    assert results[1]["outcome"] == "D"

    # Three books x three selections x (closing + pre-closing), minus the books
    # absent from this sample.
    assert len(quotes) > 0
    assert {q["selection"] for q in quotes} == {"H", "D", "A"}


def test_football_data_marks_closing_odds_as_unknowable_before_kickoff(store, tmp_path):
    """The specific leak that makes bad models look profitable."""
    path = tmp_path / "D1.csv"
    path.write_text(CSV_SAMPLE)
    source = FootballDataSource(store, raw_dir=tmp_path, delay=0)
    from bet.ingest.base import IngestResult
    matches, _, quotes = source._parse_season(
        pd.read_csv(path), "bundesliga", "2024-25", IngestResult(source="fd"))

    kickoff = matches[0]["kickoff_utc"]
    closing = [q for q in quotes if q["is_closing"] and q["match_id"] == matches[0]["match_id"]]
    assert closing
    for quote in closing:
        assert quote["known_at"] >= kickoff

    pre = [q for q in quotes if not q["is_closing"] and q["match_id"] == matches[0]["match_id"]]
    for quote in pre:
        assert quote["known_at"] < kickoff


def test_football_data_marks_results_as_knowable_only_after_the_match(store, tmp_path):
    path = tmp_path / "D1.csv"
    path.write_text(CSV_SAMPLE)
    source = FootballDataSource(store, raw_dir=tmp_path, delay=0)
    from bet.ingest.base import IngestResult
    matches, results, _ = source._parse_season(
        pd.read_csv(path), "bundesliga", "2024-25", IngestResult(source="fd"))
    for match, result in zip(matches, results):
        assert result["known_at"] > match["kickoff_utc"]


def test_unknown_club_is_recorded_as_an_error_not_silently_dropped(store, tmp_path):
    bad = CSV_SAMPLE + "D1,25/08/2024,14:30,Real Madrid,Bayern Munich,0,1,A,0,0,D,3.0,3.5,2.4,3.0,3.5,2.4,3.0,3.5,2.4\n"
    path = tmp_path / "bad.csv"
    path.write_text(bad)
    source = FootballDataSource(store, raw_dir=tmp_path, delay=0)
    from bet.ingest.base import IngestResult
    result = IngestResult(source="fd")
    matches, _, _ = source._parse_season(pd.read_csv(path), "bundesliga", "2024-25", result)
    assert len(matches) == 2
    assert any("Real Madrid" in e for e in result.errors)


def test_match_ids_are_stable_across_sources():
    # The same fixture ingested from two sources must collapse onto one row.
    a = make_match_id("bundesliga", "2024-25", "bayern_munich", "sc_freiburg")
    b = make_match_id("bundesliga", "2024-25", "bayern_munich", "sc_freiburg")
    assert a == b


def test_understat_hex_escaped_payload_is_decoded():
    html = r"""var shotsData = JSON.parse('\x7B\x22h\x22:[{\x22id\x22:\x2299\x22,\x22xG\x22:\x220.31\x22}]\x7D')"""
    data = extract_json_var(html, "shotsData")
    assert data["h"][0]["xG"] == "0.31"


def test_understat_missing_variable_raises():
    with pytest.raises(ValueError, match="not found"):
        extract_json_var("<html></html>", "shotsData")
