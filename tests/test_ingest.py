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

CSV_SAMPLE = """Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,HS,AS,HST,AST,HF,AF,HC,AC,HY,AY,HR,AR,PSH,PSD,PSA,PSCH,PSCD,PSCA,AvgCH,AvgCD,AvgCA
D1,24/08/2024,14:30,Bayern Munich,Wolfsburg,3,2,H,1,1,D,18,7,9,3,11,14,8,2,1,3,0,0,1.25,6.50,11.0,1.22,6.80,12.0,1.24,6.60,11.5
D1,24/08/2024,14:30,Dortmund,M'gladbach,1,1,D,0,1,A,12,11,4,5,13,12,5,6,2,1,0,0,1.80,3.90,4.20,1.85,3.85,4.10,1.83,3.88,4.15
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
    matches, results, quotes, team_stats = source._parse_season(
        frame, "bundesliga", "2024-25", result)

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

    # Shots, corners, cards and fouls ship in the same CSV and were previously
    # discarded. Corners and cards are among the softest markets available free.
    assert len(team_stats) == 4
    bayern = next(s for s in team_stats if s["team_id"] == "bayern_munich")
    assert bayern["shots"] == 18
    assert bayern["shots_on_target"] == 9
    assert bayern["corners"] == 8
    assert bayern["yellow_cards"] == 1
    assert bayern["at_home"] is True


def test_football_data_marks_closing_odds_as_unknowable_before_kickoff(store, tmp_path):
    """The specific leak that makes bad models look profitable."""
    path = tmp_path / "D1.csv"
    path.write_text(CSV_SAMPLE)
    source = FootballDataSource(store, raw_dir=tmp_path, delay=0)
    from bet.ingest.base import IngestResult
    matches, _, quotes, _ = source._parse_season(
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
    matches, results, _, _ = source._parse_season(
        pd.read_csv(path), "bundesliga", "2024-25", IngestResult(source="fd"))
    for match, result in zip(matches, results):
        assert result["known_at"] > match["kickoff_utc"]


def test_unknown_club_is_recorded_as_an_error_not_silently_dropped(store, tmp_path):
    bad = CSV_SAMPLE + "D1,25/08/2024,14:30,Real Madrid,Bayern Munich,0,1,A,0,0,D,9,14,3,6,10,11,4,7,1,2,0,0,3.0,3.5,2.4,3.0,3.5,2.4,3.0,3.5,2.4\n"
    path = tmp_path / "bad.csv"
    path.write_text(bad)
    source = FootballDataSource(store, raw_dir=tmp_path, delay=0)
    from bet.ingest.base import IngestResult
    result = IngestResult(source="fd")
    matches, _, _, _ = source._parse_season(pd.read_csv(path), "bundesliga", "2024-25", result)
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


# --------------------------------------------------------------- openligadb

# The api.openligadb.de payload, camelCase, as the live API serves it. The
# adapter was written against the older PascalCase spelling and there was no
# test here at all, so every entry raised `KeyError: 'MatchDateTimeUTC'` and a
# whole season parsed to nothing -- with 306 identical errors in the report.
OLDB_CAMEL = [
    {
        "matchID": 1,
        "matchDateTime": "2024-08-24T15:30:00",
        "matchDateTimeUTC": "2024-08-24T13:30:00Z",
        "team1": {"teamId": 40, "teamName": "Bayern München"},
        "team2": {"teamId": 131, "teamName": "VfL Wolfsburg"},
        "matchIsFinished": True,
        "matchResults": [
            {"resultTypeID": 1, "pointsTeam1": 1, "pointsTeam2": 1},
            {"resultTypeID": 2, "pointsTeam1": 3, "pointsTeam2": 2},
        ],
    },
    {
        "matchID": 2,
        "matchDateTimeUTC": "2024-08-24T13:30:00Z",
        "team1": {"teamName": "Borussia Dortmund"},
        "team2": {"teamName": "Borussia Mönchengladbach"},
        "matchIsFinished": False,
        "matchResults": [],
    },
]


def _to_pascal(node):
    """The same payload under the original spelling."""
    if isinstance(node, dict):
        return {key[:1].upper() + key[1:]: _to_pascal(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_to_pascal(item) for item in node]
    return node


def _parse_oldb(store, payload):
    from bet.ingest.base import IngestResult
    from bet.ingest.openligadb import OpenLigaDBSource

    source = OpenLigaDBSource(store)
    result = IngestResult(source="openligadb")
    matches, results = source._parse_season(payload, "bundesliga", "2024-25", result)
    return matches, results, result


@pytest.mark.parametrize("spelling", ["camelCase", "PascalCase"])
def test_openligadb_parses_either_capitalisation(store, spelling):
    """The API has served both. Neither may break the adapter."""
    payload = OLDB_CAMEL if spelling == "camelCase" else _to_pascal(OLDB_CAMEL)
    matches, results, result = _parse_oldb(store, payload)

    assert result.errors == []
    assert len(matches) == 2
    assert matches[0]["home_team_id"] == "bayern_munich"
    assert matches[0]["away_team_id"] == "vfl_wolfsburg"

    # Only the finished match yields a result.
    assert len(results) == 1
    assert (results[0]["home_goals"], results[0]["away_goals"]) == (3, 2)
    assert results[0]["outcome"] == "H"
    assert (results[0]["ht_home"], results[0]["ht_away"]) == (1, 1)


@pytest.mark.parametrize("spelling", ["camelCase", "PascalCase"])
def test_openligadb_reads_kickoff_as_utc_not_german_local_time(store, spelling):
    """`matchDateTime` is local; using it would shift every fixture."""
    payload = OLDB_CAMEL if spelling == "camelCase" else _to_pascal(OLDB_CAMEL)
    matches, _, _ = _parse_oldb(store, payload)
    assert matches[0]["kickoff_utc"] == datetime(2024, 8, 24, 13, 30)


def test_openligadb_falls_back_to_local_time_when_utc_is_absent(store):
    entry = dict(OLDB_CAMEL[0])
    entry.pop("matchDateTimeUTC")
    matches, _, result = _parse_oldb(store, [entry])
    assert matches[0]["kickoff_utc"] == datetime(2024, 8, 24, 15, 30)
    assert result.errors == []


def test_openligadb_collapses_repeated_failures_and_names_the_fields(store):
    """A renamed field breaks all 306 entries identically.

    One line per fixture would bury every other source's error; and the bare
    `KeyError` repr said a key was missing without saying which keys exist,
    which is the one fact that identifies the fix.
    """
    broken = [{"whenever": "2024-08-24T13:30:00Z", "home": "x", "away": "y"}] * 306
    matches, _, result = _parse_oldb(store, broken)

    assert matches == []
    assert len(result.errors) == 1
    message = result.errors[0]
    assert "skipped 306 of 306" in message
    assert "whenever" in message and "home" in message      # the keys on offer
    assert "MatchDateTime" in message                       # the key wanted


def test_openligadb_reports_an_unknown_club_by_name(store):
    entry = dict(OLDB_CAMEL[0], team2={"teamName": "Racing Club de Nowhere"})
    matches, _, result = _parse_oldb(store, [entry])
    assert matches == []
    assert "Nowhere" in result.errors[0]


# ------------------------------------------------------------ fetch retries

class _FakeResponse:
    def __init__(self, status, text="body"):
        self.status_code = status
        self.text = text

    def raise_for_status(self):
        import requests
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Server Error", response=self)


class _FakeSession:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.headers = {}
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        status = self.statuses.pop(0)
        if isinstance(status, Exception):
            raise status
        return _FakeResponse(status)


def _source(store, tmp_path, session):
    from bet.ingest.base import Source

    class _Probe(Source):
        name = "probe"

        def ingest(self, **kwargs):
            raise NotImplementedError

    return _Probe(store, session=session, raw_dir=tmp_path, delay=0)


def test_a_gateway_error_is_retried(store, tmp_path):
    """ClubElo returned 502 for three snapshots and the source was written off."""
    store.init_schema()
    session = _FakeSession([502, 502, 200])
    text, _ = _source(store, tmp_path, session).fetch(
        "http://api.clubelo.com/2026-09-01", cache=False)

    assert text == "body"
    assert len(session.calls) == 3


def test_a_not_found_is_not_retried(store, tmp_path):
    """A 404 will still be a 404; repeating it only annoys a free service."""
    import requests

    store.init_schema()
    session = _FakeSession([404, 200])
    with pytest.raises(requests.HTTPError):
        _source(store, tmp_path, session).fetch("http://example.invalid/x", cache=False)

    assert len(session.calls) == 1


def test_a_forbidden_is_not_retried(store, tmp_path):
    import requests

    store.init_schema()
    session = _FakeSession([403, 200])
    with pytest.raises(requests.HTTPError):
        _source(store, tmp_path, session).fetch("http://example.invalid/x", cache=False)

    assert len(session.calls) == 1


def test_persistent_failure_reports_how_many_attempts_were_made(store, tmp_path):
    store.init_schema()
    session = _FakeSession([503, 503, 503])
    with pytest.raises(RuntimeError, match="after 3 attempts") as exc:
        _source(store, tmp_path, session).fetch("http://example.invalid/x", cache=False)

    assert len(session.calls) == 3
    # Callers prefix the URL and the underlying error carries it, so repeating
    # it here produced "clubelo: <url>: <url>: still failing ...".
    assert str(exc.value).count("http://example.invalid/x") <= 1


def test_a_connection_error_is_retried(store, tmp_path):
    """A dropped connection has no status code and is worth another try."""
    import requests

    store.init_schema()
    session = _FakeSession([requests.ConnectionError("reset by peer"), 200])
    text, _ = _source(store, tmp_path, session).fetch("http://example.invalid/x", cache=False)

    assert text == "body"
    assert len(session.calls) == 2


# ---------------------------------------------------------------- clubelo

CLUBELO_CSV = (
    "Rank,Club,Country,Level,Elo,From,To\n"
    "1,Bayern,GER,1,1990.5,2026-08-26,2026-09-02\n"
    "8,Leverkusen,GER,1,1850.1,2026-08-26,2026-09-02\n"
)


def _clubelo(store, tmp_path, session):
    from bet.ingest.clubelo import ClubEloSource
    return ClubEloSource(store, session=session, raw_dir=tmp_path, delay=0)


class _SchemeSession(_FakeSession):
    """Answers on https only; http is a gateway error, as reported."""

    def __init__(self):
        super().__init__([])

    def get(self, url, timeout=None):
        self.calls.append(url)
        if url.startswith("http://"):
            return _FakeResponse(502)
        return _FakeResponse(200, CLUBELO_CSV)


def test_clubelo_falls_back_to_the_other_scheme(store, tmp_path):
    """http returned 502 for every snapshot, and the source was written off."""
    from datetime import date

    store.init_schema()
    session = _SchemeSession()
    result = _clubelo(store, tmp_path, session).ingest(
        start=date(2026, 9, 1), end=date(2026, 9, 1), cache=False)

    assert result.errors == []
    assert result.rows_written["team_rating"] == 2
    assert any(c.startswith("https://") for c in session.calls)


def test_clubelo_probes_the_other_scheme_only_once(store, tmp_path):
    """Latching on keeps the probe at one extra request, not one per week."""
    from datetime import date

    store.init_schema()
    session = _SchemeSession()
    _clubelo(store, tmp_path, session).ingest(
        start=date(2026, 9, 1), end=date(2026, 9, 22), cache=False)

    http_calls = [c for c in session.calls if c.startswith("http://")]
    # One request, for one date: the probe is not retried, and once https has
    # answered the remaining weeks never touch http again.
    assert http_calls == ["http://api.clubelo.com/2026-09-01"]
    assert len([c for c in session.calls if c.startswith("https://")]) >= 3


def test_clubelo_reports_both_schemes_when_neither_answers(store, tmp_path):
    from datetime import date

    class _AllDown(_FakeSession):
        def __init__(self):
            super().__init__([])

        def get(self, url, timeout=None):
            self.calls.append(url)
            return _FakeResponse(502)

    store.init_schema()
    result = _clubelo(store, tmp_path, _AllDown()).ingest(
        start=date(2026, 9, 1), end=date(2026, 9, 1), cache=False)

    assert len(result.errors) == 1
    message = result.errors[0]
    assert "no scheme answered" in message
    assert "http://api.clubelo.com" in message
    assert "https://api.clubelo.com" in message
