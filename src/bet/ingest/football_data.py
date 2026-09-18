"""football-data.co.uk: results and historical closing odds.

The most valuable free resource for this project, and the reason the backtest
harness can exist at all at zero cost. One CSV per league per season, back to
the 1990s, carrying full-time results and — critically — odds from several
books including Pinnacle, in both opening and closing form.

Without historical closing prices there is no closing-line-value measurement,
and without that there is no way to know whether a model has an edge. Ingest
this first.

Column conventions in these files:
    FTHG / FTAG / FTR   full-time home goals, away goals, result
    PSH  / PSD  / PSA   Pinnacle, pre-closing
    PSCH / PSCD / PSCA  Pinnacle, closing  (the 'C' marks closing throughout)
    AvgCH / AvgCD / AvgCA   market average closing
    MaxCH / MaxCD / MaxCA   best available closing price
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from bet.config import FOOTBALL_DATA_LEAGUES, SETTINGS
from bet.ingest.base import IngestResult, Source, make_match_id, season_code, season_label
from bet.teams import UnknownTeamError, resolve

BASE_URL = "https://www.football-data.co.uk/mmz4281"

# book key -> (home column, draw column, away column), closing prices only.
CLOSING_ODDS_COLUMNS = {
    "pinnacle": ("PSCH", "PSCD", "PSCA"),
    "bet365": ("B365CH", "B365CD", "B365CA"),
    "market_avg": ("AvgCH", "AvgCD", "AvgCA"),
    "market_max": ("MaxCH", "MaxCD", "MaxCA"),
    "william_hill": ("WHCH", "WHCD", "WHCA"),
}

# Pre-closing prices. Knowable before kickoff, so usable as a model input,
# unlike the closing columns above.
OPENING_ODDS_COLUMNS = {
    "pinnacle": ("PSH", "PSD", "PSA"),
    "bet365": ("B365H", "B365D", "B365A"),
    "market_avg": ("AvgH", "AvgD", "AvgA"),
    "william_hill": ("WHH", "WHD", "WHA"),
}


def _parse_kickoff(date_value, time_value) -> datetime | None:
    """Parse the dd/mm/yy(yy) date plus optional HH:MM time.

    Files before roughly 2019 carry no kickoff time; those default to 15:00 UTC,
    a typical Saturday afternoon slot. The imprecision is harmless because the
    prediction cut-off is an hour out and matchdays are clustered anyway.
    """
    if pd.isna(date_value):
        return None
    date = None
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            date = datetime.strptime(str(date_value).strip(), fmt)
            break
        except ValueError:
            continue
    if date is None:
        return None

    if pd.isna(time_value) or not str(time_value).strip():
        return date.replace(hour=15, minute=0)
    try:
        hh, mm = str(time_value).strip().split(":")[:2]
        return date.replace(hour=int(hh), minute=int(mm))
    except (ValueError, IndexError):
        return date.replace(hour=15, minute=0)


class FootballDataSource(Source):
    name = "football_data"

    def ingest(self, seasons: list[int], league: str = "bundesliga",
               cache: bool = True) -> IngestResult:
        result = IngestResult(source=self.name)
        code = FOOTBALL_DATA_LEAGUES.get(league)
        if code is None:
            raise ValueError(f"unknown league {league!r}; expected one of {list(FOOTBALL_DATA_LEAGUES)}")

        matches, results, quotes = [], [], []

        for start_year in seasons:
            url = f"{BASE_URL}/{season_code(start_year)}/{code}.csv"
            try:
                text, path = self.fetch(url, suffix=".csv", cache=cache)
            except Exception as exc:
                result.errors.append(f"{url}: {exc}")
                continue
            result.documents_fetched += 1

            try:
                frame = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
            except Exception as exc:
                result.errors.append(f"parse {url}: {exc}")
                continue

            season = season_label(start_year)
            m, r, q = self._parse_season(frame, league, season, result)
            matches.extend(m)
            results.extend(r)
            quotes.extend(q)

        if matches:
            result.rows_written["match"] = self.store.upsert(
                "match", pd.DataFrame(matches).drop_duplicates("match_id"), ["match_id"])
        if results:
            result.rows_written["match_result"] = self.store.upsert(
                "match_result", pd.DataFrame(results).drop_duplicates("match_id"), ["match_id"])
        if quotes:
            quote_frame = pd.DataFrame(quotes).drop_duplicates(
                ["match_id", "book", "market", "selection", "quoted_at"])
            result.rows_written["odds_quote"] = self.store.upsert(
                "odds_quote", quote_frame, ["match_id", "book", "market", "selection", "quoted_at"])

        return result

    def _parse_season(self, frame: pd.DataFrame, league: str, season: str,
                      result: IngestResult) -> tuple[list, list, list]:
        matches, results, quotes = [], [], []
        time_column = "Time" if "Time" in frame.columns else None

        for row in frame.itertuples(index=False):
            record = row._asdict()
            if pd.isna(record.get("HomeTeam")) or pd.isna(record.get("FTHG")):
                continue

            kickoff = _parse_kickoff(record.get("Date"), record.get(time_column) if time_column else None)
            if kickoff is None:
                continue

            try:
                home = resolve(str(record["HomeTeam"]))
                away = resolve(str(record["AwayTeam"]))
            except UnknownTeamError as exc:
                result.errors.append(str(exc))
                continue

            match_id = make_match_id(league, season, home, away)

            # A fixture list is published well before the season starts; a
            # month's lead is a safe, deliberately conservative stand-in.
            matches.append({
                "match_id": match_id, "source": self.name, "league": league, "season": season,
                "kickoff_utc": kickoff, "home_team_id": home, "away_team_id": away,
                "known_at": kickoff - timedelta(days=30),
            })

            try:
                home_goals, away_goals = int(record["FTHG"]), int(record["FTAG"])
            except (TypeError, ValueError):
                continue

            outcome = "H" if home_goals > away_goals else ("D" if home_goals == away_goals else "A")
            results.append({
                "match_id": match_id,
                "home_goals": home_goals, "away_goals": away_goals, "outcome": outcome,
                "ht_home": int(record["HTHG"]) if not pd.isna(record.get("HTHG")) else None,
                "ht_away": int(record["HTAG"]) if not pd.isna(record.get("HTAG")) else None,
                # Public only once the match has finished.
                "known_at": kickoff + timedelta(seconds=SETTINGS.result_known_after_seconds),
            })

            quotes.extend(self._extract_odds(record, match_id, kickoff))

        return matches, results, quotes

    def _extract_odds(self, record: dict, match_id: str, kickoff: datetime) -> list[dict]:
        """Pull 1X2 prices, tagging closing and pre-closing correctly.

        The `known_at` on a closing price is the kickoff itself, which keeps it
        out of every pre-match model's view while leaving it available for
        scoring. Getting this wrong is precisely the leak that makes a bad model
        look profitable.
        """
        quotes: list[dict] = []

        for is_closing, columns in ((True, CLOSING_ODDS_COLUMNS), (False, OPENING_ODDS_COLUMNS)):
            # Pre-closing prices are treated as knowable a day out. That is an
            # approximation: the files give no timestamp for when they were taken.
            quoted_at = kickoff if is_closing else kickoff - timedelta(days=1)

            for book, (h_col, d_col, a_col) in columns.items():
                values = [record.get(h_col), record.get(d_col), record.get(a_col)]
                if any(v is None or pd.isna(v) for v in values):
                    continue
                try:
                    odds = [float(v) for v in values]
                except (TypeError, ValueError):
                    continue
                if any(not np.isfinite(o) or o <= 1.0 for o in odds):
                    continue

                for selection, price in zip(("H", "D", "A"), odds):
                    quotes.append({
                        "match_id": match_id, "book": book, "market": "1x2",
                        "selection": selection, "decimal_odds": price,
                        "quoted_at": quoted_at, "is_closing": is_closing,
                        "known_at": quoted_at,
                    })

        return quotes
