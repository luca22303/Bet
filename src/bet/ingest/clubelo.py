"""ClubElo: long-run power ratings, free via a plain CSV API.

Gives the backtest a credible non-market baseline and, more usefully, a prior
for promoted sides. A Dixon-Coles model fitted on Bundesliga results alone knows
nothing about a team that was in the 2. Bundesliga last May, and will produce
nonsense for it for roughly ten matchdays every season. ClubElo tracks clubs
across divisions, so it carries exactly the information that is missing.

The API serves a snapshot of every club's rating on a given date, so ratings are
ingested by walking dates. `known_at` is the snapshot date: a rating dated the
7th was not available on the 6th.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from io import StringIO

import pandas as pd

from bet.ingest.base import IngestResult, Source
from bet.teams import resolve

# ClubElo documents its API as plain http. When http returned 502 for every
# snapshot in a run -- persistently, not as a blip -- the open question was
# whether their http endpoint is broken rather than whether the API moved, so
# both schemes are tried before the source is written off.
BASE_URLS = ("http://api.clubelo.com", "https://api.clubelo.com")
BASE_URL = BASE_URLS[0]


class ClubEloSource(Source):
    name = "clubelo"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._base: str | None = None

    def ingest(self, start: date, end: date | None = None, step_days: int = 7,
               cache: bool = True) -> IngestResult:
        """Walk weekly snapshots between two dates.

        Weekly is enough: Elo moves only when a club plays, and the store keeps
        the most recent rating at or before any `as_of`.
        """
        result = IngestResult(source=self.name)
        end = end or date.today()
        rows: list[dict] = []

        current = start
        while current <= end:
            try:
                text, url = self._fetch_snapshot(current, cache=cache)
            except Exception as exc:
                result.errors.append(str(exc))
                current += timedelta(days=step_days)
                continue
            result.documents_fetched += 1

            try:
                frame = pd.read_csv(StringIO(text))
            except Exception as exc:
                result.errors.append(f"parse {url}: {exc}")
                current += timedelta(days=step_days)
                continue

            rows.extend(self._parse_snapshot(frame, current))
            current += timedelta(days=step_days)

        if rows:
            frame = pd.DataFrame(rows).drop_duplicates(["team_id", "source", "valid_from"])
            result.rows_written["team_rating"] = self.store.upsert(
                "team_rating", frame, ["team_id", "source", "valid_from"])
        return result

    def _fetch_snapshot(self, day: date, *, cache: bool) -> tuple[str, str]:
        """One daily snapshot, over whichever scheme the host answers on.

        The working base is remembered for the rest of the run, so probing
        costs one extra request in total rather than one per week.
        """
        bases = (self._base,) if self._base else BASE_URLS
        attempts: list[str] = []

        for index, base in enumerate(bases):
            url = f"{base}/{day.isoformat()}"
            # While still deciding which scheme works, spend one request per
            # candidate rather than the full retry budget on each: the point of
            # the probe is to move on quickly, and the winner gets the retries.
            probing = len(bases) > 1 and index < len(bases) - 1
            try:
                text, _ = self.fetch(url, suffix=".csv", cache=cache,
                                     attempts=1 if probing else 3)
            except Exception as exc:
                attempts.append(f"{base} -> {exc}")
                continue
            self._base = base
            return text, url

        raise RuntimeError(
            f"clubelo {day.isoformat()}: no scheme answered — " + "; ".join(attempts))

    def _parse_snapshot(self, frame: pd.DataFrame, snapshot: date) -> list[dict]:
        rows = []
        for row in frame.itertuples(index=False):
            record = row._asdict()
            club = record.get("Club")
            if club is None or pd.isna(club):
                continue
            # Unknown clubs are skipped rather than raising: the snapshot covers
            # every club in Europe and only the German ones concern us.
            team_id = resolve(str(club), strict=False)
            if team_id is None:
                continue
            try:
                rating = float(record["Elo"])
            except (KeyError, TypeError, ValueError):
                continue

            valid_from = pd.to_datetime(record.get("From")).date() if not pd.isna(record.get("From")) else snapshot
            valid_to = pd.to_datetime(record.get("To")).date() if not pd.isna(record.get("To")) else None

            rows.append({
                "team_id": team_id, "source": self.name, "rating": rating,
                "valid_from": valid_from, "valid_to": valid_to,
                # A rating dated in the past was still only published at the
                # snapshot we fetched it from; never claim earlier knowledge.
                "known_at": datetime.combine(max(valid_from, snapshot), datetime.min.time()),
            })
        return rows
