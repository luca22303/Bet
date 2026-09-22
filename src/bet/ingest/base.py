"""Shared plumbing for source adapters.

Two habits worth the small cost:

Archive before parsing. Every fetched payload is written to disk and recorded
in `raw_document` before anything reads it. When a site changes its markup —
and they all do — re-parsing a decade of archived files takes seconds, while
re-scraping it takes days and may be impossible if the source has since pulled
the data.

Be slow on purpose. These are free public sources run by people who do not owe
you anything, and hammering them gets everyone blocked. The default delay is
three seconds and there is no parallel fetching.
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from bet.config import SETTINGS


@dataclass
class IngestResult:
    source: str
    rows_written: dict[str, int] = field(default_factory=dict)
    documents_fetched: int = 0
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        written = ", ".join(f"{t}={n}" for t, n in sorted(self.rows_written.items())) or "nothing"
        tail = f" [{len(self.errors)} errors]" if self.errors else ""
        return f"{self.source}: {written} ({self.documents_fetched} documents){tail}"


class Source(ABC):
    """Base class for ingest adapters."""

    name: str = "source"

    def __init__(self, store, session: requests.Session | None = None,
                 raw_dir: Path | None = None, delay: float | None = None) -> None:
        self.store = store
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", SETTINGS.user_agent)
        self.raw_dir = Path(raw_dir or SETTINGS.raw_dir) / self.name
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.delay = SETTINGS.request_delay_seconds if delay is None else delay
        self._last_request = 0.0

    # ------------------------------------------------------------ fetching

    def fetch(self, url: str, *, suffix: str = ".txt", cache: bool = True,
              timeout: int = 45, attempts: int = 3) -> tuple[str, Path]:
        """Fetch a URL, archive the body, and return (text, path).

        With `cache=True` an already-archived URL is read from disk. Re-running
        ingestion during development should not re-hit the source.
        """
        doc_id = hashlib.sha256(url.encode()).hexdigest()[:20]
        path = self.raw_dir / f"{doc_id}{suffix}"

        if cache and path.exists():
            return path.read_text(encoding="utf-8", errors="replace"), path

        text = self._get_with_retries(url, timeout=timeout, attempts=attempts)

        path.write_text(text, encoding="utf-8")
        self._record_document(doc_id, url, text, path)
        return text, path

    # Statuses worth trying again. A 5xx or a 429 says the server could not
    # answer right now; a 404 or a 403 says it will not, and repeating those
    # just annoys a free service that owes us nothing.
    RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})

    def _get_with_retries(self, url: str, *, timeout: int, attempts: int) -> str:
        """GET with a bounded retry on transient failures.

        ClubElo returned 502 for three consecutive weekly snapshots and the
        whole source was written off, when a gateway error is the textbook
        case for trying again a moment later.
        """
        last: Exception | None = None

        for attempt in range(1, attempts + 1):
            elapsed = time.monotonic() - self._last_request
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)

            try:
                response = self.session.get(url, timeout=timeout)
                self._last_request = time.monotonic()
                if response.status_code in self.RETRYABLE_STATUSES:
                    response.raise_for_status()
                    raise requests.HTTPError(                 # pragma: no cover
                        f"{response.status_code} for url: {url}", response=response)
                response.raise_for_status()
                return response.text
            except requests.RequestException as exc:
                self._last_request = time.monotonic()
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and status not in self.RETRYABLE_STATUSES:
                    raise
                last = exc
                if attempt < attempts:
                    # 2s, 4s: long enough for a gateway blip, short enough that
                    # a refresh of a dozen URLs does not become a coffee break.
                    time.sleep(self.delay * (2 ** attempt))

        raise RuntimeError(
            f"{url}: still failing after {attempts} attempts — {last}") from last

    def _record_document(self, doc_id: str, url: str, text: str, path: Path) -> None:
        frame = pd.DataFrame([{
            "doc_id": doc_id,
            "source": self.name,
            "url": url,
            "fetched_at": datetime.now(timezone.utc).replace(tzinfo=None),
            "content_hash": hashlib.sha256(text.encode()).hexdigest(),
            "path": str(path),
        }])
        self.store.upsert("raw_document", frame, ["doc_id"])

    # ------------------------------------------------------------- contract

    @abstractmethod
    def ingest(self, **kwargs) -> IngestResult:
        """Fetch, parse and load. Must be idempotent."""


def season_label(start_year: int) -> str:
    """2024 -> '2024-25'."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def season_code(start_year: int) -> str:
    """2024 -> '2425', the football-data.co.uk path segment."""
    return f"{str(start_year)[-2:]}{str(start_year + 1)[-2:]}"


def make_match_id(league: str, season: str, home_team_id: str, away_team_id: str) -> str:
    """Deterministic id so re-ingestion updates rows rather than duplicating.

    Built from the fixture rather than a source's own key, so the same match
    arriving from football-data.co.uk and Understat lands on one row.
    """
    return f"{league}:{season}:{home_team_id}:{away_team_id}"
