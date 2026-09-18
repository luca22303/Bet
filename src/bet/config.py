"""Project-wide paths and constants.

Everything that a user might reasonably want to change lives here rather than
being scattered through the ingest and evaluation modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("BET_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "db" / "bet.duckdb"

# The three 1X2 outcomes, in the fixed order used by every probability vector,
# every odds triplet and every metric in this codebase. Order matters: the
# ranked probability score treats these as ordered categories.
OUTCOMES = ("H", "D", "A")

# football-data.co.uk league codes.
FOOTBALL_DATA_LEAGUES = {"bundesliga": "D1", "bundesliga2": "D2"}

# Germany levies a tax on sports betting stakes under the
# Rennwett- und Lotteriegesetz. Confirm how your operator actually applies it
# to your account before trusting any expected-value number.
GERMAN_STAKE_TAX = 0.053


@dataclass(frozen=True)
class Settings:
    db_path: Path = DB_PATH
    raw_dir: Path = RAW_DIR
    # Seconds before kickoff at which a model is allowed to see the world.
    # Confirmed line-ups land roughly an hour out; predicting earlier than this
    # is a deliberate choice, not an accident.
    prediction_lead_seconds: int = 60 * 60
    # How long after kickoff a result becomes public knowledge.
    result_known_after_seconds: int = 60 * 115
    user_agent: str = "bet/0.1 (personal analytics research)"
    request_delay_seconds: float = 3.0
    leagues: tuple[str, ...] = field(default_factory=lambda: ("bundesliga",))

    def ensure_dirs(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)


SETTINGS = Settings()
