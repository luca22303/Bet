"""Schemas for turning unstructured team news into validated facts.

Every field here exists to make a specific failure mode impossible or visible.

`player_name` is free text because a news report will not know your player ids;
it is resolved against the actual squad in a later step, and an unresolvable
name is dropped rather than invented.

`source_confidence` and `reasoning` are required because an extraction with no
stated basis is indistinguishable from a hallucination, and the judge needs
something to check. A model that must quote its evidence hallucinates less than
one that only has to state a conclusion.

`expected_return_date` is a date rather than a duration in days, because "out
for two weeks" is ambiguous about when the clock started and resolving it at
extraction time, against the article's own date, is more reliable than doing it
downstream.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class ExtractedStatus(str, Enum):
    FIT = "FIT"
    DOUBTFUL = "DOUBTFUL"
    OUT = "OUT"
    SUSPENDED = "SUSPENDED"


class ExtractedAvailability(BaseModel):
    """One player's availability, as stated by a source."""

    player_name: str = Field(
        description="Player name exactly as written in the source text.")
    status: ExtractedStatus = Field(
        description=("OUT if the player will definitely not play. DOUBTFUL if the "
                     "source expresses uncertainty ('a doubt', 'will be assessed', "
                     "'race against time'). SUSPENDED for bans or red-card "
                     "suspensions. FIT only if the source says a previously "
                     "doubtful or injured player is now available."))
    reason: str | None = Field(
        default=None,
        description="The injury, illness or suspension, in a few words.")
    expected_return_date: date | None = Field(
        default=None,
        description=("Resolved calendar date the player is expected back, if the "
                     "source gives one. Resolve relative phrases such as 'out for "
                     "three weeks' against the article's publication date."))
    source_confidence: float = Field(
        ge=0.0, le=1.0,
        description=("How firmly the source states this. 0.95 for an official club "
                     "statement or a manager quote, 0.7 for a confident report by a "
                     "named journalist, 0.4 for speculation or an unsourced rumour."))
    reasoning: str = Field(
        description=("The specific sentence or phrase from the source that supports "
                     "this. Quote it. Do not paraphrase and do not infer beyond it."))

    @field_validator("player_name")
    @classmethod
    def name_must_be_substantive(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) < 2:
            raise ValueError("player_name is too short to resolve")
        return cleaned


class TeamAvailabilityReport(BaseModel):
    """Everything extracted from one source document about one team."""

    team_name: str = Field(description="Club name as written in the source.")
    as_of: datetime = Field(description="Publication time of the source.")
    entries: list[ExtractedAvailability] = Field(default_factory=list)

    @field_validator("entries")
    @classmethod
    def no_duplicate_players(cls, entries: list[ExtractedAvailability]):
        seen = set()
        unique = []
        for entry in entries:
            key = entry.player_name.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(entry)
        return unique


class JudgeVerdict(BaseModel):
    """A second model's assessment of one extracted entry.

    The judge exists because the extractor's incentives are wrong: asked to
    find team news, a model will find team news, including in an article that
    contains none. A separate pass that only has to answer 'is this actually
    supported' catches inferences the extractor presented as facts.
    """

    supported: bool = Field(
        description=("True only if the quoted reasoning genuinely appears in the "
                     "source and genuinely supports the stated status."))
    player_in_squad: bool = Field(
        description="True if the named player appears in the supplied squad list.")
    correct_status: ExtractedStatus | None = Field(
        default=None,
        description="If the status is wrong but the player is real, the right one.")
    problem: str | None = Field(
        default=None,
        description="What is wrong with this entry, if anything.")


class ExtractionResult(BaseModel):
    """Extraction plus verification, with everything that was rejected."""

    report: TeamAvailabilityReport
    accepted: list[ExtractedAvailability] = Field(default_factory=list)
    rejected: list[dict] = Field(default_factory=list)
    model: str = ""
    judged: bool = False

    @property
    def acceptance_rate(self) -> float:
        total = len(self.report.entries)
        return len(self.accepted) / total if total else 0.0

    def summary(self) -> str:
        return (f"{self.report.team_name}: {len(self.accepted)} accepted, "
                f"{len(self.rejected)} rejected "
                f"({'judged' if self.judged else 'unjudged'})")
