"""Extraction pipeline: read team news, verify it, write validated facts.

Three passes, backend-agnostic:

    1.  Extract. Schema-constrained generation turns prose into entries, each
        carrying the sentence it relied on.
    2.  Judge. A second call, given the actual squad list, decides whether each
        entry is genuinely supported. The extractor's incentives are wrong --
        asked to find team news, a model will find team news even in an article
        that contains none -- and a pass that only has to answer "is this
        supported" catches inferences presented as fact. With a small local
        model this step is doing real work, not ceremony.
    3.  Resolve. Accepted names are matched against the real squad. An
        unresolvable name is dropped, never guessed.

Everything rejected is kept. A pipeline that silently discards what it could not
verify gives no way to distinguish a quiet week from a broken extractor.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pandas as pd

from bet.extract.base import LLMBackend, closed_schema
from bet.extract.schemas import (
    ExtractedAvailability,
    ExtractionResult,
    JudgeVerdict,
    TeamAvailabilityReport,
)
from bet.players import resolve_within_squad

EXTRACTION_SYSTEM = """\
You extract player availability facts from football team news. You output JSON \
matching the given schema and nothing else.

Extract only what the source states. If an article mentions a player without \
saying anything about his availability, he is not an entry. An absence of news \
is a valid result: returning zero entries is correct far more often than \
inventing one.

Quote your evidence. The `reasoning` field must contain the actual sentence or \
phrase from the source. If you cannot quote something specific, do not create \
the entry.

Distinguish uncertainty from fact. "Will not travel" is OUT. "Faces a late \
fitness test" is DOUBTFUL. Do not resolve a doubt into a certainty either way.

Calibrate confidence honestly. An official club statement or direct manager \
quote is about 0.95. A named journalist reporting confidently is about 0.7. \
Speculation or an unattributed rumour is 0.4 or below.

Resolve relative dates against the publication date, which is given to you.

Sources may be in German. "verletzt" is injured, "fraglich" is doubtful, \
"fällt aus" means will not play, "gesperrt" is suspended, "angeschlagen" \
means carrying a knock and is usually DOUBTFUL."""

JUDGE_SYSTEM = """\
You verify extracted player availability entries against their source. You \
output JSON matching the given schema and nothing else.

For each entry decide:

Is the quoted reasoning actually present in the source, and does it actually \
support the stated status? A quote that has been paraphrased, stitched together \
or invented fails.

Does the named player appear in the supplied squad list? Match on surname, \
allowing for accents and short forms. A player genuinely not in the squad fails.

Is the status right? If the player and quote are real but the status misreads \
them, give the correct one.

Be strict. A false rejection costs one data point. A false acceptance puts a \
wrong fact into a model that will price bets on it."""


def extract_availability(backend: LLMBackend, text: str, team_name: str,
                         published_at: datetime, *, max_tokens: int = 4000) -> TeamAvailabilityReport:
    """Pass 1: prose to structured entries."""
    prompt = (
        f"Team: {team_name}\n"
        f"Publication date: {published_at.date().isoformat()}\n\n"
        f"Source text:\n{text}\n\n"
        f"Extract every player availability fact this source states about {team_name}. "
        "Return an empty entries list if it states none."
    )
    payload = backend.complete_json(
        EXTRACTION_SYSTEM, prompt, closed_schema(TeamAvailabilityReport),
        max_tokens=max_tokens)

    payload.setdefault("team_name", team_name)
    payload.setdefault("as_of", published_at.isoformat())
    return TeamAvailabilityReport.model_validate(payload)


def judge_entries(backend: LLMBackend, entries: list[ExtractedAvailability],
                  source_text: str, team_name: str, squad_names: list[str],
                  *, max_tokens: int = 3000) -> list[JudgeVerdict]:
    """Pass 2: verify entries against the source and the real squad."""
    if not entries:
        return []

    prompt = (
        f"Squad list for {team_name}:\n"
        + "\n".join(f"- {n}" for n in squad_names)
        + f"\n\nSource text:\n{source_text}\n\nExtracted entries to verify:\n"
        + json.dumps([e.model_dump(mode="json") for e in entries], indent=2)
        + "\n\nReturn exactly one verdict per entry, in the same order."
    )
    schema = {
        "type": "object",
        "properties": {"verdicts": {"type": "array", "items": closed_schema(JudgeVerdict)}},
        "required": ["verdicts"],
        "additionalProperties": False,
    }
    payload = backend.complete_json(JUDGE_SYSTEM, prompt, schema, max_tokens=max_tokens)
    return [JudgeVerdict.model_validate(v) for v in payload.get("verdicts", [])]


def extract_and_verify(backend: LLMBackend, text: str, team_name: str,
                       published_at: datetime, squad_names: list[str],
                       *, min_confidence: float = 0.5, judge: bool = True) -> ExtractionResult:
    """Run the full pipeline and report what survived."""
    report = extract_availability(backend, text, team_name, published_at)
    result = ExtractionResult(report=report, model=backend.name, judged=judge)

    candidates = []
    for entry in report.entries:
        if entry.source_confidence < min_confidence:
            # No point paying for a judge call on something the extractor
            # itself called a rumour.
            result.rejected.append({
                "entry": entry.model_dump(mode="json"),
                "reason": f"confidence {entry.source_confidence:.2f} below {min_confidence:.2f}",
            })
        else:
            candidates.append(entry)

    if not judge or not candidates:
        result.accepted = candidates
        return result

    verdicts = judge_entries(backend, candidates, text, team_name, squad_names)

    if len(verdicts) != len(candidates):
        # A misaligned judge response cannot be mapped onto entries, so nothing
        # is accepted rather than guessing the alignment.
        result.rejected.extend({
            "entry": e.model_dump(mode="json"),
            "reason": f"judge returned {len(verdicts)} verdicts for {len(candidates)} entries",
        } for e in candidates)
        return result

    for entry, verdict in zip(candidates, verdicts):
        if not verdict.supported or not verdict.player_in_squad:
            result.rejected.append({
                "entry": entry.model_dump(mode="json"),
                "reason": verdict.problem or "not supported by the source",
            })
            continue
        if verdict.correct_status and verdict.correct_status != entry.status:
            entry = entry.model_copy(update={"status": verdict.correct_status})
        result.accepted.append(entry)

    return result


def to_availability_rows(result: ExtractionResult, store, team_id: str,
                         as_of: datetime, *, source: str | None = None) -> pd.DataFrame:
    """Turn accepted entries into `player_availability` rows.

    Names resolve against the real squad; anything unresolvable is dropped
    rather than written under an invented id, which would create a player who
    never plays and quietly distort that team's replacement-level estimates.
    """
    if not result.accepted:
        return pd.DataFrame()

    squad = store.squad_as_of(as_of, team_id)
    rows = []
    for entry in result.accepted:
        player_id = resolve_within_squad(entry.player_name, squad)
        if player_id is None:
            result.rejected.append({
                "entry": entry.model_dump(mode="json"),
                "reason": f"could not resolve {entry.player_name!r} to a squad member",
            })
            continue
        rows.append({
            "player_id": player_id,
            "team_id": team_id,
            "source": source or result.model,
            "status": entry.status.value,
            "reason": entry.reason,
            "expected_return": entry.expected_return_date,
            "confidence": entry.source_confidence,
            # Stamped when read, never at the match it refers to: a report is
            # knowable from publication, and backdating it would leak.
            "known_at": as_of,
        })
    return pd.DataFrame(rows)
