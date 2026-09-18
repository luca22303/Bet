"""Extraction pipeline, exercised with a scripted backend.

No Ollama and no network. A fake backend returns canned payloads, which lets the
pipeline's actual logic be tested: confidence filtering, judge verification,
misaligned responses, and name resolution against a real squad.
"""

from datetime import date, datetime

import pandas as pd
import pytest

from bet.extract.base import ExtractionUnavailable, LLMBackend, closed_schema
from bet.extract.ollama import OllamaBackend
from bet.extract.pipeline import extract_and_verify, extract_availability, to_availability_rows
from bet.extract.schemas import (
    ExtractedAvailability,
    ExtractedStatus,
    JudgeVerdict,
    TeamAvailabilityReport,
)

ARTICLE = """
Bayern Munich manager confirmed on Friday that Harry Kane faces a late fitness
test after rolling his ankle in training. Jamal Musiala will not travel, having
picked up a hamstring problem that is expected to keep him out for three weeks.
Joshua Kimmich trained fully and is available.
"""

SQUAD = ["Harry Kane", "Jamal Musiala", "Joshua Kimmich", "Manuel Neuer"]


class ScriptedBackend(LLMBackend):
    """Returns queued payloads in order, recording what it was asked."""

    name = "scripted"

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def complete_json(self, system, prompt, schema, *, max_tokens=4000):
        self.calls.append({"system": system, "prompt": prompt, "schema": schema})
        if not self.payloads:
            raise AssertionError("backend called more times than scripted")
        return self.payloads.pop(0)


def _report_payload(entries):
    return {
        "team_name": "Bayern Munich",
        "as_of": "2026-03-27T10:00:00",
        "entries": entries,
    }


def _entry(name, status, confidence=0.9, reasoning="quoted evidence", **kwargs):
    return {"player_name": name, "status": status, "source_confidence": confidence,
            "reasoning": reasoning, **kwargs}


def _verdict(supported=True, in_squad=True, correct=None, problem=None):
    return {"supported": supported, "player_in_squad": in_squad,
            "correct_status": correct, "problem": problem}


# ------------------------------------------------------------------- schemas


def test_schema_is_closed_against_invented_fields():
    """Matters more for constrained decoding than for validation.

    An open schema lets a small model pad the object with plausible fields.
    """
    assert closed_schema(TeamAvailabilityReport)["additionalProperties"] is False


# ---------------------------------------------------------------- extraction


def test_extraction_parses_into_the_schema():
    backend = ScriptedBackend([_report_payload([
        _entry("Harry Kane", "DOUBTFUL", 0.9, "faces a late fitness test"),
        _entry("Jamal Musiala", "OUT", 0.95, "will not travel"),
    ])])
    report = extract_availability(backend, ARTICLE, "Bayern Munich", datetime(2026, 3, 27))
    assert len(report.entries) == 2
    assert report.entries[0].status is ExtractedStatus.DOUBTFUL
    assert report.entries[1].status is ExtractedStatus.OUT


def test_prompt_carries_the_publication_date():
    """Relative phrases like 'out for three weeks' need an anchor."""
    backend = ScriptedBackend([_report_payload([])])
    extract_availability(backend, ARTICLE, "Bayern Munich", datetime(2026, 3, 27))
    assert "2026-03-27" in backend.calls[0]["prompt"]


def test_empty_extraction_is_a_valid_result():
    """Most articles contain no availability news; inventing one is the failure."""
    backend = ScriptedBackend([_report_payload([])])
    report = extract_availability(backend, "A match preview with no team news.",
                                  "Bayern Munich", datetime(2026, 3, 27))
    assert report.entries == []


# -------------------------------------------------------------- verification


def test_low_confidence_entries_are_dropped_before_judging():
    """No point paying for a judge call on something called a rumour."""
    backend = ScriptedBackend([
        _report_payload([
            _entry("Harry Kane", "OUT", 0.95),
            _entry("Manuel Neuer", "OUT", 0.2),
        ]),
        {"verdicts": [_verdict()]},
    ])
    result = extract_and_verify(backend, ARTICLE, "Bayern Munich",
                                datetime(2026, 3, 27), SQUAD, min_confidence=0.5)
    assert len(result.accepted) == 1
    assert len(result.rejected) == 1
    assert "confidence" in result.rejected[0]["reason"]


def test_judge_rejects_unsupported_entries():
    """The extractor's incentives are wrong; the judge's are not."""
    backend = ScriptedBackend([
        _report_payload([_entry("Harry Kane", "OUT", 0.9)]),
        {"verdicts": [_verdict(supported=False, problem="source says doubtful, not out")]},
    ])
    result = extract_and_verify(backend, ARTICLE, "Bayern Munich",
                                datetime(2026, 3, 27), SQUAD)
    assert result.accepted == []
    assert "doubtful" in result.rejected[0]["reason"]


def test_judge_rejects_players_not_in_the_squad():
    """A hallucinated player is the failure this pass exists to catch."""
    backend = ScriptedBackend([
        _report_payload([_entry("Erling Haaland", "OUT", 0.9)]),
        {"verdicts": [_verdict(in_squad=False, problem="not in the squad list")]},
    ])
    result = extract_and_verify(backend, ARTICLE, "Bayern Munich",
                                datetime(2026, 3, 27), SQUAD)
    assert result.accepted == []


def test_judge_can_correct_a_status():
    backend = ScriptedBackend([
        _report_payload([_entry("Harry Kane", "OUT", 0.9)]),
        {"verdicts": [_verdict(correct="DOUBTFUL")]},
    ])
    result = extract_and_verify(backend, ARTICLE, "Bayern Munich",
                                datetime(2026, 3, 27), SQUAD)
    assert result.accepted[0].status is ExtractedStatus.DOUBTFUL


def test_misaligned_judge_response_rejects_everything():
    """Verdicts that cannot be mapped to entries must not be guessed at."""
    backend = ScriptedBackend([
        _report_payload([_entry("Harry Kane", "OUT"), _entry("Jamal Musiala", "OUT")]),
        {"verdicts": [_verdict()]},          # one verdict for two entries
    ])
    result = extract_and_verify(backend, ARTICLE, "Bayern Munich",
                                datetime(2026, 3, 27), SQUAD)
    assert result.accepted == []
    assert len(result.rejected) == 2


def test_judging_can_be_skipped():
    backend = ScriptedBackend([_report_payload([_entry("Harry Kane", "OUT", 0.9)])])
    result = extract_and_verify(backend, ARTICLE, "Bayern Munich",
                                datetime(2026, 3, 27), SQUAD, judge=False)
    assert len(result.accepted) == 1
    assert not result.judged


def test_rejections_are_kept_not_discarded():
    """A pipeline that silently drops what it cannot verify hides its own failure."""
    backend = ScriptedBackend([
        _report_payload([_entry("Harry Kane", "OUT", 0.1)]),
    ])
    result = extract_and_verify(backend, ARTICLE, "Bayern Munich",
                                datetime(2026, 3, 27), SQUAD)
    assert result.rejected
    assert result.acceptance_rate == 0.0


# ----------------------------------------------------------------- resolution


def test_accepted_entries_resolve_to_real_players(store_with_players):
    store = store_with_players
    as_of = datetime(2024, 1, 1)
    team = "bayern_munich"
    squad = store.squad_as_of(as_of, team)
    name = next(iter(squad.values()))

    backend = ScriptedBackend([
        _report_payload([_entry(name, "OUT", 0.95)]),
        {"verdicts": [_verdict()]},
    ])
    result = extract_and_verify(backend, ARTICLE, team, as_of, list(squad.values()))
    rows = to_availability_rows(result, store, team, as_of)

    assert len(rows) == 1
    assert rows.iloc[0]["player_id"] in squad
    assert rows.iloc[0]["status"] == "OUT"
    # Stamped when read, never backdated to the match it refers to.
    assert rows.iloc[0]["known_at"] == as_of


def test_unresolvable_names_are_dropped_and_recorded(store_with_players):
    """Writing an invented player id would create a player who never plays."""
    store = store_with_players
    as_of = datetime(2024, 1, 1)
    backend = ScriptedBackend([
        _report_payload([_entry("Cristiano Ronaldo", "OUT", 0.95)]),
        {"verdicts": [_verdict()]},
    ])
    squad = store.squad_as_of(as_of, "bayern_munich")
    result = extract_and_verify(backend, ARTICLE, "bayern_munich", as_of, list(squad.values()))
    rows = to_availability_rows(result, store, "bayern_munich", as_of)

    assert rows.empty
    assert any("resolve" in r["reason"] for r in result.rejected)


# -------------------------------------------------------------------- ollama


def test_ollama_health_reports_a_useful_error_when_absent():
    backend = OllamaBackend(host="http://localhost:1")
    healthy, message = backend.health()
    assert not healthy
    assert "cannot reach Ollama" in message


def test_ollama_backend_names_itself_by_model():
    assert OllamaBackend(model="llama3.1:8b").name == "ollama:llama3.1:8b"
