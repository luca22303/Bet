from bet.extract.base import ExtractionUnavailable, LLMBackend, closed_schema
from bet.extract.ollama import OllamaBackend
from bet.extract.pipeline import (
    extract_and_verify,
    extract_availability,
    judge_entries,
    to_availability_rows,
)
from bet.extract.schemas import (
    ExtractedAvailability,
    ExtractedStatus,
    ExtractionResult,
    JudgeVerdict,
    TeamAvailabilityReport,
)

__all__ = [
    "ExtractionUnavailable", "LLMBackend", "closed_schema",
    "OllamaBackend",
    "extract_and_verify", "extract_availability", "judge_entries", "to_availability_rows",
    "ExtractedAvailability", "ExtractedStatus", "ExtractionResult",
    "JudgeVerdict", "TeamAvailabilityReport",
]


def default_backend(**kwargs) -> LLMBackend:
    """Ollama, the project default: local, free, no rate limit.

    `ClaudeBackend` is importable from `bet.extract.claude` when a harder
    document warrants it.
    """
    return OllamaBackend(**kwargs)
