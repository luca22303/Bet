"""Provider-agnostic interface for the extraction layer.

Extraction is the one place in this project where a language model earns its
place: turning "Kane faces a late fitness test" into a row a statistical model
can use. It is a small, well-bounded job, and a small local model does it fine
provided two things are true.

The output must be schema-constrained at decode time, not merely validated
afterwards. A 7B model asked politely for JSON will produce prose, markdown
fences and trailing commentary; the same model decoding against a grammar
produces valid JSON every time. Ollama's `format` parameter does this, which is
what makes the local path viable rather than merely cheap.

And the judge pass matters more with a small model, not less. A smaller model
infers more freely from thin evidence, so the verification step is doing real
work here rather than acting as a formality.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ExtractionUnavailable(RuntimeError):
    """Raised when a backend cannot be reached or is not installed."""


class LLMBackend(ABC):
    """Minimal contract: constrained JSON out, given a system prompt and a prompt."""

    name: str = "backend"

    @abstractmethod
    def complete_json(self, system: str, prompt: str, schema: dict[str, Any],
                      *, max_tokens: int = 4000) -> dict:
        """Return a JSON object conforming to `schema`."""

    def health(self) -> tuple[bool, str]:
        """Whether the backend is reachable, and why not if it is not."""
        return True, "assumed available"

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


def closed_schema(model_class) -> dict[str, Any]:
    """JSON schema for a Pydantic model, closed against invented fields.

    `additionalProperties: false` matters more for constrained decoding than for
    validation: it stops a model padding the object with plausible-looking
    fields that would then be silently discarded.
    """
    schema = model_class.model_json_schema()
    schema["additionalProperties"] = False
    return schema
