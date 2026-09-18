"""Claude backend: optional, for when extraction quality matters more than cost.

Ollama is the default in this project and handles routine team news fine. This
exists for the cases where it does not: a long ambiguous press conference
transcript, a backfill where the extraction will be reused for years, or a
disagreement worth adjudicating with a stronger model.

It implements the same `LLMBackend` interface, so switching is a constructor
change and nothing downstream knows the difference.
"""

from __future__ import annotations

import json
from typing import Any

from bet.extract.base import ExtractionUnavailable, LLMBackend

DEFAULT_MODEL = "claude-opus-5"


class ClaudeBackend(LLMBackend):
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None,
                 client=None) -> None:
        self.model = model
        self.api_key = api_key
        self._client = client
        self.name = f"claude:{model}"

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:
            raise ExtractionUnavailable(
                "the anthropic package is required: pip install anthropic") from exc
        try:
            self._client = (anthropic.Anthropic(api_key=self.api_key)
                            if self.api_key else anthropic.Anthropic())
        except Exception as exc:
            raise ExtractionUnavailable(f"could not construct a Claude client: {exc}") from exc
        return self._client

    def health(self) -> tuple[bool, str]:
        try:
            self._ensure_client()
        except ExtractionUnavailable as exc:
            return False, str(exc)
        return True, f"{self.name} ready"

    def complete_json(self, system: str, prompt: str, schema: dict[str, Any],
                      *, max_tokens: int = 4000) -> dict:
        client = self._ensure_client()
        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            thinking={"type": "adaptive"},
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": prompt}],
        )

        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            raise ExtractionUnavailable(
                f"model declined: {getattr(details, 'category', 'unknown')}")

        for block in response.content:
            if getattr(block, "type", None) == "text":
                try:
                    # Always parsed, never string-matched: escaping in model
                    # output varies between models and versions.
                    return json.loads(block.text)
                except json.JSONDecodeError:
                    continue
        raise ValueError("no JSON object found in the response")

    def generate_text(self, system: str, prompt: str, *, max_tokens: int = 2000) -> str:
        client = self._ensure_client()
        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in response.content
                       if getattr(b, "type", None) == "text").strip()
