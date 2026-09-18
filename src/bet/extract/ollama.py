"""Ollama backend: local, free, schema-constrained.

The default for this project. Extraction is high-volume (every club, every
matchday, all season) and low-difficulty (read a paragraph, fill four fields),
which is the exact shape where a small local model beats a frontier API: the
marginal cost is zero, nothing leaves the machine, and there is no rate limit to
design around.

What makes it reliable is Ollama's `format` parameter, which constrains
generation to a JSON schema at decode time. Without it, a 7B model returns
markdown fences and commentary often enough to be unusable. With it, the output
is valid JSON every time and the only question left is whether the content is
right -- which is what the judge pass is for.

Model choice: `qwen2.5:7b-instruct` is the default because German-language team
news is a meaningful share of the sources here and Qwen handles non-English
input better than similarly sized Llama variants. `llama3.1:8b` works. Below
about 7B, the judge starts rejecting most of what the extractor produces, which
is the system working correctly but not usefully.

    ollama serve
    ollama pull qwen2.5:7b-instruct
"""

from __future__ import annotations

import json
from typing import Any

import requests

from bet.extract.base import ExtractionUnavailable, LLMBackend

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b-instruct"


class OllamaBackend(LLMBackend):
    """Talks to a local Ollama server over its HTTP API.

    Uses `requests` directly rather than the `ollama` package: the API is two
    endpoints and avoiding the dependency keeps the local path installable with
    what the project already has.
    """

    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST,
                 timeout: int = 180, temperature: float = 0.0,
                 num_ctx: int = 8192) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        # Extraction is not a creative task. Temperature 0 makes a rerun on the
        # same article reproduce the same rows, which matters when a pipeline
        # is re-run after a parser fix.
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.name = f"ollama:{model}"

    def health(self) -> tuple[bool, str]:
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.RequestException as exc:
            return False, f"cannot reach Ollama at {self.host}: {exc}"

        available = [m.get("name", "") for m in response.json().get("models", [])]
        if not available:
            return False, f"Ollama is running but has no models; try: ollama pull {self.model}"

        # Ollama reports tags as name:tag; a bare name should still match.
        base = self.model.split(":")[0]
        if not any(m == self.model or m.split(":")[0] == base for m in available):
            return False, (f"model {self.model!r} not installed "
                           f"(available: {', '.join(sorted(available))}); "
                           f"try: ollama pull {self.model}")
        return True, f"{self.name} ready"

    def complete_json(self, system: str, prompt: str, schema: dict[str, Any],
                      *, max_tokens: int = 4000) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            # The schema constrains decoding, so the response is valid JSON by
            # construction rather than by hope.
            "format": schema,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens,
            },
        }

        try:
            response = requests.post(f"{self.host}/api/chat", json=payload,
                                     timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ExtractionUnavailable(
                f"Ollama request failed ({self.host}, model {self.model}): {exc}"
            ) from exc

        content = response.json().get("message", {}).get("content", "")
        if not content:
            raise ExtractionUnavailable("Ollama returned an empty response")

        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            # Should not happen with `format` set, but a truncated response at
            # num_predict will land here and the cause is worth naming.
            raise ValueError(
                f"Ollama returned invalid JSON (possibly truncated at "
                f"{max_tokens} tokens): {content[:200]}"
            ) from exc

    def generate_text(self, system: str, prompt: str, *, max_tokens: int = 2000) -> str:
        """Unconstrained text, for narrating a matchday brief.

        Separate from `complete_json` on purpose: prose wants no schema, and a
        schema-constrained call is the wrong tool for it.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.2, "num_ctx": self.num_ctx,
                        "num_predict": max_tokens},
        }
        try:
            response = requests.post(f"{self.host}/api/chat", json=payload,
                                     timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ExtractionUnavailable(f"Ollama request failed: {exc}") from exc
        return response.json().get("message", {}).get("content", "").strip()
