"""Google Gemini adapter (google-genai SDK). SDK imported lazily; key from GOOGLE_API_KEY.

Reasoning ("thinking") tokens are billed as output but reported separately, so they are
added to ``output_tokens``. If they were ignored the ledger would under-estimate cost and
the settle step would look cheaper than the invoice.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from llm_router.providers.base import Completion, add_tokens, api_key_from_env


@dataclass
class GoogleProvider:
    name: str = "google"
    api_key_env: str = "GOOGLE_API_KEY"
    timeout_ms: int = 90_000
    reasoning_models: tuple[str, ...] = field(default_factory=tuple)
    reasoning_headroom: int = 0
    thinking_level: str | None = None  # e.g. "low"

    def _reasons(self, model: str) -> bool:
        return bool(self.reasoning_models) and model.startswith(self.reasoning_models)

    def billable_max_tokens(self, model: str, max_tokens: int) -> int:
        """Output ceiling actually sent to the API; the router quotes the reservation with it."""
        return max_tokens + (self.reasoning_headroom if self._reasons(model) else 0)

    def complete(self, model: str, system: str, user: str, *, max_tokens: int) -> Completion:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key_from_env(self.api_key_env),
                              http_options={"timeout": self.timeout_ms, "retry_options": {"attempts": 1}})
        extra = {}
        if self._reasons(model) and self.thinking_level:
            extra["thinking_config"] = types.ThinkingConfig(thinking_level=self.thinking_level)
        max_tokens = self.billable_max_tokens(model, max_tokens)
        resp = client.models.generate_content(
            model=model, contents=user,
            config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=max_tokens, **extra))
        meta = getattr(resp, "usage_metadata", None)
        out = add_tokens(getattr(meta, "candidates_token_count", None), getattr(meta, "thoughts_token_count", 0))
        return Completion(resp.text or "", getattr(meta, "prompt_token_count", None), out)
