"""Anthropic adapter (Messages API, streamed). SDK imported lazily; key from ANTHROPIC_API_KEY.

Streaming avoids HTTP timeouts on long outputs: a timeout that fires after the provider
already generated the answer still bills, and a blind retry pays twice. Models that
reason before answering spend ``max_tokens`` on reasoning too, so ``reasoning_headroom``
is added for the models listed in ``reasoning_models`` (prefix match). The router's
reservation must then be quoted with the larger value, which is why the headroom is
explicit configuration rather than hidden inside the adapter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm_router.providers.base import Completion, add_tokens, api_key_from_env


@dataclass
class AnthropicProvider:
    name: str = "anthropic"
    api_key_env: str = "ANTHROPIC_API_KEY"
    timeout_s: float = 300.0
    reasoning_models: tuple[str, ...] = field(default_factory=tuple)
    reasoning_headroom: int = 0
    effort: str | None = None  # e.g. "low"; only sent to reasoning models

    def _reasons(self, model: str) -> bool:
        return bool(self.reasoning_models) and model.startswith(self.reasoning_models)

    def billable_max_tokens(self, model: str, max_tokens: int) -> int:
        """Output ceiling actually sent to the API; the router quotes the reservation with it."""
        return max_tokens + (self.reasoning_headroom if self._reasons(model) else 0)

    def request_kwargs(self, model: str, max_tokens: int) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"max_tokens": self.billable_max_tokens(model, max_tokens)}
        if self._reasons(model) and self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        return kwargs

    def complete(self, model: str, system: str, user: str, *, max_tokens: int) -> Completion:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key_from_env(self.api_key_env), max_retries=0,
                                     timeout=self.timeout_s)
        with client.messages.stream(model=model, system=system, messages=[{"role": "user", "content": user}],
                                    **self.request_kwargs(model, max_tokens)) as stream:
            resp = stream.get_final_message()
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        usage = getattr(resp, "usage", None)
        tokens_in = getattr(usage, "input_tokens", None)
        # Cached input is still input for budgeting purposes (conservative).
        for attr in ("cache_creation_input_tokens", "cache_read_input_tokens"):
            tokens_in = add_tokens(tokens_in, getattr(usage, attr, 0))
        return Completion(text, tokens_in, getattr(usage, "output_tokens", None))
