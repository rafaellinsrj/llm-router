"""OpenAI adapter (Chat Completions, JSON mode). SDK imported lazily; key from OPENAI_API_KEY."""
from __future__ import annotations

from dataclasses import dataclass

from llm_router.providers.base import Completion, api_key_from_env


@dataclass
class OpenAIProvider:
    name: str = "openai"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_s: float = 90.0
    json_mode: bool = True

    def complete(self, model: str, system: str, user: str, *, max_tokens: int) -> Completion:
        from openai import OpenAI

        client = OpenAI(api_key=api_key_from_env(self.api_key_env), max_retries=0, timeout=self.timeout_s)
        extra = {"response_format": {"type": "json_object"}} if self.json_mode else {}
        resp = client.chat.completions.create(
            model=model, max_completion_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **extra)
        usage = getattr(resp, "usage", None)
        return Completion(resp.choices[0].message.content or "",
                          getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None))
