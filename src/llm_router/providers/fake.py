"""Deterministic in-process provider for tests and demos. Never touches the network."""
from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from llm_router.providers.base import Completion

Response = str | dict[str, Any] | BaseException | Completion
Responder = Callable[[str, str, str], Response]


@dataclass
class FakeCall:
    model: str
    system: str
    user: str
    max_tokens: int


@dataclass
class FakeProvider:
    """Scripted provider.

    ``script`` maps a model name to a list of responses consumed in order (the last one
    repeats), or to a callable ``(model, system, user) -> response``. A response may be a
    dict (returned as JSON), a raw string, a ``Completion`` or an exception to raise.
    Usage is estimated as ``len(text) // 4`` so costs are realistic and deterministic.
    """

    name: str
    script: dict[str, list[Response] | Responder] = field(default_factory=dict)
    default: Response = field(default_factory=lambda: {"ok": True})
    calls: list[FakeCall] = field(default_factory=list)

    def complete(self, model: str, system: str, user: str, *, max_tokens: int) -> Completion:
        self.calls.append(FakeCall(model, system, user, max_tokens))
        entry = self.script.get(model, [self.default])
        if callable(entry):
            response = entry(model, system, user)
        else:
            queue = entry
            response = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, Completion):
            return response
        text = json.dumps(response, ensure_ascii=False) if isinstance(response, dict) else str(response)
        return Completion(text, max(1, (len(system) + len(user)) // 4), max(1, len(text) // 4))

    def models_called(self) -> list[str]:
        return [c.model for c in self.calls]


def fake_providers(names: Iterable[str] = ("openai", "anthropic", "google")) -> dict[str, FakeProvider]:
    return {n: FakeProvider(n) for n in names}
