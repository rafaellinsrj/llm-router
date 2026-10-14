"""Provider interface shared by every adapter."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Completion:
    """Raw text plus token usage. ``None`` usage means "unknown", never "zero"."""

    text: str
    input_tokens: int | None
    output_tokens: int | None

    def usage(self) -> dict[str, int | None]:
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens}


@runtime_checkable
class Provider(Protocol):
    name: str

    def complete(self, model: str, system: str, user: str, *, max_tokens: int) -> Completion:
        """One request, no internal retries (the router owns retries and backoff)."""
        ...


class MissingAPIKey(RuntimeError):
    pass


def api_key_from_env(var: str) -> str:
    """API keys come only from the environment; they are never passed through config objects."""
    value = os.environ.get(var, "").strip()
    if not value:
        raise MissingAPIKey(f"environment variable {var} is not set")
    return value


def add_tokens(base: int | None, extra: object) -> int | None:
    """Add an optional token counter (cache, reasoning) without turning unknown into zero."""
    extra = extra or 0
    if type(base) is int and type(extra) is int and extra >= 0:
        return base + extra
    return None if extra else base
