"""Deterministic gates that sit ABOVE the judge score.

An LLM judge returns a score, and scores are persuasive: a 95 looks safe. But some
failures must block regardless of the number (a refusal returned in place of the output,
a high-severity issue the judge itself listed, a sensitive claim without a primary source,
sources that contradict each other). These checks are plain code: testable without a
network, identical on every run, and impossible to talk out of with a well-written draft.

Gates run in order; the first one that returns a result decides. Only when no gate fires
does the score threshold apply.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

Decision = Literal["accept", "regenerate", "reject"]


@dataclass(frozen=True)
class Issue:
    kind: str
    severity: Literal["low", "medium", "high"] = "medium"
    detail: str = ""


@dataclass(frozen=True)
class Verdict:
    """Normalised judge output. ``score`` is 0 to 100."""

    score: int
    issues: tuple[Issue, ...] = ()
    valid: bool = True

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Verdict:
        """Parse ``{"score": int, "issues": [{"kind", "severity", "detail"}]}``. Invalid -> ``valid=False``."""
        try:
            score = int(data["score"])
            if not 0 <= score <= 100:
                raise ValueError
            issues = tuple(Issue(str(i.get("kind", "?")), str(i.get("severity", "medium")).lower(),  # type: ignore[arg-type]
                                 str(i.get("detail", ""))) for i in data.get("issues") or [])
            return cls(score, issues)
        except (KeyError, TypeError, ValueError, AttributeError):
            return cls(0, (), valid=False)


@dataclass(frozen=True)
class GateContext:
    verdict: Verdict
    text: str
    facts: dict[str, Any] = field(default_factory=dict)  # caller-supplied signals (sensitive, sources...)


@dataclass(frozen=True)
class GateResult:
    decision: Decision
    score: int
    flags: tuple[str, ...] = ()
    reason: str | None = None


Gate = Callable[[GateContext], GateResult | None]

_REFUSAL = re.compile(
    r"(?i)^\s*(?:i(?:'m| am) sorry|i can(?:'t|not) (?:help|assist|comply)|as an ai\b|"
    r"desculpe,? (?:mas )?n[aã]o posso|lo siento,? (?:pero )?no puedo)")


def invalid_verdict(ctx: GateContext) -> GateResult | None:
    """No trustworthy verdict means no acceptance: regenerate and judge again."""
    if not ctx.verdict.valid:
        return GateResult("regenerate", 0, ("invalid_verdict",), "judge output invalid or incomplete")
    return None


def refusal(ctx: GateContext) -> GateResult | None:
    if _REFUSAL.search(ctx.text):
        return GateResult("reject", ctx.verdict.score, ("refusal",), "model refusal returned as output")
    return None


def sensitive_without_primary_source(ctx: GateContext) -> GateResult | None:
    if ctx.facts.get("sensitive") and not ctx.facts.get("primary_source"):
        return GateResult("reject", ctx.verdict.score, ("sensitive",), "sensitive topic without a primary source")
    return None


def conflicting_sources(ctx: GateContext) -> GateResult | None:
    if ctx.facts.get("conflicting_sources"):
        return GateResult("reject", ctx.verdict.score, ("conflict",), "sources disagree and nothing resolves it")
    return None


def high_severity(ctx: GateContext) -> GateResult | None:
    """A high-severity issue listed by the judge wins over the judge's own score.
    One rewrite pointing at the issues; if it persists, the caller rejects."""
    high = [i.kind for i in ctx.verdict.issues if i.severity == "high"]
    if high:
        return GateResult("regenerate", ctx.verdict.score, tuple(high), "high-severity issue reported by the judge")
    return None


DEFAULT_GATES: tuple[Gate, ...] = (invalid_verdict, refusal, sensitive_without_primary_source,
                                   conflicting_sources, high_severity)


def apply_gates(verdict: Verdict, text: str, *, facts: dict[str, Any] | None = None,
                gates: Sequence[Gate] = DEFAULT_GATES, accept_threshold: int = 70) -> GateResult:
    """Pure decision function: gates first, score threshold last."""
    ctx = GateContext(verdict, text, facts or {})
    for gate in gates:
        result = gate(ctx)
        if result is not None:
            return result
    if verdict.score < accept_threshold:
        return GateResult("reject", verdict.score, ("score",), f"score below {accept_threshold}")
    return GateResult("accept", verdict.score)
