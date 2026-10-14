"""Typed configuration: stages, routes and the router settings."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum


class Stage(StrEnum):
    """Generic pipeline stages. Any string works as a stage; these are the common ones."""

    EXTRACT = "extract"        # pull structured facts out of raw input (cheap, fast models)
    SYNTHESIZE = "synthesize"  # write the main output (quality matters most)
    VERIFY = "verify"          # judge the synthesis; must run on a different provider
    SUMMARIZE = "summarize"    # short derived texts
    CLASSIFY = "classify"      # labels, routing hints


def stage_name(stage: Stage | str) -> str:
    return stage.value if isinstance(stage, Stage) else str(stage)


@dataclass(frozen=True, order=True)
class Route:
    """One (provider, model) pair. Rendered as ``provider/model``."""

    provider: str
    model: str

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.model.strip():
            raise ValueError("provider and model are required")

    @classmethod
    def parse(cls, value: str | Route | tuple[str, str]) -> Route:
        if isinstance(value, Route):
            return value
        if isinstance(value, tuple):
            return cls(*value)
        provider, sep, model = value.partition("/")
        if not sep:
            raise ValueError(f"route must look like 'provider/model': {value!r}")
        return cls(provider, model)

    def __str__(self) -> str:
        return f"{self.provider}/{self.model}"


def dedupe(routes: Iterable[Route]) -> list[Route]:
    """Remove repeated routes, preserving the first occurrence."""
    seen: set[Route] = set()
    out: list[Route] = []
    for route in routes:
        if route not in seen:
            seen.add(route)
            out.append(route)
    return out


@dataclass
class RouterConfig:
    """All knobs of the router in one typed object (no environment lookups inside the library).

    chains:
        Ordered fallback chain per stage. The selector may reorder it; it never adds
        routes that are not in the chain unless a tournament approved them.
    verify_stages:
        Stages that act as a judge. For them the router refuses to fall back to the
        provider being judged unless ``allow_degraded_verification`` is True.
    """

    chains: dict[str, list[Route]]
    retries: int = 3
    backoff_s: float = 2.0
    max_tokens: int = 4096
    daily_budget_usd: Decimal = Decimal("20")
    stage_budgets_usd: dict[str, Decimal] = field(default_factory=dict)
    verify_stages: frozenset[str] = frozenset({Stage.VERIFY.value})
    allow_degraded: bool = True
    allow_degraded_verification: bool = False

    def __post_init__(self) -> None:
        self.chains = {stage_name(k): dedupe(Route.parse(r) for r in v) for k, v in self.chains.items()}
        self.stage_budgets_usd = {stage_name(k): Decimal(v) for k, v in self.stage_budgets_usd.items()}
        self.daily_budget_usd = Decimal(self.daily_budget_usd)
        self.verify_stages = frozenset(stage_name(s) for s in self.verify_stages)
        if self.retries < 1:
            raise ValueError("retries must be >= 1")
        if self.backoff_s < 0:
            raise ValueError("backoff_s must be >= 0")
        if self.daily_budget_usd <= 0:
            raise ValueError("daily_budget_usd must be positive")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> RouterConfig:
        """Build from a plain dict (e.g. parsed TOML/JSON). Chains use 'provider/model' strings."""
        data = dict(raw)
        chains = {k: [Route.parse(r) for r in v] for k, v in dict(data.pop("chains")).items()}  # type: ignore[arg-type]
        if "daily_budget_usd" in data:
            data["daily_budget_usd"] = Decimal(str(data["daily_budget_usd"]))
        if "stage_budgets_usd" in data:
            data["stage_budgets_usd"] = {k: Decimal(str(v)) for k, v in dict(data["stage_budgets_usd"]).items()}  # type: ignore[arg-type]
        if "verify_stages" in data:
            data["verify_stages"] = frozenset(data["verify_stages"])  # type: ignore[arg-type]
        return cls(chains=chains, **data)  # type: ignore[arg-type]

    def chain(self, stage: Stage | str) -> list[Route]:
        name = stage_name(stage)
        if name not in self.chains:
            raise KeyError(f"no chain configured for stage {name!r}")
        return list(self.chains[name])

    def stage_cap(self, stage: Stage | str) -> Decimal:
        return self.stage_budgets_usd.get(stage_name(stage), self.daily_budget_usd)
