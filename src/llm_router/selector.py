"""Model selector: learn which model should run each task from measured outcomes.

The selector never calls a model. It only reorders a chain the router already allows
(or, when a tournament approved models, picks from that approved pool). Layers, in order:

1. Hard eligibility gates (deterministic): disabled model, wrong language or modality,
   excluded provider, above the cost cap, or a sensitive request and a model below the
   quality/reliability floor. A gated model is never ranked, however good its score.
2. Rejection lock (deterministic): a model rejected downstream in ``block_rate`` or more
   of at least ``block_min`` pieces is demoted to the end of the chain (kept only as a
   last-resort fallback).
3. Ranking by reward: reward = ``quality_weight`` * judge quality + the rest * engagement
   percentile (engagement only counts once the piece is mature). Among models with enough
   samples whose reward is within ``cheaper_margin`` of the best, the cheapest wins.
   Price is the *effective* price: price per attempt divided by the approval rate, so a
   model that is often rejected is as expensive as it really is.
4. Exploration: ``explore_rate`` of the calls (``cold_explore_rate`` while some candidate
   has fewer than ``min_sample`` samples) go to another candidate, so the statistics keep
   improving and a new model gets a fair chance.
"""
from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from llm_router.config import Route, Stage, stage_name
from llm_router.errors import SelectorError
from llm_router.storage import MetricRow, Storage

PriceFn = Callable[[Route], float]


@dataclass(frozen=True)
class SelectorConfig:
    min_sample: int = 15            # samples for a model's score to be trusted
    provisional_sample: int = 3     # samples for a provisional default when nobody is "ready"
    explore_rate: float = 0.15
    cold_explore_rate: float = 0.30
    cheaper_margin: float = 0.05    # "cheapest wins if quality is within 5 points"
    clear_win_margin: float = 0.10  # a provisional model this far ahead wins right away
    quality_weight: float = 0.4     # rest of the reward is engagement
    mature_hours: int = 72          # engagement counts only after this
    segment_min_sample: int = 8     # segment (e.g. language/topic) score replaces the broad one
    block_rate: float = 0.70        # rejection lock
    block_min: int = 30
    reject_weight: float = 1.0      # how much the rejection rate discounts the reward
    min_approval: float = 0.2       # floor for the effective price denominator

    def __post_init__(self) -> None:
        for name in ("explore_rate", "cold_explore_rate", "cheaper_margin", "clear_win_margin",
                     "quality_weight", "reject_weight", "min_approval"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise SelectorError(f"{name} must be between 0 and 1")


@dataclass(frozen=True)
class ModelStats:
    reward: float
    n: int


@dataclass(frozen=True)
class Selection:
    chain: list[Route]
    chosen: Route
    reason: str
    explored: bool = False
    gated: tuple[str, ...] = ()
    demoted: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"chosen": str(self.chosen), "reason": self.reason, "explored": self.explored,
                "chain": [str(r) for r in self.chain], "gated": list(self.gated),
                "demoted": list(self.demoted)}


# --------------------------------------------------------------------- 1. hard gates

@dataclass(frozen=True)
class CandidateProfile:
    """Static capabilities of a model. Used by the deterministic eligibility gates."""

    route: Route
    languages: tuple[str, ...] = ("*",)
    modalities: tuple[str, ...] = ("text",)
    quality: Decimal = Decimal("0.5")
    reliability: Decimal = Decimal("0.5")
    enabled: bool = True

    def __post_init__(self) -> None:
        for value in (self.quality, self.reliability):
            if not isinstance(value, Decimal) or not value.is_finite() or not 0 <= value <= 1:
                raise SelectorError("quality and reliability must be Decimals between 0 and 1")
        if not self.languages or not self.modalities:
            raise SelectorError("languages and modalities must be non-empty")


@dataclass(frozen=True)
class SelectionRequest:
    language: str = "*"
    modality: str = "text"
    sensitive: bool = False
    max_unit_price: float | None = None
    excluded_providers: tuple[str, ...] = ()
    sensitive_floor: Decimal = Decimal("0.8")


def gate_reason(profile: CandidateProfile, request: SelectionRequest, price: PriceFn | None = None) -> str | None:
    """Why the model is not eligible (``None`` = eligible). Order is fixed and deterministic."""
    if not profile.enabled:
        return "disabled"
    if request.language not in profile.languages and "*" not in profile.languages and request.language != "*":
        return "language"
    if request.modality not in profile.modalities:
        return "modality"
    if profile.route.provider in request.excluded_providers:
        return "excluded_provider"
    if request.max_unit_price is not None and price is not None and price(profile.route) > request.max_unit_price:
        return "cost_cap"
    if request.sensitive and (profile.quality < request.sensitive_floor
                              or profile.reliability < request.sensitive_floor):
        return "sensitive_floor"
    return None


# --------------------------------------------------------------------- 2. rejection lock

def blocked_by_rejection(rates: Mapping[Route, tuple[float, int]], config: SelectorConfig,
                         exempt: Iterable[Route] = ()) -> set[Route]:
    """Routes rejected in ``block_rate`` or more of at least ``block_min`` pieces.

    ``exempt`` lets a model under an explicit evaluation period keep competing.
    """
    skip = set(exempt)
    return {route for route, (rate, total) in rates.items()
            if route not in skip and total >= config.block_min and rate >= config.block_rate}


def demote(chain: Sequence[Route], blocked: set[Route]) -> list[Route]:
    """Blocked routes go to the end; if every route is blocked the chain is unchanged."""
    keep = [r for r in chain if r not in blocked]
    return (keep + [r for r in chain if r in blocked]) if keep and blocked else list(chain)


def effective_price(base: PriceFn, rejection_rate: Mapping[Route, float], config: SelectorConfig) -> PriceFn:
    """Cost per *accepted* output: price per attempt / approval rate (floored)."""
    def price(route: Route) -> float:
        approval = max(config.min_approval, 1.0 - rejection_rate.get(route, 0.0))
        return base(route) / approval
    return price


# --------------------------------------------------------------------- 3 + 4. ranking

def pick(chain: Sequence[Route], stats: Mapping[Route, ModelStats], *, config: SelectorConfig,
         price: PriceFn, pool: Sequence[Route] | None = None,
         rng: random.Random | None = None) -> Selection:
    """Reorder ``chain`` so the chosen route goes first. ``pool`` restricts the candidates."""
    rng = rng or random.Random()
    chain = list(chain)
    if not chain:
        raise SelectorError("empty chain")
    candidates = list(pool) if pool else list(chain)
    if len(candidates) < 2:
        only = candidates[0] if candidates else chain[0]
        return Selection([only, *[r for r in chain if r != only]], only, "not enough candidates")
    sampled = {c: stats.get(c, ModelStats(0.0, 0)) for c in candidates}
    cold = [c for c, s in sampled.items() if s.n < config.min_sample]
    ready = [c for c, s in sampled.items() if s.n >= config.min_sample]
    provisional = [c for c, s in sampled.items() if s.n >= config.provisional_sample]

    if ready:
        best = max(ready, key=lambda c: sampled[c].reward)
        top = sampled[best].reward
        close = [c for c in ready if sampled[c].reward >= top - config.cheaper_margin]
        best = min(close, key=lambda c: (price(c), -sampled[c].reward, str(c)))
        reason = "best reward" if best == max(ready, key=lambda c: sampled[c].reward) else "cheaper within margin"
        challengers = [c for c in provisional if c not in ready
                       and sampled[c].reward > sampled[best].reward + config.clear_win_margin]
        if challengers:
            best = max(challengers, key=lambda c: sampled[c].reward)
            reason = "provisional clear win"
    elif provisional:
        best = max(provisional, key=lambda c: sampled[c].reward)
        reason = "best provisional reward"
    else:
        best = candidates[0]
        reason = "default (no data)"

    explored = False
    if rng.random() < (config.cold_explore_rate if cold else config.explore_rate):
        others = [c for c in cold if c != best] or [c for c in candidates if c != best]
        if others:
            best, reason, explored = rng.choice(others), "exploring", True
    return Selection([best, *[r for r in chain if r != best]], best, reason, explored)


# --------------------------------------------------------------------- rewards

@dataclass(frozen=True)
class Sample:
    """One produced piece: who wrote it, its judge quality (0..1) and later engagement."""

    route: Route
    segment: str
    quality: float
    created_at: datetime
    engagement: float | None = None
    subsegment: str | None = None  # e.g. topic; counted in "segment/subsegment" too
    cohort: str | None = None      # engagement percentile is computed inside a cohort


def _percentile(value: float, population: Sequence[float]) -> float:
    if not population:
        return 0.5
    below = sum(1 for v in population if v < value)
    equal = sum(1 for v in population if v == value)
    return (below + 0.5 * equal) / len(population)


def compute_rewards(samples: Iterable[Sample], config: SelectorConfig,
                    now: datetime | None = None) -> dict[tuple[Route, str], dict[str, float]]:
    """Mean reward per (route, segment). Engagement is a percentile within its cohort,
    so audiences of different sizes are comparable; immature pieces count quality only."""
    now = now or datetime.now(UTC)
    cutoff = now - timedelta(hours=config.mature_hours)
    samples = list(samples)
    by_cohort: dict[str, list[float]] = {}
    for s in samples:
        if s.created_at <= cutoff and s.engagement is not None:
            by_cohort.setdefault(s.cohort or s.segment, []).append(s.engagement)
    out: dict[tuple[Route, str], dict[str, float]] = {}
    for s in samples:
        if not 0.0 <= s.quality <= 1.0:
            raise SelectorError("quality must be between 0 and 1")
        mature = s.created_at <= cutoff and s.engagement is not None
        if mature:
            eng = _percentile(s.engagement, by_cohort.get(s.cohort or s.segment, []))  # type: ignore[arg-type]
            reward = config.quality_weight * s.quality + (1 - config.quality_weight) * eng
        else:
            reward = s.quality
        for seg in [s.segment, *([f"{s.segment}/{s.subsegment}"] if s.subsegment else [])]:
            acc = out.setdefault((s.route, seg), {"n": 0, "sum": 0.0, "quality_sum": 0.0, "mature": 0})
            acc["n"] += 1
            acc["sum"] += reward
            acc["quality_sum"] += s.quality
            acc["mature"] += int(mature)
    for acc in out.values():
        acc["reward"] = acc["sum"] / acc["n"]
        acc["quality"] = acc["quality_sum"] / acc["n"]
    return out


def merge_segment_stats(broad: Mapping[Route, ModelStats], narrow: Mapping[Route, ModelStats],
                        config: SelectorConfig) -> dict[Route, ModelStats]:
    """A narrow segment score replaces the broad one once it has enough samples."""
    merged = dict(broad)
    for route, stat in narrow.items():
        if stat.n >= config.segment_min_sample:
            merged[route] = ModelStats(stat.reward, max(stat.n, broad.get(route, ModelStats(0, 0)).n))
    return merged


# --------------------------------------------------------------------- facade

class Selector:
    """Stateful facade used by the router: reads metrics from storage and picks a route.

    ``stage_tasks`` says which stages are learned (others pass through unchanged).
    ``approved`` is an optional callback (usually ``Tournament.approved``) returning the
    pool allowed to compete for a task and segment, or ``None`` when there is no recent
    tournament (then the configured chain is the pool).
    """

    def __init__(self, storage: Storage, price: PriceFn, config: SelectorConfig | None = None, *,
                 stage_tasks: Mapping[Stage | str, str] | None = None,
                 profiles: Iterable[CandidateProfile] = (),
                 approved: Callable[[str, str], list[Route] | None] | None = None,
                 rng: random.Random | None = None,
                 exempt_from_lock: Iterable[Route] = ()) -> None:
        self.storage = storage
        self.price = price
        self.config = config or SelectorConfig()
        self.stage_tasks = {stage_name(k): v for k, v in (stage_tasks or {}).items()}
        self.profiles = {p.route: p for p in profiles}
        self.approved = approved
        self.rng = rng or random.Random()
        self.exempt = set(exempt_from_lock)

    def stats(self, task: str, segment: str) -> tuple[dict[Route, ModelStats], dict[Route, tuple[float, int]]]:
        stats: dict[Route, ModelStats] = {}
        rejections: dict[Route, tuple[float, int]] = {}
        for row in self.storage.latest_metrics(task, segment):
            route = Route(row.provider, row.model)
            stats[route] = ModelStats(row.reward, row.sample_n)
            if row.evidence.get("reject_rate") is not None:
                rejections[route] = (float(row.evidence["reject_rate"]),
                                     int(row.evidence.get("reject_total", row.sample_n)))
        return stats, rejections

    def choose(self, stage: Stage | str, chain: Sequence[Route], *, segment: str = "*",
               subsegment: str | None = None, request: SelectionRequest | None = None) -> Selection:
        name = stage_name(stage)
        request = request or SelectionRequest(language=segment)
        task = self.stage_tasks.get(name)

        gated: dict[Route, str] = {}
        for route in chain:
            profile = self.profiles.get(route)
            if profile is not None and (why := gate_reason(profile, request, self.price)):
                gated[route] = why
        if request.excluded_providers:
            gated.update({r: "excluded_provider" for r in chain if r.provider in request.excluded_providers})
        allowed = [r for r in chain if r not in gated]
        gated_desc = tuple(f"{r}:{why}" for r, why in gated.items())
        if not allowed:
            # Gates never invent routes: with nothing eligible the router sees the original
            # chain and its own independence rules still apply.
            return Selection(list(chain), chain[0], "no eligible candidate", gated=gated_desc)
        if task is None:
            return Selection(allowed, allowed[0], "stage not learned", gated=gated_desc)

        stats, rejections = self.stats(task, segment)
        if subsegment:
            narrow, _ = self.stats(task, f"{segment}/{subsegment}")
            stats = merge_segment_stats(stats, narrow, self.config)
        blocked = blocked_by_rejection(rejections, self.config, self.exempt)
        pool = self.approved(task, segment) if self.approved else None
        if pool is not None:
            pool = [r for r in pool if r not in gated and r not in blocked]
        allowed = demote(allowed, blocked)
        price = effective_price(self.price, {r: rate for r, (rate, _n) in rejections.items()}, self.config)
        selection = pick(allowed, stats, config=self.config, price=price, pool=pool or None, rng=self.rng)
        ranked = demote(selection.chain, blocked)
        return Selection(ranked, ranked[0], selection.reason, selection.explored, gated_desc,
                         tuple(str(r) for r in blocked if r in ranked))

    def refresh(self, task: str, samples: Iterable[Sample],
                rejections: Mapping[tuple[Route, str], tuple[float, int]] | None = None,
                now: datetime | None = None) -> int:
        """Recompute rewards from samples and append one metric row per (route, segment).

        ``rejections`` maps (route, segment) to (rejection rate, total pieces); the reward is
        discounted by ``reject_weight * rate`` and the rate is stored for the lock.
        """
        rewards = compute_rewards(samples, self.config, now)
        for (route, segment), acc in rewards.items():
            reward = acc["reward"]
            evidence: dict[str, Any] = {"mature": int(acc["mature"]), "quality_weight": self.config.quality_weight}
            rej = (rejections or {}).get((route, segment.split("/")[0]))
            if rej:
                rate, total = rej
                reward *= 1 - self.config.reject_weight * rate
                evidence.update(reject_rate=round(rate, 4), reject_total=total)
            self.storage.add_metric(MetricRow(task=task, segment=segment, provider=route.provider,
                                              model=route.model, reward=round(reward, 4),
                                              quality=round(acc["quality"], 4), sample_n=int(acc["n"]),
                                              evidence=evidence))
        return len(rewards)


__all__ = ["CandidateProfile", "ModelStats", "Sample", "Selection", "SelectionRequest", "Selector",
           "SelectorConfig", "blocked_by_rejection", "compute_rewards", "demote", "effective_price",
           "gate_reason", "merge_segment_stats", "pick"]
