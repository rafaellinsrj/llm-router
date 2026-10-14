"""Offline tournament: every model produces the SAME pieces from the SAME inputs.

Production statistics are biased (each model saw different inputs, on different days).
A periodic tournament removes that bias: all candidates get identical inputs, a judge
scores each output with the same rubric, and the result becomes a quality/cost board per
task and segment. Nothing is published.

Only models *approved* in a recent tournament compete in production:
approved = at least ``min_samples`` scored pieces, mean quality >= ``quality_floor`` and
no more than ``quality_gap`` below the best model for that task and segment.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from llm_router.config import Route, Stage
from llm_router.errors import BudgetExceeded, RouterError
from llm_router.router import FORCED_ROUTE, Router
from llm_router.storage import MetricRow, Storage

# judge(task, input_item, output_data) -> quality in [0, 1], or None when it cannot score
Judge = Callable[[str, dict[str, Any], dict[str, Any]], float | None]
# builder(task, input_item) -> (stage, system prompt, payload, max_tokens)
RequestBuilder = Callable[[str, dict[str, Any]], tuple[Stage | str, str, Any, int]]


@dataclass(frozen=True)
class TournamentConfig:
    min_samples: int = 2
    quality_floor: float = 0.55
    quality_gap: float = 0.15
    fresh_days: int = 14
    max_cost_usd: Decimal = Decimal("10")
    task_prefix: str = "tournament:"


@dataclass
class _Acc:
    scores: list[float] = field(default_factory=list)
    failures: int = 0
    cost: Decimal = Decimal(0)


@dataclass(frozen=True)
class TournamentRow:
    task: str
    segment: str
    route: Route
    quality: float
    n: int
    failures: int
    cost_per_piece_usd: Decimal

    @property
    def reliability(self) -> float:
        return self.n / max(1, self.n + self.failures)


class Tournament:
    def __init__(self, router: Router, storage: Storage, judge: Judge, build: RequestBuilder,
                 config: TournamentConfig | None = None,
                 clock: Callable[[], datetime] | None = None) -> None:
        self.router = router
        self.storage = storage
        self.judge = judge
        self.build = build
        self.config = config or TournamentConfig()
        self.clock = clock or (lambda: datetime.now(UTC))

    def run(self, models: Sequence[Route], tasks: Iterable[str],
            inputs: dict[str, list[dict[str, Any]]]) -> list[TournamentRow]:
        """``inputs`` maps segment (e.g. language) to the shared input items.

        Each (task, item, model) runs with a forced route: no fallback, so a failure counts
        against the model that failed instead of being hidden by another provider.
        Stops early at ``max_cost_usd`` (sum of real estimated costs, all attempts).
        """
        results: dict[tuple[str, str, Route], _Acc] = {}
        spent = Decimal(0)
        for segment, items in inputs.items():
            for task in tasks:
                for item in items:
                    for route in models:
                        if spent >= self.config.max_cost_usd:
                            return self._save(results)
                        acc = results.setdefault((task, segment, route), _Acc())
                        stage, system, payload, max_tokens = self.build(task, item)
                        token = FORCED_ROUTE.set(route)
                        seen = len(self.router.storage.list_costs())
                        try:
                            data = self.router.call(stage, system, payload, max_tokens=max_tokens,
                                                    segment=segment).data
                        except BudgetExceeded:
                            return self._save(results)
                        except RouterError:
                            data = None
                        finally:
                            FORCED_ROUTE.reset(token)
                        # Cost from the ledger's cost log: includes failed and retried attempts.
                        cost = sum((c.cost_usd for c in self.router.storage.list_costs()[seen:]), Decimal(0))
                        spent += cost
                        acc.cost += cost
                        score = self._score(task, item, data) if data is not None else None
                        if score is None:
                            acc.failures += 1
                        else:
                            acc.scores.append(score)
        return self._save(results)

    def _score(self, task: str, item: dict[str, Any], data: dict[str, Any]) -> float | None:
        try:
            score = self.judge(task, item, data)
        except Exception:  # a judge failure is a missing score, not a crash of the whole run
            return None
        return None if score is None else max(0.0, min(1.0, float(score)))

    def _save(self, results: dict[tuple[str, str, Route], _Acc]) -> list[TournamentRow]:
        rows = []
        for (task, segment, route), acc in results.items():
            n = len(acc.scores)
            quality = sum(acc.scores) / n if n else 0.0
            per_piece = acc.cost / max(1, n + acc.failures)
            row = TournamentRow(task, segment, route, round(quality, 4), n, acc.failures, per_piece)
            rows.append(row)
            self.storage.add_metric(MetricRow(
                task=self.config.task_prefix + task, segment=segment, provider=route.provider,
                model=route.model, reward=row.quality, quality=row.quality, sample_n=n,
                evidence={"failures": acc.failures, "reliability": round(row.reliability, 4),
                          "cost_per_piece_usd": str(per_piece)},
                at=self.clock().isoformat()))
        return rows

    def approved(self, task: str, segment: str) -> list[Route] | None:
        """Routes approved in the latest fresh tournament, or ``None`` without a fresh one."""
        rows = self.storage.latest_metrics(self.config.task_prefix + task, segment)
        cutoff = self.clock() - timedelta(days=self.config.fresh_days)
        fresh = [r for r in rows if datetime.fromisoformat(r.at) > cutoff]
        if not fresh:
            return None
        best = max(r.quality for r in fresh)
        bar = max(self.config.quality_floor, best - self.config.quality_gap)
        return [Route(r.provider, r.model) for r in fresh if r.sample_n >= self.config.min_samples and r.quality >= bar]


def leaderboard(rows: Iterable[TournamentRow]) -> list[TournamentRow]:
    """Sorted by task, segment, quality (desc), then cost per piece (asc)."""
    return sorted(rows, key=lambda r: (r.task, r.segment, -r.quality, r.cost_per_piece_usd))
