"""End-to-end demo with FakeProvider: no API keys, no network.

    python examples/demo.py

Shows: selector choice with exploration, a provider outage and fallback, invalid JSON
retried, independent verification, deterministic gates over the judge score, a budget
ceiling stopping a call before it is paid, and a small offline tournament.
"""
from __future__ import annotations

import random
from datetime import UTC, datetime
from decimal import Decimal

from llm_router import (
    BudgetExceeded,
    InMemoryStorage,
    Route,
    RouterConfig,
    Sample,
    Selector,
    SelectorConfig,
    Stage,
    Tournament,
    Verdict,
    apply_gates,
    build_router,
    example_price_table,
)
from llm_router.providers import FakeProvider


def line(title: str) -> None:
    print(f"\n== {title} " + "=" * max(0, 60 - len(title)))


def show(label: str, result) -> None:
    sel = f" | selector: {result.selection.reason}" if result.selection else ""
    flag = " | DEGRADED" if result.degraded else ""
    if result.total_estimated_cost_usd is None:
        flag += " | reconciliation required (some attempt may be billed)"
    print(f"{label:<12} -> {result.provider}/{result.model:<15} attempts={result.attempts} "
          f"known cost=${result.known_cost_usd:.6f}{sel}{flag}")


def main() -> None:
    config = RouterConfig(
        chains={
            Stage.EXTRACT: ["openai/gpt-small", "anthropic/claude-small"],
            Stage.SYNTHESIZE: ["openai/gpt-large", "anthropic/claude-large", "google/gemini-flash"],
            Stage.VERIFY: ["openai/gpt-small", "anthropic/claude-small", "google/gemini-flash-lite"],
            Stage.SUMMARIZE: ["google/gemini-flash-lite", "anthropic/claude-small"],
        },
        backoff_s=0.0,
        daily_budget_usd=Decimal("1.00"),
        stage_budgets_usd={"summarize": Decimal("0.0015")},
    )
    prices = example_price_table()
    storage = InMemoryStorage()

    openai = FakeProvider("openai", script={
        # first call: extraction; next calls: acting as the judge
        "gpt-small": [{"facts": ["a", "b"]},
                      {"score": 91, "issues": [{"kind": "unsupported_claim", "severity": "high"}]}],
        "gpt-large": [RuntimeError("503 from provider"), RuntimeError("503 again"), RuntimeError("503")],
    })
    anthropic = FakeProvider("anthropic", script={
        "claude-large": ["Sure! I will write the draft now.", {"title": "Demo", "body": "..."}],
    })
    google = FakeProvider("google", default={"summary": "short"})
    providers = {"openai": openai, "anthropic": anthropic, "google": google}

    # Selector history: gpt-large and claude-large are within 5 points, so the cheaper one
    # (in the example price table) leads; gemini-flash is cheap but clearly worse.
    selector = Selector(storage, prices.unit_price, SelectorConfig(), stage_tasks={Stage.SYNTHESIZE: "synthesis"},
                        rng=random.Random(42))
    now = datetime.now(UTC)
    history = ([Sample(Route("openai", "gpt-large"), "en", 0.90, now)] * 20
               + [Sample(Route("anthropic", "claude-large"), "en", 0.88, now)] * 20
               + [Sample(Route("google", "gemini-flash"), "en", 0.60, now)] * 20)
    selector.refresh("synthesis", history)

    router = build_router(config, providers, storage=storage, prices=prices, selector=selector)

    line("1. extract (healthy first route)")
    show("extract", router.call(Stage.EXTRACT, "Extract facts as JSON.", {"text": "..."}))

    line("2. synthesize (selector + outage + invalid JSON retry)")
    synthesis = router.call(Stage.SYNTHESIZE, "Write a JSON draft.", {"facts": ["a", "b"]}, segment="en")
    show("synthesize", synthesis)
    audit = storage.get_audit(synthesis.call_id)
    for d in audit["attempt_details"]:
        print(f"   attempt {d['attempt']}: {d['provider']}/{d['model']:<14} {d['status']:<15} "
              f"cost={d['estimated_cost_usd']}")

    line("3. verify (independent provider) + gates over the score")
    verdict_call = router.verify(synthesis, "Judge the draft. Return JSON.", {"draft": synthesis.data})
    show("verify", verdict_call)
    print(f"   judge provider != writer provider: {verdict_call.provider != synthesis.provider}")
    verdict = Verdict.from_json(verdict_call.data)
    gate = apply_gates(verdict, str(synthesis.data))
    print(f"   judge score={verdict.score} -> decision={gate.decision} ({gate.reason})")

    line("4. budget ceiling (summarize capped at $0.0015/day)")
    for i in range(1, 8):
        try:
            show(f"summarize #{i}", router.call(Stage.SUMMARIZE, "Summarize.", "x" * 2000, max_tokens=256))
        except BudgetExceeded as exc:
            print(f"summarize #{i} -> STOPPED before calling any provider: {exc}")
            break

    line("5. offline tournament (same inputs, forced route)")
    quality = {"gpt-large": 0.90, "claude-large": 0.87, "gemini-flash": 0.50}
    for p in providers.values():
        p.script = {m: (lambda model, s, u: {"model": model}) for m in quality}
    tour = Tournament(router, storage, judge=lambda task, item, data: quality[data["model"]],
                      build=lambda task, item: (Stage.SYNTHESIZE, "Write.", item, 512))
    rows = tour.run([Route("openai", "gpt-large"), Route("anthropic", "claude-large"),
                     Route("google", "gemini-flash")], ["synthesis"], {"en": [{"t": 1}, {"t": 2}]})
    for r in sorted(rows, key=lambda r: -r.quality):
        print(f"   {r.route!s:<24} quality={r.quality:.2f} n={r.n} cost/piece=${r.cost_per_piece_usd:.6f}")
    print(f"   approved to compete: {[str(r) for r in tour.approved('synthesis', 'en') or []]}")

    line("ledger")
    for scope, row in router.ledger.view().items():
        print(f"   {scope:<11} spent=${Decimal(row['spent_usd']):.6f} reserved=${Decimal(row['reserved_usd']):.6f} "
              f"cap=${row['cap_usd']}")
    print(f"   cost records: {len(storage.list_costs())}")
    print("   'reserved' on synthesize = attempts that raised: they may have been billed, so the money")
    print("   stays reserved until someone reconciles it with the invoice (ledger.reconcile).")


if __name__ == "__main__":
    main()
