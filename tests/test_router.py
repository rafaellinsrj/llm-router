from __future__ import annotations

from decimal import Decimal

import pytest

from llm_router import BudgetExceeded, LLMError, Route, Stage
from llm_router.providers import FakeProvider, fake_providers
from llm_router.providers.base import Completion


def test_first_route_wins_when_healthy(router_factory):
    providers = fake_providers()
    providers["openai"].script["gpt-large"] = [{"text": "hello"}]
    router = router_factory(providers=providers)

    result = router.call(Stage.SYNTHESIZE, "system", {"input": "x"})

    assert result.route == Route("openai", "gpt-large")
    assert result.data == {"text": "hello"}
    assert result.attempts == 1
    assert providers["anthropic"].calls == []


def test_fallback_follows_chain_order_after_retries(router_factory):
    providers = fake_providers()
    providers["openai"].script["gpt-large"] = [RuntimeError("down")]
    providers["anthropic"].script["claude-large"] = [TimeoutError("slow")]
    providers["google"].script["gemini-flash"] = [{"ok": 1}]
    router = router_factory(providers=providers, retries=2)

    result = router.call(Stage.SYNTHESIZE, "s", "u")

    assert result.route == Route("google", "gemini-flash")
    assert providers["openai"].models_called() == ["gpt-large", "gpt-large"]
    assert providers["anthropic"].models_called() == ["claude-large", "claude-large"]
    assert result.attempts == 5
    audit = router.storage.get_audit(result.call_id)
    assert [d["status"] for d in audit["attempt_details"]] == ["provider_error"] * 4 + ["completed"]


def test_backoff_is_linear_per_attempt(router_factory):
    providers = fake_providers()
    providers["openai"].script["gpt-small"] = [RuntimeError("x"), RuntimeError("x"), {"ok": 1}]
    router = router_factory(providers=providers)
    router.config.backoff_s = 2.0
    slept = []
    router.sleep = slept.append

    router.call(Stage.EXTRACT, "s", "u")

    assert slept == [2.0, 4.0]


def test_missing_provider_is_skipped_without_cost(router_factory):
    providers = {"anthropic": FakeProvider("anthropic")}
    router = router_factory(providers=providers)

    result = router.call(Stage.EXTRACT, "s", "u")

    assert result.provider == "anthropic"
    assert len(router.storage.list_costs()) == 1


def test_all_routes_failing_raises(router_factory):
    providers = fake_providers()
    for p in providers.values():
        p.default = RuntimeError("down")
    router = router_factory(providers=providers, retries=1)

    with pytest.raises(LLMError, match="all attempts failed"):
        router.call(Stage.SYNTHESIZE, "s", "u")


# --------------------------------------------------------------- independence


def test_verification_never_uses_the_synthesis_provider(router_factory):
    providers = fake_providers()
    router = router_factory(providers=providers)
    synthesis = router.call(Stage.SYNTHESIZE, "s", "u")
    assert synthesis.provider == "openai"

    verdict = router.verify(synthesis, "judge", {"draft": synthesis.data})

    assert verdict.provider != synthesis.provider
    assert verdict.route == Route("anthropic", "claude-small")
    assert not verdict.degraded
    assert providers["openai"].models_called() == ["gpt-large"]


def test_verification_fails_closed_when_only_the_same_provider_is_left(router_factory):
    providers = fake_providers()
    providers["anthropic"].default = RuntimeError("no credit")
    providers["google"].default = RuntimeError("no key")
    router = router_factory(providers=providers, retries=1)
    synthesis = router.call(Stage.SYNTHESIZE, "s", "u")

    with pytest.raises(LLMError):
        router.verify(synthesis, "judge", "draft")

    assert providers["openai"].models_called() == ["gpt-large"]  # the judge never ran on openai


def test_degraded_verification_is_opt_in_and_flagged(router_factory):
    providers = fake_providers()
    providers["anthropic"].default = RuntimeError("no credit")
    providers["google"].default = RuntimeError("no key")
    router = router_factory(providers=providers, retries=1, allow_degraded_verification=True)
    synthesis = router.call(Stage.SYNTHESIZE, "s", "u")

    verdict = router.verify(synthesis, "judge", "draft")

    assert verdict.degraded is True
    assert verdict.provider == "openai" and verdict.model != synthesis.model


def test_verify_stage_requires_exclusion(router_factory):
    router = router_factory()
    with pytest.raises(ValueError, match="exclude_provider"):
        router.call(Stage.VERIFY, "judge", "draft")


# --------------------------------------------------------------- budget


def test_budget_exceeded_stops_before_calling_the_provider(router_factory):
    providers = fake_providers()
    router = router_factory(providers=providers, budget="0.0001")

    with pytest.raises(BudgetExceeded):
        router.call(Stage.SYNTHESIZE, "s", "u" * 1000)

    assert all(p.calls == [] for p in providers.values())
    assert router.storage.list_costs() == []


def test_stage_ceiling_is_independent_of_global(router_factory):
    providers = fake_providers()
    router = router_factory(providers=providers, stage_budgets={"extract": "0.0001"})

    with pytest.raises(BudgetExceeded):
        router.call(Stage.EXTRACT, "s", "u")
    assert router.call(Stage.SYNTHESIZE, "s", "u").provider == "openai"
    assert providers["openai"].models_called() == ["gpt-large"]


def test_budget_runs_out_mid_day(router_factory):
    # Each call reserves the worst case; once spent + next reservation > cap, calls stop.
    probe = router_factory()
    reserve = probe.ledger.quote(Route("openai", "gpt-small"), "s", "u", 4096).reserved_usd
    router = router_factory(budget=str(reserve + Decimal("0.00002")))
    calls = 0
    with pytest.raises(BudgetExceeded):
        for _ in range(10):
            router.call(Stage.EXTRACT, "s", "u")
            calls += 1
    assert calls == 2  # each real call costs ~0.000014: the third reservation no longer fits
    view = router.ledger.view()
    assert Decimal(view["*"]["spent_usd"]) + Decimal(view["*"]["reserved_usd"]) <= reserve + Decimal("0.00002")


# --------------------------------------------------------------- JSON and cost


def test_invalid_json_is_retried_and_each_attempt_paid_once(router_factory):
    providers = fake_providers()
    providers["openai"].script["gpt-small"] = ["sorry, no json here", '```json\n{"ok": true}\n```']
    router = router_factory(providers=providers)

    result = router.call(Stage.EXTRACT, "s", "u")

    assert result.data == {"ok": True}
    assert result.attempts == 2
    costs = router.storage.list_costs()
    assert len(costs) == 2
    assert len({c.key for c in costs}) == 2
    assert result.total_estimated_cost_usd == sum(c.cost_usd for c in costs)


def test_cost_recorded_once_per_successful_call(router_factory):
    router = router_factory()
    result = router.call(Stage.SYNTHESIZE, "s", "u")

    costs = router.storage.list_costs()
    assert len(costs) == 1
    assert costs[0].cost_usd == result.cost_usd
    # Settling the same reservation again (e.g. a retried bookkeeping job) is idempotent.
    reservation_id = router.storage.get_audit(result.call_id)["attempt_details"][0]["reservation_id"]
    router.ledger.settle(reservation_id, result.usage)
    assert len(router.storage.list_costs()) == 1
    assert router.ledger.view()["*"]["spent_usd"] == str(result.cost_usd)


def test_unknown_usage_keeps_reservation_and_blocks_retry_on_invalid_json(router_factory):
    providers = fake_providers()
    providers["openai"].script["gpt-small"] = [Completion("not json", None, None)]
    router = router_factory(providers=providers)

    with pytest.raises(LLMError, match="unsettled cost"):
        router.call(Stage.EXTRACT, "s", "u")

    assert len(providers["openai"].calls) == 1  # never paid twice blindly
    view = router.ledger.view()
    assert float(view["*"]["reserved_usd"]) > 0  # money stays reserved until reconciled


def test_unpriced_route_is_skipped(router_factory):
    chains = {Stage.EXTRACT: ["openai/unknown-model", "anthropic/claude-small"]}
    providers = fake_providers()
    router = router_factory(chains=chains, providers=providers)

    result = router.call(Stage.EXTRACT, "s", "u")

    assert result.provider == "anthropic"
    assert providers["openai"].calls == []


def test_audit_never_stores_prompts(router_factory):
    router = router_factory()
    result = router.call(Stage.EXTRACT, "SECRET-SYSTEM", "SECRET-USER")
    assert "SECRET" not in str(router.storage.get_audit(result.call_id))


def test_router_on_sqlite_storage(router_factory, tmp_path):
    from llm_router import SQLiteStorage

    storage = SQLiteStorage(tmp_path / "router.db")
    router = router_factory(storage=storage)
    result = router.call(Stage.SYNTHESIZE, "s", "u")
    audit = storage.get_audit(result.call_id)
    assert audit["status"] == "completed" and audit["reconciliation_required"] is False
    assert len(storage.list_costs()) == 1
