from __future__ import annotations

from decimal import Decimal

import pytest

from llm_router import BudgetLedger, InMemoryStorage, Route, Router, RouterConfig, Stage, example_price_table
from llm_router.providers import fake_providers

CHAINS = {
    Stage.EXTRACT: ["openai/gpt-small", "anthropic/claude-small"],
    Stage.SYNTHESIZE: ["openai/gpt-large", "anthropic/claude-large", "google/gemini-flash"],
    Stage.VERIFY: ["openai/gpt-small", "anthropic/claude-small", "google/gemini-flash-lite"],
}


def make_router(*, chains=None, budget="20", stage_budgets=None, providers=None, storage=None,
                selector=None, **config_kwargs):
    config = RouterConfig(chains=chains or CHAINS, backoff_s=0, daily_budget_usd=Decimal(budget),
                          stage_budgets_usd=stage_budgets or {}, **config_kwargs)
    storage = storage or InMemoryStorage()
    providers = providers if providers is not None else fake_providers()
    ledger = BudgetLedger(storage, example_price_table(), config)
    return Router(config=config, providers=providers, ledger=ledger, storage=storage, selector=selector)


@pytest.fixture
def router_factory():
    return make_router


@pytest.fixture
def routes():
    return {name: Route.parse(name) for chain in CHAINS.values() for name in chain}
