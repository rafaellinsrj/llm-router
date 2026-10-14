from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from llm_router import BudgetLedger, LedgerBlocked, Route, RouterConfig, SQLiteStorage, UnpricedModel
from llm_router.pricing import ModelPrice, PriceTable, money

ROUTE = Route("openai", "gpt-small")


def ledger(storage=None, cap="1", prices=None, clock=None):
    config = RouterConfig(chains={"extract": [ROUTE]}, daily_budget_usd=Decimal(cap))
    prices = prices or PriceTable("t1", datetime.now(UTC) + timedelta(days=1),
                                  {str(ROUTE): ModelPrice(Decimal("1"), Decimal("2"), 100_000, 4_000,
                                                          Decimal("0.001"))})
    return BudgetLedger(storage or SQLiteStorage(), prices, config, clock=clock)


def test_money_rejects_floats_and_rounds_up():
    with pytest.raises(LedgerBlocked):
        money(0.1)
    assert money("0.00000000001") == Decimal("0.0000000001")


def test_reserve_settle_releases_the_unused_part():
    led = ledger()
    quote = led.quote(ROUTE, "s", "u", 1000)
    rid = led.reserve("extract", ROUTE, quote)
    assert Decimal(led.view()["*"]["reserved_usd"]) == quote.reserved_usd

    cost = led.settle(rid, {"input_tokens": 100, "output_tokens": 50})

    assert cost == Decimal("0.0002")
    view = led.view()["*"]
    assert Decimal(view["spent_usd"]) == cost
    assert Decimal(view["reserved_usd"]) == Decimal("0.001")  # surcharge pending reconciliation


def test_reconcile_requires_evidence_and_closes_pending():
    led = ledger()
    rid = led.reserve("extract", ROUTE, led.quote(ROUTE, "s", "u", 100))
    led.settle(rid, {"input_tokens": 10, "output_tokens": 10})
    with pytest.raises(LedgerBlocked):
        led.reconcile(rid, "0.0001", "ops", "short")
    led.reconcile(rid, "0.0001", "ops", "invoice line 42, provider dashboard export")
    view = led.view()["*"]
    assert Decimal(view["reserved_usd"]) == 0
    assert Decimal(view["spent_usd"]) == Decimal("0.0001")


def test_usage_above_quote_freezes_the_day():
    led = ledger()
    rid = led.reserve("extract", ROUTE, led.quote(ROUTE, "s", "u", 100))
    led.settle(rid, {"input_tokens": 10, "output_tokens": 5000})  # adapter ignored max_tokens
    assert led.view()["*"]["frozen"] is True
    with pytest.raises(LedgerBlocked, match="frozen"):
        led.reserve("extract", ROUTE, led.quote(ROUTE, "s", "u", 100))
    led.unfreeze(led.today(), "ops", "adapter fixed")
    led.reserve("extract", ROUTE, led.quote(ROUTE, "s", "u", 100))


def test_expired_price_table_refuses_to_quote():
    prices = PriceTable("old", datetime.now(UTC) - timedelta(seconds=1),
                        {str(ROUTE): ModelPrice(Decimal(1), Decimal(1), 1000, 100)})
    with pytest.raises(UnpricedModel, match="expired"):
        ledger(prices=prices).quote(ROUTE, "s", "u", 10)


def test_reservation_stays_on_its_original_day():
    now = [datetime(2026, 1, 1, 23, 59, tzinfo=UTC)]
    led = ledger(clock=lambda: now[0])
    rid = led.reserve("extract", ROUTE, led.quote(ROUTE, "s", "u", 100))
    now[0] += timedelta(minutes=5)
    led.settle(rid, {"input_tokens": 1, "output_tokens": 1})
    assert Decimal(led.view("2026-01-01")["*"]["spent_usd"]) > 0
    assert Decimal(led.view("2026-01-02")["*"]["spent_usd"]) == 0


def test_sqlite_storage_persists_across_connections(tmp_path):
    path = tmp_path / "ledger.db"
    led = ledger(SQLiteStorage(path))
    rid = led.reserve("extract", ROUTE, led.quote(ROUTE, "s", "u", 100))
    led.settle(rid, {"input_tokens": 10, "output_tokens": 10})

    reopened = SQLiteStorage(path)
    assert reopened.get_reservation(rid).state == "estimated"
    assert len(reopened.list_costs()) == 1
