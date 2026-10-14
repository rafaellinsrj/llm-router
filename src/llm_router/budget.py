"""Budget ledger: reserve the worst case BEFORE the paid call, settle with real usage AFTER.

Lifecycle of one attempt::

    quote (price table) -> reserve (atomic check against global and stage ceilings)
        -> provider call -> settle (usage -> estimated cost, release the rest)
        -> optional reconcile (human, with evidence from the invoice)

Rules kept from production:

* The reservation is committed before the call, so a crash mid-call never leaves an
  invisible spend: the money stays reserved until someone settles or reconciles it.
* Missing token usage never releases the reservation. An adapter exception does not
  prove the provider did not bill, so uncertainty is kept, not optimistically freed.
* A settle that comes out above the reservation freezes the day: something in the price
  table or the adapter is wrong, and no new reservation is accepted until it is fixed.
* A per-call surcharge (``max_extra_usd``) stays reserved after settle until reconciled.
* A reservation that crosses midnight stays on its original day.
"""
from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from decimal import Decimal

from llm_router.config import Route, RouterConfig, Stage, stage_name
from llm_router.errors import LedgerBlocked
from llm_router.pricing import PriceTable, Quote, estimate, money
from llm_router.storage import CostRecord, Reservation, Storage

GLOBAL = "*"
_STAGE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")


class BudgetLedger:
    def __init__(self, storage: Storage, prices: PriceTable, config: RouterConfig,
                 clock: Callable[[], datetime] | None = None) -> None:
        self.storage = storage
        self.prices = prices
        self.config = config
        self.clock = clock or (lambda: datetime.now(UTC))

    def today(self) -> str:
        return self.clock().astimezone(UTC).strftime("%Y-%m-%d")

    def _caps(self, stage: str) -> dict[str, Decimal]:
        if not _STAGE_RE.fullmatch(stage):
            raise LedgerBlocked(f"invalid stage name {stage!r}")
        return {GLOBAL: money(self.config.daily_budget_usd, positive=True),
                stage: money(self.config.stage_cap(stage), positive=True)}

    # ------------------------------------------------------------------ attempts

    def quote(self, route: Route, system: str, user: str, max_tokens: int) -> Quote:
        return self.prices.quote(route, system, user, max_tokens, now=self.clock())

    def reserve(self, stage: Stage | str, route: Route, quote: Quote, subject: str | None = None) -> str:
        """Atomically reserve ``quote.reserved_usd`` on the global and the stage ceilings."""
        name = stage_name(stage)
        caps = self._caps(name)
        amount = money(quote.reserved_usd, positive=True)
        day = self.today()
        with self.storage.transaction():
            balances = {scope: self.storage.get_balance(day, scope) for scope in caps}
            for scope, bal in balances.items():
                if bal.frozen:
                    raise LedgerBlocked(f"budget day {day} is frozen ({scope})")
                if bal.reserved_usd + bal.spent_usd + amount > caps[scope]:
                    raise LedgerBlocked(
                        f"daily ceiling for {'all stages' if scope == GLOBAL else scope} "
                        f"would be exceeded ({bal.reserved_usd + bal.spent_usd} + {amount} > {caps[scope]})")
            for bal in balances.values():
                bal.reserved_usd += amount
                self.storage.put_balance(bal)
            identifier = str(uuid.uuid4())
            self.storage.put_reservation(Reservation(
                id=identifier, day=day, stage=name, provider=route.provider, model=route.model,
                quote=quote.to_dict(), reserved_usd=amount, subject=subject))
        return identifier

    def settle(self, reservation_id: str, usage: Mapping[str, object]) -> Decimal:
        """Turn the reservation into an estimated cost. Idempotent for the same usage."""
        tokens = {k: usage.get(k) for k in ("input_tokens", "output_tokens")}
        with self.storage.transaction():
            res = self.storage.get_reservation(reservation_id)
            if res is None:
                raise LedgerBlocked("unknown reservation")
            quote = Quote.from_dict(res.quote)
            cost = estimate(quote, tokens)
            if res.state != "reserved":
                if res.state == "estimated" and res.usage == tokens:
                    return res.estimated_usd or cost
                raise LedgerBlocked("reservation already closed with a different result")
            exceeded = (cost > res.reserved_usd
                        or int(tokens["input_tokens"]) > quote.input_token_bound  # type: ignore[call-overload]
                        or int(tokens["output_tokens"]) > quote.max_output_tokens)  # type: ignore[call-overload]
            pending = min(quote.max_extra_usd, res.reserved_usd)
            for scope in (GLOBAL, res.stage):
                bal = self.storage.get_balance(res.day, scope)
                bal.reserved_usd = bal.reserved_usd - res.reserved_usd + pending
                bal.spent_usd += cost
                bal.frozen = bal.frozen or exceeded
                self.storage.put_balance(bal)
            res.state, res.estimated_usd, res.usage = "estimated", cost, tokens  # type: ignore[assignment]
            self.storage.put_reservation(res)
            # Same transaction, idempotency key = reservation id: the cost is logged exactly once.
            self.storage.add_cost(CostRecord(key=res.id, stage=res.stage, provider=res.provider,
                                             model=res.model, cost_usd=cost, usage=dict(tokens)))  # type: ignore[arg-type]
        return cost

    def reconcile(self, reservation_id: str, observed_usd: str | Decimal, actor: str, evidence: str) -> None:
        """Human action with proof (invoice line, provider dashboard export). Never automatic."""
        if not isinstance(evidence, str) or not 20 <= len(evidence.strip()) <= 4000:
            raise LedgerBlocked("reconciliation requires evidence (20 to 4000 characters)")
        amount = money(observed_usd)
        with self.storage.transaction():
            res = self.storage.get_reservation(reservation_id)
            if res is None:
                raise LedgerBlocked("unknown reservation")
            if res.state == "reconciled":
                if res.observed_usd == amount:
                    return
                raise LedgerBlocked("already reconciled with a different amount")
            quote = Quote.from_dict(res.quote)
            pending = res.reserved_usd if res.state == "reserved" else min(quote.max_extra_usd, res.reserved_usd)
            adjustment = amount - (res.estimated_usd or Decimal(0))
            for scope in (GLOBAL, res.stage):
                bal = self.storage.get_balance(res.day, scope)
                bal.reserved_usd -= pending
                bal.spent_usd += adjustment
                bal.frozen = bal.frozen or amount > res.reserved_usd
                self.storage.put_balance(bal)
            res.state, res.observed_usd, res.reconciled_by, res.evidence = "reconciled", amount, actor, evidence
            self.storage.put_reservation(res)

    def unfreeze(self, day: str, actor: str, reason: str) -> None:
        """Explicit operator action after fixing the price table or adapter."""
        if not reason.strip() or not actor.strip():
            raise LedgerBlocked("unfreeze requires an actor and a reason")
        with self.storage.transaction():
            for scope in {GLOBAL, *self.config.chains}:
                bal = self.storage.get_balance(day, scope)
                if bal.frozen:
                    bal.frozen = False
                    self.storage.put_balance(bal)

    def view(self, day: str | None = None) -> dict[str, dict[str, str | bool]]:
        day = day or self.today()
        out: dict[str, dict[str, str | bool]] = {}
        for scope in (GLOBAL, *self.config.chains):
            bal = self.storage.get_balance(day, scope)
            out[scope] = {"reserved_usd": str(bal.reserved_usd), "spent_usd": str(bal.spent_usd),
                          "cap_usd": str(self.config.daily_budget_usd if scope == GLOBAL
                                         else self.config.stage_cap(scope)),
                          "frozen": bal.frozen}
        return out
