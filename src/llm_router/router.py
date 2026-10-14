"""Single entry point for every LLM call.

For one ``call(stage, ...)`` the router:

1. takes the stage chain from config and lets the selector reorder it (optional);
2. splits it into *eligible* routes and *degraded* routes (same provider as the one being
   judged; tried last, only if allowed, and flagged in the result);
3. for each route and attempt: reserves the worst-case cost in the ledger BEFORE calling,
   calls the provider, settles the real cost, parses JSON;
4. retries with linear backoff on provider errors and invalid JSON, then falls back to
   the next route;
5. writes an audit snapshot after every state change (metadata only, never prompts).

Invariants (each one has a test):

* a call whose reservation does not fit the ceiling raises ``BudgetExceeded`` without
  contacting any provider;
* a verify stage never runs on the provider it is judging unless degraded mode is
  explicitly allowed, and even then it is marked ``degraded=True``;
* the cost of each attempt is recorded exactly once (idempotency key = reservation id);
* a paid answer whose cost could not be settled is never retried automatically: retrying
  would pay again for an amount nobody can see.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from llm_router.audit import CallAudit
from llm_router.budget import BudgetLedger
from llm_router.config import Route, RouterConfig, Stage, stage_name
from llm_router.errors import BudgetExceeded, LedgerBlocked, LLMError, UnpricedModel
from llm_router.parsing import parse_json
from llm_router.providers.base import Provider
from llm_router.selector import Selection, SelectionRequest, Selector
from llm_router.storage import Storage

log = logging.getLogger("llm_router")

# Forced single route without fallback, used by the tournament to compare models on the
# same input. Never set in normal operation.
FORCED_ROUTE: ContextVar[Route | None] = ContextVar("llm_router_forced_route", default=None)


@dataclass(frozen=True)
class CallResult:
    data: dict[str, Any]
    provider: str
    model: str
    usage: dict[str, int | None]
    cost_usd: Decimal | None
    degraded: bool
    call_id: str
    attempts: int
    known_cost_usd: Decimal  # sum of the attempts whose cost is known
    total_estimated_cost_usd: Decimal | None  # all attempts; None if some cost is unknown
    reconciliation_required: bool
    selection: Selection | None = None

    @property
    def route(self) -> Route:
        return Route(self.provider, self.model)


@dataclass
class Router:
    config: RouterConfig
    providers: Mapping[str, Provider]
    ledger: BudgetLedger
    storage: Storage
    selector: Selector | None = None
    sleep: Callable[[float], None] = field(default=time.sleep)

    # ------------------------------------------------------------------ helpers

    def _persist(self, audit: CallAudit, status: str, **extra: Any) -> dict[str, Any]:
        snapshot = audit.snapshot(status, **extra)
        try:
            self.storage.put_audit(audit.call_id, snapshot)
        except Exception as exc:  # audit failure must never trigger a second paid call
            log.warning("audit_unavailable call=%s type=%s", audit.call_id, type(exc).__name__)
        return snapshot

    def _backoff(self, attempt: int) -> None:
        if self.config.backoff_s:
            self.sleep(self.config.backoff_s * (attempt + 1))

    def _plan(self, stage: str, exclude_provider: str | None, exclude_model: str | None,
              allow_degraded: bool, segment: str, subsegment: str | None,
              sensitive: bool) -> tuple[list[Route], list[Route], Selection | None]:
        base = self.config.chain(stage)
        forced = FORCED_ROUTE.get()
        if forced is not None:
            return [forced], [], None
        eligible = [r for r in base if r.provider != exclude_provider]
        # Degraded routes come from the configured chain, never from the selector: they are a
        # last resort and must stay predictable. The exact model being judged is never used.
        degraded = [r for r in base if r.provider == exclude_provider and r.model != exclude_model] \
            if allow_degraded else []
        selection = None
        if self.selector is not None and eligible:
            request = SelectionRequest(language=segment,
                                       sensitive=sensitive or stage in self.config.verify_stages,
                                       excluded_providers=(exclude_provider,) if exclude_provider else ())
            try:
                selection = self.selector.choose(stage, eligible, segment=segment, subsegment=subsegment,
                                                 request=request)
                allowed = set(eligible)
                eligible = [r for r in selection.chain if r in allowed] or eligible
            except Exception as exc:  # selector failure keeps the configured chain
                log.warning("selector_unavailable stage=%s type=%s", stage, type(exc).__name__)
        return eligible, degraded, selection

    # ------------------------------------------------------------------ public API

    def call(self, stage: Stage | str, system: str, payload: str | Mapping[str, Any], *,
             max_tokens: int | None = None, exclude_provider: str | None = None,
             exclude_model: str | None = None, allow_degraded: bool | None = None,
             segment: str = "*", subsegment: str | None = None, sensitive: bool = False,
             subject: str | None = None) -> CallResult:
        """Run ``stage`` and return parsed JSON plus routing and cost metadata.

        ``exclude_provider`` is the independence rule: for a judge, pass the provider that
        produced the text being judged (``verify`` does it for you).
        """
        name = stage_name(stage)
        max_tokens = max_tokens or self.config.max_tokens
        if allow_degraded is None:
            allow_degraded = (self.config.allow_degraded_verification if name in self.config.verify_stages
                              else self.config.allow_degraded)
        if name in self.config.verify_stages and exclude_provider is None:
            raise ValueError(f"stage {name!r} is a verify stage: exclude_provider is required")
        user = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, default=str)
        eligible, degraded, selection = self._plan(name, exclude_provider, exclude_model, allow_degraded,
                                                   segment, subsegment, sensitive)
        if FORCED_ROUTE.get() is not None:
            exclude_provider = None
        audit = CallAudit(str(uuid.uuid4()), name)
        if selection is not None:
            self._persist(audit, "running", selection=selection.to_dict())
        last_err: BaseException | None = None

        for route in eligible + degraded:
            provider = self.providers.get(route.provider)
            for attempt in range(self.config.retries):
                detail = audit.start(route.provider, route.model)
                if provider is None:
                    # Never reached an adapter: nothing could have been billed.
                    detail.update(estimated_cost_usd="0", budget_recorded=True)
                    audit.finish(detail, "provider_unavailable")
                    last_err = LLMError(f"provider {route.provider!r} not configured")
                    self._persist(audit, "running")
                    break
                billable = getattr(provider, "billable_max_tokens", None)
                quoted_tokens = billable(route.model, max_tokens) if callable(billable) else max_tokens
                try:
                    quote = self.ledger.quote(route, system, user, quoted_tokens)
                except UnpricedModel as exc:
                    detail.update(estimated_cost_usd="0", budget_recorded=True)
                    audit.finish(detail, "unpriced", exc)
                    last_err = exc
                    self._persist(audit, "running")
                    break
                try:
                    reservation_id = self.ledger.reserve(name, route, quote, subject)
                except BudgetExceeded as exc:
                    # Stop BEFORE any paid call. Trying the next route would only hide the ceiling.
                    detail.update(estimated_cost_usd="0", budget_recorded=True)
                    audit.finish(detail, "budget_exceeded", exc)
                    self._persist(audit, "budget_exceeded", error_type=type(exc).__name__)
                    raise
                detail["reservation_id"] = reservation_id
                self._persist(audit, "running")

                try:
                    completion = provider.complete(route.model, system, user, max_tokens=max_tokens)
                except Exception as exc:
                    # An adapter error does not prove there was no charge: the reservation stays
                    # open (reconciliation_required) and the attempt is retried after backoff.
                    audit.finish(detail, "provider_error", exc)
                    last_err = exc
                    self._persist(audit, "running")
                    self._backoff(attempt)
                    continue

                usage = completion.usage()
                detail["usage"] = usage
                cost: Decimal | None = None
                try:
                    cost = self.ledger.settle(reservation_id, usage)
                    detail.update(estimated_cost_usd=str(cost), budget_recorded=True)
                except LedgerBlocked as exc:
                    detail["accounting_error_type"] = type(exc).__name__
                    log.warning("cost_incomplete call=%s attempt=%s", audit.call_id, detail["attempt"])

                try:
                    data = parse_json(completion.text)
                except LLMError as exc:
                    audit.finish(detail, "invalid_json", exc)
                    last_err = exc
                    self._persist(audit, "running")
                    if not detail["budget_recorded"]:
                        self._persist(audit, "failed", error_type="UnsettledCost")
                        raise LLMError("invalid answer with an unsettled cost; reconcile the reservation "
                                       f"{reservation_id} before retrying") from None
                    self._backoff(attempt)
                    continue

                audit.finish(detail, "completed")
                is_degraded = exclude_provider is not None and route.provider == exclude_provider
                snap = self._persist(audit, "completed", provider=route.provider, model=route.model,
                                     degraded=is_degraded)
                if is_degraded:
                    log.error("DEGRADED VERIFICATION: %s judged output of its own provider", route)
                total = snap["total_estimated_cost_usd"]
                return CallResult(data=data, provider=route.provider, model=route.model, usage=usage,
                                  cost_usd=cost, degraded=is_degraded, call_id=audit.call_id,
                                  attempts=len(audit.details),
                                  known_cost_usd=Decimal(snap["known_estimated_cost_usd"]),
                                  total_estimated_cost_usd=Decimal(total) if total is not None else None,
                                  reconciliation_required=snap["reconciliation_required"],
                                  selection=selection)

        self._persist(audit, "failed", error_type=type(last_err).__name__ if last_err else "NoRoute")
        if not eligible and not degraded:
            raise LLMError(f"no independent route available for stage {name!r}")
        raise LLMError(f"all attempts failed for stage {name!r}: {type(last_err).__name__}") from last_err

    def verify(self, judged: CallResult, system: str, payload: str | Mapping[str, Any], *,
               stage: Stage | str = Stage.VERIFY, **kwargs: Any) -> CallResult:
        """Judge ``judged`` on a different provider (and never on the same model)."""
        return self.call(stage, system, payload, exclude_provider=judged.provider,
                         exclude_model=judged.model, **kwargs)
