"""Versioned price table (rate card) with validity and per-model token limits.

Money is handled with ``Decimal`` and rounded *up* (ROUND_CEILING): a reservation that
rounds down can let the real spend exceed the ceiling by accumulation.

IMPORTANT: ``example_price_table()`` uses placeholder model IDs and illustrative prices.
They are NOT official prices of any provider. Load your own table, checked against the
providers' pricing pages, before using this in production.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import ROUND_CEILING, Decimal, InvalidOperation

from llm_router.config import Route
from llm_router.errors import LedgerBlocked, UnpricedModel

UNIT = Decimal("0.0000000001")
MILLION = Decimal(1_000_000)
MAX_AMOUNT = Decimal(1_000_000_000)
# Fixed overhead added to the input upper bound (message framing, role tokens, etc.).
INPUT_OVERHEAD_TOKENS = 1024


def money(value: object, *, positive: bool = False) -> Decimal:
    """Exact, finite, non-negative decimal rounded up to 10 decimal places.

    Floats are rejected on purpose: ``0.1`` as float is not 0.1, and a ledger must be exact.
    """
    if isinstance(value, bool) or not isinstance(value, (str, int, Decimal)):
        raise LedgerBlocked("amounts must be str, int or Decimal")
    try:
        amount = Decimal(value)
    except InvalidOperation:
        raise LedgerBlocked("invalid amount") from None
    if not amount.is_finite() or amount < 0 or amount > MAX_AMOUNT or (positive and amount <= 0):
        raise LedgerBlocked("amount out of range")
    return amount.quantize(UNIT, rounding=ROUND_CEILING)


@dataclass(frozen=True)
class ModelPrice:
    input_usd_per_million: Decimal
    output_usd_per_million: Decimal
    input_token_limit: int
    output_token_limit: int
    max_extra_usd: Decimal = Decimal(0)  # per-call surcharge that may appear on the invoice

    def __post_init__(self) -> None:
        for limit in (self.input_token_limit, self.output_token_limit):
            if type(limit) is not int or not 1 <= limit <= 10_000_000:
                raise ValueError("token limits must be integers between 1 and 10,000,000")


@dataclass(frozen=True)
class Quote:
    """Worst-case price of one attempt, frozen at reservation time."""

    table_version: str
    input_usd_per_million: Decimal
    output_usd_per_million: Decimal
    input_token_bound: int
    max_output_tokens: int
    max_extra_usd: Decimal
    reserved_usd: Decimal

    def to_dict(self) -> dict[str, str | int]:
        return {k: (str(v) if isinstance(v, Decimal) else v) for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> Quote:
        return cls(
            table_version=str(raw["table_version"]),
            input_usd_per_million=Decimal(str(raw["input_usd_per_million"])),
            output_usd_per_million=Decimal(str(raw["output_usd_per_million"])),
            input_token_bound=int(raw["input_token_bound"]),  # type: ignore[call-overload]
            max_output_tokens=int(raw["max_output_tokens"]),  # type: ignore[call-overload]
            max_extra_usd=Decimal(str(raw["max_extra_usd"])),
            reserved_usd=Decimal(str(raw["reserved_usd"])),
        )


class PriceTable:
    """Prices keyed by ``provider/model``. Expired tables refuse to quote."""

    def __init__(self, version: str, valid_until: datetime, models: Mapping[str, ModelPrice]):
        if not version.strip():
            raise ValueError("price table needs a version")
        if valid_until.tzinfo is None:
            raise ValueError("valid_until must be timezone-aware")
        self.version = version
        self.valid_until = valid_until
        self.models = {str(Route.parse(k)): v for k, v in models.items()}

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> PriceTable:
        models = {}
        for key, entry in dict(raw["models"]).items():  # type: ignore[call-overload]
            models[key] = ModelPrice(
                input_usd_per_million=money(str(entry["input_usd_per_million"])),
                output_usd_per_million=money(str(entry["output_usd_per_million"])),
                input_token_limit=entry["input_token_limit"],
                output_token_limit=entry["output_token_limit"],
                max_extra_usd=money(str(entry.get("max_extra_usd", "0"))),
            )
        return cls(str(raw["version"]), datetime.fromisoformat(str(raw["valid_until"])), models)

    @classmethod
    def from_json(cls, text: str) -> PriceTable:
        return cls.from_dict(json.loads(text))

    def price(self, route: Route) -> ModelPrice:
        try:
            return self.models[str(route)]
        except KeyError:
            raise UnpricedModel(f"no price for {route}") from None

    def unit_price(self, route: Route) -> float:
        """Blended reference price (USD per 1M tokens, 1:1 in/out). For ranking, not billing."""
        p = self.price(route)
        return float((p.input_usd_per_million + p.output_usd_per_million) / 2)

    def quote(self, route: Route, system: str, user: str, max_tokens: int,
              *, now: datetime | None = None) -> Quote:
        """Worst-case cost of one attempt.

        Input bound: UTF-8 byte length + fixed overhead. With byte-level BPE tokenizers a
        token covers at least one byte in practice, so bytes are a conservative upper bound
        that needs no tokenizer. Output bound: ``max_tokens``, which the adapter enforces.
        """
        now = now or datetime.now(UTC)
        if self.valid_until <= now:
            raise UnpricedModel(f"price table {self.version} expired at {self.valid_until.isoformat()}")
        p = self.price(route)
        if type(max_tokens) is not int or not 1 <= max_tokens <= p.output_token_limit:
            raise UnpricedModel(f"max_tokens={max_tokens} outside the limit of {route}")
        bound = len(system.encode()) + len(user.encode()) + INPUT_OVERHEAD_TOKENS
        if bound > p.input_token_limit:
            raise UnpricedModel(f"input too large for {route}")
        reserved = money((p.input_usd_per_million * bound + p.output_usd_per_million * max_tokens) / MILLION
                         + p.max_extra_usd, positive=True)
        return Quote(self.version, p.input_usd_per_million, p.output_usd_per_million, bound,
                     max_tokens, p.max_extra_usd, reserved)


def estimate(quote: Quote, usage: Mapping[str, object]) -> Decimal:
    """Cost of an attempt from reported token usage, using the prices frozen in the quote."""
    tokens = {k: usage.get(k) for k in ("input_tokens", "output_tokens")}
    if any(type(v) is not int or v < 0 for v in tokens.values()):  # type: ignore[operator]
        raise LedgerBlocked("token usage missing or invalid; reservation kept for reconciliation")
    return money((quote.input_usd_per_million * tokens["input_tokens"]  # type: ignore[operator]
                  + quote.output_usd_per_million * tokens["output_tokens"]) / MILLION)  # type: ignore[operator]


def example_price_table(valid_days: int = 365) -> PriceTable:
    """Placeholder models with ILLUSTRATIVE prices (USD per 1M tokens). Not official prices."""
    def mp(i: str, o: str, in_limit: int = 200_000, out_limit: int = 32_000) -> ModelPrice:
        return ModelPrice(Decimal(i), Decimal(o), in_limit, out_limit)

    return PriceTable(
        version="example-1",
        valid_until=datetime.now(UTC) + timedelta(days=valid_days),
        models={
            "openai/gpt-large": mp("2.50", "15.00"),
            "openai/gpt-small": mp("0.75", "4.50"),
            "anthropic/claude-large": mp("3.00", "15.00"),
            "anthropic/claude-small": mp("1.00", "5.00"),
            "google/gemini-flash": mp("0.75", "3.75", 1_000_000),
            "google/gemini-flash-lite": mp("0.25", "1.50", 1_000_000),
        },
    )
