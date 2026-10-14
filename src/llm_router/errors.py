"""Exception hierarchy. Every error raised by the library derives from ``RouterError``."""
from __future__ import annotations


class RouterError(Exception):
    """Base class for all library errors."""


class LLMError(RouterError):
    """No route in the chain produced a valid answer (or a paid answer cannot be retried safely)."""


class InvalidJSON(LLMError):
    """The model answered, but no JSON object could be extracted from the text."""


class BudgetExceeded(RouterError):
    """The daily ceiling (global or per stage) does not cover the worst-case cost of the next call."""


class LedgerBlocked(BudgetExceeded):
    """The ledger refused to reserve or settle (frozen day, invalid amount, closed reservation...)."""


class UnpricedModel(RouterError):
    """The route has no valid entry in the price table, so its worst-case cost is unknown.

    The router skips such routes instead of calling them: a call that cannot be priced
    in advance cannot be reserved, and an unreserved paid call breaks the budget guarantee.
    """


class SelectorError(RouterError):
    """Invalid selector input (candidate profile, request or statistics)."""
