"""llm_router: budget-aware, auditable LLM routing with independent verification."""
from llm_router.budget import BudgetLedger
from llm_router.config import Route, RouterConfig, Stage
from llm_router.errors import (
    BudgetExceeded,
    InvalidJSON,
    LedgerBlocked,
    LLMError,
    RouterError,
    SelectorError,
    UnpricedModel,
)
from llm_router.gates import GateResult, Issue, Verdict, apply_gates
from llm_router.pricing import ModelPrice, PriceTable, example_price_table
from llm_router.providers import FakeProvider
from llm_router.router import CallResult, Router
from llm_router.selector import CandidateProfile, Sample, Selection, SelectionRequest, Selector, SelectorConfig
from llm_router.storage import InMemoryStorage, SQLiteStorage, Storage
from llm_router.tournament import Tournament, TournamentConfig

__version__ = "0.1.0"

__all__ = [
    "BudgetExceeded", "BudgetLedger", "CallResult", "CandidateProfile", "FakeProvider", "GateResult",
    "InMemoryStorage", "InvalidJSON", "Issue", "LLMError", "LedgerBlocked", "ModelPrice", "PriceTable",
    "Route", "Router", "RouterConfig", "RouterError", "SQLiteStorage", "Sample", "Selection",
    "SelectionRequest", "Selector", "SelectorConfig", "SelectorError", "Stage", "Storage", "Tournament",
    "TournamentConfig", "UnpricedModel", "Verdict", "apply_gates", "build_router", "example_price_table",
]


def build_router(config: RouterConfig, providers: dict, *, storage: Storage | None = None,
                 prices: PriceTable | None = None, selector: Selector | None = None, **kwargs: object) -> Router:
    """Convenience wiring: storage + ledger + router in one call."""
    storage = storage or InMemoryStorage()
    ledger = BudgetLedger(storage, prices or example_price_table(), config)
    return Router(config=config, providers=providers, ledger=ledger, storage=storage, selector=selector,
                  **kwargs)  # type: ignore[arg-type]
