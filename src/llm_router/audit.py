"""Per-call audit trail: metadata of every attempt. Never stores prompts or responses."""
from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat()


class CallAudit:
    """Builds monotonic snapshots (``revision`` always grows) of one router call.

    ``reconciliation_required`` is True whenever some attempt may have been billed
    without a settled cost: the call is still running, a cost is unknown, or a
    reservation was opened and not settled.
    """

    SCHEMA_VERSION = 1

    def __init__(self, call_id: str, stage: str) -> None:
        self.call_id = call_id
        self.stage = stage
        self.details: list[dict[str, Any]] = []
        self.revision = 0
        self.started_at = _now()

    def start(self, provider: str, model: str) -> dict[str, Any]:
        item: dict[str, Any] = {"attempt": len(self.details) + 1, "provider": provider, "model": model,
                                "status": "started", "started_at": _now(),
                                "estimated_cost_usd": None, "budget_recorded": False,
                                "reservation_id": None}
        self.details.append(item)
        return item

    @staticmethod
    def finish(item: dict[str, Any], status: str, error: BaseException | None = None) -> None:
        item.update(status=status, finished_at=_now())
        if error is not None:
            item["error_type"] = type(error).__name__

    def snapshot(self, status: str, **extra: Any) -> dict[str, Any]:
        self.revision += 1
        complete = all(item["estimated_cost_usd"] is not None for item in self.details)
        known = sum((Decimal(str(item["estimated_cost_usd"])) for item in self.details
                     if item["estimated_cost_usd"] is not None), Decimal(0))
        pending = any(item["reservation_id"] and not item["budget_recorded"] for item in self.details)
        return {
            "schema_version": self.SCHEMA_VERSION, "call_id": self.call_id, "stage": self.stage,
            "revision": self.revision, "status": status,
            "started_at": self.started_at, "updated_at": _now(),
            "attempts": len(self.details), "attempt_details": deepcopy(self.details),
            "known_estimated_cost_usd": str(known),
            "total_estimated_cost_usd": str(known) if complete else None,
            "cost_complete": complete,
            "reconciliation_required": status == "running" or not complete or pending,
            **extra,
        }
