"""Pluggable persistence for the ledger, cost log, call audit and model metrics.

The ledger needs check-and-reserve to be atomic across workers, so the interface exposes
``transaction()``: everything inside it must be serialised against other writers.
``InMemoryStorage`` uses a re-entrant lock (single process). ``SQLiteStorage`` uses
``BEGIN IMMEDIATE``, which takes the database write lock up front, so several processes
sharing one file cannot both pass the balance check.
"""
from __future__ import annotations

import copy
import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

ZERO = Decimal(0)


@dataclass
class Balance:
    day: str
    scope: str  # "*" for the global ceiling, or a stage name
    reserved_usd: Decimal = ZERO
    spent_usd: Decimal = ZERO
    frozen: bool = False


@dataclass
class Reservation:
    id: str
    day: str
    stage: str
    provider: str
    model: str
    quote: dict[str, Any]
    reserved_usd: Decimal
    state: str = "reserved"  # reserved -> estimated -> reconciled
    estimated_usd: Decimal | None = None
    observed_usd: Decimal | None = None
    usage: dict[str, int] | None = None
    reconciled_by: str | None = None
    evidence: str | None = None
    subject: str | None = None


@dataclass
class CostRecord:
    key: str  # idempotency key (the reservation id): a cost is written at most once
    stage: str
    provider: str
    model: str
    cost_usd: Decimal
    usage: dict[str, int] = field(default_factory=dict)
    estimated: bool = True
    at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class MetricRow:
    """Aggregated score of one model for one task and segment (e.g. language, or language/topic)."""

    task: str
    segment: str
    provider: str
    model: str
    reward: float
    quality: float
    sample_n: int
    evidence: dict[str, Any] = field(default_factory=dict)
    at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@runtime_checkable
class Storage(Protocol):
    def transaction(self) -> Any: ...
    def get_balance(self, day: str, scope: str) -> Balance: ...
    def put_balance(self, balance: Balance) -> None: ...
    def get_reservation(self, reservation_id: str) -> Reservation | None: ...
    def put_reservation(self, reservation: Reservation) -> None: ...
    def add_cost(self, record: CostRecord) -> bool: ...
    def list_costs(self) -> list[CostRecord]: ...
    def put_audit(self, call_id: str, snapshot: dict[str, Any]) -> bool: ...
    def get_audit(self, call_id: str) -> dict[str, Any] | None: ...
    def add_metric(self, row: MetricRow) -> None: ...
    def latest_metrics(self, task: str, segment: str) -> list[MetricRow]: ...


def _audit_can_advance(old: dict[str, Any] | None, new: dict[str, Any]) -> bool:
    """Snapshots only move forward, and a final status is immutable."""
    if old is None:
        return True
    return old.get("status") == "running" and int(new.get("revision", 0)) > int(old.get("revision", 0))


class InMemoryStorage:
    """Thread-safe, process-local storage. Ideal for tests, demos and single-process jobs."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._balances: dict[tuple[str, str], Balance] = {}
        self._reservations: dict[str, Reservation] = {}
        self._costs: dict[str, CostRecord] = {}
        self._audits: dict[str, dict[str, Any]] = {}
        self._metrics: list[MetricRow] = []

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._lock:
            yield

    def get_balance(self, day: str, scope: str) -> Balance:
        with self._lock:
            return copy.copy(self._balances.get((day, scope)) or Balance(day, scope))

    def put_balance(self, balance: Balance) -> None:
        with self._lock:
            self._balances[(balance.day, balance.scope)] = copy.copy(balance)

    def get_reservation(self, reservation_id: str) -> Reservation | None:
        with self._lock:
            found = self._reservations.get(reservation_id)
            return copy.deepcopy(found) if found else None

    def put_reservation(self, reservation: Reservation) -> None:
        with self._lock:
            self._reservations[reservation.id] = copy.deepcopy(reservation)

    def add_cost(self, record: CostRecord) -> bool:
        with self._lock:
            if record.key in self._costs:
                return False
            self._costs[record.key] = copy.deepcopy(record)
            return True

    def list_costs(self) -> list[CostRecord]:
        with self._lock:
            return [copy.deepcopy(c) for c in self._costs.values()]

    def put_audit(self, call_id: str, snapshot: dict[str, Any]) -> bool:
        with self._lock:
            if not _audit_can_advance(self._audits.get(call_id), snapshot):
                return False
            self._audits[call_id] = copy.deepcopy(snapshot)
            return True

    def get_audit(self, call_id: str) -> dict[str, Any] | None:
        with self._lock:
            found = self._audits.get(call_id)
            return copy.deepcopy(found) if found else None

    def add_metric(self, row: MetricRow) -> None:
        with self._lock:
            self._metrics.append(copy.deepcopy(row))

    def latest_metrics(self, task: str, segment: str) -> list[MetricRow]:
        with self._lock:
            latest: dict[tuple[str, str], MetricRow] = {}
            for row in self._metrics:  # insertion order: later rows win
                if row.task == task and row.segment == segment:
                    latest[(row.provider, row.model)] = row
            return [copy.deepcopy(r) for r in latest.values()]


# --------------------------------------------------------------------------- SQLite

def _dumps(obj: Any) -> str:
    return json.dumps(asdict(obj) if hasattr(obj, "__dataclass_fields__") else obj,
                      default=str, ensure_ascii=False, sort_keys=True)


def _dec(value: Any) -> Decimal | None:
    return None if value is None else Decimal(str(value))


_SCHEMA = """
CREATE TABLE IF NOT EXISTS balances (day TEXT, scope TEXT, data TEXT NOT NULL, PRIMARY KEY (day, scope));
CREATE TABLE IF NOT EXISTS reservations (id TEXT PRIMARY KEY, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS costs (key TEXT PRIMARY KEY, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audits (call_id TEXT PRIMARY KEY, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS metrics (id INTEGER PRIMARY KEY AUTOINCREMENT, task TEXT, segment TEXT,
                                    provider TEXT, model TEXT, data TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS metrics_lookup ON metrics (task, segment, provider, model, id);
"""


class SQLiteStorage:
    """Durable storage on a single SQLite file, safe for several processes on one host."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False, timeout=30)
        self._lock = threading.RLock()
        self._depth = 0
        self._conn.execute("PRAGMA journal_mode=WAL" if str(path) != ":memory:" else "PRAGMA journal_mode=MEMORY")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._lock:
            outer = self._depth == 0
            if outer:
                self._conn.execute("BEGIN IMMEDIATE")
            self._depth += 1
            try:
                yield
            except BaseException:
                self._depth -= 1
                if outer:
                    self._conn.execute("ROLLBACK")
                raise
            self._depth -= 1
            if outer:
                self._conn.execute("COMMIT")

    def _one(self, sql: str, args: tuple[Any, ...]) -> Any:
        with self._lock:
            row = self._conn.execute(sql, args).fetchone()
        return json.loads(row[0]) if row else None

    def _write(self, sql: str, args: tuple[Any, ...]) -> int:
        with self.transaction():
            return self._conn.execute(sql, args).rowcount

    def get_balance(self, day: str, scope: str) -> Balance:
        raw = self._one("SELECT data FROM balances WHERE day=? AND scope=?", (day, scope))
        if not raw:
            return Balance(day, scope)
        return Balance(day, scope, Decimal(raw["reserved_usd"]), Decimal(raw["spent_usd"]), bool(raw["frozen"]))

    def put_balance(self, balance: Balance) -> None:
        self._write("INSERT OR REPLACE INTO balances (day, scope, data) VALUES (?, ?, ?)",
                    (balance.day, balance.scope, _dumps(balance)))

    def get_reservation(self, reservation_id: str) -> Reservation | None:
        raw = self._one("SELECT data FROM reservations WHERE id=?", (reservation_id,))
        if not raw:
            return None
        for key in ("reserved_usd", "estimated_usd", "observed_usd"):
            raw[key] = _dec(raw[key])
        return Reservation(**raw)

    def put_reservation(self, reservation: Reservation) -> None:
        self._write("INSERT OR REPLACE INTO reservations (id, data) VALUES (?, ?)",
                    (reservation.id, _dumps(reservation)))

    def add_cost(self, record: CostRecord) -> bool:
        return self._write("INSERT OR IGNORE INTO costs (key, data) VALUES (?, ?)",
                           (record.key, _dumps(record))) == 1

    def list_costs(self) -> list[CostRecord]:
        with self._lock:
            rows = self._conn.execute("SELECT data FROM costs ORDER BY rowid").fetchall()
        out = []
        for (data,) in rows:
            raw = json.loads(data)
            raw["cost_usd"] = Decimal(raw["cost_usd"])
            out.append(CostRecord(**raw))
        return out

    def put_audit(self, call_id: str, snapshot: dict[str, Any]) -> bool:
        with self.transaction():
            old = self._one("SELECT data FROM audits WHERE call_id=?", (call_id,))
            if not _audit_can_advance(old, snapshot):
                return False
            self._conn.execute("INSERT OR REPLACE INTO audits (call_id, data) VALUES (?, ?)",
                               (call_id, json.dumps(snapshot, default=str)))
            return True

    def get_audit(self, call_id: str) -> dict[str, Any] | None:
        result: dict[str, Any] | None = self._one("SELECT data FROM audits WHERE call_id=?", (call_id,))
        return result

    def add_metric(self, row: MetricRow) -> None:
        self._write("INSERT INTO metrics (task, segment, provider, model, data) VALUES (?, ?, ?, ?, ?)",
                    (row.task, row.segment, row.provider, row.model, _dumps(row)))

    def latest_metrics(self, task: str, segment: str) -> list[MetricRow]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT m.data FROM metrics m JOIN (SELECT provider, model, max(id) AS id FROM metrics "
                "WHERE task=? AND segment=? GROUP BY provider, model) last ON last.id = m.id ORDER BY m.id",
                (task, segment)).fetchall()
        return [MetricRow(**json.loads(data)) for (data,) in rows]
