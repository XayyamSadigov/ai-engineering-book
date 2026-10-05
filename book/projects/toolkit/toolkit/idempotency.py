# path: book/projects/toolkit/toolkit/idempotency.py
"""Duplicate suppression for side-effecting tools.

A key names one logical action. The first caller reserves it (`in_progress`), runs the
side effect, and records the result (`succeeded`). Any later caller with the same key gets
the recorded result instead of a second side effect. If the outcome could not be
determined (a timeout after the request left), the record becomes `unknown` and the
action is not repeated automatically.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

IdemStatus = Literal["in_progress", "succeeded", "unknown"]


class IdempotencyRecord(BaseModel):
    key: str
    tool_name: str
    args_hash: str
    status: IdemStatus
    result: Any = None
    created_at: float
    updated_at: float
    expires_at: float


@runtime_checkable
class IdempotencyStore(Protocol):
    def begin(self, key: str, tool_name: str, args_hash: str, ttl_s: float) -> IdempotencyRecord | None:
        """Reserve `key`. Returns None if this caller now owns it, else the existing record."""

    def get(self, key: str) -> IdempotencyRecord | None: ...

    def complete(self, key: str, result: Any) -> None: ...

    def mark_unknown(self, key: str) -> None: ...

    def release(self, key: str) -> None:
        """Forget a reservation whose action definitely did not happen, so it may be retried."""


class InMemoryIdempotencyStore:
    """Single-process store. Correct for tests and one-replica deployments only."""

    def __init__(self, *, lease_s: float = 120.0, clock: Callable[[], float] = time.time) -> None:
        self.lease_s = lease_s
        self._clock = clock
        self._items: dict[str, IdempotencyRecord] = {}
        self._lock = threading.Lock()

    def begin(self, key: str, tool_name: str, args_hash: str, ttl_s: float) -> IdempotencyRecord | None:
        now = self._clock()
        with self._lock:
            rec = self._items.get(key)
            if rec is not None and rec.expires_at <= now:
                rec = None
            if rec is None:
                self._items[key] = IdempotencyRecord(key=key, tool_name=tool_name, args_hash=args_hash,
                                                     status="in_progress", created_at=now, updated_at=now,
                                                     expires_at=now + ttl_s)
                return None
            if rec.status == "in_progress" and now - rec.updated_at > self.lease_s:
                rec.status, rec.updated_at = "unknown", now  # owner died mid-flight
            return rec.model_copy()

    def get(self, key: str) -> IdempotencyRecord | None:
        with self._lock:
            rec = self._items.get(key)
            return rec.model_copy() if rec else None

    def complete(self, key: str, result: Any) -> None:
        with self._lock:
            rec = self._items[key]
            rec.status, rec.result, rec.updated_at = "succeeded", result, self._clock()

    def mark_unknown(self, key: str) -> None:
        with self._lock:
            if key in self._items:
                self._items[key].status = "unknown"
                self._items[key].updated_at = self._clock()

    def release(self, key: str) -> None:
        with self._lock:
            self._items.pop(key, None)


class SQLiteIdempotencyStore:
    """Durable store. The PRIMARY KEY plus BEGIN IMMEDIATE makes `begin` atomic across
    threads and processes sharing the file. The same SQL ports to PostgreSQL with
    ``INSERT ... ON CONFLICT DO NOTHING``.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS tool_idempotency (
        key TEXT PRIMARY KEY,
        tool_name TEXT NOT NULL,
        args_hash TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('in_progress','succeeded','unknown')),
        result_json TEXT,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        expires_at REAL NOT NULL
    )"""

    def __init__(self, path: str | Path = ":memory:", *, lease_s: float = 120.0,
                 clock: Callable[[], float] = time.time) -> None:
        self.lease_s = lease_s
        self._clock = clock
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None, timeout=10)
        self._conn.execute("PRAGMA journal_mode=WAL") if str(path) != ":memory:" else None
        self._conn.execute(self._SCHEMA)

    def _row(self, key: str) -> IdempotencyRecord | None:
        row = self._conn.execute(
            "SELECT key, tool_name, args_hash, status, result_json, created_at, updated_at, expires_at "
            "FROM tool_idempotency WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return IdempotencyRecord(key=row[0], tool_name=row[1], args_hash=row[2], status=row[3],
                                 result=json.loads(row[4]) if row[4] is not None else None,
                                 created_at=row[5], updated_at=row[6], expires_at=row[7])

    def begin(self, key: str, tool_name: str, args_hash: str, ttl_s: float) -> IdempotencyRecord | None:
        now = self._clock()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute("DELETE FROM tool_idempotency WHERE key = ? AND expires_at <= ?", (key, now))
                cur = self._conn.execute(
                    "INSERT INTO tool_idempotency (key, tool_name, args_hash, status, created_at, updated_at, expires_at) "
                    "VALUES (?, ?, ?, 'in_progress', ?, ?, ?) ON CONFLICT(key) DO NOTHING",
                    (key, tool_name, args_hash, now, now, now + ttl_s))
                if cur.rowcount == 1:
                    self._conn.execute("COMMIT")
                    return None
                self._conn.execute(
                    "UPDATE tool_idempotency SET status='unknown', updated_at=? "
                    "WHERE key=? AND status='in_progress' AND ? - updated_at > ?",
                    (now, key, now, self.lease_s))
                rec = self._row(key)
                self._conn.execute("COMMIT")
                return rec
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def get(self, key: str) -> IdempotencyRecord | None:
        with self._lock:
            return self._row(key)

    def complete(self, key: str, result: Any) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE tool_idempotency SET status='succeeded', result_json=?, updated_at=? WHERE key=?",
                (json.dumps(result, default=str), self._clock(), key))

    def mark_unknown(self, key: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE tool_idempotency SET status='unknown', updated_at=? WHERE key=?",
                               (self._clock(), key))

    def release(self, key: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM tool_idempotency WHERE key=?", (key,))

    def close(self) -> None:
        self._conn.close()


__all__ = ["IdemStatus", "IdempotencyRecord", "IdempotencyStore", "InMemoryIdempotencyStore",
           "SQLiteIdempotencyStore"]
