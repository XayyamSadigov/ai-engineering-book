# path: book/projects/toolkit/toolkit/audit.py
"""Append-only audit events for every tool decision and execution.

Events carry the arguments *hash* by default, not the arguments: the hash is enough to
correlate a proposal, its approval, and its execution, and it keeps personal data out of
a log that many people can read. Enable `include_arguments` only for sinks with the access
controls and retention of the system of record.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

AuditEventType = Literal[
    "tool.proposed",
    "tool.denied",
    "tool.invalid",
    "tool.approval_requested",
    "tool.approval_rejected",
    "tool.duplicate_suppressed",
    "tool.retry",
    "tool.executed",
    "tool.failed",
]


class AuditEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    ts: float = Field(default_factory=time.time)
    event_type: AuditEventType
    tool_name: str
    call_id: str | None = None
    user_id: str | None = None
    tenant: str | None = None
    session_id: str | None = None
    request_id: str | None = None
    args_hash: str | None = None
    tool_fingerprint: str | None = None
    policy_version: str | None = None
    verdict: str | None = None
    rules: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    error_category: str | None = None
    error_code: str | None = None
    attempt: int | None = None
    latency_ms: float | None = None
    approval_id: str | None = None
    idempotency_key: str | None = None
    arguments: dict[str, Any] | None = None


class AuditSink(Protocol):
    def emit(self, event: AuditEvent) -> None: ...


class InMemoryAuditLog:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []
        self._lock = threading.Lock()

    def emit(self, event: AuditEvent) -> None:
        with self._lock:
            self.events.append(event)

    def types(self) -> list[str]:
        return [e.event_type for e in self.events]

    def of_type(self, event_type: str) -> list[AuditEvent]:
        return [e for e in self.events if e.event_type == event_type]


class JsonlAuditLog:
    """One JSON object per line, append-only. Ship the file to your log pipeline."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, event: AuditEvent) -> None:
        line = json.dumps(event.model_dump(mode="json", exclude_none=True), ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


class NullAuditLog:
    def emit(self, event: AuditEvent) -> None:
        return None


__all__ = ["AuditEventType", "AuditEvent", "AuditSink", "InMemoryAuditLog", "JsonlAuditLog", "NullAuditLog"]
