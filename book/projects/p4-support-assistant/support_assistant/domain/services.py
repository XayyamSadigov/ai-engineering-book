# path: book/projects/p4-support-assistant/support_assistant/domain/services.py
"""Service status, reply drafts, and the outbox (the external system send_reply reaches)."""
from __future__ import annotations

import itertools
import threading
from datetime import datetime, timezone

from pydantic import BaseModel
from toolkit import ToolError

SERVICES = ("vpn", "email", "pos-payments", "tracking-api", "route-planner", "warehouse-scanners", "identity")


class StatusBoard:
    """Stand-in for a status-page API. `fail_next(n)` simulates an upstream outage."""

    def __init__(self) -> None:
        self._status = {s: "operational" for s in SERVICES}
        self._status.update({"vpn": "degraded", "tracking-api": "operational"})
        self._notes = {"vpn": "Intermittent tunnel drops for retail stores; network team investigating (INC-2026-0412)."}
        self._failures = 0
        self._lock = threading.Lock()

    def set(self, service: str, status: str, note: str = "") -> None:
        self._status[service] = status
        self._notes[service] = note

    def fail_next(self, n: int) -> None:
        self._failures = n

    def get(self, service: str) -> dict[str, str]:
        with self._lock:
            if self._failures > 0:
                self._failures -= 1
                raise ToolError.transient("status_api_unavailable", "status API returned 503", retry_after_s=0.01)
        return {"service": service, "status": self._status[service], "note": self._notes.get(service, ""),
                "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


class Draft(BaseModel):
    id: str
    ticket_id: str
    to: str
    subject: str
    body: str
    author: str


class DraftStore:
    def __init__(self) -> None:
        self._items: dict[str, Draft] = {}
        self._ids = itertools.count(1)

    def create(self, **fields: str) -> Draft:
        d = Draft(id=f"DRF-{next(self._ids):04d}", **fields)
        self._items[d.id] = d
        return d

    def all(self) -> list[Draft]:
        return list(self._items.values())


class SentMessage(BaseModel):
    message_id: str
    to: str
    subject: str
    body: str
    ticket_id: str | None
    sent_by: str
    idempotency_key: str | None


class Outbox:
    """The external email gateway. Accepts an idempotency key, as good APIs do."""

    def __init__(self) -> None:
        self.sent: list[SentMessage] = []
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def send(self, *, to: str, subject: str, body: str, ticket_id: str | None, sent_by: str,
             idempotency_key: str | None) -> SentMessage:
        with self._lock:
            for m in self.sent:  # provider-side dedupe, the last line of defense
                if idempotency_key and m.idempotency_key == idempotency_key:
                    return m
            m = SentMessage(message_id=f"MSG-{next(self._ids):05d}", to=to, subject=subject, body=body,
                            ticket_id=ticket_id, sent_by=sent_by, idempotency_key=idempotency_key)
            self.sent.append(m)
            return m

    def find(self, idempotency_key: str) -> SentMessage | None:
        return next((m for m in self.sent if m.idempotency_key == idempotency_key), None)


__all__ = ["SERVICES", "StatusBoard", "Draft", "DraftStore", "SentMessage", "Outbox"]
