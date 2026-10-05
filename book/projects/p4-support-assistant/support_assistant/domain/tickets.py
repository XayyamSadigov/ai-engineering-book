# path: book/projects/p4-support-assistant/support_assistant/domain/tickets.py
"""Ticket store over the shared Northwind tickets plus tickets created by the assistant."""
from __future__ import annotations

import json
import re
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel


def load_shared_tickets(shared_data_dir: Path, extra_path: Path | None = None) -> list["TicketRecord"]:
    """Load tickets.jsonl through the shared-data loader, then any local fixtures."""
    if str(shared_data_dir) not in sys.path:
        sys.path.insert(0, str(shared_data_dir))
    from shared_data import Ticket, load_tickets  # noqa: PLC0415 - path set above

    tickets = load_tickets(shared_data_dir / "tickets.jsonl")
    if extra_path is not None and extra_path.exists():
        for line in extra_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                tickets.append(Ticket.model_validate(json.loads(line)))
    return [TicketRecord(**t.model_dump()) for t in tickets]


class TicketRecord(BaseModel):
    id: str
    created_at: str
    tenant: str
    requester_role: str
    channel: str
    subject: str
    body: str
    category: str
    priority: str
    status: str
    resolution: str | None = None
    created_by: str | None = None
    idempotency_key: str | None = None


_WORD = re.compile(r"[a-z0-9]+")


class TicketStore:
    def __init__(self, tickets: list[TicketRecord]) -> None:
        self._tickets = {t.id: t for t in tickets}
        self._lock = threading.Lock()
        self._seq = 1000

    def get(self, ticket_id: str, *, tenant: str) -> TicketRecord | None:
        t = self._tickets.get(ticket_id)
        return t if t is not None and t.tenant in (tenant, "shared") else None

    def search(self, query: str, *, tenant: str, status: str = "any", limit: int = 5) -> list[TicketRecord]:
        terms = set(_WORD.findall(query.lower()))
        scored = []
        for t in self._tickets.values():
            if t.tenant not in (tenant, "shared") or (status != "any" and t.status != status):
                continue
            words = set(_WORD.findall(f"{t.subject} {t.body} {t.category}".lower()))
            score = len(terms & words)
            if score:
                scored.append((score, t.created_at, t))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [t for _, _, t in scored[:limit]]

    def create(self, *, subject: str, body: str, category: str, priority: str, tenant: str,
               created_by: str, idempotency_key: str | None) -> TicketRecord:
        with self._lock:
            self._seq += 1
            t = TicketRecord(id=f"TCK-2026-{self._seq:04d}", created_at=datetime.now(timezone.utc).isoformat(),
                             tenant=tenant, requester_role="employee", channel="chat", subject=subject, body=body,
                             category=category, priority=priority, status="open", created_by=created_by,
                             idempotency_key=idempotency_key)
            self._tickets[t.id] = t
            return t

    def find_by_idempotency_key(self, key: str) -> TicketRecord | None:
        return next((t for t in self._tickets.values() if key and t.idempotency_key == key), None)

    def created(self) -> list[TicketRecord]:
        return [t for t in self._tickets.values() if t.created_by is not None]


__all__ = ["TicketRecord", "TicketStore", "load_shared_tickets"]
