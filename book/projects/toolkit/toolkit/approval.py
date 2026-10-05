# path: book/projects/toolkit/toolkit/approval.py
"""Human approval bound to one exact action.

An approval is not "the user said yes earlier". It is a record that names a tool and the
hash of its validated arguments, has an expiry, can be used once, and is decided through
a channel the model cannot write to (an API endpoint, not the chat transcript).
"""
from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .policy import ToolContext
from .registry import args_hash


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CONSUMED = "consumed"   # approved and executed; cannot be reused


class ApprovalRequest(BaseModel):
    id: str
    tool_name: str
    arguments: dict[str, Any]          # validated, normalized; exactly what will run
    args_hash: str
    user_id: str
    tenant: str
    session_id: str | None = None
    reasons: list[str] = Field(default_factory=list)
    summary: str = ""                  # human-readable rendering shown to the approver
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: float
    expires_at: float
    decided_by: str | None = None
    decided_at: float | None = None
    note: str | None = None


class ApprovalError(Exception):
    """Raised when an approval cannot authorize the call. `code` is machine-readable."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ApprovalManager:
    """In-memory approval queue. Swap for a table with the same methods in production.

    `allow_self_approval=False` enforces four-eyes: the requester cannot approve their own
    action. Leave it on for assistants where the human operator is the requester and the
    threat is the *model* approving, which this design already prevents.
    """

    def __init__(self, *, ttl_s: float = 900.0, allow_self_approval: bool = True,
                 clock: Callable[[], float] = time.time) -> None:
        self.ttl_s = ttl_s
        self.allow_self_approval = allow_self_approval
        self._clock = clock
        self._items: dict[str, ApprovalRequest] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------- lifecycle
    def request(self, tool_name: str, arguments: dict[str, Any], ctx: ToolContext, *,
                reasons: list[str] | None = None, summary: str = "") -> ApprovalRequest:
        """Create (or return the existing pending) request for this exact action."""
        h = args_hash(tool_name, arguments)
        now = self._clock()
        with self._lock:
            for item in self._items.values():
                self._expire_if_needed(item, now)
                if (item.status == ApprovalStatus.PENDING and item.args_hash == h
                        and item.user_id == ctx.user_id):
                    return item
            item = ApprovalRequest(
                id=f"apr_{uuid.uuid4().hex[:12]}", tool_name=tool_name, arguments=arguments,
                args_hash=h, user_id=ctx.user_id, tenant=ctx.tenant, session_id=ctx.session_id,
                reasons=list(reasons or []), summary=summary or f"{tool_name}({arguments})",
                created_at=now, expires_at=now + self.ttl_s,
            )
            self._items[item.id] = item
            return item

    def approve(self, request_id: str, approver_id: str, note: str | None = None) -> ApprovalRequest:
        return self._decide(request_id, approver_id, ApprovalStatus.APPROVED, note)

    def reject(self, request_id: str, approver_id: str, note: str | None = None) -> ApprovalRequest:
        return self._decide(request_id, approver_id, ApprovalStatus.REJECTED, note)

    def verify(self, request_id: str, tool_name: str, arguments_hash: str, ctx: ToolContext) -> ApprovalRequest:
        """Check that `request_id` authorizes exactly this call by this principal right now."""
        with self._lock:
            item = self._items.get(request_id)
            if item is None:
                raise ApprovalError("approval_not_found", f"no approval {request_id}")
            self._expire_if_needed(item, self._clock())
            if item.tool_name != tool_name or item.args_hash != arguments_hash:
                raise ApprovalError("approval_mismatch",
                                    "approval was granted for a different action or different arguments")
            if item.user_id != ctx.user_id or item.tenant != ctx.tenant:
                raise ApprovalError("approval_mismatch", "approval belongs to another principal")
            if item.status != ApprovalStatus.APPROVED:
                raise ApprovalError(f"approval_{item.status.value}", f"approval is {item.status.value}")
            return item

    def consume(self, request_id: str) -> None:
        """Mark an approval used after the action succeeded. Single use."""
        with self._lock:
            item = self._items[request_id]
            if item.status == ApprovalStatus.APPROVED:
                item.status = ApprovalStatus.CONSUMED

    # ---------------------------------------------------------------------- queries
    def get(self, request_id: str) -> ApprovalRequest | None:
        with self._lock:
            item = self._items.get(request_id)
            if item is not None:
                self._expire_if_needed(item, self._clock())
            return item

    def pending(self, *, tenant: str | None = None, user_id: str | None = None) -> list[ApprovalRequest]:
        now = self._clock()
        with self._lock:
            out = []
            for item in self._items.values():
                self._expire_if_needed(item, now)
                if item.status != ApprovalStatus.PENDING:
                    continue
                if tenant is not None and item.tenant != tenant:
                    continue
                if user_id is not None and item.user_id != user_id:
                    continue
                out.append(item)
            return sorted(out, key=lambda i: i.created_at)

    # -------------------------------------------------------------------- internals
    def _decide(self, request_id: str, approver_id: str, status: ApprovalStatus, note: str | None) -> ApprovalRequest:
        with self._lock:
            item = self._items.get(request_id)
            if item is None:
                raise ApprovalError("approval_not_found", f"no approval {request_id}")
            self._expire_if_needed(item, self._clock())
            if item.status != ApprovalStatus.PENDING:
                raise ApprovalError(f"approval_{item.status.value}", f"approval is already {item.status.value}")
            if not self.allow_self_approval and approver_id == item.user_id:
                raise ApprovalError("self_approval", "requester cannot approve their own action")
            item.status, item.decided_by, item.decided_at, item.note = status, approver_id, self._clock(), note
            return item

    @staticmethod
    def _expire_if_needed(item: ApprovalRequest, now: float) -> None:
        if item.status in (ApprovalStatus.PENDING, ApprovalStatus.APPROVED) and now >= item.expires_at:
            item.status = ApprovalStatus.EXPIRED


__all__ = ["ApprovalStatus", "ApprovalRequest", "ApprovalError", "ApprovalManager"]
