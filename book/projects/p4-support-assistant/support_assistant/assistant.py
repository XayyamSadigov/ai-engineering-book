# path: book/projects/p4-support-assistant/support_assistant/assistant.py
"""Application service: conversations, the tool loop, and the approval channel."""
from __future__ import annotations

import threading
import uuid

from aie_core.llm.types import Message
from pydantic import BaseModel
from toolkit import ApprovalError, ApprovalRequest, ToolResult

from .domain.directory import STAFF, StaffAccount
from .prompts import SYSTEM_PROMPT
from .wiring import Container


class UnknownUser(Exception):
    pass


class NotAuthorized(Exception):
    pass


class ToolCallView(BaseModel):
    tool: str
    status: str
    error: dict | None = None
    duplicate: bool = False
    approval_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    stop_reason: str
    tool_calls: list[ToolCallView]
    pending_approvals: list[ApprovalRequest]


class SupportAssistant:
    def __init__(self, container: Container, staff: dict[str, StaffAccount] | None = None) -> None:
        self.c = container
        self.staff = staff or STAFF
        self._sessions: dict[str, list[Message]] = {}
        self._owners: dict[str, str] = {}
        self._lock = threading.Lock()

    def account(self, user_id: str) -> StaffAccount:
        try:
            return self.staff[user_id]
        except KeyError:
            raise UnknownUser(user_id) from None

    # ------------------------------------------------------------------------ chat
    def chat(self, user_id: str, text: str, session_id: str | None = None,
             request_id: str | None = None) -> ChatResponse:
        acct = self.account(user_id)
        with self._lock:
            session_id = session_id or f"ses_{uuid.uuid4().hex[:10]}"
            owner = self._owners.setdefault(session_id, user_id)
            if owner != user_id:
                raise NotAuthorized("session belongs to another user")
            history = self._sessions.setdefault(session_id, [Message.system(SYSTEM_PROMPT)])
        ctx = acct.to_context(session_id=session_id, request_id=request_id)
        result = self.c.loop.run([*history, Message.user(text)], ctx, task_tags=["support"],
                                 metadata={"session_id": session_id})
        with self._lock:
            self._sessions[session_id] = result.messages
        pending = [a for a in (self.c.approvals.get(i) for i in result.pending_approvals) if a is not None]
        return ChatResponse(session_id=session_id, reply=result.final_text, stop_reason=result.stop_reason,
                            tool_calls=[_view(r) for r in result.tool_results], pending_approvals=pending)

    # ------------------------------------------------------------------- approvals
    def pending(self, approver_id: str) -> list[ApprovalRequest]:
        acct = self.account(approver_id)
        if "replies:approve" in acct.scopes:
            return self.c.approvals.pending(tenant=acct.tenant)
        return self.c.approvals.pending(tenant=acct.tenant, user_id=approver_id)

    def approve(self, approval_id: str, approver_id: str, note: str | None = None) -> ToolResult:
        """Record the decision, then run exactly the approved action as the requester."""
        req = self._approval_for(approval_id, approver_id)
        self.c.approvals.approve(approval_id, approver_id, note)
        requester = self.account(req.user_id).to_context(session_id=req.session_id)
        result = self.c.executor.execute_approved(approval_id, requester)
        self._note(req, f"[approval {approval_id} approved by {approver_id}; result: {result.content}]")
        return result

    def reject(self, approval_id: str, approver_id: str, note: str | None = None) -> ApprovalRequest:
        self._approval_for(approval_id, approver_id)
        req = self.c.approvals.reject(approval_id, approver_id, note)
        self._note(req, f"[approval {approval_id} rejected by {approver_id}: {note or 'no reason given'}]")
        return req

    # -------------------------------------------------------------------- internals
    def _approval_for(self, approval_id: str, approver_id: str) -> ApprovalRequest:
        """Leads decide any approval in their tenant; requesters decide their own only when
        self-approval is enabled. ApprovalManager enforces four-eyes again underneath."""
        acct = self.account(approver_id)
        req = self.c.approvals.get(approval_id)
        if req is None or req.tenant != acct.tenant:
            raise ApprovalError("approval_not_found", f"no approval {approval_id}")
        is_lead = "replies:approve" in acct.scopes
        is_own = approver_id == req.user_id and self.c.approvals.allow_self_approval
        if not (is_lead or is_own):
            raise NotAuthorized("only a lead, or the requester when self-approval is enabled, may decide")
        return req

    def _note(self, req: ApprovalRequest, text: str) -> None:
        # Decisions enter the transcript as system-authored notes, so the model sees the
        # outcome on the next turn but can never author an approval itself.
        if req.session_id and req.session_id in self._sessions:
            with self._lock:
                self._sessions[req.session_id].append(Message.system(text))

    def transcript(self, session_id: str) -> list[Message]:
        return list(self._sessions.get(session_id, []))


def _view(r: ToolResult) -> ToolCallView:
    return ToolCallView(tool=r.tool_name, status=r.status, error=r.error, duplicate=r.duplicate,
                        approval_id=r.approval_id)


__all__ = ["SupportAssistant", "ChatResponse", "ToolCallView", "UnknownUser", "NotAuthorized"]
