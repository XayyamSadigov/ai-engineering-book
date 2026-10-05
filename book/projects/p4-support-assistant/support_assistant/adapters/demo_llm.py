# path: book/projects/p4-support-assistant/support_assistant/adapters/demo_llm.py
"""A keyword-driven stand-in model so the service runs end to end without an API key.

It is deliberately dumb: it maps phrases to tool calls and turns tool results into a short
summary. Its only job is to exercise the real harness (policy, approvals, idempotency,
audit) from a browser or curl. Tests script FakeLLM directly instead.
"""
from __future__ import annotations

import json
import re
import uuid

from aie_core.llm.providers import FakeLLM
from aie_core.llm.types import CompletionRequest, Role, ToolCall

from ..domain.services import SERVICES

_TICKET = re.compile(r"TCK-\d{4}-\d{4}")
_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[A-Za-z]+")


def _call(name: str, **args) -> list[ToolCall]:
    return [ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name=name, arguments=args)]


def demo_handler(req: CompletionRequest):
    last = req.messages[-1]
    if last.role == Role.TOOL or req.tool_choice == "none":
        lines = []
        for m in reversed(req.messages):
            if m.role != Role.TOOL:
                break
            payload = json.loads(m.text)
            if payload.get("status") == "pending_approval":
                lines.append(f"Waiting for approval {payload['approval_id']}; nothing has been sent yet.")
            elif payload.get("ok"):
                lines.append("Result: " + json.dumps(payload["result"])[:600])
            else:
                err = payload["error"]
                lines.append(f"The tool refused ({err['category']}/{err['code']}): {err['message']}")
        return "\n".join(reversed(lines)) or "Done."
    text = last.text
    low = text.lower()
    ticket = _TICKET.search(text)
    email = _EMAIL.search(text)
    if "status" in low:
        service = next((s for s in SERVICES if s.split("-")[0] in low), "vpn")
        return _call("get_service_status", service=service)
    if low.startswith("send") and ticket and email:
        body = text.split(":", 1)[1].strip() if ":" in text else "Your issue has been resolved."
        return _call("send_reply", ticket_id=ticket.group(), to=email.group(), subject=f"Re: {ticket.group()}", body=body)
    if low.startswith("draft") and ticket and email:
        body = text.split(":", 1)[1].strip() if ":" in text else "We are looking into your issue."
        return _call("draft_reply", ticket_id=ticket.group(), to=email.group(), subject=f"Re: {ticket.group()}", body=body)
    if low.startswith("create ticket"):
        summary = text.split(":", 1)[1].strip() if ":" in text else "Issue reported via assistant"
        return _call("create_ticket", subject=summary[:120], body=f"Reported via chat: {summary}",
                     category="vpn_network" if "vpn" in low else "hardware", priority="P3")
    if low.startswith("who is") or "employee" in low:
        return _call("lookup_employee", query=text.split()[-1].strip("?"))
    return _call("search_tickets", query=text[:200], status="any", limit=3)


def make_demo_llm() -> FakeLLM:
    return FakeLLM(handler=demo_handler, model="demo-keyword-model")


__all__ = ["demo_handler", "make_demo_llm"]
