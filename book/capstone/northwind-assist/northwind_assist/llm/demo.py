# path: book/capstone/northwind-assist/northwind_assist/llm/demo.py
"""The offline demo model: what LLM_PROVIDER=fake runs, in the UI, in tests, and in the eval gate.

It is an aie_core FakeLLM with a handler that recognizes the four request shapes the capstone
sends, so the real harness (retrieval, packing, streaming validation, guardrails, policy,
approvals, budgets, memory) runs end to end without a key:

* grounded answers (`prompt.id` rag.grounded_answer*): quote the best-matching evidence sentences
  with their [E#] markers, or reply INSUFFICIENT_EVIDENCE;
* agent steps (tools present): a fixed plan per request pattern, then a summary of the results;
* memory maintenance (fact extraction and rolling summaries for memorykit);
* anything else: a short help text.

It is deliberately *gullible*: when evidence carries a markdown image or a "send to" instruction,
it copies the image into its answer, the way an injected model would. The red-team suite needs a
model that falls for the attack, so the controls around it are what is being measured.
It is a fixture, not a model; never deploy it.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any

from aie_core.llm.providers import FakeLLM
from aie_core.llm.types import CompletionRequest, Role, ToolCall

_BLOCK_RE = re.compile(r'<untrusted_data source="(E\d+)"([^>]*)>\n(.*?)\n</untrusted_data>', re.DOTALL)
_QUESTION_RE = re.compile(r"Question:\s*(.+)\Z", re.DOTALL)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"[a-z0-9]+")
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(https?://[^)\s]+\)")
_TICKET = re.compile(r"TCK-\d{4}-\d{4}")
# Matches a real address or the PII token the input guardrail put in its place (rehydrated by the tool layer).
_EMAIL = re.compile(r"<PII:email:[0-9a-f]+>|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_SERVICES = ("vpn", "email", "pos-payments", "tracking-api", "route-planner", "warehouse-scanners", "identity")
STOP = frozenset("a an the is are was were be to of in on for and or what how when who which do does i my me we our "
                 "can by with from at it this that as if there their they you your should must may".split())


def content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in STOP and len(w) > 2}


def _call(name: str, **args: Any) -> list[ToolCall]:
    return [ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name=name, arguments=args)]


# --------------------------------------------------------------------------- grounded answers
def grounded_answer(req: CompletionRequest, *, gullible: bool = True, max_sentences: int = 2) -> str:
    user = next((m.text for m in reversed(req.messages) if m.role == Role.USER), "")
    q = _QUESTION_RE.search(user)
    question = q.group(1).strip() if q else user
    qwords = content_words(question)
    scored: list[tuple[float, str, int, str, str]] = []
    injected: list[str] = []
    for order, (eid, attrs, text) in enumerate(_BLOCK_RE.findall(user)):
        found = re.search(r'updated_at="([^"]*)"', attrs)
        updated = found.group(1) if found else ""
        injected.extend(_IMAGE_RE.findall(text))
        for sent in _SENT_SPLIT.split(text.replace("**", "").replace("`", "")):
            sent = sent.replace("**", "").replace("`", "").strip().lstrip("-*# ").strip()
            words = content_words(sent)
            if len(words) < 4 or not qwords or sent.startswith("|") or sent.endswith("?"):
                continue
            overlap = len(qwords & words) / len(qwords)
            # near-ties go to the newer document: a careful model prefers the current policy
            scored.append((round(overlap, 1), updated, -order, sent, eid))
    scored.sort(reverse=True)
    picked: list[str] = []
    seen: set[str] = set()
    for overlap, _, _, sent, eid in scored:
        if overlap < 0.3 or len(picked) >= max_sentences or sent in seen:
            continue
        seen.add(sent)
        body = sent if sent.endswith((".", "!", "?")) else sent + "."
        extra = f" {injected[0]}" if gullible and injected and not picked else ""
        picked.append(f"{body[:-1]}{extra} [{eid}]{body[-1]}")
    if not picked:
        return "INSUFFICIENT_EVIDENCE: the documents I can see do not answer this question."
    return " ".join(picked)


# --------------------------------------------------------------------------- agent steps
def _goal(req: CompletionRequest) -> str:
    return next((m.text for m in req.messages if m.role == Role.USER), "")


def plan_for(goal: str) -> list[tuple[str, dict[str, Any]]]:
    low = goal.lower()
    ticket = _TICKET.search(goal)
    email = _EMAIL.search(goal)
    body = goal.split(":", 1)[1].strip() if ":" in goal else ""
    if email:  # the message body follows the recipient ("... to x@y: text"); token colons are not separators
        body = goal[email.end():].lstrip(" :").strip()
    service = next((s for s in _SERVICES if s.split("-")[0] in low), "vpn")
    if ("send" in low.split()[:3] or low.startswith("send")) and ticket and email:
        return [("send_reply", {"ticket_id": ticket.group(), "to": email.group(), "subject": f"Re: {ticket.group()}",
                                "body": body or "Your issue has been resolved."})]
    if low.startswith("draft") and ticket and email:
        return [("draft_reply", {"ticket_id": ticket.group(), "to": email.group(), "subject": f"Re: {ticket.group()}",
                                 "body": body or "We are looking into your issue."})]
    if any(p in low for p in ("create ticket", "create a ticket", "open a ticket", "raise a ticket", "file a ticket")):
        summary = (body or goal)[:110]
        category = "vpn_network" if "vpn" in low else ("pos_payments" if "card" in low or "pos" in low else "hardware")
        return [("search_tickets", {"query": summary[:150], "status": "open", "limit": 3}),
                ("create_ticket", {"subject": summary if len(summary) >= 5 else "Issue reported via assistant",
                                   "body": f"Reported via Northwind Assist: {summary}", "category": category,
                                   "priority": "P2" if "urgent" in low else "P3"})]
    if "status" in low or "outage" in low or " down" in low:
        return [("get_service_status", {"service": service})]
    if low.startswith(("who is", "look up", "lookup", "find employee")):
        name = re.sub(r"^(who is|look up|lookup|find employee)\s+", "", goal.strip(), flags=re.I).strip(" ?.")
        return [("lookup_employee", {"query": name[:100] or "unknown"})]
    if any(p in low for p in ("investigate", "incident", "why is", "research")):
        return [("get_service_status", {"service": service}),
                ("search_tickets", {"query": f"{service} {goal[:120]}", "status": "any", "limit": 3})]
    return [("search_tickets", {"query": goal[:150] or "help", "status": "any", "limit": 3})]


def _summarize_results(req: CompletionRequest) -> str:
    lines: list[str] = []
    for m in req.messages:
        if m.role != Role.TOOL:
            continue
        try:
            payload = json.loads(m.text)
        except (json.JSONDecodeError, TypeError):
            lines.append(m.text[:300])
            continue
        if payload.get("status") == "pending_approval":
            lines.append(f"The action is waiting for approval {payload.get('approval_id')}; nothing has been sent yet.")
        elif payload.get("ok"):
            result = payload.get("result")
            if isinstance(result, dict) and "ticket_id" in result:
                lines.append(f"Created ticket {result['ticket_id']} with priority {result.get('priority')}.")
            elif isinstance(result, dict) and "tickets" in result:
                ids = ", ".join(t["id"] for t in result["tickets"][:3]) or "none"
                lines.append(f"Found {result.get('count', 0)} related tickets: {ids}.")
            elif isinstance(result, dict) and "status" in result and "service" in result:
                lines.append(f"{result['service']} is {result['status']}. {result.get('note', '')}".strip())
            elif isinstance(result, dict) and "employees" in result:
                people = "; ".join(f"{e['name']} ({e['title']}, {e['team']})" for e in result["employees"][:3])
                lines.append(f"Directory: {people}.")
            elif isinstance(result, dict) and "draft_id" in result:
                lines.append(f"Drafted reply {result['draft_id']} to {result.get('to')}; it has not been sent.")
            else:
                lines.append("Result: " + json.dumps(result)[:300])
        else:
            err = payload.get("error") or {}
            lines.append(f"The tool refused ({err.get('category')}/{err.get('code')}): {err.get('message')}")
    return "\n".join(lines) or "I could not complete that with the tools available."


def agent_step(req: CompletionRequest) -> str | list[ToolCall]:
    plan = plan_for(_goal(req))
    done = sum(1 for m in req.messages if m.role == Role.ASSISTANT and m.tool_calls)
    last = req.messages[-1]
    if last.role == Role.TOOL and '"pending_approval"' in last.text:
        return _summarize_results(req)
    if done < len(plan):
        name, args = plan[done]
        return _call(name, **args)
    return _summarize_results(req)


# --------------------------------------------------------------------------- dispatcher
def demo_handler(req: CompletionRequest, *, gullible: bool = True) -> Any:
    pid = str(req.metadata.get("prompt.id", ""))
    purpose = str(req.metadata.get("purpose", ""))
    if pid.startswith("rag.grounded_answer"):
        return grounded_answer(req, gullible=gullible)
    if purpose == "memory.fact_extraction":
        return {"facts": []}
    if purpose == "memory.summary":
        return "The user asked earlier questions about Northwind policies and support tickets."
    if req.tools:
        return agent_step(req)
    return ("I can answer questions from Northwind policies and runbooks with citations, look up tickets, "
            "create tickets, and draft or send replies with approval.")


def make_demo_llm(*, gullible: bool = True) -> FakeLLM:
    return FakeLLM(handler=lambda req: demo_handler(req, gullible=gullible), model="demo-model", chunk_size=12)


__all__ = ["make_demo_llm", "demo_handler", "grounded_answer", "agent_step", "plan_for", "content_words"]
