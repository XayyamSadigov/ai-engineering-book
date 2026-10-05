# path: book/projects/examples/ch25/taskevals/standins.py
"""Deterministic stand-in "models" for the offline suites, built on aie_core's FakeLLM.

They exercise the real code paths (CompletionRequest, complete_structured, tool calls) with
behavior we control, so the suites, the gate, and the CI job can be demonstrated and tested
without a provider. Each stand-in has versions that mirror a realistic change:

- classifier: v1 keyword rules (from Chapter 24's worked example); v3 adds security phrases
  without obeying the word "security" inside a ticket.
- extractor: baseline misses PO numbers in letter-format invoices and cites the subtotal line
  as evidence for the total on statements; candidate fixes both; regressed reads the subtotal
  as the total on receipts.
- planner: baseline follows the expert procedure with one redundant search on AG-003;
  candidate drops the redundant call; regressed loops on AG-002 and files AG-001 at P2.
- tool selector: keyword routing over the request text.

Replace any of them with `aie_core.make_llm_client()` to evaluate a real model on the same suites.
"""
from __future__ import annotations

import re
from typing import Any, Literal

from aie_core import CompletionRequest, FakeLLM, Role, ToolCall

from .extraction import locate_value

System = Literal["baseline", "candidate", "regressed"]

# ------------------------------------------------------------------ classifier
RULES_V1: list[tuple[str, list[str]]] = [
    ("password_mfa", ["password", "locked", "mfa", "cant get in"]),
    ("warehouse_scanner", ["scanner", "sh-", "receiving quantity"]),
    ("pos_payments", ["register", "store server", "price change", "gift card not accepted"]),
    ("returns", ["return", "refund", "final-sale", "ret-"]),
    ("shipment_tracking", ["tracking", "webhook", "route", "assignment rate", "trk-"]),
    ("security_report", ["suspicious", "personal data", "unknown login", "instructions for the ai"]),
    ("hardware", ["laptop", "monitor", "macbook"]),
    ("account_access", ["access", "group", "activation", "console"]),
    ("vpn_network", ["vpn"]),
    ("expenses_travel", ["expense", "per diem", "mileage", "reimburs", "travel", "hotel", "taxi"]),
    ("benefits_leave", ["parental", "sick leave", "bank details"]),
    ("time_off", ["pto", "portugal", "blackout"]),
]
RULES_V3: list[tuple[str, list[str]]] = [
    ("security_report", ["suspicious", "personal data", "unknown login", "instructions for the ai",
                         "conflict of interest", "hr documents"]),
    *[(c, kw) for c, kw in RULES_V1 if c != "security_report"],
]
CLASSIFIER_RULES = {"baseline": RULES_V1, "candidate": RULES_V3, "regressed": RULES_V1}


def _hits(text: str, keywords: list[str]) -> int:
    return sum(1 for k in keywords if re.search(r"(?<![a-z0-9])" + re.escape(k), text))


def classifier_model(system: System) -> FakeLLM:
    rules = CLASSIFIER_RULES[system]

    def handler(req: CompletionRequest) -> dict[str, Any]:
        ticket = req.messages[-1].text.lower().split("<ticket>", 1)[-1]
        for category, keywords in rules:
            hits = _hits(ticket, keywords)
            if hits:
                return {"category": category, "confidence": round(min(0.95, 0.55 + 0.2 * hits), 2)}
        return {"category": "account_access", "confidence": 0.3}

    return FakeLLM(handler=handler, model=f"fake-classifier-{system}")


# ------------------------------------------------------------------ extractor
EXTRACT_FIELDS = ["vendor", "invoice_number", "invoice_date", "due_date", "po_number", "currency",
                  "subtotal", "tax_amount", "total"]


def extractor_model(system: System, invoices: dict[str, dict[str, Any]]) -> FakeLLM:
    """`invoices` maps invoice id -> {"text", "format", "expected"}; the request carries the id in metadata."""

    def handler(req: CompletionRequest) -> dict[str, Any]:
        inv = invoices[req.metadata["invoice_id"]]
        text, fmt, gold = inv["text"], inv["format"], inv["expected"]
        fields = {f: gold.get(f) for f in EXTRACT_FIELDS}
        evidence = {f: locate_value(text, v) for f, v in fields.items() if v is not None}
        if system == "baseline" and fmt == "letter":
            fields["po_number"] = None  # PO mentioned in running prose is missed
            evidence.pop("po_number", None)
        if system in ("baseline", "regressed") and fmt == "statement":
            evidence["total"] = locate_value(text, gold["subtotal"])  # cites the wrong line
        if system == "regressed" and fmt == "receipt":
            fields["total"] = gold["subtotal"]
            evidence["total"] = locate_value(text, gold["subtotal"])
        return {"fields": fields, "line_items": gold["line_items"],
                "evidence": {k: v for k, v in evidence.items() if v}}

    return FakeLLM(handler=handler, model=f"fake-extractor-{system}")


# ------------------------------------------------------------------ planner
# Expert procedures per agent task: (tool, arguments). The planner stand-in replays one per task.
_PLANS: dict[str, list[tuple[str, dict[str, Any]]]] = {
    "AG-001": [
        ("get_service_status", {"service": "paybridge-adapter"}),
        ("search_tickets", {"query": "register declines cards paybridge", "limit": 5}),
        ("create_ticket", {"subject": "Store 0412 register 3 declines all cards",
                           "body": "Register 3 at store 0412 declines every card; PayBridge adapter suspected.",
                           "category": "pos_payments", "priority": "P1"}),
        ("draft_reply", {"ticket_id": "TCK-2026-0001", "to": "store0412@northwind.example",
                         "subject": "Register 3 card declines",
                         "body": "We restarted the PayBridge adapter for register 3."}),
        ("send_reply", {"ticket_id": "TCK-2026-0001", "to": "store0412@northwind.example",
                        "subject": "Register 3 card declines",
                        "body": "We restarted the PayBridge adapter for register 3."}),
    ],
    "AG-002": [
        ("get_service_status", {"service": "vpn-gateway"}),
        ("search_tickets", {"query": "vpn drops harbor city warehouse", "limit": 5}),
        ("create_ticket", {"subject": "VPN drops for Harbor City warehouse staff",
                           "body": "Warehouse staff at Harbor City report repeated VPN drops; vpn-gateway degraded.",
                           "category": "vpn_network", "priority": "P2"}),
    ],
    "AG-003": [
        ("lookup_employee", {"query": "E-1042"}),
        ("search_tickets", {"query": "pto carryover", "limit": 3}),
        ("draft_reply", {"ticket_id": "TCK-2026-0040", "to": "dana.ortiz@northwind.example",
                         "subject": "PTO carryover into 2026",
                         "body": "Under PTO policy v3.0 you can carry over up to 10 days into 2026."}),
    ],
    "AG-004": [
        ("get_service_status", {"service": "scanner-fleet"}),
        ("create_ticket", {"subject": "Scanner fleet degraded at Harbor City warehouse",
                           "body": "Handheld scanners at Harbor City are failing; scanner-fleet reports degraded.",
                           "category": "warehouse_scanner", "priority": "P1"}),
        ("draft_reply", {"ticket_id": "TCK-2026-0051", "to": "harbor-city-ops@northwind.example",
                         "subject": "Scanner outage", "body": "Scanner outage confirmed; a P1 ticket is open."}),
        ("send_reply", {"ticket_id": "TCK-2026-0051", "to": "harbor-city-ops@northwind.example",
                        "subject": "Scanner outage", "body": "Scanner outage confirmed; a P1 ticket is open."}),
    ],
}
FINAL_ANSWERS = {
    "AG-001": "Opened a P1 ticket for register 3 and replied to the store.",
    "AG-002": "Opened a P2 VPN ticket for the logistics team; no reply sent.",
    "AG-003": "Drafted a reply: up to 10 PTO days carry over under policy v3.0.",
    "AG-004": "Opened a P1 scanner ticket and replied to the warehouse.",
}


def plan_for(task_id: str, system: System) -> list[tuple[str, dict[str, Any]]]:
    plan = [(t, dict(a)) for t, a in _PLANS[task_id]]
    if system == "candidate" and task_id == "AG-003":
        plan = [p for p in plan if p[0] != "search_tickets"]  # the redundant search is gone
    if system == "regressed" and task_id == "AG-002":
        search = plan[1]
        plan = [plan[0], search, search, search, search, plan[2]]  # retry loop on an unchanged query
    if system == "regressed" and task_id == "AG-001":
        plan[2][1]["priority"] = "P2"
    return plan


def planner_model(system: System, goals: dict[str, str] | None = None) -> FakeLLM:
    """Emits the next planned tool call, one per turn, then the final answer.

    The task is identified from request metadata (`task_id`) or, inside agentkit's runtime,
    from the goal text via `goals` (goal -> task id). Progress is the number of tool messages
    (results and denials) already in the conversation.
    """

    def handler(req: CompletionRequest) -> Any:
        task_id = req.metadata.get("task_id")
        if task_id is None:
            first_user = next(m.text for m in req.messages if m.role == Role.USER)
            task_id = (goals or {})[first_user]
        plan = plan_for(task_id, system)
        done = sum(1 for m in req.messages if m.role == Role.TOOL)
        if done >= len(plan):
            return FINAL_ANSWERS[task_id]
        tool, args = plan[done]
        return [ToolCall(id=f"c{done + 1}", name=tool, arguments=args)]

    return FakeLLM(handler=handler, model=f"fake-planner-{system}")


# ------------------------------------------------------------------ tool selector
_TOOL_RULES: list[tuple[str, list[str]]] = [
    ("get_service_status", ["status", "is it down", "outage"]),
    ("lookup_employee", ["who is", "manager of", "employee"]),
    ("create_ticket", ["open a ticket", "file a ticket", "create a ticket"]),
    ("search_tickets", ["similar tickets", "past tickets", "has anyone"]),
    ("query_metrics", ["how many", "average", "per week"]),
]


def tool_selector_model(system: System) -> FakeLLM:
    def handler(req: CompletionRequest) -> Any:
        text = req.messages[-1].text.lower()
        for tool, kws in _TOOL_RULES:
            if any(k in text for k in kws):
                args: dict[str, Any] = {}
                if tool == "get_service_status":
                    m = re.search(r"([a-z]+-[a-z]+)", text)
                    args = {"service": m.group(1) if m else "unknown"}
                elif tool == "create_ticket":
                    prio = "P1" if "urgent" in text else "P3"
                    args = {"subject": text[:60], "body": text, "priority": prio,
                            "category": "hardware" if "laptop" in text else "account_access"}
                    if system == "regressed":
                        args["priority"] = "high"  # outside the enum: an argument-validity regression
                elif tool == "lookup_employee":
                    args = {"query": text.split()[-1].strip("?")}
                else:
                    args = {"query" if tool == "search_tickets" else "sql": text}
                return [ToolCall(id="t1", name=tool, arguments=args)]
        return "I can answer that directly."

    return FakeLLM(handler=handler, model=f"fake-tools-{system}")


__all__ = ["System", "RULES_V1", "RULES_V3", "classifier_model", "EXTRACT_FIELDS", "extractor_model", "plan_for",
           "planner_model", "FINAL_ANSWERS", "tool_selector_model"]
