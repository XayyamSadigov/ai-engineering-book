# path: book/projects/examples/ch32/northwind_triage/adapters/simulated_model.py
"""A deterministic stand-in for a real classifier, used when LLM_PROVIDER=fake.

It lets the offline eval gate, the CI pipeline, and the demo run without keys. It is a
keyword heuristic, not a model; its accuracy numbers say nothing about any real model.
"""
from __future__ import annotations

import json
import re

from aie_core.llm.providers import FakeLLM
from aie_core.llm.types import CompletionRequest

KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("security_report", ("phish", "suspicious", "malware", "stolen", "lost laptop", "credential",
                         "breach", "virus", "scam", "signed in from", "login alert", "personal data",
                         "instructions for the ai", "automated readers", "personal address",
                         "conflict of interest")),
    ("pos_payments", ("register", "till", "card", "payment", "pos ", "declin")),
    ("warehouse_scanner", ("scanner", "handheld", "barcode", "scan gun")),
    ("shipment_tracking", ("shipment", "tracking", "parcel", "consignment", "pallet", "route", "stops",
                           "webhook")),
    ("returns", ("return", "refund", "rma")),
    ("password_mfa", ("password", "mfa", "authenticator", "locked out", "2fa", "one-time code")),
    ("vpn_network", ("vpn", "wifi", "wi-fi", "network", "internet")),
    ("account_access", ("access", "permission", "role", "console", "approve")),
    ("expenses_travel", ("expense", "travel", "receipt", "per diem", "hotel", "mileage", "flight")),
    ("time_off", ("pto", "vacation", "time off", "holiday request", "day off")),
    ("benefits_leave", ("benefit", "parental", "leave", "insurance", "pension")),
    ("hardware", ("laptop", "monitor", "keyboard", "printer", "dock", "screen", "headset")),
]
URGENT = ("all cards", "every card", "queue", "store closed", "cannot ship", "outage", "stopped",
          "stolen", "an hour ago", "minutes ago")
QUESTION = ("?", "request", "how do i", "can i", "please add")


def classify_text(text: str) -> dict[str, object]:
    lowered = text.lower()
    scores = []
    for category, words in KEYWORDS:
        hits = sum(1 for w in words if w in lowered)
        if hits:
            scores.append((hits, category))
    if not scores:
        return {"category": "hardware", "priority": "P4", "confidence": 0.3, "rationale": "no signal"}
    scores.sort(key=lambda s: -s[0])
    best_hits, category = scores[0]
    tied = len(scores) > 1 and scores[1][0] == best_hits
    if any(u in lowered for u in URGENT):
        priority = "P1"
    elif category == "security_report":
        priority = "P2"
    elif any(q in lowered for q in QUESTION):
        priority = "P4"
    else:
        priority = "P3"
    confidence = 0.55 if tied else min(0.95, 0.6 + 0.1 * best_hits)
    return {"category": category, "priority": priority, "confidence": round(confidence, 2),
            "rationale": f"matched {best_hits} keyword(s) for {category}"}


def _ticket_text(req: CompletionRequest) -> str:
    user = next((m.text for m in reversed(req.messages) if m.role.value == "user"), "")
    match = re.search(r"<ticket[^>]*>(.*)</ticket>", user, flags=re.S)
    return match.group(1) if match else user


def simulated_llm(model: str = "sim-classifier") -> FakeLLM:
    return FakeLLM(handler=lambda req: json.dumps(classify_text(_ticket_text(req))),
                   model=model, provider="simulated")
