# path: book/projects/examples/ch17/triage_domain.py
"""Shared domain for the Northwind ticket-triage example.

Both the fixed pipeline and the graph import from here, so the comparison
between the two is about orchestration only, not about step logic.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["refund", "shipping", "account", "other"]
Verdict = Literal["ok", "fixable", "escalate"]

# A model is just a callable in this chapter. In the full stack this is
# `gateway.complete(CompletionRequest(...))` from aie_core; see Chapter 3.
Model = Callable[[str, str], str]  # (task, prompt) -> text

POLICIES: dict[str, str] = {
    "refund": "Retail refund policy: full refund within 30 days with receipt; "
              "store credit up to 90 days. Never promise a refund for opened software.",
    "shipping": "Logistics policy: standard delivery 3-5 business days; "
                "delays over 7 days qualify for free re-shipment.",
    "account": "Account policy: identity must be verified before any account change; "
               "never share account details in a reply.",
}


class ValidationReport(BaseModel):
    verdict: Verdict
    reasons: list[str] = Field(default_factory=list)


class TriageState(BaseModel):
    ticket_id: str
    tenant: Literal["retail", "logistics"]
    text: str
    category: Category | None = None
    policy: str | None = None
    draft: str | None = None
    report: ValidationReport | None = None
    draft_attempts: int = 0
    approved: bool | None = None
    approval_note: str | None = None
    sent: bool = False
    outcome: Literal["sent", "escalated"] | None = None
    log: list[str] = Field(default_factory=list)


@dataclass
class ModelCall:
    task: str
    duration_ms: float


@dataclass
class FakeModel:
    """Scripted model. `scripts[task]` is a list of replies consumed in order;
    the last reply is repeated when the list is exhausted."""

    scripts: dict[str, list[str]]
    simulated_latency_ms: float = 0.0
    calls: list[ModelCall] = field(default_factory=list)
    _cursor: dict[str, int] = field(default_factory=dict)

    def __call__(self, task: str, prompt: str) -> str:
        started = time.perf_counter()
        replies = self.scripts[task]
        idx = min(self._cursor.get(task, 0), len(replies) - 1)
        self._cursor[task] = idx + 1
        if self.simulated_latency_ms:
            time.sleep(self.simulated_latency_ms / 1000)
        self.calls.append(ModelCall(task, (time.perf_counter() - started) * 1000))
        return replies[idx]


# --- the five steps, as pure functions of (state, model) --------------------
def classify(state: TriageState, model: Model) -> TriageState:
    raw = model("classify", f"Classify this support ticket into refund/shipping/account/other:\n{state.text}")
    category = raw.strip().lower()
    if category not in ("refund", "shipping", "account", "other"):
        raise ValueError(f"classifier returned an unknown label: {raw!r}")
    state.category = category  # type: ignore[assignment]
    state.log.append(f"classified as {category}")
    return state


def retrieve_policy(state: TriageState) -> TriageState:
    """Deterministic step: no model involved. A dict stands in for retrieval."""
    assert state.category is not None
    state.policy = POLICIES.get(state.category, "No specific policy; answer generically and offer escalation.")
    state.log.append("policy retrieved")
    return state


def draft_reply(state: TriageState, model: Model) -> TriageState:
    prompt = (f"Policy:\n{state.policy}\n\nTicket:\n{state.text}\n\n"
              f"Write a short reply. Previous validation: {state.report.model_dump() if state.report else 'none'}")
    state.draft = model("draft", prompt)
    state.draft_attempts += 1
    state.log.append(f"draft #{state.draft_attempts}")
    return state


def validate(state: TriageState, model: Model) -> TriageState:
    """Two layers: deterministic checks first, model judgment second."""
    reasons: list[str] = []
    assert state.draft is not None
    if "guarantee" in state.draft.lower() or "promise" in state.draft.lower():
        reasons.append("draft makes a promise the policy does not allow")
    if state.category == "account" and "@" in state.draft:
        reasons.append("draft may leak account data")
    raw = model("validate", f"Policy:\n{state.policy}\nDraft:\n{state.draft}\nReturn JSON {{verdict, reasons}}.")
    judged = json.loads(raw)
    verdict: Verdict = judged["verdict"]
    reasons.extend(judged.get("reasons", []))
    if reasons and verdict == "ok":
        verdict = "fixable"
    state.report = ValidationReport(verdict=verdict, reasons=reasons)
    state.log.append(f"validated: {verdict}")
    return state


def needs_human(state: TriageState) -> bool:
    """Policy decision in code, not in the model: account changes and refunds
    are side effects a human signs off on."""
    return state.category in ("account", "refund")


# (ticket_id, body, idempotency_key): the sender must deliver at most once per key.
Sender = Callable[[str, str, str], None]


def send(state: TriageState, sender: Sender, key: str | None = None) -> TriageState:
    """The only irreversible step. The key lets the sender suppress a duplicate
    when the step is re-executed after a crash or an ambiguous failure."""
    assert state.draft is not None
    sender(state.ticket_id, state.draft, key or f"{state.ticket_id}:send")
    state.sent = True
    state.outcome = "sent"
    state.log.append("sent")
    return state


def escalate(state: TriageState) -> TriageState:
    state.outcome = "escalated"
    state.log.append("escalated to a human agent")
    return state
