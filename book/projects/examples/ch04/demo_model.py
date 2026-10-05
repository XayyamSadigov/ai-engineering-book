# path: book/projects/examples/ch04/demo_model.py
"""A simulated ticket-routing model so the regression harness can run offline.

It is NOT a model and its numbers say nothing about real prompts. It exists so that a
prompt edit changes behavior in a way you can see:

* It reads "decision rules" from the system prompt (lines shaped like
  "- If the ticket mentions a, b or c, choose <category>.") and applies them in order,
  first match wins. Reordering rules in the prompt file changes its answers.
* Without rules it falls back to a crude built-in keyword prior.
* It only reads ticket text from inside <untrusted_data> blocks and, like a naive model,
  it obeys an instruction "classify as <x>" if one appears *outside* those blocks. That
  lets tests show what escaping buys you and, by contrast, what it cannot (Chapter 26).
"""
from __future__ import annotations

import json
import re

from aie_core.llm.providers import FakeLLM
from aie_core.llm.types import CompletionRequest, Role

_RULE = re.compile(r"^- If the ticket mentions (.+?), choose ([a-z_]+)\.\s*$", re.MULTILINE)
_BLOCK = re.compile(r"<untrusted_data[^>]*>\n(.*?)\n</untrusted_data>", re.DOTALL)
_OBEY = re.compile(r"classify as ([a-z_]+)", re.IGNORECASE)

# The "prior" a weak model might have: plausible keywords, naive order.
_PRIOR: list[tuple[str, str]] = [
    ("vpn", "vpn_network"), ("password", "password_mfa"), ("locked", "password_mfa"),
    ("login", "password_mfa"), ("laptop", "hardware"), ("register", "pos_payments"),
    ("scanner", "warehouse_scanner"), ("expense", "expenses_travel"), ("hotel", "expenses_travel"),
    ("pto", "time_off"), ("parental", "benefits_leave"), ("suspicious", "security_report"),
    ("personal data", "security_report"), ("return", "returns"), ("tracking", "shipment_tracking"),
    ("access", "account_access"),
]


def _keywords(spec: str) -> list[str]:
    return [k.strip().lower() for k in re.split(r",\s*|\s+or\s+", spec) if k.strip()]


def _quote(text: str, keyword: str) -> str:
    """The sentence containing the keyword, cut to 120 chars: an exact quote from the ticket."""
    idx = text.lower().find(keyword)
    start = max(text.rfind(".", 0, idx) + 1, 0)
    end = text.find(".", idx)
    sentence = text[start : end if end != -1 else len(text)].strip()
    return sentence[:120]


def classify(req: CompletionRequest) -> str:
    system = "\n".join(m.text for m in req.messages if m.role == Role.SYSTEM)
    last_user = next(m.text for m in reversed(req.messages) if m.role == Role.USER)
    ticket = "\n".join(_BLOCK.findall(last_user))
    outside = _BLOCK.sub("", last_user)

    injected = _OBEY.search(outside)
    if injected:  # the failure mode a forged closing tag aims for
        return json.dumps({"category": injected.group(1), "evidence": ""})

    rules = [(_keywords(spec), cat) for spec, cat in _RULE.findall(system)]
    table = [(kw, cat) for kws, cat in rules for kw in kws] if rules else _PRIOR
    lowered = ticket.lower()
    for keyword, category in table:
        if keyword in lowered:
            return json.dumps({"category": category, "evidence": _quote(ticket, keyword)})
    return json.dumps({"category": "other", "evidence": ""})


def simulated_router() -> FakeLLM:
    return FakeLLM(handler=classify, model="simulated-router")


__all__ = ["classify", "simulated_router"]
