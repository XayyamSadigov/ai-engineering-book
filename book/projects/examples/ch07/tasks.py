# path: book/projects/examples/ch07/tasks.py
"""The evaluation task used in Chapter 7: classify a Northwind support ticket.

Cases are built from book/projects/shared-data/tickets.jsonl. Each case carries the exact
CompletionRequest a candidate model receives and the expected label, so the selection
harness and the cascade evaluator never format prompts themselves.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from aie_core import Completion, CompletionRequest, Message

TICKETS_PATH = Path(__file__).resolve().parents[2] / "shared-data" / "tickets.jsonl"

CATEGORIES: tuple[str, ...] = (
    "account_access", "benefits_leave", "expenses_travel", "hardware", "password_mfa",
    "pos_payments", "returns", "security_report", "shipment_tracking", "time_off",
    "vpn_network", "warehouse_scanner",
)

SYSTEM_PROMPT = (
    "You classify Northwind IT and operations support tickets.\n"
    f"Allowed categories: {', '.join(CATEGORIES)}.\n"
    'Answer with JSON only: {"category": <one allowed category>, "confidence": <0..1>}.'
)

LABEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["category", "confidence"],
    "additionalProperties": False,
}


class Label(BaseModel):
    category: str
    confidence: float = Field(ge=0.0, le=1.0)


class EvalCase(BaseModel):
    id: str
    request: CompletionRequest
    expected: str
    tags: dict[str, str] = Field(default_factory=dict)


def ticket_request(subject: str, body: str, *, tenant: str = "shared") -> CompletionRequest:
    return CompletionRequest(
        messages=[Message.system(SYSTEM_PROMPT), Message.user(f"Subject: {subject}\n\n{body}")],
        max_tokens=64,
        response_schema=LABEL_SCHEMA,
        metadata={"task": "classify_ticket", "tenant": tenant},
    )


def load_ticket_cases(path: Path = TICKETS_PATH) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            t = json.loads(line)
            cases.append(EvalCase(
                id=t["id"],
                request=ticket_request(t["subject"], t["body"], tenant=t["tenant"]),
                expected=t["category"],
                tags={"tenant": t["tenant"], "priority": t["priority"]},
            ))
    return cases


def parse_label(completion: Completion) -> Label | None:
    """Parse a model answer. None means malformed or outside the label set: always wrong."""
    try:
        label = Label.model_validate_json(completion.text)
    except ValidationError:
        return None
    return label if label.category in CATEGORIES else None


def score_label(case: EvalCase, completion: Completion) -> float:
    label = parse_label(completion)
    return 1.0 if label is not None and label.category == case.expected else 0.0


def label_confidence(completion: Completion) -> float:
    """Self-reported confidence. Malformed output counts as zero confidence, so it escalates."""
    label = parse_label(completion)
    return 0.0 if label is None else label.confidence


__all__ = [
    "CATEGORIES", "SYSTEM_PROMPT", "LABEL_SCHEMA", "Label", "EvalCase", "ticket_request",
    "load_ticket_cases", "parse_label", "score_label", "label_confidence", "TICKETS_PATH",
]
