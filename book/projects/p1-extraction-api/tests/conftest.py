# path: book/projects/p1-extraction-api/tests/conftest.py
"""Shared fixtures. Everything is offline: FakeLLM scripts, InMemoryTracer, in-memory queue."""
from __future__ import annotations

import copy
import json
from datetime import date
from pathlib import Path
from typing import Any, Callable

import pytest
from aie_core import FakeLLM
from aie_core.observability import InMemoryTracer

from extraction_api.adapters import InMemoryReviewQueue
from extraction_api.application import ExtractionService
from extraction_api.domain import RoutingPolicy

FIXTURE = Path(__file__).resolve().parents[1] / "extraction_api" / "eval" / "fixtures" / "invoices_sample.jsonl"
TODAY = date(2026, 10, 4)


def load_fixture() -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["id"]: r for r in rows}


GOLD = load_fixture()
INV1_TEXT: str = GOLD["INV-001"]["text"]

# A correct extraction of INV-001, as a careful model would return it.
INV1_DRAFT: dict[str, Any] = {
    "vendor": "Brightline Supply Co.",
    "invoice_number": "BL-INV-2026-00231",
    "invoice_date": "2026-01-14",
    "due_date": "2026-02-13",
    "po_number": "PO-NW-2026-10412",
    "currency": "USD",
    "line_items": [
        {"description": "Recycled corrugated box, M (BL-BOX-M-R)", "quantity": "2,400", "unit_price": "$0.62", "amount": "$1,488.00"},
        {"description": "Thermal label roll 100x150 (BL-LBL-1015)", "quantity": 120, "unit_price": 7.90, "amount": 948.00},
        {"description": "Tote lid, stackable (BL-TOTE-LID)", "quantity": 300, "unit_price": 2.15, "amount": 645.00},
    ],
    "subtotal": "3,081.00",
    "tax_rate": "8%",
    "tax_amount": 246.48,
    "total": "$3,327.48",
    "evidence": [
        {"field": "vendor", "quote": "Brightline Supply Co.", "confidence": 0.97},
        {"field": "invoice_number", "quote": "INVOICE BL-INV-2026-00231", "confidence": 0.98},
        {"field": "invoice_date", "quote": "Invoice date: 2026-01-14", "confidence": 0.97},
        {"field": "currency", "quote": "Currency: USD.", "confidence": 0.95},
        {"field": "total", "quote": "TOTAL DUE     $3,327.48", "confidence": 0.96},
        {"field": "po_number", "quote": "PO-NW-2026-10412", "confidence": 0.95},
    ],
}

CLASSIFY_INVOICE = {"doc_type": "invoice", "confidence": 0.96, "reason": "supplier invoice with totals"}


def draft(**changes: Any) -> dict[str, Any]:
    d = copy.deepcopy(INV1_DRAFT)
    d.update(changes)
    return d


@pytest.fixture
def tracer() -> InMemoryTracer:
    return InMemoryTracer()


@pytest.fixture
def queue() -> InMemoryReviewQueue:
    return InMemoryReviewQueue()


@pytest.fixture
def make_service(tracer: InMemoryTracer, queue: InMemoryReviewQueue) -> Callable[..., tuple[ExtractionService, FakeLLM]]:
    def _make(responses: list[Any] | None = None, *, handler: Any = None, **kw: Any) -> tuple[ExtractionService, FakeLLM]:
        llm = FakeLLM(responses=responses, handler=handler) if handler else FakeLLM(responses=responses or [])
        policy = kw.pop("policy", RoutingPolicy(accept_threshold=0.8, max_rule_repairs=1))
        svc = ExtractionService(llm, queue, policy=policy, tracer=tracer, today=lambda: TODAY, **kw)
        return svc, llm
    return _make
