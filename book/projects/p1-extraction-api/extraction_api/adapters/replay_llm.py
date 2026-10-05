# path: book/projects/p1-extraction-api/extraction_api/adapters/replay_llm.py
"""ReplayLLM: an offline stand-in model that answers from labeled sample data.

It exists so that `uvicorn`, the Docker image, and the evaluation CLI do something useful
with LLM_PROVIDER=fake. Given a document it recognizes from shared-data (an invoice or a
ticket), it returns what a careful model would: values copied as printed, verbatim evidence
quotes, and the document's own inconsistent totals left unchanged. Unknown documents are
classified as "other". It is a demo and test fixture, not a model; never deploy it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from aie_core import CompletionRequest, FakeLLM

_DOC_RE = re.compile(r"<document>\n(.*)\n</document>", re.DOTALL)


def _document_text(req: CompletionRequest) -> str:
    for m in req.messages:
        if m.role.value == "user":
            found = _DOC_RE.search(m.text)
            if found:
                return found.group(1)
    return ""


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _money_variants(v: float) -> list[str]:
    return [f"{v:,.2f}", f"{v:.2f}", f"{v:,.0f}" if v == int(v) else f"{v:,.2f}"]


def _first_in(text: str, candidates: list[str]) -> str | None:
    for c in candidates:
        if c and c in text:
            return c
    return None


class ReplayLLM(FakeLLM):
    def __init__(self, invoices: list[dict[str, Any]], tickets: list[dict[str, Any]], confidence: float = 0.95) -> None:
        super().__init__(handler=self._handle, provider="replay", model="replay")
        self.invoices = invoices
        self.tickets = tickets
        self.confidence = confidence

    @classmethod
    def from_shared_data(cls, root: str | Path | None = None) -> "ReplayLLM":
        base = Path(root) if root else Path(__file__).resolve().parents[3] / "shared-data"
        invoices = _load_jsonl(base / "invoices.jsonl")
        if not invoices:  # fall back to the project's own fixture
            invoices = _load_jsonl(Path(__file__).resolve().parents[1] / "eval" / "fixtures" / "invoices_sample.jsonl")
        return cls(invoices, _load_jsonl(base / "tickets.jsonl"))

    # ----------------------------------------------------------------- lookup
    def _find_invoice(self, text: str) -> dict[str, Any] | None:
        head = text.strip()[:200]
        return next((r for r in self.invoices if head and head in r["text"]), None)

    def _find_ticket(self, text: str) -> dict[str, Any] | None:
        return next((t for t in self.tickets if t["body"][:120] in text), None)

    # ---------------------------------------------------------------- answers
    def _handle(self, req: CompletionRequest) -> dict[str, Any]:
        task = str(req.metadata.get("task", ""))
        text = _document_text(req)
        if task == "classify":
            if self._find_invoice(text):
                return {"doc_type": "invoice", "confidence": self.confidence, "reason": "supplier invoice layout"}
            if self._find_ticket(text):
                return {"doc_type": "support_ticket", "confidence": self.confidence, "reason": "employee request"}
            return {"doc_type": "other", "confidence": 0.3, "reason": "not recognized"}
        if task.endswith("invoice"):
            return self._invoice_draft(text)
        if task.endswith("support_ticket"):
            return self._ticket_draft(text)
        return {"error": f"unknown task {task}"}

    def _invoice_draft(self, text: str) -> dict[str, Any]:
        row = self._find_invoice(text)
        if row is None:
            keys = ["vendor", "invoice_number", "invoice_date", "due_date", "po_number", "currency",
                    "subtotal", "tax_rate", "tax_amount", "total"]
            return {**{k: None for k in keys}, "line_items": [], "evidence": []}
        e = row["expected"]
        draft: dict[str, Any] = {
            "vendor": e["vendor"], "invoice_number": e["invoice_number"],
            "invoice_date": e["invoice_date"], "due_date": e.get("due_date"),
            "po_number": e.get("po_number"), "currency": e["currency"],
            "line_items": [dict(li) for li in e["line_items"]],
            "subtotal": e.get("subtotal"), "tax_rate": e.get("tax_rate"),
            "tax_amount": e.get("tax_amount"), "total": e.get("total"),
        }
        evidence = []
        for field in ("vendor", "invoice_number", "invoice_date", "due_date", "po_number", "currency",
                      "subtotal", "tax_rate", "tax_amount", "total"):
            value = draft[field]
            if value is None:
                continue
            if isinstance(value, (int, float)) and field == "tax_rate":
                quote = _first_in(text, [f"{value * 100:g}%", f"{value:.2f}"])
            elif isinstance(value, (int, float)):
                quote = _first_in(text, _money_variants(float(value)))
            else:
                quote = _first_in(text, [str(value)])
            if quote:
                evidence.append({"field": field, "quote": quote, "confidence": self.confidence})
        draft["evidence"] = evidence
        return draft

    def _ticket_draft(self, text: str) -> dict[str, Any]:
        t = self._find_ticket(text)
        if t is None:
            return {"category": "other", "priority": "P4", "summary": "Unrecognized request",
                    "affected_system": None, "contains_personal_data": False, "entities": [], "evidence": []}
        first_sentence = re.split(r"(?<=[.!?])\s", t["body"], maxsplit=1)[0][:200]
        return {
            "category": t["category"], "priority": t["priority"], "summary": t["subject"][:200],
            "affected_system": None, "contains_personal_data": False, "entities": [],
            "evidence": [
                {"field": "category", "quote": t["subject"], "confidence": self.confidence},
                {"field": "priority", "quote": first_sentence, "confidence": self.confidence},
            ],
        }


__all__ = ["ReplayLLM"]
