# path: book/projects/p1-extraction-api/extraction_api/eval/dataset.py
"""Gold data for invoice extraction: shared-data/invoices.jsonl if present, else a local fixture."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

_PROJECT = Path(__file__).resolve().parents[2]
SHARED_INVOICES = _PROJECT.parent / "shared-data" / "invoices.jsonl"
FIXTURE_INVOICES = Path(__file__).resolve().parent / "fixtures" / "invoices_sample.jsonl"


class GoldInvoice(BaseModel):
    id: str
    text: str
    expected: dict[str, Any]
    format: str = "unknown"          # slice key: table, letter, statement, receipt
    tenant: str | None = None
    validation: dict[str, Any] = Field(default_factory=dict)

    @property
    def document_is_consistent(self) -> bool:
        return bool(self.validation.get("is_consistent", True))


def default_gold_path() -> Path:
    override = os.environ.get("EXTRACT_GOLD_PATH")
    if override:
        return Path(override)
    return SHARED_INVOICES if SHARED_INVOICES.exists() else FIXTURE_INVOICES


def load_invoice_gold(path: str | Path | None = None, limit: int | None = None) -> list[GoldInvoice]:
    p = Path(path) if path else default_gold_path()
    rows = [GoldInvoice.model_validate(json.loads(line))
            for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


__all__ = ["GoldInvoice", "load_invoice_gold", "default_gold_path", "SHARED_INVOICES", "FIXTURE_INVOICES"]
