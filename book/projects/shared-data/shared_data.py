# path: book/projects/shared-data/shared_data.py
"""Typed loader for the Northwind Assist shared dataset.

Usage from any project in the book:

    import sys; sys.path.insert(0, "../shared-data")   # or install as a path dependency
    from shared_data import load_docs, load_tickets, load_invoices, load_retrieval_gold

Everything is read from disk relative to this file; no network, no API keys.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Iterator, Literal

from pydantic import BaseModel, Field, field_validator

DATA_DIR = Path(__file__).resolve().parent
DOCS_DIR = DATA_DIR / "docs"
EVAL_DIR = DATA_DIR / "eval"

Tenant = Literal["retail", "logistics", "shared"]

TICKET_CATEGORIES: tuple[str, ...] = (
    "account_access",
    "vpn_network",
    "hardware",
    "password_mfa",
    "time_off",
    "expenses_travel",
    "benefits_leave",
    "pos_payments",
    "returns",
    "shipment_tracking",
    "warehouse_scanner",
    "security_report",
)

_FRONT_MATTER_RE = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)


# --------------------------------------------------------------------------- models
class DocMeta(BaseModel):
    """Front-matter fields shared by every document and by manifest.json entries."""

    id: str
    title: str
    version: str
    updated_at: date
    owner: str
    tenant: Tenant
    acl_groups: list[str] = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)

    @field_validator("version", mode="before")
    @classmethod
    def _version_as_str(cls, v: Any) -> str:
        return str(v)


class Doc(DocMeta):
    path: str
    body: str
    sha256: str

    @property
    def text(self) -> str:
        """Body without front matter; what most indexers want to chunk."""
        return self.body

    def visible_to(self, user_groups: list[str] | set[str], tenant: str | None = None) -> bool:
        """ACL check used throughout the book: any shared group grants access; tenant must match or be shared."""
        if tenant is not None and self.tenant not in ("shared", tenant):
            return False
        return bool(set(self.acl_groups) & set(user_groups))


class ManifestEntry(DocMeta):
    path: str
    sha256: str
    bytes: int


class Manifest(BaseModel):
    generated_at: date
    doc_count: int
    docs: list[ManifestEntry]


class Ticket(BaseModel):
    id: str
    created_at: str
    tenant: Tenant
    requester_role: str
    channel: Literal["portal", "email", "chat", "phone"]
    subject: str
    body: str
    category: str
    priority: Literal["P1", "P2", "P3", "P4"]
    status: Literal["open", "closed"]
    resolution: str | None = None

    @field_validator("category")
    @classmethod
    def _known_category(cls, v: str) -> str:
        if v not in TICKET_CATEGORIES:
            raise ValueError(f"unknown ticket category {v!r}")
        return v


class LineItem(BaseModel):
    description: str
    quantity: float
    unit_price: float
    amount: float


class InvoiceExpected(BaseModel):
    vendor: str
    invoice_number: str
    invoice_date: date
    due_date: date
    po_number: str | None
    currency: Literal["USD", "EUR", "GBP"]
    line_items: list[LineItem]
    subtotal: float
    tax_rate: float
    tax_amount: float
    total: float
    total_excluding_tax: float


class ValidationIssue(BaseModel):
    code: Literal["TOTAL_MISMATCH", "MISSING_PO", "LINE_SUM_MISMATCH"]
    detail: str


class InvoiceValidation(BaseModel):
    is_consistent: bool
    issues: list[ValidationIssue]


class Invoice(BaseModel):
    id: str
    tenant: Tenant
    format: Literal["table", "letter", "statement", "receipt"]
    text: str
    expected: InvoiceExpected
    validation: InvoiceValidation


class RetrievalQuestion(BaseModel):
    id: str
    question: str
    required_doc_ids: list[str]
    acceptable_doc_ids: list[str] = Field(default_factory=list)
    answer_rubric: list[str] = Field(min_length=1, max_length=3)
    user_groups: list[str] = Field(min_length=1)
    tenant: Tenant
    tags: list[str]

    @property
    def expects_abstain(self) -> bool:
        return "abstain" in self.tags


class ClassificationLabel(BaseModel):
    id: str
    category: str


# --------------------------------------------------------------------------- helpers
def _parse_scalar(raw: str) -> Any:
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(p) for p in inner.split(",")]
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        return raw[1:-1]
    return raw


def parse_front_matter(content: str) -> tuple[dict[str, Any], str]:
    """Minimal YAML front-matter parser (flat keys, scalars and inline lists). No PyYAML dependency."""
    m = _FRONT_MATTER_RE.match(content)
    if not m:
        raise ValueError("document has no YAML front matter")
    meta: dict[str, Any] = {}
    for line in m.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = _parse_scalar(value)
    return meta, m.group(2)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# --------------------------------------------------------------------------- loaders
def load_docs(docs_dir: Path = DOCS_DIR) -> list[Doc]:
    docs: list[Doc] = []
    for path in sorted(docs_dir.glob("*.md")):
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        docs.append(Doc(**meta, path=f"docs/{path.name}", body=body, sha256=sha256_of(path)))
    return docs


def load_manifest(path: Path = DATA_DIR / "manifest.json") -> Manifest:
    return Manifest.model_validate_json(path.read_text(encoding="utf-8"))


def load_tickets(path: Path = DATA_DIR / "tickets.jsonl") -> list[Ticket]:
    return [Ticket.model_validate(r) for r in _iter_jsonl(path)]


def load_invoices(path: Path = DATA_DIR / "invoices.jsonl") -> list[Invoice]:
    return [Invoice.model_validate(r) for r in _iter_jsonl(path)]


def load_retrieval_gold(path: Path = EVAL_DIR / "retrieval_gold.jsonl") -> list[RetrievalQuestion]:
    return [RetrievalQuestion.model_validate(r) for r in _iter_jsonl(path)]


def load_classification_gold(path: Path = EVAL_DIR / "classification_gold.jsonl") -> list[ClassificationLabel]:
    return [ClassificationLabel.model_validate(r) for r in _iter_jsonl(path)]


__all__ = [
    "DATA_DIR",
    "DOCS_DIR",
    "TICKET_CATEGORIES",
    "Doc",
    "DocMeta",
    "Manifest",
    "ManifestEntry",
    "Ticket",
    "Invoice",
    "InvoiceExpected",
    "LineItem",
    "RetrievalQuestion",
    "ClassificationLabel",
    "parse_front_matter",
    "sha256_of",
    "load_docs",
    "load_manifest",
    "load_tickets",
    "load_invoices",
    "load_retrieval_gold",
    "load_classification_gold",
]
