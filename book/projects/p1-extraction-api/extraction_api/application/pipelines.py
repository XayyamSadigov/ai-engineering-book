# path: book/projects/p1-extraction-api/extraction_api/application/pipelines.py
"""One spec per document type. Adding a type means adding a spec, not editing the service."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

from pydantic import BaseModel, TypeAdapter

from ..domain.common import DocumentType, EvidenceSpan, RuleViolation
from ..domain.invoice import CRITICAL_INVOICE_FIELDS, InvoiceDraft, build_invoice
from ..domain.routing import document_score, field_scores
from ..domain.rules import check_evidence, check_invoice, check_ticket
from ..domain.ticket import CRITICAL_TICKET_FIELDS, SupportTicketDraft, build_ticket
from .prompts import INVOICE_SYSTEM, TICKET_SYSTEM

_JSON = TypeAdapter(dict[str, Any])


@dataclass(frozen=True)
class ProcessContext:
    today: date
    require_po: bool = True
    dollar_means: str = "USD"


@dataclass
class Processed:
    """The output of deterministic post-processing for one draft."""

    data: dict[str, Any]              # JSON-safe normalized values (partial if invalid)
    valid: bool                       # did the strict domain record build?
    violations: list[RuleViolation]
    field_scores: dict[str, float]
    score: float


@dataclass(frozen=True)
class DocTypeSpec:
    doc_type: DocumentType
    system_prompt: str
    draft_schema: type[BaseModel]
    critical_fields: tuple[str, ...]
    process: Callable[[Any, str, ProcessContext], Processed]
    max_tokens: int = 2048


def _present(data: dict[str, Any], fields: tuple[str, ...]) -> set[str]:
    return {f for f in fields if data.get(f) not in (None, "", [])}


def process_invoice(draft: InvoiceDraft, text: str, ctx: ProcessContext) -> Processed:
    data, invoice, violations = build_invoice(draft, text, dollar_means=ctx.dollar_means)
    present = _present(data, CRITICAL_INVOICE_FIELDS)
    if invoice is not None:
        violations += check_invoice(invoice, today=ctx.today, require_po=ctx.require_po)
        spans = invoice.evidence
        data = invoice.model_dump(mode="json")
    else:
        spans = [EvidenceSpan.model_validate(e) for e in data["evidence"]]
        data = _JSON.dump_python(data, mode="json")
    violations += check_evidence(spans, present, CRITICAL_INVOICE_FIELDS)
    scores = field_scores(spans, CRITICAL_INVOICE_FIELDS, present)
    return Processed(data=data, valid=invoice is not None, violations=violations,
                     field_scores=scores, score=document_score(scores))


def process_ticket(draft: SupportTicketDraft, text: str, ctx: ProcessContext) -> Processed:
    data, ticket, violations = build_ticket(draft, text)
    violations += check_ticket(ticket)
    present = _present(data, CRITICAL_TICKET_FIELDS)
    violations += check_evidence(ticket.evidence, present, CRITICAL_TICKET_FIELDS)
    scores = field_scores(ticket.evidence, CRITICAL_TICKET_FIELDS, present)
    return Processed(data=data, valid=True, violations=violations,
                     field_scores=scores, score=document_score(scores))


SPECS: dict[DocumentType, DocTypeSpec] = {
    DocumentType.INVOICE: DocTypeSpec(
        doc_type=DocumentType.INVOICE, system_prompt=INVOICE_SYSTEM, draft_schema=InvoiceDraft,
        critical_fields=CRITICAL_INVOICE_FIELDS, process=process_invoice, max_tokens=3000,
    ),
    DocumentType.SUPPORT_TICKET: DocTypeSpec(
        doc_type=DocumentType.SUPPORT_TICKET, system_prompt=TICKET_SYSTEM, draft_schema=SupportTicketDraft,
        critical_fields=CRITICAL_TICKET_FIELDS, process=process_ticket, max_tokens=1200,
    ),
}

__all__ = ["ProcessContext", "Processed", "DocTypeSpec", "SPECS", "process_invoice", "process_ticket"]
