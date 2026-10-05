# path: book/projects/p1-extraction-api/extraction_api/domain/__init__.py
"""Pure domain logic: schemas, normalization, business rules, routing. No I/O, no model calls."""
from .common import DocumentType, EvidenceSpan, Route, RuleViolation, Severity
from .invoice import Currency, Invoice, InvoiceDraft, LineItem, build_invoice
from .routing import RouteDecision, RoutingPolicy, decide, document_score, field_scores
from .rules import check_evidence, check_invoice, check_ticket
from .ticket import Priority, SupportTicket, SupportTicketDraft, TicketCategory, build_ticket

__all__ = [
    "DocumentType", "EvidenceSpan", "Route", "RuleViolation", "Severity",
    "Currency", "Invoice", "InvoiceDraft", "LineItem", "build_invoice",
    "RouteDecision", "RoutingPolicy", "decide", "document_score", "field_scores",
    "check_evidence", "check_invoice", "check_ticket",
    "Priority", "SupportTicket", "SupportTicketDraft", "TicketCategory", "build_ticket",
]
