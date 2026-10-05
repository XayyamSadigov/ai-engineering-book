# path: book/projects/p1-extraction-api/extraction_api/domain/ticket.py
"""Support ticket schemas: classification (category, priority) plus entity extraction.

Two extraction techniques meet here. Well-formed identifiers (emails, phone numbers,
Northwind error and return codes) are found by regular expressions, which are exact and
free. Fuzzy entities (people, systems, locations) come from the model and are kept only if
their quote is actually in the ticket text.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .common import EvidenceSpan, RuleViolation
from .normalize import clean_text, locate


class TicketCategory(str, Enum):
    ACCOUNT_ACCESS = "account_access"
    PASSWORD_MFA = "password_mfa"
    VPN_NETWORK = "vpn_network"
    HARDWARE = "hardware"
    POS_PAYMENTS = "pos_payments"
    RETURNS = "returns"
    SHIPMENT_TRACKING = "shipment_tracking"
    WAREHOUSE_SCANNER = "warehouse_scanner"
    EXPENSES_TRAVEL = "expenses_travel"
    TIME_OFF = "time_off"
    BENEFITS_LEAVE = "benefits_leave"
    SECURITY_REPORT = "security_report"
    OTHER = "other"  # the explicit abstention value; always routed to a person


class Priority(str, Enum):
    P1 = "P1"  # business stopped, many users or revenue affected now
    P2 = "P2"  # degraded, workaround exists
    P3 = "P3"  # single user, normal queue
    P4 = "P4"  # question or request


EntityType = Literal[
    "person", "store", "system", "location", "client_account", "route", "error_code",
    "return_id", "email", "phone", "other",
]
TicketFieldName = Literal["category", "priority", "summary", "affected_system"]
CRITICAL_TICKET_FIELDS: tuple[str, ...] = ("category", "priority")


# ------------------------------------------------------------------ wire schema
class TicketEntityDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: EntityType
    value: str = Field(description="Canonical value, for example 0412 for 'Store 0412'.")
    quote: str = Field(max_length=200, description="Verbatim excerpt containing the entity.")


class TicketEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: TicketFieldName
    quote: str = Field(max_length=300)
    confidence: float = Field(ge=0.0, le=1.0)


class SupportTicketDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: TicketCategory = Field(description="Best category; use other if none fits.")
    priority: Priority
    summary: str = Field(max_length=200, description="One sentence, no personal data.")
    affected_system: str | None = Field(description="Named system or application, null if none.")
    contains_personal_data: bool = Field(description="True if the text includes contact details or personal records.")
    entities: list[TicketEntityDraft]
    evidence: list[TicketEvidence]


# ---------------------------------------------------------------- domain record
class TicketEntity(BaseModel):
    type: EntityType
    value: str
    start: int
    end: int
    source: Literal["model", "pattern"]


class SupportTicket(BaseModel):
    category: TicketCategory
    priority: Priority
    summary: str = Field(min_length=1, max_length=200)
    affected_system: str | None = None
    contains_personal_data: bool = False
    entities: list[TicketEntity] = Field(default_factory=list)
    evidence: list[EvidenceSpan] = Field(default_factory=list)


# --------------------------------------------------------- deterministic entities
_PATTERNS: list[tuple[EntityType, re.Pattern[str]]] = [
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("phone", re.compile(r"\+?\d{1,3}[-\s]\d{3}[-\s]\d{4}\b")),
    ("return_id", re.compile(r"\bRET-\d{8}-\d{4,}\b")),
    ("error_code", re.compile(r"\b(?:SH|RET|TRK|POS|VPN)-\d{3}\b")),
    ("store", re.compile(r"(?<=\bStore )\d{4}\b")),
]


def pattern_entities(text: str) -> list[TicketEntity]:
    found: list[TicketEntity] = []
    for etype, rx in _PATTERNS:
        for m in rx.finditer(text):
            found.append(TicketEntity(type=etype, value=m.group(0), start=m.start(), end=m.end(), source="pattern"))
    return found


def _overlaps(a: TicketEntity, b: TicketEntity) -> bool:
    return a.start < b.end and b.start < a.end


# ------------------------------------------------------------------- the bridge
def build_ticket(
    draft: SupportTicketDraft, text: str
) -> tuple[dict[str, Any], SupportTicket, list[RuleViolation]]:
    v: list[RuleViolation] = []
    entities = pattern_entities(text)
    for ent in draft.entities:
        pos = locate(ent.quote, text) or locate(ent.value, text)
        if pos is None:
            # an entity we cannot point at is worse than a missing one: drop it, record why
            v.append(RuleViolation(code="ENTITY_NOT_GROUNDED", field="entities",
                                   detail=f"{ent.type} {ent.value!r} not found in text", severity="warning"))
            continue
        candidate = TicketEntity(type=ent.type, value=clean_text(ent.value) or ent.value,
                                 start=pos[0], end=pos[1], source="model")
        if not any(_overlaps(candidate, e) and e.type == candidate.type for e in entities):
            entities.append(candidate)
    entities.sort(key=lambda e: e.start)

    has_contact = any(e.type in ("email", "phone") for e in entities)
    contains_pd = draft.contains_personal_data or has_contact
    if has_contact and not draft.contains_personal_data:
        v.append(RuleViolation(code="PII_FLAG_CORRECTED", field="contains_personal_data",
                               detail="contact details found by pattern; flag set by code", severity="warning"))

    spans = []
    for ev in draft.evidence:
        pos = locate(ev.quote, text)
        spans.append(EvidenceSpan(field=ev.field, quote=ev.quote, model_confidence=ev.confidence,
                                  start=pos[0] if pos else None, end=pos[1] if pos else None))

    ticket = SupportTicket(
        category=draft.category,
        priority=draft.priority,
        summary=clean_text(draft.summary) or "(no summary)",
        affected_system=clean_text(draft.affected_system),
        contains_personal_data=contains_pd,
        entities=entities,
        evidence=spans,
    )
    return ticket.model_dump(mode="json"), ticket, v


__all__ = [
    "TicketCategory", "Priority", "EntityType", "TicketFieldName", "CRITICAL_TICKET_FIELDS",
    "TicketEntityDraft", "TicketEvidence", "SupportTicketDraft", "TicketEntity", "SupportTicket",
    "pattern_entities", "build_ticket",
]
