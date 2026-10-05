# path: book/projects/examples/ch32/northwind_triage/domain/models.py
"""Domain vocabulary for ticket triage. Pure: no I/O, no provider types, no framework imports.

pydantic is allowed here because it is a value-validation library, not infrastructure.
Nothing in this package knows that a language model exists.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Category(str, Enum):
    ACCOUNT_ACCESS = "account_access"
    BENEFITS_LEAVE = "benefits_leave"
    EXPENSES_TRAVEL = "expenses_travel"
    HARDWARE = "hardware"
    PASSWORD_MFA = "password_mfa"
    POS_PAYMENTS = "pos_payments"
    RETURNS = "returns"
    SECURITY_REPORT = "security_report"
    SHIPMENT_TRACKING = "shipment_tracking"
    TIME_OFF = "time_off"
    VPN_NETWORK = "vpn_network"
    WAREHOUSE_SCANNER = "warehouse_scanner"


class Priority(str, Enum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"

    @property
    def urgency(self) -> int:
        """Higher is more urgent: P1 -> 4, P4 -> 1."""
        return 5 - int(self.value[1])

    @classmethod
    def from_urgency(cls, urgency: int) -> "Priority":
        return cls(f"P{5 - max(1, min(4, urgency))}")


class Ticket(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    tenant: Literal["retail", "logistics", "shared"]
    subject: str
    body: str
    requester_role: str = "employee"


class TriageDecision(BaseModel):
    """What the classifier proposes. Validated, but not yet trusted for routing."""

    model_config = ConfigDict(frozen=True)

    category: Category
    priority: Priority
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(default="", max_length=500)


Route = Literal["auto", "human_review"]


class FinalTriage(BaseModel):
    """What the system does. Produced only by deterministic code from a TriageDecision."""

    model_config = ConfigDict(frozen=True)

    ticket_id: str
    category: Category | None
    priority: Priority
    route: Route
    reasons: tuple[str, ...] = ()


# Business rules live here, in code, where they are testable and reviewable. The model
# proposes; these functions enforce invariants the business cares about.
MIN_URGENCY: dict[Category, Priority] = {
    Category.SECURITY_REPORT: Priority.P2,
    Category.POS_PAYMENTS: Priority.P3,
}
AUTO_ROUTE_MIN_CONFIDENCE = 0.6
ALWAYS_HUMAN: frozenset[Category] = frozenset({Category.SECURITY_REPORT})


def apply_business_rules(ticket: Ticket, decision: TriageDecision) -> FinalTriage:
    reasons: list[str] = []
    priority = decision.priority
    floor = MIN_URGENCY.get(decision.category)
    if floor is not None and priority.urgency < floor.urgency:
        reasons.append(f"priority raised from {priority.value} to {floor.value} by category floor")
        priority = floor

    route: Route = "auto"
    if decision.category in ALWAYS_HUMAN:
        route = "human_review"
        reasons.append(f"{decision.category.value} always gets a human")
    if decision.confidence < AUTO_ROUTE_MIN_CONFIDENCE:
        route = "human_review"
        reasons.append(f"confidence {decision.confidence:.2f} below {AUTO_ROUTE_MIN_CONFIDENCE}")
    return FinalTriage(
        ticket_id=ticket.id,
        category=decision.category,
        priority=priority,
        route=route,
        reasons=tuple(reasons),
    )


def fallback_triage(ticket: Ticket, reason: str) -> FinalTriage:
    """The safe answer when the classifier is down or its output cannot be parsed."""
    return FinalTriage(
        ticket_id=ticket.id,
        category=None,
        priority=Priority.P3,
        route="human_review",
        reasons=(reason,),
    )
