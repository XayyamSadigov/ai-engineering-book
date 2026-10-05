# path: book/projects/examples/ch32/northwind_triage/adapters/tools.py
"""A tool with an explicit, testable contract: create_ticket.

The JSON Schema the model sees is generated from the pydantic model the handler validates
with, so the two cannot drift. The contract also declares side-effect class, approval
requirement, a schema version, and examples. Contract tests (tests/test_tool_contracts.py)
check all of it, and tool_schemas.lock pins the schema hash per version. Chapter 16 owns the
full tool registry and policy engine.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from aie_core.llm.types import ToolSpec
from pydantic import BaseModel, ConfigDict, Field

from ..domain import Category, Priority


class CreateTicketArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant: Literal["retail", "logistics", "shared"]
    subject: str = Field(min_length=3, max_length=120)
    body: str = Field(min_length=1, max_length=4000)
    category: Category
    priority: Priority
    idempotency_key: str = Field(pattern=r"^[A-Za-z0-9_-]{8,64}$",
                                 description="Client-generated; retries with the same key are no-ops.")


class CreateTicketResult(BaseModel):
    ticket_id: str
    created: bool  # False when the idempotency key was already used


@dataclass
class InMemoryTicketRepo:
    tickets: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_key: dict[str, str] = field(default_factory=dict)

    def create(self, args: CreateTicketArgs) -> CreateTicketResult:
        scoped_key = f"{args.tenant}:{args.idempotency_key}"
        if scoped_key in self.by_key:
            return CreateTicketResult(ticket_id=self.by_key[scoped_key], created=False)
        ticket_id = f"TCK-NEW-{len(self.tickets) + 1:04d}"
        self.tickets[ticket_id] = args.model_dump(mode="json")
        self.by_key[scoped_key] = ticket_id
        return CreateTicketResult(ticket_id=ticket_id, created=True)


@dataclass(frozen=True)
class ToolContract:
    name: str
    version: str
    description: str
    args_model: type[BaseModel]
    result_model: type[BaseModel]
    side_effect: Literal["read", "write"]
    requires_approval: bool
    valid_examples: tuple[dict[str, Any], ...]
    invalid_examples: tuple[dict[str, Any], ...]

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description,
                        parameters=self.args_model.model_json_schema())

    def schema_hash(self) -> str:
        canonical = json.dumps(self.spec().model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    @property
    def label(self) -> str:
        return f"{self.version}#{self.schema_hash()}"


CREATE_TICKET = ToolContract(
    name="create_ticket",
    version="2.0.0",
    description=("Create a support ticket for the requester's tenant. Use only after the user "
                 "confirmed the summary. Retrying with the same idempotency_key is safe."),
    args_model=CreateTicketArgs,
    result_model=CreateTicketResult,
    side_effect="write",
    requires_approval=False,
    valid_examples=(
        {"tenant": "retail", "subject": "Register 3 declines cards", "body": "Store 0412, since 08:00.",
         "category": "pos_payments", "priority": "P1", "idempotency_key": "conv-81f2-msg-4"},
        {"tenant": "shared", "subject": "VPN drops hourly", "body": "Since the update.",
         "category": "vpn_network", "priority": "P3", "idempotency_key": "conv-0001-msg-2"},
    ),
    invalid_examples=(
        {"tenant": "acme", "subject": "x" * 10, "body": "b", "category": "hardware",
         "priority": "P3", "idempotency_key": "abcdefgh"},                       # unknown tenant
        {"tenant": "retail", "subject": "ok subject", "body": "b", "category": "hardware",
         "priority": "P9", "idempotency_key": "abcdefgh"},                       # bad priority
        {"tenant": "retail", "subject": "ok subject", "body": "b", "category": "hardware",
         "priority": "P3"},                                                      # missing key
        {"tenant": "retail", "subject": "ok subject", "body": "b", "category": "hardware",
         "priority": "P3", "idempotency_key": "abcdefgh", "assignee": "root"},   # extra field
    ),
)


def make_create_ticket_handler(repo: InMemoryTicketRepo) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """The executor-facing handler: raw model arguments in, JSON-able result out."""

    def handler(raw_args: dict[str, Any]) -> dict[str, Any]:
        args = CreateTicketArgs.model_validate(raw_args)   # raises ValidationError on bad input
        return repo.create(args).model_dump(mode="json")

    return handler


CONTRACTS: tuple[ToolContract, ...] = (CREATE_TICKET,)
