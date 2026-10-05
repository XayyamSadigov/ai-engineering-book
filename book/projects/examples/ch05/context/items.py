# path: book/projects/examples/ch05/context/items.py
"""The unit of context: one typed, attributed, budgeted piece of text.

Everything that can enter a prompt (instructions, a retrieved chunk, a tool result, a
conversation turn, a remembered fact) becomes a ContextItem before the builder sees it.
The builder never handles raw strings, so every token in the final prompt can be traced
back to a source and a reason.
"""
from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class Trust(str, Enum):
    TRUSTED = "trusted"  # written by us: system contract, policies we own, schemas
    UNTRUSTED = "untrusted"  # anything a user, document author, or tool could influence


class Section(str, Enum):
    """Where an item lands in the rendered prompt. Order here is render order."""

    SYSTEM = "system"  # stable instructions, policies, schemas, examples (cacheable prefix)
    STATE = "state"  # structured facts and the compacted summary
    HISTORY = "history"  # recent conversation turns, verbatim
    TOOL_RESULTS = "tool_results"
    EVIDENCE = "evidence"  # retrieved documents
    QUERY = "query"  # the current user request: always last, nearest the decision


Kind = Literal[
    "instructions", "policy", "schema", "example",
    "fact", "summary", "turn",
    "tool_result", "evidence", "memory",
    "query",
]

KIND_TO_SECTION: dict[str, Section] = {
    "instructions": Section.SYSTEM,
    "policy": Section.SYSTEM,
    "schema": Section.SYSTEM,
    "example": Section.SYSTEM,
    "fact": Section.STATE,
    "summary": Section.STATE,
    "memory": Section.STATE,
    "turn": Section.HISTORY,
    "tool_result": Section.TOOL_RESULTS,
    "evidence": Section.EVIDENCE,
    "query": Section.QUERY,
}

# Kinds whose content is expected to be identical across requests. The builder renders them
# first so the provider or serving engine can reuse the computed prefix.
STABLE_KINDS = frozenset({"instructions", "policy", "schema", "example"})


class ContextItem(BaseModel):
    kind: Kind
    content: str
    source_id: str  # where it came from: "kb:hr-pto-policy#2", "tool:search_tickets:call_7", "turn:12"
    priority: float = 0.5  # higher wins when the budget is tight; retrieval score, recency, or a constant
    trust: Trust = Trust.UNTRUSTED  # default to the safe side; trusted must be asserted
    tokens: int | None = None  # filled by the builder (rendered size, labels included)
    pinned: bool = False  # must be included verbatim or the build fails; never compacted or dropped
    role: Literal["user", "assistant"] | None = None  # only for kind="turn"
    metadata: dict[str, Any] = Field(default_factory=dict)  # tenant, acl_groups, score, turn index...
    id: str = ""

    @model_validator(mode="after")
    def _derive(self) -> "ContextItem":
        if not self.id:
            digest = hashlib.sha1(f"{self.kind}|{self.source_id}|{self.content}".encode()).hexdigest()
            self.id = f"{self.kind}:{digest[:10]}"
        if self.kind == "turn" and self.role is None:
            raise ValueError("turn items need a role")
        if self.kind in STABLE_KINDS and self.trust is Trust.UNTRUSTED:
            # An instruction we did not write is not an instruction; it is data pretending to be one.
            raise ValueError(f"{self.kind} items must be trusted; label external text as evidence instead")
        return self

    @property
    def section(self) -> Section:
        return KIND_TO_SECTION[self.kind]

    @property
    def stable(self) -> bool:
        return self.kind in STABLE_KINDS


__all__ = ["Trust", "Section", "Kind", "ContextItem", "KIND_TO_SECTION", "STABLE_KINDS"]
