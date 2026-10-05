# path: book/projects/memorykit/memorykit/models.py
"""The memory record: one row of durable memory, with everything needed to trust, rank,
expire, and delete it.

A memory without provenance cannot be corrected, a memory without an owner cannot be
deleted on request, and a memory without a source cannot be told apart from a model guess.
Every field below exists because one of those operations needs it.
"""
from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

Clock = Callable[[], datetime]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MemoryKind(str, Enum):
    PROFILE = "profile"        # durable facts about a user: preferences, role, location
    SEMANTIC = "semantic"      # facts and learned summaries, retrieved by meaning
    EPISODIC = "episodic"      # what happened on a past task and how it ended
    PROCEDURAL = "procedural"  # reusable instructions and workflows, authored by people


class Source(str, Enum):
    USER_STATED = "user_stated"            # the user said it, verbatim evidence exists
    SYSTEM_OF_RECORD = "system_of_record"  # read from an authoritative system (HRIS, CMDB, harness)
    MODEL_INFERRED = "model_inferred"      # a model concluded it; a guess until confirmed
    RETRIEVED_CONTENT = "retrieved_content"  # text from a document the assistant read
    TOOL_OUTPUT = "tool_output"            # free text returned by a tool that is not a system of record


#: Sources that must never become durable memory on their own (Chapter 26: memory poisoning).
UNTRUSTED_SOURCES = frozenset({Source.RETRIEVED_CONTENT, Source.TOOL_OUTPUT})

#: Conflict precedence: a higher number wins a same-key conflict.
SOURCE_PRECEDENCE: dict[Source, int] = {
    Source.SYSTEM_OF_RECORD: 3,
    Source.USER_STATED: 2,
    Source.MODEL_INFERRED: 1,
    Source.RETRIEVED_CONTENT: 0,
    Source.TOOL_OUTPUT: 0,
}


class Sensitivity(str, Enum):
    """Mirrors the Northwind data classification policy."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class MemoryStatus(str, Enum):
    ACTIVE = "active"          # readable by retrieval
    PENDING = "pending"        # proposed, waiting for user confirmation; never retrieved
    SUPERSEDED = "superseded"  # replaced by a newer record for the same key; kept for audit


class Owner(BaseModel, frozen=True):
    """Every record belongs to a tenant. `user=None` means tenant-wide (shared) memory."""

    tenant: str
    user: str | None = None

    @property
    def is_shared(self) -> bool:
        return self.user is None

    def shared(self) -> "Owner":
        return Owner(tenant=self.tenant)

    def __str__(self) -> str:
        return f"{self.tenant}/{self.user or '*'}"


def new_id() -> str:
    return "mem_" + uuid.uuid4().hex[:16]


_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _WS.sub(" ", text.strip().lower())


class MemoryRecord(BaseModel):
    id: str = Field(default_factory=new_id)
    owner: Owner
    kind: MemoryKind
    key: str | None = None            # slot for structured facts ("preferred_language"); None for free text
    content: str                      # human-readable statement, what gets rendered into context
    value: Any = None                 # structured value when there is one
    source: Source
    provenance: list[str] = Field(default_factory=list)  # "turn:sess-7#4", "hris:emp-1042", "run:r-88", "mem:<id>"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    salience: float = Field(default=0.5, ge=0.0, le=1.0)  # how much it matters when it is relevant
    sensitivity: Sensitivity = Sensitivity.INTERNAL
    status: MemoryStatus = MemoryStatus.ACTIVE
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime | None = None
    version: int = 1
    embedding: list[float] | None = None
    embedding_model: str | None = None

    def is_expired(self, now: datetime) -> bool:
        return self.expires_at is not None and self.expires_at <= now

    def value_text(self) -> str:
        if self.key is not None and self.value is not None:
            return str(self.value)
        return self.content

    def derived_from(self) -> list[str]:
        return [p.removeprefix("mem:") for p in self.provenance if p.startswith("mem:")]


class Tombstone(BaseModel):
    """Proof that a memory existed and was deleted, without its content.

    The fingerprint lets the write path refuse to re-create a deleted fact when an old
    transcript or a re-run extraction proposes it again.
    """

    record_id: str
    owner: Owner
    kind: MemoryKind
    key: str | None
    fingerprint: str
    deleted_at: datetime
    reason: str


__all__ = [
    "Clock",
    "utcnow",
    "MemoryKind",
    "Source",
    "UNTRUSTED_SOURCES",
    "SOURCE_PRECEDENCE",
    "Sensitivity",
    "MemoryStatus",
    "Owner",
    "MemoryRecord",
    "Tombstone",
    "new_id",
    "normalize_text",
]
