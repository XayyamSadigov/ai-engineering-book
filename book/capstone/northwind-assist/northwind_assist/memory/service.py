# path: book/capstone/northwind-assist/northwind_assist/memory/service.py
"""Memory (Chapter 21): per-session conversation memory and a per-user profile.

Writes go through memorykit's WritePolicy, which rejects retrieved content and free-text tool
output as sources (memory poisoning, Chapter 26), secrets, directive-shaped text, and unapproved
PII. What the *user* states explicitly ("remember that my team is Store 0412") is stored as a
user-stated fact. What the assistant *infers* ("I prefer short answers" -> answer_style=short)
is stored as PENDING and shown to the user for confirmation; only a confirmation turns it into a
fact. Profile facts are rendered into the agent's context as data, never as instructions.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any

from aie_core.llm.client import LLMClient
from memorykit import (
    ConversationMemory,
    InMemoryStore,
    MemoryStatus,
    Owner,
    ProfileError,
    Source,
    SQLiteStore,
    UserProfileMemory,
    WriteOutcome,
)

from ..config import Settings

_REMEMBER = re.compile(r"^\s*remember(?: that)? my ([a-z][a-z ]{1,30}?) (?:is|are) (.{1,120}?)\.?\s*$", re.I)
_PREFER = re.compile(r"\bi (?:prefer|like) (short|brief|detailed|long) (?:answers|replies)\b", re.I)
_LANG = re.compile(r"\b(?:answer|reply) (?:me )?in (english|spanish|portuguese|polish|german|french)\b", re.I)


@dataclass
class MemoryEvent:
    kind: str            # stored | pending | rejected
    key: str
    value: Any
    record_id: str | None
    reasons: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "key": self.key, "value": self.value, "record_id": self.record_id,
                "reasons": self.reasons}


def _event(outcome: WriteOutcome, key: str, value: Any) -> MemoryEvent:
    rec = outcome.record
    if not outcome.stored or rec is None:
        return MemoryEvent("rejected", key, value, None, outcome.reasons)
    kind = "pending" if rec.status == MemoryStatus.PENDING else "stored"
    return MemoryEvent(kind, key, value, rec.id, outcome.reasons)


class MemoryService:
    def __init__(self, settings: Settings) -> None:
        secret = settings.memory_secret.get_secret_value().encode()
        self.store = SQLiteStore(settings.memory_db, fingerprint_secret=secret) if settings.memory_db \
            else InMemoryStore(fingerprint_secret=secret)
        self.profile = UserProfileMemory(self.store)
        self._sessions: dict[tuple[str, str, str], ConversationMemory] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ conversation
    def conversation(self, owner: Owner, session_id: str, llm: LLMClient) -> ConversationMemory:
        key = (owner.tenant, owner.user or "", session_id)
        with self._lock:
            mem = self._sessions.get(key)
            if mem is None:
                mem = ConversationMemory(llm, session_id=session_id, window_turns=6, trigger_turns=12)
                self._sessions[key] = mem
            mem.llm = llm          # the routed client of the current request pays for compaction
            return mem

    # ------------------------------------------------------------------ profile writes
    def observe_user_turn(self, owner: Owner, text: str, *, turn_ref: str) -> list[MemoryEvent]:
        """Deterministic extraction from the user's own words; nothing from documents or tools."""
        events: list[MemoryEvent] = []
        m = _REMEMBER.match(text)
        if m:
            key = re.sub(r"\W+", "_", m.group(1).strip().lower())
            out = self.profile.propose(owner, key, m.group(2).strip(), source=Source.USER_STATED, provenance=[turn_ref])
            events.append(_event(out, key, m.group(2).strip()))
        for rx, key, mapping in ((_PREFER, "answer_style", {"brief": "short", "long": "detailed"}), (_LANG, "language", {})):
            hit = rx.search(text)
            if hit:
                value = mapping.get(hit.group(1).lower(), hit.group(1).lower())
                out = self.profile.propose(owner, key, value, source=Source.MODEL_INFERRED, confidence=0.8,
                                           provenance=[turn_ref])
                events.append(_event(out, key, value))
        return events

    def propose_from(self, owner: Owner, key: str, value: Any, *, source: Source, provenance: str) -> MemoryEvent:
        """The single entry point for every other writer (tools, documents, the model). The write
        policy, not the caller, decides; untrusted sources are rejected by construction."""
        return _event(self.profile.propose(owner, key, value, source=source, provenance=[provenance]), key, value)

    def confirm(self, owner: Owner, record_id: str, *, turn_ref: str) -> MemoryEvent:
        out = self.profile.confirm(owner, record_id, provenance=turn_ref)
        rec = out.record
        return _event(out, rec.key if rec and rec.key else "?", rec.value if rec else None)

    def reject(self, owner: Owner, record_id: str) -> int:
        return len(self.profile.reject(owner, record_id))

    # ------------------------------------------------------------------ reads
    def facts(self, owner: Owner) -> dict[str, Any]:
        return {k: r.value for k, r in self.profile.facts(owner).items()}

    def pending(self, owner: Owner) -> list[dict[str, Any]]:
        return [{"record_id": r.id, "key": r.key, "value": r.value, "source": r.source.value}
                for r in self.profile.pending(owner)]

    def render(self, owner: Owner) -> str:
        return self.profile.render(owner)

    def forget(self, owner: Owner, key: str) -> int:
        return len(self.profile.forget(owner, key))


__all__ = ["MemoryService", "MemoryEvent", "ProfileError"]
