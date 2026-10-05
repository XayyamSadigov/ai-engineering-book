# path: book/projects/memorykit/memorykit/policy.py
"""Write policy and consolidation: the only path from "a component wants to remember X"
to a durable row.

The policy answers "may this be stored, in what form, for how long?" Consolidation answers
"given what is already stored, is this new, a duplicate, an update, or a losing conflict?"
Memory classes never call `store.put` for new memories directly; they call `write()`.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime, timedelta
from enum import Enum

from pydantic import BaseModel, Field

from .models import (
    SOURCE_PRECEDENCE,
    UNTRUSTED_SOURCES,
    MemoryKind,
    MemoryRecord,
    MemoryStatus,
    Sensitivity,
    Source,
    normalize_text,
)
from .store import MemoryStore


class Decision(str, Enum):
    ACCEPT = "accept"
    PENDING = "pending"  # stored, but invisible to retrieval until the user confirms it
    REJECT = "reject"


class WriteDecision(BaseModel):
    decision: Decision
    reasons: list[str] = Field(default_factory=list)
    record: MemoryRecord | None = None  # the record as it would be written: TTL set, PII redacted


# Illustrative defaults. Tune per product and per legal retention schedule.
DEFAULT_TTL: dict[MemoryKind, timedelta | None] = {
    MemoryKind.PROFILE: timedelta(days=365),
    MemoryKind.SEMANTIC: timedelta(days=180),
    MemoryKind.EPISODIC: timedelta(days=90),
    MemoryKind.PROCEDURAL: None,  # versioned and reviewed by people instead of expiring
}
DEFAULT_PENDING_TTL = timedelta(days=14)

SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("password", re.compile(r"(?i)\b(pass(word|wd|phrase)?|pwd)\b\s*(is|:|=)\s*\S+")),
    ("api_key", re.compile(r"(?i)\b(api[_-]?key|secret|token|bearer)\b\s*(is|:|=)?\s*[A-Za-z0-9_\-\.]{16,}")),
    ("provider_key", re.compile(r"\b(sk|pk|rk)-[A-Za-z0-9_\-]{16,}\b")),
    ("cloud_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
]
PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("phone", re.compile(r"(?<!\w)\+?\d[\d\s\-().]{7,}\d(?!\w)")),
]
_CARD = re.compile(r"\b(?:\d[ \-]?){13,19}\b")

# Content that addresses the assistant instead of describing the world. A memory is a fact,
# not a directive; directive-shaped text in a memory is either a bug or an attack. This is a
# heuristic second line: the source rule above it is what actually stops poisoning.
INSTRUCTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)\b(ignore|disregard|forget)\b.{0,40}\b(instructions|rules|guidelines|restrictions|access control)\b"),
    re.compile(r"(?i)\byou are (now|an? (ai|assistant))\b"),
    re.compile(r"(?i)\b(system prompt|developer message)\b"),
    re.compile(r"(?i)\b(send|forward|email|upload|post)\b.{0,80}\bto\b.{0,40}(@|https?://)"),
    re.compile(r"(?i)\b(does not|doesn't|do not|no) (require|need)\b.{0,30}\b(confirmation|approval)\b"),
    re.compile(r"(?i)\bfrom now on\b|\balways (reply|respond|send|include|forward)\b"),
]

_SENSITIVITY_ORDER = [Sensitivity.PUBLIC, Sensitivity.INTERNAL, Sensitivity.CONFIDENTIAL, Sensitivity.RESTRICTED]


def _at_least(current: Sensitivity, floor: Sensitivity) -> Sensitivity:
    return max(current, floor, key=_SENSITIVITY_ORDER.index)


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def find_secrets(text: str) -> list[str]:
    hits = [name for name, rx in SECRET_PATTERNS if rx.search(text)]
    for m in _CARD.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            hits.append("card_number")
            break
    return hits


def find_pii(text: str) -> list[str]:
    return [name for name, rx in PII_PATTERNS if rx.search(text)]


def redact_pii(text: str) -> str:
    for name, rx in PII_PATTERNS:
        text = rx.sub(f"[REDACTED:{name}]", text)
    return text


def looks_like_instruction(text: str) -> bool:
    return any(rx.search(text) for rx in INSTRUCTION_PATTERNS)


class WritePolicy:
    """Decides whether a candidate memory may be stored, and in what form.

    Rules, in order (the first rejection wins):
      1. untrusted sources (retrieved documents, free-text tool output) never write memory;
      2. secrets are never stored, in any kind, from any source;
      3. directive-shaped content is rejected outside procedural memory;
      4. procedural memory is written only from a system of record (reviewed by people);
      5. a fact the user deleted is not re-created (tombstone fingerprint match);
      6. PII: allowed only in profile slots on an allow-list from trusted sources; redacted
         from free-text kinds; rejected elsewhere;
      7. model inferences need a minimum confidence, and profile inferences wait for
         confirmation as PENDING with a short TTL;
      8. every accepted record gets an expiry capped by its kind's TTL.
    """

    def __init__(
        self,
        *,
        ttl_by_kind: dict[MemoryKind, timedelta | None] | None = None,
        pending_ttl: timedelta = DEFAULT_PENDING_TTL,
        min_inferred_confidence: float = 0.7,
        pii_allowed_keys: frozenset[str] = frozenset({"work_email", "work_phone"}),
    ) -> None:
        self.ttl_by_kind = {**DEFAULT_TTL, **(ttl_by_kind or {})}
        self.pending_ttl = pending_ttl
        self.min_inferred_confidence = min_inferred_confidence
        self.pii_allowed_keys = pii_allowed_keys

    def evaluate(self, record: MemoryRecord, *, now: datetime, store: MemoryStore | None = None) -> WriteDecision:
        r = record.model_copy(deep=True)
        text = f"{r.content}\n{r.value if r.value is not None else ''}"

        def reject(reason: str) -> WriteDecision:
            return WriteDecision(decision=Decision.REJECT, reasons=[reason])

        if r.source in UNTRUSTED_SOURCES:
            return reject(f"untrusted_source:{r.source.value}")
        secrets = find_secrets(text)
        if secrets:
            return reject("secret:" + ",".join(secrets))
        if r.kind != MemoryKind.PROCEDURAL and looks_like_instruction(text):
            return reject("instruction_like_content")
        if r.kind == MemoryKind.PROCEDURAL and r.source != Source.SYSTEM_OF_RECORD:
            return reject("procedural_requires_system_of_record")
        if store is not None and store.is_suppressed(r):
            return reject("suppressed_by_user_deletion")

        reasons: list[str] = []
        pii = find_pii(text)
        if pii:
            trusted = r.source in (Source.USER_STATED, Source.SYSTEM_OF_RECORD)
            if r.kind == MemoryKind.PROFILE:
                if r.key not in self.pii_allowed_keys or not trusted:
                    return reject("pii_not_allowed:" + ",".join(pii))
                r.sensitivity = _at_least(r.sensitivity, Sensitivity.CONFIDENTIAL)
                reasons.append(f"pii_allowed:{r.key}")
            else:
                r.content = redact_pii(r.content)
                if isinstance(r.value, str):
                    r.value = redact_pii(r.value)
                reasons.append("redacted:" + ",".join(pii))

        decision = Decision.ACCEPT
        if r.source == Source.MODEL_INFERRED:
            if r.confidence < self.min_inferred_confidence:
                return reject(f"low_confidence:{r.confidence:.2f}")
            if r.kind == MemoryKind.PROFILE:
                decision = Decision.PENDING
                r.status = MemoryStatus.PENDING
                r.expires_at = now + self.pending_ttl
                reasons.append("needs_user_confirmation")

        ttl = self.ttl_by_kind.get(r.kind)
        if ttl is not None:
            cap = now + ttl
            r.expires_at = min(r.expires_at, cap) if r.expires_at else cap
        r.updated_at = now
        return WriteDecision(decision=decision, reasons=reasons, record=r)


# ------------------------------------------------------------------------------ consolidation
class ConsolidationAction(str, Enum):
    INSERTED = "inserted"            # new memory
    REFRESHED = "refreshed"          # exact duplicate: provenance merged, expiry renewed
    MERGED = "merged"                # near duplicate of a free-text memory: folded into it
    SUPERSEDED = "superseded"        # same key, candidate won: old record marked superseded
    KEPT_EXISTING = "kept_existing"  # same key, existing record won: candidate dropped


class ConsolidationResult(BaseModel):
    action: ConsolidationAction
    record: MemoryRecord             # the record that is now authoritative for this content
    replaced: list[str] = Field(default_factory=list)
    conflict: bool = False


NearDuplicate = Callable[[MemoryRecord, MemoryRecord], bool]


def candidate_wins(new: MemoryRecord, old: MemoryRecord) -> bool:
    """Higher source precedence wins; within the same precedence, the newer record wins."""
    pn, po = SOURCE_PRECEDENCE[new.source], SOURCE_PRECEDENCE[old.source]
    if pn != po:
        return pn > po
    return new.updated_at >= old.updated_at


def _same_value(a: MemoryRecord, b: MemoryRecord) -> bool:
    return a.key == b.key and normalize_text(a.value_text()) == normalize_text(b.value_text())


def _merge_provenance(a: list[str], b: list[str]) -> list[str]:
    return list(dict.fromkeys([*a, *b]))


def consolidate(
    store: MemoryStore,
    candidate: MemoryRecord,
    *,
    now: datetime,
    near_duplicate: NearDuplicate | None = None,
) -> ConsolidationResult:
    existing = store.query(candidate.owner, kinds=[candidate.kind], key=candidate.key, now=now)
    if candidate.key is None:
        existing = [e for e in existing if e.key is None]

    for e in existing:
        if _same_value(e, candidate):
            better_source = candidate.source if SOURCE_PRECEDENCE[candidate.source] > SOURCE_PRECEDENCE[e.source] else e.source
            refreshed = e.model_copy(update={
                "updated_at": now,
                "confidence": max(e.confidence, candidate.confidence),
                "salience": max(e.salience, candidate.salience),
                "provenance": _merge_provenance(e.provenance, candidate.provenance),
                "source": better_source,
                "expires_at": candidate.expires_at,
                "version": e.version + 1,
            })
            store.put(refreshed, expected_version=e.version)
            return ConsolidationResult(action=ConsolidationAction.REFRESHED, record=refreshed)

    if candidate.status == MemoryStatus.PENDING:
        # Unconfirmed proposals never displace anything; they wait beside the active value.
        store.put(candidate)
        return ConsolidationResult(action=ConsolidationAction.INSERTED, record=candidate)

    if candidate.key is not None and existing:
        old = existing[-1]
        if not candidate_wins(candidate, old):
            return ConsolidationResult(action=ConsolidationAction.KEPT_EXISTING, record=old, conflict=True)
        for e in existing:
            store.put(
                e.model_copy(update={"status": MemoryStatus.SUPERSEDED, "updated_at": now, "version": e.version + 1}),
                expected_version=e.version,
            )
        # "supersedes:" rather than "mem:" so deleting the old record does not cascade into the new one.
        winner = candidate.model_copy(update={"provenance": [*candidate.provenance, *(f"supersedes:{e.id}" for e in existing)]})
        store.put(winner)
        return ConsolidationResult(
            action=ConsolidationAction.SUPERSEDED, record=winner, replaced=[e.id for e in existing], conflict=True
        )

    if candidate.key is None and near_duplicate is not None:
        for e in existing:
            if near_duplicate(e, candidate):
                newer_wins = candidate_wins(candidate, e)
                merged = e.model_copy(update={
                    "content": candidate.content if newer_wins else e.content,
                    "embedding": candidate.embedding if newer_wins else e.embedding,
                    "embedding_model": candidate.embedding_model if newer_wins else e.embedding_model,
                    "provenance": _merge_provenance(e.provenance, candidate.provenance),
                    "confidence": max(e.confidence, candidate.confidence),
                    "salience": max(e.salience, candidate.salience),
                    "updated_at": now,
                    "expires_at": candidate.expires_at,
                    "version": e.version + 1,
                })
                store.put(merged, expected_version=e.version)
                return ConsolidationResult(action=ConsolidationAction.MERGED, record=merged, replaced=[candidate.id])

    store.put(candidate)
    return ConsolidationResult(action=ConsolidationAction.INSERTED, record=candidate)


class WriteOutcome(BaseModel):
    decision: Decision
    reasons: list[str] = Field(default_factory=list)
    action: ConsolidationAction | None = None
    record: MemoryRecord | None = None
    replaced: list[str] = Field(default_factory=list)
    conflict: bool = False

    @property
    def stored(self) -> bool:
        return self.decision != Decision.REJECT and self.action != ConsolidationAction.KEPT_EXISTING


def write(
    store: MemoryStore,
    policy: WritePolicy,
    record: MemoryRecord,
    *,
    now: datetime,
    near_duplicate: NearDuplicate | None = None,
    prepare: Callable[[MemoryRecord], MemoryRecord] | None = None,
) -> WriteOutcome:
    """Policy first, then consolidation. The single entry point for new memories.

    `prepare` runs between the two, on the policy-approved (possibly redacted) record. Use it
    for work that must not see rejected content, such as computing an embedding: sending a
    rejected secret to an embedding provider is itself a leak.
    """
    decision = policy.evaluate(record, now=now, store=store)
    if decision.decision == Decision.REJECT or decision.record is None:
        return WriteOutcome(decision=Decision.REJECT, reasons=decision.reasons)
    approved = prepare(decision.record) if prepare is not None else decision.record
    result = consolidate(store, approved, now=now, near_duplicate=near_duplicate)
    return WriteOutcome(
        decision=decision.decision,
        reasons=decision.reasons,
        action=result.action,
        record=result.record,
        replaced=result.replaced,
        conflict=result.conflict,
    )


__all__ = [
    "Decision",
    "WriteDecision",
    "WritePolicy",
    "DEFAULT_TTL",
    "DEFAULT_PENDING_TTL",
    "find_secrets",
    "find_pii",
    "redact_pii",
    "looks_like_instruction",
    "ConsolidationAction",
    "ConsolidationResult",
    "consolidate",
    "candidate_wins",
    "WriteOutcome",
    "write",
]
