# path: book/projects/examples/ch05/context/state.py
"""Conversation state with compaction.

Three stores, three rules:
- the turn log is append-only and is the source of truth; compaction never deletes from it;
- facts are exact values (IDs, amounts, decisions, commitments) kept as structured data and
  rendered verbatim on every request; they never pass through a summarizer;
- the summary is lossy narrative for old turns, produced by an LLM, guarded, and rebuildable
  from the log.
The last `keep_last` turns, and any pinned turn, stay verbatim.

State outlives the request, so it is persisted between turns. `snapshot()` / `from_snapshot()`
give it a serializable form, `version` increases on every write, and `InMemoryStateStore.save`
shows the compare-and-set a real store needs so two workers cannot silently overwrite each other.
"""
from __future__ import annotations

import re
import threading
from collections.abc import Callable
from typing import Literal, Protocol

from pydantic import BaseModel

from aie_core.llm.client import LLMClient
from aie_core.llm.tokens import count_tokens
from aie_core.llm.types import CompletionRequest, Message

from .items import ContextItem, Trust

FactCategory = Literal["identifier", "amount", "decision", "preference", "commitment", "open_task", "constraint"]


class Turn(BaseModel):
    index: int
    role: Literal["user", "assistant"]
    content: str
    pinned: bool = False
    message_id: str | None = None  # client-supplied id; a retried request does not append twice


class Fact(BaseModel):
    key: str
    value: str
    category: FactCategory
    source_turn: int | None = None  # provenance: which turn established it
    trust: Trust = Trust.UNTRUSTED  # TRUSTED only when read from a system of record, not from chat


class CompactionReport(BaseModel):
    accepted: bool
    reason: str
    folded_turns: list[int] = []
    tokens_before: int = 0
    tokens_after: int = 0
    novel_literals: list[str] = []


class Summarizer(Protocol):
    def summarize(self, previous_summary: str, turns: list[Turn], facts: list[Fact]) -> str: ...


SUMMARIZER_PROMPT = """You compress the older part of a support conversation so it can be dropped from context.
Write a short narrative of what happened: the user's goal, what was tried, what was decided, what is still open.
Rules:
- Exact values (IDs, amounts, dates, codes) are stored separately under the keys listed below. Refer to them by key, never restate or alter them.
- Do not add anything that is not in the previous summary or the turns.
- Keep unresolved questions and commitments explicit.
- At most {max_words} words. Plain text, no preamble."""


class LLMSummarizer:
    """Summarizer backed by any aie_core LLMClient (a ModelGateway in production, FakeLLM in tests)."""

    def __init__(self, client: LLMClient, *, model: str | None = None, max_words: int = 120) -> None:
        self.client = client
        self.model = model
        self.max_words = max_words

    def summarize(self, previous_summary: str, turns: list[Turn], facts: list[Fact]) -> str:
        fact_keys = ", ".join(f.key for f in facts) or "(none)"
        transcript = "\n".join(f"[{t.index}] {t.role}: {t.content}" for t in turns)
        user = (
            f"Fact keys: {fact_keys}\n\n"
            f"Previous summary:\n{previous_summary or '(none)'}\n\n"
            f"Turns to fold in:\n{transcript}"
        )
        req = CompletionRequest(
            messages=[Message.system(SUMMARIZER_PROMPT.format(max_words=self.max_words)), Message.user(user)],
            model=self.model,
            temperature=0.0,
            max_tokens=self.max_words * 2,
            metadata={"purpose": "context.compaction"},
        )
        return self.client.complete(req).text.strip()


_LITERAL_RE = re.compile(r"\b[A-Z]{2,}-\d+\b|\$?\d[\d,]*(?:\.\d+)?\b")


def literals(text: str) -> set[str]:
    """IDs like INC-4821 and numbers like 1,250.00: the things a summary must not invent.

    Single digits are ignored; they are too common in prose ("2 options") to be a useful signal.
    """
    found = {m.replace(",", "").lstrip("$") for m in _LITERAL_RE.findall(text)}
    return {x for x in found if len(x) > 1}


class ConversationState:
    def __init__(
        self,
        *,
        keep_last: int = 6,
        trigger_tokens: int = 1500,
        counter: Callable[[str], int] | None = None,
    ) -> None:
        self.keep_last = keep_last
        self.trigger_tokens = trigger_tokens
        self.count = counter or count_tokens
        self.log: list[Turn] = []
        self.facts: dict[str, Fact] = {}
        self.fact_history: list[Fact] = []  # superseded values, for audit
        self.summary: str = ""
        self.summary_through: int = -1  # highest turn index folded into the summary
        self.version: int = 0  # bumped on every write
        self.loaded_version: int = 0  # version the store held when this object was loaded (CAS key)

    # ---------------------------------------------------------------- writes
    def add_turn(
        self,
        role: Literal["user", "assistant"],
        content: str,
        *,
        pinned: bool = False,
        message_id: str | None = None,
    ) -> Turn:
        if message_id is not None:
            existing = next((t for t in self.log if t.message_id == message_id), None)
            if existing is not None:
                return existing  # client retry of a turn we already logged
        turn = Turn(index=len(self.log), role=role, content=content, pinned=pinned, message_id=message_id)
        self.log.append(turn)
        self.version += 1
        return turn

    def remember(
        self,
        key: str,
        value: str,
        category: FactCategory,
        *,
        source_turn: int | None = None,
        trust: Trust = Trust.UNTRUSTED,
    ) -> Fact:
        if key in self.facts:
            self.fact_history.append(self.facts[key])
        fact = Fact(key=key, value=value, category=category, source_turn=source_turn, trust=trust)
        self.facts[key] = fact
        self.version += 1
        return fact

    def forget(self, key: str) -> None:
        if key in self.facts:
            self.fact_history.append(self.facts.pop(key))
            self.version += 1

    # ----------------------------------------------------------------- reads
    def active_turns(self) -> list[Turn]:
        return [t for t in self.log if t.index > self.summary_through or t.pinned]

    def history_tokens(self) -> int:
        return sum(self.count(t.content) for t in self.active_turns()) + self.count(self.summary)

    def needs_compaction(self) -> bool:
        return self.history_tokens() > self.trigger_tokens

    def turns_between(self, start: int, end: int) -> list[Turn]:
        """Rehydrate original turns covered by the summary, for audit or on-demand detail."""
        return [t for t in self.log if start <= t.index <= end]

    # ------------------------------------------------------------ compaction
    def _foldable(self) -> list[Turn]:
        recent = {t.index for t in self.log[-self.keep_last :]} if self.keep_last else set()
        return [t for t in self.log if t.index > self.summary_through and not t.pinned and t.index not in recent]

    def compact(self, summarizer: Summarizer, *, force: bool = False) -> CompactionReport:
        if not force and not self.needs_compaction():
            return CompactionReport(accepted=False, reason="under_trigger")
        fold = self._foldable()
        if not fold:
            return CompactionReport(accepted=False, reason="nothing_to_fold")
        before = self.count(self.summary) + sum(self.count(t.content) for t in fold)
        facts = list(self.facts.values())
        candidate = summarizer.summarize(self.summary, fold, facts)
        report = self._check(candidate, fold, facts, before)
        if report.accepted:
            self.summary = candidate
            self.summary_through = max(t.index for t in fold)
            self.version += 1
        return report

    def rebuild_summary(self, summarizer: Summarizer) -> CompactionReport:
        """Re-summarize from the log instead of from the previous summary, to stop drift."""
        if self.summary_through < 0:
            return CompactionReport(accepted=False, reason="nothing_to_fold")
        covered = [t for t in self.log if t.index <= self.summary_through and not t.pinned]
        facts = list(self.facts.values())
        before = sum(self.count(t.content) for t in covered)
        candidate = summarizer.summarize("", covered, facts)
        report = self._check(candidate, covered, facts, before, previous="")
        if report.accepted:
            self.summary = candidate
            self.version += 1
        return report

    def _check(
        self, candidate: str, fold: list[Turn], facts: list[Fact], before: int, previous: str | None = None
    ) -> CompactionReport:
        previous = self.summary if previous is None else previous
        allowed = literals(previous) | {x for t in fold for x in literals(t.content)}
        allowed |= {str(t.index) for t in fold}  # "in turn 12" is a reference, not an invention
        allowed |= {x for f in facts for x in literals(f.value) | literals(f.key)}
        novel = sorted(literals(candidate) - allowed)
        after = self.count(candidate)
        folded = [t.index for t in fold]
        if not candidate.strip():
            return CompactionReport(accepted=False, reason="empty_summary", folded_turns=folded, tokens_before=before)
        if novel:
            # The summarizer introduced a number or ID that appears nowhere in its sources.
            return CompactionReport(
                accepted=False, reason="novel_literals", folded_turns=folded,
                tokens_before=before, tokens_after=after, novel_literals=novel,
            )
        if after >= before:
            return CompactionReport(
                accepted=False, reason="no_gain", folded_turns=folded, tokens_before=before, tokens_after=after
            )
        return CompactionReport(
            accepted=True, reason="ok", folded_turns=folded, tokens_before=before, tokens_after=after
        )

    # ----------------------------------------------------------- persistence
    def snapshot(self) -> StateSnapshot:
        return StateSnapshot(
            version=self.version,
            keep_last=self.keep_last,
            trigger_tokens=self.trigger_tokens,
            log=list(self.log),
            facts=list(self.facts.values()),
            fact_history=list(self.fact_history),
            summary=self.summary,
            summary_through=self.summary_through,
        )

    @classmethod
    def from_snapshot(cls, snap: StateSnapshot, *, counter: Callable[[str], int] | None = None) -> ConversationState:
        state = cls(keep_last=snap.keep_last, trigger_tokens=snap.trigger_tokens, counter=counter)
        state.log = list(snap.log)
        state.facts = {f.key: f for f in snap.facts}
        state.fact_history = list(snap.fact_history)
        state.summary = snap.summary
        state.summary_through = snap.summary_through
        state.version = snap.version
        state.loaded_version = snap.version
        return state

    # -------------------------------------------------------------- to items
    def to_items(self) -> list[ContextItem]:
        items: list[ContextItem] = []
        for trust in (Trust.TRUSTED, Trust.UNTRUSTED):
            group = [f for f in self.facts.values() if f.trust is trust]
            if not group:
                continue
            header = (
                "Exact facts from systems of record (authoritative, copy values verbatim):"
                if trust is Trust.TRUSTED
                else "Facts stated in this conversation (copy values verbatim, verify before acting):"
            )
            lines = [header] + [f"- {f.key} [{f.category}]: {f.value}" for f in group]
            items.append(
                ContextItem(
                    kind="fact",
                    content="\n".join(lines),
                    source_id=f"state:facts:{trust.value}",
                    trust=trust,
                    pinned=True,  # exact state is never dropped and never compacted
                    priority=1.0,
                )
            )
        if self.summary:
            items.append(
                ContextItem(
                    kind="summary",
                    content=f"Summary of turns 0-{self.summary_through}:\n{self.summary}",
                    source_id=f"state:summary:0-{self.summary_through}",
                    trust=Trust.UNTRUSTED,  # derived from user text by a model
                    priority=0.9,
                    metadata={"covers": [0, self.summary_through]},
                )
            )
        active = self.active_turns()
        n = len(active)
        for rank, turn in enumerate(active):
            items.append(
                ContextItem(
                    kind="turn",
                    role=turn.role,
                    content=turn.content,
                    source_id=f"turn:{turn.index}",
                    pinned=turn.pinned,
                    priority=0.4 + 0.5 * (rank + 1) / n,  # newer turns matter more
                    metadata={"turn": turn.index},
                )
            )
        return items


class StateSnapshot(BaseModel):
    """Serializable ConversationState: one row (or document) per session."""

    version: int
    keep_last: int
    trigger_tokens: int
    log: list[Turn]
    facts: list[Fact]
    fact_history: list[Fact]
    summary: str
    summary_through: int


class StaleStateError(Exception):
    """Another writer saved this session since it was loaded. Reload, re-apply, retry."""


class InMemoryStateStore:
    """Reference store with compare-and-set on `version`.

    A SQL store does the same with `UPDATE sessions SET body = :body, version = :new
    WHERE id = :id AND version = :expected` and treats zero updated rows as StaleStateError.
    """

    def __init__(self) -> None:
        self._rows: dict[str, StateSnapshot] = {}
        self._lock = threading.Lock()

    def load(self, session_id: str, *, counter: Callable[[str], int] | None = None) -> ConversationState:
        with self._lock:
            snap = self._rows.get(session_id)
        if snap is None:
            return ConversationState(counter=counter)
        return ConversationState.from_snapshot(snap, counter=counter)

    def save(self, session_id: str, state: ConversationState) -> None:
        expected = state.loaded_version
        with self._lock:
            row = self._rows.get(session_id)
            current = 0 if row is None else row.version
            if current != expected:
                raise StaleStateError(f"session {session_id}: loaded v{expected}, store has v{current}")
            snap = state.snapshot()
            self._rows[session_id] = snap
        state.loaded_version = snap.version


__all__ = [
    "ConversationState", "Turn", "Fact", "FactCategory", "CompactionReport",
    "Summarizer", "LLMSummarizer", "literals", "SUMMARIZER_PROMPT",
    "StateSnapshot", "StaleStateError", "InMemoryStateStore",
]
