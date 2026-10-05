# path: book/projects/ragkit/ragkit/generation/stream.py
"""Safe streaming of grounded answers.

Streaming raw tokens to a user means showing text before anyone checked it. A sentence with a
fabricated citation, or a fabricated figure, is on screen before the validator runs, and
retracting text that a user already read is worse than never showing it. The compromise used
here is sentence-level buffering:

* the model writes plain sentences that end with [E#] markers (no JSON, so it streams well);
* tokens accumulate until a sentence and its trailing markers are complete;
* each sentence is checked (ids exist, has a citation, lexical support) and then emitted as a
  `text` event, preceded by a `citation` event the first time an evidence id appears;
* a sentence that fails is withheld and reported as a `withheld` event for logs;
* the final `done` event carries a GroundedAnswer assembled from the emitted sentences.

The cost is latency: time to first visible token becomes time to first complete sentence.
For a target such as p95 time-to-first-token under 2 s (illustrative), budget for it.
"""
from __future__ import annotations

import re
from typing import Iterable, Iterator, Literal

from aie_core.llm.client import LLMClient
from aie_core.llm.types import Message
from pydantic import BaseModel, Field

from .generator import CONTRACT_RULES, GroundedGenerator
from .packer import PackedEvidence
from .schema import AnswerStatus, Claim, GroundedAnswer, ResolvedCitation, ValidationIssue
from .support import content_tokens, lexical_support, markers, strip_markers

STREAM_PROMPT_ID = "rag.grounded_answer.stream"
INSUFFICIENT_TOKEN = "INSUFFICIENT_EVIDENCE"
CONFLICT_TOKEN = "CONFLICT:"

STREAM_SYSTEM_PROMPT = f"""You are Northwind Assist. You answer employee questions using only the evidence in this request.

Rules:
{CONTRACT_RULES}
6. Write plain sentences, no lists, no JSON. End every factual sentence with its markers, for example
   "Unused days can be carried over up to 10 days [E1]."
7. If the evidence does not answer the question, reply with exactly one line:
   {INSUFFICIENT_TOKEN}: <what is missing>
8. If rule 5 applies, start the reply with "{CONFLICT_TOKEN}" and then answer."""

EventType = Literal["status", "citation", "text", "withheld", "error", "done"]


class AnswerEvent(BaseModel):
    type: EventType
    text: str | None = None
    eids: list[str] = Field(default_factory=list)
    status: AnswerStatus | None = None
    citation: ResolvedCitation | None = None
    issue: ValidationIssue | None = None
    answer: GroundedAnswer | None = None


# A sentence is complete at . ! or ? plus any trailing markers, once whitespace and the first
# character of the next sentence (not a marker) have arrived. Waiting for that next character
# is what keeps "[E" + "1]" split across deltas from being emitted half-finished.
_BOUNDARY_RE = re.compile(r"(?<=[.!?])((?:\s*\[\s*E\d+(?:\s*,\s*E\d+)*\s*\])*)\s+(?=[^\s\[])")


class SentenceBuffer:
    def __init__(self) -> None:
        self._buf = ""

    def feed(self, delta: str) -> list[str]:
        self._buf += delta
        out: list[str] = []
        start = 0
        for m in _BOUNDARY_RE.finditer(self._buf):
            end = m.start() + len(m.group(1))
            sentence = self._buf[start:end].strip()
            if sentence:
                out.append(sentence)
            start = m.end()
        self._buf = self._buf[start:]
        return out

    def flush(self) -> list[str]:
        rest, self._buf = self._buf.strip(), ""
        return [rest] if rest else []


class GroundedStreamer:
    def __init__(self, llm: LLMClient, *, generator: GroundedGenerator | None = None,
                 check_support: bool = True, min_support: float = 0.6, uncited_min_tokens: int = 4) -> None:
        self.llm = llm
        self.generator = generator or GroundedGenerator(llm)
        self.check_support = check_support
        self.min_support = min_support
        self.uncited_min_tokens = uncited_min_tokens

    def stream(self, question: str, packed: PackedEvidence) -> Iterator[AnswerEvent]:
        if packed.is_empty:
            yield from self.stream_text([f"{INSUFFICIENT_TOKEN}: no relevant documents were retrieved."], packed)
            return
        req = self.generator.build_request(question, packed)
        req = req.model_copy(update={
            "messages": [Message.system(STREAM_SYSTEM_PROMPT), *req.messages[1:]],
            "metadata": {**req.metadata, "prompt.id": STREAM_PROMPT_ID},
        })
        deltas: list[str] = []

        def text_deltas() -> Iterator[str]:
            for ev in self.llm.stream(req):
                if ev.type == "text_delta" and ev.text:
                    yield ev.text
                elif ev.type == "error":
                    deltas.append(ev.error or "stream error")
                    return

        yield from self.stream_text(text_deltas(), packed)
        if deltas:
            yield AnswerEvent(type="error", text=deltas[0])

    def stream_text(self, deltas: Iterable[str], packed: PackedEvidence) -> Iterator[AnswerEvent]:
        """The testable core: turn raw text deltas into validated answer events."""
        buffer = SentenceBuffer()
        state = _StreamState(packed)
        for delta in deltas:
            for sentence in buffer.feed(delta):
                yield from self._handle(sentence, state)
        for sentence in buffer.flush():
            yield from self._handle(sentence, state)
        if state.status is None:
            state.status = "insufficient_evidence"
            state.missing = "The model returned no answer."
        yield AnswerEvent(type="done", status=state.final_status(), answer=state.answer())

    def _handle(self, sentence: str, state: "_StreamState") -> Iterator[AnswerEvent]:
        if state.status is None:
            if sentence.startswith(INSUFFICIENT_TOKEN):
                state.status = "insufficient_evidence"
                state.missing = sentence[len(INSUFFICIENT_TOKEN):].lstrip(" :") or "Not covered by the evidence."
                yield AnswerEvent(type="status", status=state.status)
                return
            if sentence.startswith(CONFLICT_TOKEN):
                state.status = "conflict"
                sentence = sentence[len(CONFLICT_TOKEN):].strip()
            else:
                state.status = "answered"
            yield AnswerEvent(type="status", status=state.status)
            if not sentence:
                return
        if state.status == "insufficient_evidence":
            return  # anything after an abstention line is ignored

        issue = self._check(sentence, state.packed)
        if issue is not None:
            state.withheld += 1
            yield AnswerEvent(type="withheld", text=sentence, issue=issue)
            return
        ids = markers(sentence)
        for eid in ids:
            if eid not in state.cited:
                state.cited.append(eid)
                yield AnswerEvent(type="citation", eids=[eid], citation=state.packed.resolve(eid))
        state.sentences.append(sentence)
        state.claims.append(Claim(text=strip_markers(sentence), citations=ids))
        yield AnswerEvent(type="text", text=sentence + " ", eids=ids)

    def _check(self, sentence: str, packed: PackedEvidence) -> ValidationIssue | None:
        ids = markers(sentence)
        unknown = [e for e in ids if packed.get(e) is None]
        if unknown:
            return ValidationIssue(code="unknown_citation", severity="error", eid=unknown[0],
                                   detail=f"sentence cites {unknown}, never shown")
        if not ids:
            if len(content_tokens(sentence)) >= self.uncited_min_tokens:
                return ValidationIssue(code="uncited_sentence", severity="error", detail="factual sentence without citation")
            return None
        if self.check_support:
            blocks = [packed.get(e) for e in ids]
            clean = "\n".join(b.support_text() for b in blocks if b is not None)
            s = lexical_support(sentence, clean)
            if not s.supported(self.min_support):
                return ValidationIssue(code="unsupported_claim", severity="error",
                                       detail=f"coverage {s.coverage:.2f}, missing numbers {sorted(s.missing_numbers)}")
        return None


class _StreamState:
    def __init__(self, packed: PackedEvidence) -> None:
        self.packed = packed
        self.status: AnswerStatus | None = None
        self.missing: str = ""
        self.cited: list[str] = []
        self.sentences: list[str] = []
        self.claims: list[Claim] = []
        self.withheld = 0

    def final_status(self) -> AnswerStatus:
        if self.status == "insufficient_evidence" or not self.claims:
            return "insufficient_evidence"
        if self.withheld and self.status == "answered":
            return "partial"
        return self.status or "answered"

    def answer(self) -> GroundedAnswer:
        status = self.final_status()
        if status == "insufficient_evidence":
            return GroundedAnswer.insufficient(self.missing or "No sentence passed validation.")
        return GroundedAnswer(status=status, answer=" ".join(self.sentences), claims=self.claims,
                              confidence="low" if self.withheld else "medium")


__all__ = ["AnswerEvent", "GroundedStreamer", "INSUFFICIENT_TOKEN", "CONFLICT_TOKEN", "STREAM_SYSTEM_PROMPT",
           "SentenceBuffer"]
