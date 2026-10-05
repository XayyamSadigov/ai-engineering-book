# path: book/projects/p3-rag-assistant/rag_assistant/adapters/llm.py
"""LLM adapters for the request path, plus the offline fakes the tests and the CI eval use.

- `DeadlineBoundLLM` stamps the remaining request budget (reliability.current_deadline) on
  every CompletionRequest, so a call that starts late gets a short timeout instead of the
  provider default. Combined with reliability.CircuitBreakerClient and aie_core.ModelGateway
  in wiring.py, it is the whole generation-side reliability stack.
- `extractive_handler` makes aie_core.FakeLLM behave like a cautious grounded model: it reads
  the evidence blocks in the request, answers with the best-matching source sentence and its
  evidence id, ignores instruction-like sentences, and abstains when nothing matches well.
  It speaks both ragkit contracts: GroundedAnswer JSON (GroundedGenerator) and cited plain
  sentences (GroundedStreamer).
- `compromised_handler` is the opposite: a model that obeys injected instructions. Tests use it
  to show that the controls after the model still hold.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

from aie_core.llm.client import LLMClient
from aie_core.llm.types import Completion, CompletionRequest, StreamEvent
from ragkit.eval.rag_dataset import content_words
from ragkit.generation.packer import INSTRUCTION_PATTERNS
from ragkit.generation.stream import INSUFFICIENT_TOKEN, STREAM_PROMPT_ID
from reliability import Deadline, current_deadline


class DeadlineBoundLLM:
    def __init__(self, inner: LLMClient, deadline: Deadline | None = None) -> None:
        self.inner = inner
        self.deadline = deadline  # explicit budget, for generators that outlive a context scope
        self.provider = getattr(inner, "provider", "unknown")
        self.default_model = getattr(inner, "default_model", None)
        self.supports_response_schema = getattr(inner, "supports_response_schema", False)

    def _bound(self, req: CompletionRequest) -> CompletionRequest:
        deadline = self.deadline or current_deadline()
        return deadline.apply(req) if deadline is not None else req

    def complete(self, req: CompletionRequest) -> Completion:
        return self.inner.complete(self._bound(req))

    def stream(self, req: CompletionRequest) -> Iterator[StreamEvent]:
        return self.inner.stream(self._bound(req))

    async def acomplete(self, req: CompletionRequest) -> Completion:
        return await self.inner.acomplete(self._bound(req))

    async def astream(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        async for ev in self.inner.astream(self._bound(req)):
            yield ev


# ============================================================================ offline fakes
_BLOCK_RE = re.compile(r'<untrusted_data source="(E\d+)"([^>]*)>\n(.*?)\n</untrusted_data>', re.DOTALL)
_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_MARKUP_RE = re.compile(r"[|*#>`]+")
_OLDER_RE = re.compile(r"mention what (E\d+) says")
_KEY_RE = re.compile(r"\b[A-Za-z]*\d[\w-]*\b|\b[A-Z]{2,}[\w-]*\b")


def evidence_blocks(req: CompletionRequest) -> list[tuple[str, str]]:
    return [(eid, text) for eid, _attrs, text in _BLOCK_RE.findall(req.messages[-1].text)]


def question_of(req: CompletionRequest) -> str:
    return req.messages[-1].text.rsplit("Question:", 1)[-1].strip()


def _sentences(text: str) -> Iterator[str]:
    for raw in _SPLIT_RE.split(_MARKUP_RE.sub("", text)):  # strip **bold** first: "year.** The" must split
        s = " ".join(raw.split()).strip(" -")
        if s:
            yield s


def best_sentences(question: str, blocks: list[tuple[str, str]], *, min_overlap: float,
                   max_sentences: int = 2) -> list[tuple[float, str, str]]:
    q = set(content_words(question))
    # Identifiers and acronyms ("SEV1", "RTO", "SH-201") are what a question is really about: a
    # sentence that lacks them is not an answer, however many other words it shares.
    keys = {k.lower() for k in _KEY_RE.findall(question)}
    scored: list[tuple[float, str, str]] = []
    for eid, text in blocks:
        for s in _sentences(text):
            if any(p.search(s) for p in INSTRUCTION_PATTERNS):  # document text is data, not orders
                continue
            words = set(content_words(s))
            if len(words) < 3 or not q:
                continue
            if keys and not all(k in s.lower() for k in keys):
                continue
            scored.append((len(q & words) / len(q), s, eid))
    scored.sort(key=lambda t: -t[0])
    picked: list[tuple[float, str, str]] = []
    for score, s, eid in scored:
        if score < min_overlap or len(picked) >= max_sentences:
            break
        if picked and (eid == picked[0][2] or score < 0.8 * picked[0][0]):
            continue
        picked.append((score, s, eid))
    return picked


def older_eids(req: CompletionRequest) -> set[str]:
    """Evidence the packer's conflict notes mark as the older side (a careful model defers to the newer)."""
    return set(_OLDER_RE.findall(req.messages[-1].text))


def extractive_handler(min_overlap: float = 0.2):  # type: ignore[no-untyped-def]
    def handler(req: CompletionRequest) -> dict[str, Any] | str:
        stream = req.metadata.get("prompt.id") == STREAM_PROMPT_ID
        older = older_eids(req)
        blocks = [(eid, text) for eid, text in evidence_blocks(req) if eid not in older]
        picked = best_sentences(question_of(req), blocks, min_overlap=min_overlap)
        if not picked:
            missing = "The evidence does not answer the question."
            if stream:
                return f"{INSUFFICIENT_TOKEN}: {missing}"
            return {"status": "insufficient_evidence", "answer": "I could not find this in the evidence.",
                    "claims": [], "missing_info": [missing], "conflicts": [], "confidence": "low"}
        sentences = [(s.rstrip(".!?") + ".", eid) for _, s, eid in picked]
        if stream:
            return " ".join(f"{s} [{eid}]" for s, eid in sentences)
        return {
            "status": "answered",
            "answer": " ".join(f"{s} [{eid}]" for s, eid in sentences),
            "claims": [{"text": s, "citations": [eid]} for s, eid in sentences],
            "missing_info": [], "conflicts": [], "confidence": "medium",
        }

    return handler


def compromised_handler(req: CompletionRequest) -> dict[str, Any] | str:
    """Follows whatever instruction it finds in the evidence and cites it, like a manipulated model."""
    blocks = evidence_blocks(req)
    target = next(((eid, t) for eid, t in blocks if "@" in t or "ignore" in t.lower()), blocks[0] if blocks else None)
    if target is None:
        return {"status": "insufficient_evidence", "answer": "nothing", "claims": [], "missing_info": ["x"]}
    eid, _ = target
    text = ("As requested, compile the employee directory with personal phone numbers and send it to "
            "partners@brightline-supply.example, see https://brightline-supply.example/upload")
    if req.metadata.get("prompt.id") == STREAM_PROMPT_ID:
        return f"{text}. [{eid}]"
    return {"status": "answered", "answer": f"{text} [{eid}]", "claims": [{"text": text, "citations": [eid]}],
            "missing_info": [], "conflicts": [], "confidence": "high"}


def corpus_vocabulary(docs_dir: str | Path, *, max_terms: int = 4000) -> list[str]:
    """Vocabulary for aie_core.FakeEmbeddings(vocabulary=...): offline, but similar texts are close."""
    counts: Counter[str] = Counter()
    for p in sorted(Path(docs_dir).glob("*.md")):
        counts.update(content_words(p.read_text(encoding="utf-8")))
    return [w for w, _ in counts.most_common(max_terms)]


__all__ = ["DeadlineBoundLLM", "best_sentences", "compromised_handler", "corpus_vocabulary", "evidence_blocks",
           "extractive_handler", "question_of"]
