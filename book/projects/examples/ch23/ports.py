# path: book/projects/examples/ch23/ports.py
"""Ports and adapters for a framework-independent domain.

The domain (`AnswerService`) depends only on three Protocols it owns:
`Retriever`, `LLMClient`, `Tool`. Framework objects are wrapped in adapters
that live outside the domain. A `RecordingLLM` / `ReplayLLM` pair shows how a
framework-backed component is tested from recorded fixtures without the
framework or the network present.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


# --- domain types ---------------------------------------------------------
class Passage(BaseModel):
    id: str
    text: str
    source: str
    score: float = 0.0


class Answer(BaseModel):
    text: str
    citations: list[str]
    abstained: bool = False


# --- ports (owned by the domain) -----------------------------------------
@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[Passage]: ...


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...


@runtime_checkable
class Tool(Protocol):
    name: str
    def run(self, arguments: dict[str, Any]) -> str: ...


# --- domain service -------------------------------------------------------
@dataclass
class AnswerService:
    retriever: Retriever
    llm: LLMClient
    k: int = 4

    def answer(self, question: str) -> Answer:
        passages = self.retriever.retrieve(question, self.k)
        if not passages:
            return Answer(text="I could not find this in the knowledge base.", citations=[], abstained=True)
        evidence = "\n".join(f"[{p.id}] {p.text}" for p in passages)
        prompt = ("Answer only from the evidence. Cite passage ids in square brackets. "
                  "If the evidence is insufficient, reply exactly: INSUFFICIENT\n\n"
                  f"Evidence:\n{evidence}\n\nQuestion: {question}\nAnswer:")
        text = self.llm.complete(prompt).strip()
        if text == "INSUFFICIENT":
            return Answer(text="The knowledge base does not cover this.", citations=[], abstained=True)
        cited = [p.id for p in passages if f"[{p.id}]" in text]
        return Answer(text=text, citations=cited)


# --- a stand-in for a framework object ------------------------------------
class FrameworkRetrieverLike:
    """Pretend third-party class with its own vocabulary: `get_relevant_documents`
    returns objects with `page_content` and `metadata`. API shape at the time of writing;
    check current docs. We never let this type cross into the domain."""

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self._docs = docs

    def get_relevant_documents(self, query: str) -> list[Any]:
        words = set(query.lower().split())
        hits = [d for d in self._docs if words & set(d["page_content"].lower().split())]
        return [type("Doc", (), d)() for d in hits]


# --- adapter: framework -> port ------------------------------------------
class FrameworkRetrieverAdapter:
    def __init__(self, inner: FrameworkRetrieverLike) -> None:
        self._inner = inner

    def retrieve(self, query: str, k: int) -> list[Passage]:
        docs = self._inner.get_relevant_documents(query)[:k]
        return [Passage(id=d.metadata["id"], text=d.page_content,
                        source=d.metadata.get("source", "unknown"),
                        score=float(d.metadata.get("score", 0.0))) for d in docs]


# --- recorded fixtures ----------------------------------------------------
def _key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


@dataclass
class RecordingLLM:
    """Wrap a live client once, record prompt -> completion to a JSON file."""
    inner: LLMClient
    path: Path
    _cache: dict[str, str] = field(default_factory=dict)

    def complete(self, prompt: str) -> str:
        out = self.inner.complete(prompt)
        self._cache[_key(prompt)] = out
        self.path.write_text(json.dumps(self._cache, indent=2, sort_keys=True))
        return out


@dataclass
class ReplayLLM:
    """Serve recorded completions. A prompt that was never recorded is a
    *test failure*, not a silent miss: it means the prompt text changed."""
    path: Path

    def complete(self, prompt: str) -> str:
        table = json.loads(self.path.read_text())
        try:
            return table[_key(prompt)]
        except KeyError as exc:
            raise LookupError(f"no recorded completion for prompt hash {_key(prompt)}; "
                              "the prompt text changed, re-record or update the fixture") from exc
