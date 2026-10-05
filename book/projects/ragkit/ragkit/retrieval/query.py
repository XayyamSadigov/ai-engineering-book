# path: book/projects/ragkit/ragkit/retrieval/query.py
"""Query transformation: turn what the user typed into what the index should be searched with.

Every transformer returns a `QueryPlan` that always carries `original_text`. Transformations
are lossy and can drift from intent; keeping the original lets the pipeline rerank against it,
lets evaluation compare transformed and untransformed retrieval, and lets a trace show exactly
what was searched.

- QueryRewriter: resolves conversation references ("and for contractors?") into a standalone query.
- MultiQueryExpander: the original plus paraphrases that use the corpus's likely vocabulary.
- QueryDecomposer: splits a multi-part question into sub-questions searched separately.
- HyDEGenerator: writes a hypothetical answer passage; dense retrievers embed it instead of the
  question. The passage may be factually wrong. It is a search key, never evidence.

All use aie_core structured output and fall back to the original query when the model output is
empty, invalid, or implausibly long, so a transformer failure degrades retrieval, not availability.
"""
from __future__ import annotations

import logging
from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, Field

from aie_core.llm.client import LLMClient
from aie_core.llm.errors import LLMError
from aie_core.llm.structured import complete_structured
from aie_core.llm.types import CompletionRequest, Message

from .types import RetrievalQuery

log = logging.getLogger(__name__)


class QueryPlan(BaseModel):
    original_text: str
    queries: list[str]  # search texts for every retriever; queries[0] is the primary one
    strategy: str
    hyde_passages: list[str] = Field(default_factory=list)  # embedded by dense retrievers only
    fallback: bool = False  # True when the transformer failed and returned the original
    notes: dict[str, Any] = Field(default_factory=dict)

    @property
    def primary(self) -> str:
        """Self-contained version of the question: what rerankers judge relevance against."""
        return self.queries[0] if self.queries else self.original_text

    def search_queries(self, base: RetrievalQuery, text: str) -> RetrievalQuery:
        return base.model_copy(update={"text": text, "original_text": self.original_text})


@runtime_checkable
class QueryTransformer(Protocol):
    def transform(self, query: RetrievalQuery, history: Sequence[Message] | None = None) -> QueryPlan: ...


def _dedupe(texts: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in texts:
        t = " ".join(t.split())
        key = t.lower()
        if t and key not in seen:
            seen.add(key)
            out.append(t)
    return out


def _plausible(text: str, original: str, max_ratio: float = 4.0, slack: int = 120) -> bool:
    """Reject empty rewrites and rewrites that balloon into essays (a sign of drift)."""
    return bool(text.strip()) and len(text) <= max_ratio * len(original) + slack


def _original(query: RetrievalQuery) -> str:
    return query.original_text or query.text


class IdentityTransformer:
    name = "identity"

    def transform(self, query: RetrievalQuery, history: Sequence[Message] | None = None) -> QueryPlan:
        return QueryPlan(original_text=_original(query), queries=[query.text], strategy=self.name)


class _LLMTransformer:
    name = "llm"
    system = ""

    def __init__(self, llm: LLMClient, *, model: str | None = None, max_tokens: int = 300) -> None:
        self.llm = llm
        self.model = model
        self.max_tokens = max_tokens

    def _ask(self, user: str, schema: type[BaseModel]) -> BaseModel:
        req = CompletionRequest(
            messages=[Message.system(self.system), Message.user(user)],
            model=self.model,
            temperature=0.0,
            max_tokens=self.max_tokens,
            metadata={"purpose": f"query.{self.name}"},
        )
        parsed, _ = complete_structured(self.llm, req, schema, max_repair_attempts=1)
        return parsed

    def _fallback(self, query: RetrievalQuery, reason: str) -> QueryPlan:
        log.warning("%s fell back to the original query: %s", type(self).__name__, reason)
        return QueryPlan(original_text=_original(query), queries=[query.text], strategy=self.name,
                         fallback=True, notes={"fallback_reason": reason})


# ----------------------------------------------------------------------------- rewriting
class Rewrite(BaseModel):
    query: str = Field(description="a standalone search query")


class QueryRewriter(_LLMTransformer):
    """Conversation-aware rewriting into a standalone query. Without history, it only normalizes."""

    name = "rewrite"
    system = """Rewrite the user's latest message into one standalone search query for Northwind's
internal knowledge base. Resolve pronouns and references using the conversation. Keep every
identifier, number, product name and error code exactly as written. Do not answer the question,
do not add facts, do not broaden or narrow its scope. If it is already standalone, return it unchanged."""

    def __init__(self, llm: LLMClient, *, max_history_turns: int = 6, keep_original: bool = False, **kw: Any) -> None:
        super().__init__(llm, **kw)
        self.max_history_turns = max_history_turns
        self.keep_original = keep_original

    def transform(self, query: RetrievalQuery, history: Sequence[Message] | None = None) -> QueryPlan:
        original = _original(query)
        turns = list(history or [])[-self.max_history_turns :]
        convo = "\n".join(f"{m.role.value}: {m.text}" for m in turns) or "(no earlier messages)"
        try:
            out = self._ask(f"Conversation so far:\n{convo}\n\nLatest message: {original}", Rewrite)
        except LLMError as exc:
            return self._fallback(query, str(exc))
        assert isinstance(out, Rewrite)
        if not _plausible(out.query, original):
            return self._fallback(query, "implausible rewrite")
        queries = [out.query, original] if self.keep_original else [out.query]
        return QueryPlan(original_text=original, queries=_dedupe(queries), strategy=self.name,
                         notes={"history_turns": len(turns)})


# ----------------------------------------------------------------------------- multi-query
class Expansions(BaseModel):
    queries: list[str] = Field(description="alternative phrasings of the same question")


class MultiQueryExpander(_LLMTransformer):
    """Original query first, then up to n paraphrases. Each is searched; results are fused."""

    name = "multi_query"
    system = """Write alternative search queries for the user's question, as an employee or a
policy author might phrase it: use synonyms, expand abbreviations, use formal document vocabulary.
Keep identifiers and numbers exactly. Each alternative must ask the same thing. Do not answer."""

    def __init__(self, llm: LLMClient, *, n: int = 3, **kw: Any) -> None:
        super().__init__(llm, **kw)
        self.n = n

    def transform(self, query: RetrievalQuery, history: Sequence[Message] | None = None) -> QueryPlan:
        original = _original(query)
        try:
            out = self._ask(f"Question: {query.text}\nWrite {self.n} alternatives.", Expansions)
        except LLMError as exc:
            return self._fallback(query, str(exc))
        assert isinstance(out, Expansions)
        extra = [q for q in out.queries if _plausible(q, query.text)][: self.n]
        return QueryPlan(original_text=original, queries=_dedupe([query.text, *extra]), strategy=self.name)


# ----------------------------------------------------------------------------- decomposition
class Decomposition(BaseModel):
    sub_questions: list[str] = Field(description="self-contained sub-questions; one item if not decomposable")


class QueryDecomposer(_LLMTransformer):
    """Split multi-part questions. Each sub-question is searched; the original is kept for reranking."""

    name = "decompose"
    system = """If the question asks several things or needs facts from different documents, split
it into the smallest set of self-contained sub-questions, each answerable from one document.
Keep identifiers exactly. If the question asks one thing, return it as the only sub-question."""

    def __init__(self, llm: LLMClient, *, max_subqueries: int = 4, include_original: bool = True, **kw: Any) -> None:
        super().__init__(llm, **kw)
        self.max_subqueries = max_subqueries
        self.include_original = include_original

    def transform(self, query: RetrievalQuery, history: Sequence[Message] | None = None) -> QueryPlan:
        original = _original(query)
        try:
            out = self._ask(f"Question: {query.text}", Decomposition)
        except LLMError as exc:
            return self._fallback(query, str(exc))
        assert isinstance(out, Decomposition)
        subs = [s for s in out.sub_questions if _plausible(s, query.text)][: self.max_subqueries]
        if not subs:
            return self._fallback(query, "no usable sub-questions")
        queries = [query.text, *subs] if self.include_original else subs
        return QueryPlan(original_text=original, queries=_dedupe(queries), strategy=self.name,
                         notes={"sub_questions": subs})


# ----------------------------------------------------------------------------- HyDE
class Hypothetical(BaseModel):
    passage: str = Field(description="a short passage, written like the documentation, that would answer the question")


class HyDEGenerator(_LLMTransformer):
    """Hypothetical Document Embeddings: embed a generated answer-shaped passage, not the question.

    The plan keeps the original question as the only lexical query, so BM25 never searches with
    invented terms, and the passages are routed to dense retrievers by the pipeline.
    """

    name = "hyde"
    system = """Write a short passage (2-4 sentences) in the style of an internal policy or runbook
that would answer the question. Use the vocabulary such a document would use. If you do not know
specific values, use placeholders like X rather than inventing numbers. Do not mention the question."""

    def __init__(self, llm: LLMClient, *, n: int = 1, max_passage_chars: int = 1200, **kw: Any) -> None:
        kw.setdefault("max_tokens", 250)
        super().__init__(llm, **kw)
        self.n = n
        self.max_passage_chars = max_passage_chars

    def transform(self, query: RetrievalQuery, history: Sequence[Message] | None = None) -> QueryPlan:
        original = _original(query)
        passages: list[str] = []
        for _ in range(self.n):
            try:
                out = self._ask(f"Question: {query.text}", Hypothetical)
            except LLMError as exc:
                return self._fallback(query, str(exc))
            assert isinstance(out, Hypothetical)
            if out.passage.strip():
                passages.append(out.passage.strip()[: self.max_passage_chars])
        if not passages:
            return self._fallback(query, "empty hypothetical passage")
        return QueryPlan(original_text=original, queries=[query.text], strategy=self.name,
                         hyde_passages=_dedupe(passages))


# ----------------------------------------------------------------------------- composition
class ChainedTransformer:
    """Apply transformers in order: e.g. rewrite (uses history) then expand the standalone query.

    Each later step sees the previous step's primary query; original_text never changes. Queries
    and HyDE passages from all steps are merged.
    """

    def __init__(self, *steps: QueryTransformer) -> None:
        if not steps:
            raise ValueError("ChainedTransformer needs at least one step")
        self.steps = steps
        self.name = "+".join(getattr(s, "name", type(s).__name__) for s in steps)

    def transform(self, query: RetrievalQuery, history: Sequence[Message] | None = None) -> QueryPlan:
        original = _original(query)
        current = query
        queries: list[str] = []
        passages: list[str] = []
        fallback = False
        notes: dict[str, Any] = {}
        for i, step in enumerate(self.steps):
            plan = step.transform(current, history if i == 0 else None)
            fallback = fallback or plan.fallback
            notes[getattr(step, "name", str(i))] = plan.notes
            queries.extend(plan.queries if i == len(self.steps) - 1 else [])
            passages.extend(plan.hyde_passages)
            current = query.model_copy(update={"text": plan.primary, "original_text": original})
        return QueryPlan(original_text=original, queries=_dedupe(queries) or [current.text], strategy=self.name,
                         hyde_passages=_dedupe(passages), fallback=fallback, notes=notes)


__all__ = [
    "QueryPlan",
    "QueryTransformer",
    "IdentityTransformer",
    "QueryRewriter",
    "MultiQueryExpander",
    "QueryDecomposer",
    "HyDEGenerator",
    "ChainedTransformer",
    "Rewrite",
    "Expansions",
    "Decomposition",
    "Hypothetical",
]
