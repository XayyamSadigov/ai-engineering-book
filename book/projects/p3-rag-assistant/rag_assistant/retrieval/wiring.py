# path: book/projects/p3-rag-assistant/rag_assistant/retrieval/wiring.py
"""Assemble ragkit's RetrievalPipeline for one request, with the production adapters around it.

Adapters (each wraps a ragkit Retriever/Reranker and keeps its contract):

- `BreakerRetriever` / `BreakerReranker`: a reliability.CircuitBreaker per dependency. When the
  vector store is down, the breaker opens and dense retrieval fails in microseconds instead of
  after a timeout; RetrievalPipeline already treats a failing retriever as "skip and record
  degraded", so the request continues lexical-only.
- `TombstoneFilter`: hides documents that are logically deleted but not yet purged.
- `AuthorityReranker`: runs after the relevance reranker and applies authority metadata
  (see domain/authority.py): a small additive boost per authority level, and a hard rule that
  a chunk marked `superseded_by` a document that is also in the candidates scores below it.
"""
from __future__ import annotations

from concurrent.futures import Executor
from typing import Callable, Mapping

from aie_core.observability import Tracer
from ragkit.retrieval import (
    LexicalOverlapReranker,
    Principal,
    Reranker,
    RetrievalPipeline,
    RetrievalQuery,
    RetrievalResult,
    Retriever,
    ScoredChunk,
)
from reliability import CircuitBreaker, CircuitBreakerRegistry

from ..config import AssistantSettings
from ..domain.keys import partitions_for_principal
from .index import IndexSet


class BreakerRetriever:
    def __init__(self, inner: Retriever, breaker: CircuitBreaker) -> None:
        self.inner = inner
        self.breaker = breaker

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        return self.breaker.call(self.inner.retrieve, query)


class BreakerReranker:
    def __init__(self, inner: Reranker, breaker: CircuitBreaker) -> None:
        self.inner = inner
        self.breaker = breaker

    def rerank(self, query: RetrievalQuery, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        return self.breaker.call(self.inner.rerank, query, candidates, k)


class TombstoneFilter:
    """Drop hits from documents in `hidden` (status "deleting"). Asks the inner retriever for a
    few extra candidates so the caller still gets k results while purges are in flight."""

    def __init__(self, inner: Retriever, hidden: frozenset[str]) -> None:
        self.inner = inner
        self.hidden = hidden

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        if not self.hidden:
            return self.inner.retrieve(query)
        padded = query.model_copy(update={"k": query.k + 4 * len(self.hidden)})
        result = self.inner.retrieve(padded)
        kept = [h for h in result.hits if h.chunk.doc_id not in self.hidden][: query.k]
        kept = [h.model_copy(update={"rank": i}) for i, h in enumerate(kept, start=1)]
        trace = {**result.trace, "tombstoned_dropped": len(result.hits) - len(kept)}
        return RetrievalResult(query=query, hits=kept, trace=trace)


class AuthorityReranker:
    name = "authority"

    def __init__(self, inner: Reranker | None = None, *, weight: float = 0.05) -> None:
        self.inner = inner
        self.weight = weight

    def rerank(self, query: RetrievalQuery, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        failed = False
        try:
            ranked = self.inner.rerank(query, candidates, len(candidates)) if self.inner else list(candidates)
        except Exception:  # reranker down or breaker open: keep fused order, still apply authority
            ranked, failed = list(candidates), True
        # The boost is relative to the top score, so it means the same thing for RRF scores
        # (around 0.03), lexical-overlap scores (0 to 1.5) and cross-encoder logits.
        scale = max((abs(h.score) for h in ranked), default=1.0) or 1.0
        adjusted: list[ScoredChunk] = []
        for h in ranked:
            level = int(h.chunk.metadata.get("authority", 1) or 0)
            signals = {**h.signals, "authority": float(level), "pre_authority": round(h.score, 6)}
            if failed:
                signals["rerank_failed"] = 1.0  # RetrievalPipeline records "rerank:fallback" as degraded
            adjusted.append(h.model_copy(update={"score": h.score + self.weight * level * scale, "signals": signals}))
        best_by_doc: dict[str, float] = {}
        for h in adjusted:
            best_by_doc[h.chunk.doc_id] = max(best_by_doc.get(h.chunk.doc_id, float("-inf")), h.score)
        final: list[ScoredChunk] = []
        for h in adjusted:
            newer = [d for d in h.chunk.metadata.get("superseded_by") or [] if d in best_by_doc]
            if newer:
                ceiling = min(best_by_doc[d] for d in newer) - 1e-6
                if h.score > ceiling:
                    h = h.model_copy(update={"score": ceiling, "signals": {**h.signals, "superseded": 1.0}})
            final.append(h)
        final.sort(key=lambda x: (-x.score, x.rank))
        return [h.model_copy(update={"rank": i, "stage": "rerank"}) for i, h in enumerate(final[:k], start=1)]


def base_reranker(settings: AssistantSettings) -> Reranker | None:
    if settings.reranker == "none":
        return None
    if settings.reranker == "cross_encoder":
        from ragkit.retrieval import CrossEncoderReranker

        return CrossEncoderReranker()  # falls back to lexical overlap if the model is unavailable
    return LexicalOverlapReranker()


def build_pipeline(
    index_set: IndexSet,
    principal: Principal,
    settings: AssistantSettings,
    *,
    breakers: CircuitBreakerRegistry,
    hidden: frozenset[str] = frozenset(),
    degrade_level: int = 0,
    tracer: Tracer | None = None,
    reranker_factory: Callable[[AssistantSettings], Reranker | None] = base_reranker,
    version: str | None = None,
    executor: Executor | None = None,
) -> RetrievalPipeline:
    """Retrievers for every partition the principal may read; lexical and dense in parallel.

    Degrade level 1 (admission control under load) skips the reranker and halves candidate
    counts; level 2 is handled by the caller (no generation).
    """
    version = version or index_set.active_version
    retrievers: dict[str, Retriever] = {}
    for partition in partitions_for_principal(principal, settings.tenancy_mode):
        ix = index_set.get(version, partition)
        suffix = "" if partition == "all" else f"@{partition}"
        retrievers[f"bm25{suffix}"] = TombstoneFilter(BreakerRetriever(ix.bm25, breakers.get("bm25")), hidden)
        retrievers[f"dense{suffix}"] = TombstoneFilter(BreakerRetriever(ix.dense, breakers.get("dense")), hidden)
    inner = reranker_factory(settings) if degrade_level == 0 else None
    wrapped_inner = BreakerReranker(inner, breakers.get("reranker")) if inner is not None else None
    reranker = AuthorityReranker(wrapped_inner, weight=settings.authority_weight)
    shrink = 2 if degrade_level >= 1 else 1
    return RetrievalPipeline(
        retrievers,
        reranker=reranker,
        candidate_k=max(5, settings.candidate_k // shrink),
        rerank_k=max(5, settings.rerank_k // shrink),
        final_k=settings.final_k,
        parallel=settings.parallel_retrieval,
        retrieve_timeout_s=settings.retrieve_timeout_s,
        rerank_timeout_s=settings.rerank_timeout_s,
        executor=executor,  # one pool per service, not per request
        tracer=tracer,
    )


def dense_names(retrievers: Mapping[str, object]) -> list[str]:
    return [n for n in retrievers if n.startswith("dense")]


__all__ = ["AuthorityReranker", "BreakerReranker", "BreakerRetriever", "TombstoneFilter", "base_reranker",
           "build_pipeline"]
