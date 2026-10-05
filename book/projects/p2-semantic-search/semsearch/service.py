# path: book/projects/p2-semantic-search/semsearch/service.py
"""The search use case: principal + query -> authorized, filtered top-k.

Authorization filters are derived from the principal here, in one place, and are always
applied. The caller can narrow results (tags, k) but cannot widen them.
"""
from __future__ import annotations

import time

from aie_core.embeddings import EmbeddingClient
from aie_core.observability import NoopTracer, Tracer
from pydantic import BaseModel, Field

from .adapters.base import VectorStore
from .domain.filters import visible_tenants
from .domain.models import SearchFilter, SearchHit


class Principal(BaseModel):
    user_id: str
    tenant: str
    groups: tuple[str, ...] = Field(min_length=1)


class SearchResult(BaseModel):
    namespace: str
    query: str
    k: int
    took_ms: float
    hits: list[SearchHit]


class SearchService:
    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingClient,
        namespace: str,
        *,
        default_k: int = 8,
        max_k: int = 50,
        tracer: Tracer | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.namespace = namespace
        self.default_k = default_k
        self.max_k = max_k
        self.tracer = tracer or NoopTracer()

    def authorization_filter(self, principal: Principal, tags: list[str] | None = None) -> SearchFilter:
        return SearchFilter(
            tenants=visible_tenants(principal.tenant),
            acl_groups=tuple(principal.groups),
            tags_any=tuple(tags) if tags else None,
        )

    def search(self, query: str, principal: Principal, *, k: int | None = None, tags: list[str] | None = None) -> SearchResult:
        k = min(k or self.default_k, self.max_k)
        flt = self.authorization_filter(principal, tags)
        start = time.perf_counter()
        with self.tracer.span(
            "search",
            namespace=self.namespace,
            k=k,
            tenant=principal.tenant,
            groups=list(principal.groups),
            tags=tags or [],
            embedding_model=self.embedder.model,
        ) as span:
            t0 = time.perf_counter()
            qvec = self.embedder.embed_query(query)
            span.set_attribute("embed_ms", round((time.perf_counter() - t0) * 1000, 2))
            t1 = time.perf_counter()
            hits = self.store.search(self.namespace, qvec, k, flt)
            span.set_attribute("store_ms", round((time.perf_counter() - t1) * 1000, 2))
            span.set_attribute("returned", len(hits))
            # Fewer hits than k under a filter is the signature of an over-selective filter
            # or an ANN index that ran out of candidates; alert on its rate.
            span.set_attribute("underfilled", len(hits) < k)
            span.set_attribute("top_score", hits[0].score if hits else None)
            span.set_attribute("doc_ids", [h.doc_id for h in hits])
        return SearchResult(
            namespace=self.namespace, query=query, k=k, took_ms=round((time.perf_counter() - start) * 1000, 2), hits=hits
        )


__all__ = ["Principal", "SearchResult", "SearchService"]
