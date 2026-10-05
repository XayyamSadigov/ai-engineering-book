# path: book/projects/ragkit/ragkit/retrieval/dense.py
"""Dense retrieval over aie_core embeddings, stored in Project 2's VectorStore.

This module does not implement a vector store. It adapts ragkit chunks to semsearch's
`VectorRecord`, so the same retriever runs on `NumpyVectorStore` in tests and on
`PgVectorStore` in Project 3. What it adds is retrieval semantics:

- The namespace is `index:embedding-model:index-version` (semsearch.make_namespace), so vectors
  from two embedding models or two chunking runs never share a search space.
- Every metadata filter becomes a pre-filter the store applies before scoring: tenant and ACL
  groups from the principal, document-level filters translated into an allowed doc-id set.
- Results are re-checked with `visible()` on the way out. A store bug that returns a forbidden
  chunk is dropped and counted in the trace, never passed downstream.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

from aie_core.embeddings import EmbeddingClient

from ..documents import Chunk
from .common import Stopwatch, extract_identifiers, indexed_text, matches_filters, validate_filters
from .types import Principal, RetrievalQuery, RetrievalResult, ScoredChunk, visible

SHARED_TENANT = "shared"


def _semsearch() -> Any:
    try:
        import semsearch.adapters as adapters
        import semsearch.domain.filters as filters
        import semsearch.domain.models as models
    except ImportError as exc:  # pragma: no cover - environment problem, not logic
        raise ImportError(
            "DenseRetriever stores vectors in Project 2's semsearch package. "
            "Install it with: pip install -e ../p2-semantic-search"
        ) from exc
    return adapters, models, filters


class DenseRetriever:
    """Embed `indexed_text(chunk)` with an EmbeddingClient; search a VectorStore namespace."""

    name = "dense"

    def __init__(
        self,
        embeddings: EmbeddingClient,
        store: Any | None = None,
        *,
        index_name: str = "northwind",
        index_version: str = "v1",
        text_of: Callable[[Chunk], str] = indexed_text,
        batch_size: int = 64,
        min_score: float | None = None,
    ) -> None:
        adapters, models, _ = _semsearch()
        self.embeddings = embeddings
        self.store = store if store is not None else adapters.NumpyVectorStore(embeddings.dimensions)
        if self.store.dimensions != embeddings.dimensions:
            raise ValueError(
                f"store has {self.store.dimensions} dimensions, embedding model {embeddings.model!r} "
                f"produces {embeddings.dimensions}"
            )
        self.namespace = models.make_namespace(index_name, embeddings.model, index_version)
        self.text_of = text_of
        self.batch_size = batch_size
        self.min_score = min_score  # None keeps every hit; similarity is not calibrated (Chapter 9)
        self._catalog: dict[str, dict[str, Any]] = {}  # doc_id -> document-level metadata for filters

    # ------------------------------------------------------------------ writes
    def index(self, chunks: Iterable[Chunk]) -> int:
        """Embed and write chunks, one atomic replace per document version."""
        _, models, _ = _semsearch()
        by_doc: dict[str, list[Chunk]] = {}
        for c in chunks:
            by_doc.setdefault(c.doc_id, []).append(c)
        written = 0
        for doc_id, doc_chunks in by_doc.items():
            versions = {c.version for c in doc_chunks}
            if len(versions) != 1:
                raise ValueError(f"chunks of {doc_id} span several versions {sorted(versions)}")
            vectors: list[list[float]] = []
            for start in range(0, len(doc_chunks), self.batch_size):
                batch = doc_chunks[start : start + self.batch_size]
                vectors.extend(self.embeddings.embed([self.text_of(c) for c in batch]))
            records = [
                models.VectorRecord(
                    id=c.id,
                    namespace=self.namespace,
                    tenant=c.tenant or SHARED_TENANT,
                    doc_id=c.doc_id,
                    doc_version=c.version,
                    ordinal=c.position,
                    text=c.text,
                    vector=v,
                    acl_groups=tuple(c.acl_groups),
                    tags=tuple(c.metadata.get("tags") or ()),
                    metadata=_record_metadata(c),
                    embedding_model=self.embeddings.model,
                )
                for c, v in zip(doc_chunks, vectors)
            ]
            written += self.store.replace_document(self.namespace, doc_id, versions.pop(), records)
            self._catalog[doc_id] = dict(doc_chunks[0].metadata)
        return written

    def delete_document(self, doc_id: str) -> int:
        self._catalog.pop(doc_id, None)
        return self.store.delete_document(self.namespace, doc_id)

    # ------------------------------------------------------------------ reads
    def _store_filter(self, principal: Principal, filters: dict[str, Any]) -> Any:
        _, models, flt = _semsearch()
        validate_filters(filters)
        doc_ids: tuple[str, ...] | None = None
        if filters:
            if not self._catalog:  # e.g. a fresh process over a persistent store: fail loudly, not empty
                raise RuntimeError("DenseRetriever has no document catalog for filters; call index() or load() first")
            # every supported filter is document-level, so it reduces to an allowed doc-id set
            doc_ids = tuple(
                sorted(
                    d for d, meta in self._catalog.items()
                    if matches_filters(_probe(d, meta), filters)
                )
            )
        return models.SearchFilter(
            tenants=flt.visible_tenants(principal.tenant, SHARED_TENANT),
            acl_groups=tuple(principal.groups),
            doc_ids=doc_ids,
        )

    def search(self, text: str, principal: Principal, k: int = 10, filters: dict | None = None) -> list[ScoredChunk]:
        return self._search(text, principal, k, filters)[0]

    def _search(self, text: str, principal: Principal, k: int, filters: dict | None) -> tuple[list[ScoredChunk], int]:
        """Returns (hits, dropped) where dropped counts results that failed the final ACL check."""
        if k <= 0:
            return [], 0
        flt = self._store_filter(principal, filters or {})
        if flt.doc_ids is not None and not flt.doc_ids:
            return [], 0
        vector = self.embeddings.embed_query(text)
        raw = self.store.search(self.namespace, vector, k, flt)
        hits: list[ScoredChunk] = []
        dropped = 0
        for h in raw:
            chunk = Chunk.model_validate(h.metadata["chunk"])
            if not visible(chunk, principal):  # defense in depth: never trust one layer with ACLs
                dropped += 1
                continue
            if self.min_score is not None and h.score < self.min_score:
                continue
            hits.append(
                ScoredChunk(chunk=chunk, score=h.score, stage=self.name, rank=len(hits) + 1,
                            signals={"dense": round(h.score, 6)})
            )
        return hits, dropped

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        with Stopwatch() as sw:
            hits, dropped = self._search(query.text, query.principal, query.k, query.filters)
        trace = {
            "stage": self.name,
            "k": query.k,
            "namespace": self.namespace,
            "candidate_ids": [h.chunk.id for h in hits],
            "acl_dropped": dropped,
            "latency_ms": sw.ms,
        }
        return RetrievalResult(query=query, hits=hits, trace=trace)

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> Path:
        """Persist vectors through the store (NumpyVectorStore.save) plus a small manifest."""
        directory = Path(directory)
        if not hasattr(self.store, "save"):
            raise TypeError(f"{type(self.store).__name__} persists itself; nothing to save here")
        self.store.save(directory / "vectors")  # the store owns this folder; the manifest sits beside it
        manifest = {"namespace": self.namespace, "model": self.embeddings.model,
                    "dimensions": self.embeddings.dimensions, "catalog": self._catalog}
        (directory / "dense_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory: str | Path, embeddings: EmbeddingClient, **kwargs: Any) -> "DenseRetriever":
        adapters, _, _ = _semsearch()
        directory = Path(directory)
        manifest = json.loads((directory / "dense_manifest.json").read_text(encoding="utf-8"))
        if manifest["model"] != embeddings.model or manifest["dimensions"] != embeddings.dimensions:
            raise ValueError(
                f"index was embedded with {manifest['model']!r}/{manifest['dimensions']}d, "
                f"query client is {embeddings.model!r}/{embeddings.dimensions}d: re-embed instead"
            )
        store = adapters.NumpyVectorStore.load(directory / "vectors", dimensions=embeddings.dimensions)
        retriever = cls(embeddings, store, **kwargs)
        if retriever.namespace != manifest["namespace"]:
            raise ValueError(f"namespace mismatch: saved {manifest['namespace']!r}, built {retriever.namespace!r}")
        retriever._catalog = manifest["catalog"]
        return retriever


def _record_metadata(c: Chunk) -> dict[str, Any]:
    """The full chunk (to rebuild it on read) plus the flat fields a SQL lexical index uses
    (see sql/lexical_tsvector.sql): title, breadcrumb, context prefix, identifiers."""
    return {
        "chunk": c.model_dump(mode="json"),
        "title": str(c.metadata.get("title") or ""),
        "breadcrumb": c.context_header(),
        "context_prefix": str(c.metadata.get("context_prefix") or ""),
        "identifiers": " ".join(extract_identifiers(c.text)),
    }


def _probe(doc_id: str, meta: dict[str, Any]) -> Chunk:
    """A metadata-only stand-in chunk so document-level filters reuse `matches_filters`."""
    return Chunk(id=f"{doc_id}:probe", doc_id=doc_id, version="", text="", content_hash="", tenant=None,
                 acl_groups=[], metadata=meta, position=0, char_start=0, char_end=0, token_count=0, chunker="probe")


__all__ = ["DenseRetriever"]
