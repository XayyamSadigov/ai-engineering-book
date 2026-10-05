# path: book/projects/p2-semantic-search/semsearch/ingest/pipeline.py
"""Ingest a corpus into a vector store: load, chunk, embed, replace per document, prune.

Incremental by construction: a document whose index_version (declared version + content
hash) matches what the store already holds is skipped, so re-running ingestion costs only
the changed documents' embeddings. Documents that disappeared from the source are deleted
when prune=True; leaving them is how stale or revoked content keeps being retrieved.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

from aie_core.embeddings import EmbeddingClient
from aie_core.observability import NoopTracer, Tracer

from ..adapters.base import VectorStore
from ..domain.models import VectorRecord, chunk_id
from .loader import SourceDocument, chunk_paragraphs, embedding_text


@dataclass
class IngestReport:
    namespace: str
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    chunks_written: int = 0
    texts_embedded: int = 0
    seconds: float = 0.0

    def summary(self) -> str:
        return (
            f"namespace={self.namespace} added={len(self.added)} updated={len(self.updated)} "
            f"unchanged={len(self.unchanged)} deleted={len(self.deleted)} "
            f"chunks={self.chunks_written} embedded={self.texts_embedded} in {self.seconds:.2f}s"
        )


def build_records(
    doc: SourceDocument, chunks_vectors: Sequence[tuple[str, list[float]]], namespace: str, embedding_model: str, section_of: Sequence[str]
) -> list[VectorRecord]:
    version = doc.index_version
    return [
        VectorRecord(
            id=chunk_id(doc.doc_id, version, i),
            namespace=namespace,
            tenant=doc.tenant,
            doc_id=doc.doc_id,
            doc_version=version,
            ordinal=i,
            text=text,
            vector=vector,
            acl_groups=doc.acl_groups,
            tags=doc.tags,
            metadata={"title": doc.title, "section": section_of[i], "declared_version": doc.version, **doc.metadata},
            embedding_model=embedding_model,
        )
        for i, (text, vector) in enumerate(chunks_vectors)
    ]


def ingest(
    docs: Sequence[SourceDocument],
    store: VectorStore,
    embedder: EmbeddingClient,
    namespace: str,
    *,
    max_chars: int = 1200,
    prune: bool = False,
    batch_size: int = 64,
    tracer: Tracer | None = None,
) -> IngestReport:
    tracer = tracer or NoopTracer()
    report = IngestReport(namespace=namespace)
    start = time.perf_counter()
    with tracer.span("ingest", namespace=namespace, documents=len(docs)) as span:
        indexed = store.doc_versions(namespace)
        for doc in docs:
            current = indexed.get(doc.doc_id)
            if current is not None and current.doc_version == doc.index_version:
                report.unchanged.append(doc.doc_id)
                continue
            chunks = chunk_paragraphs(doc.body, max_chars=max_chars)
            texts = [embedding_text(doc, c) for c in chunks]
            vectors: list[list[float]] = []
            for i in range(0, len(texts), batch_size):
                vectors.extend(embedder.embed(texts[i : i + batch_size]))
            report.texts_embedded += len(texts)
            records = build_records(doc, list(zip(texts, vectors)), namespace, embedder.model, [c.section for c in chunks])
            report.chunks_written += store.replace_document(namespace, doc.doc_id, doc.index_version, records)
            (report.updated if current is not None else report.added).append(doc.doc_id)
        if prune:
            source_ids = {d.doc_id for d in docs}
            for doc_id in sorted(set(indexed) - source_ids):
                store.delete_document(namespace, doc_id)
                report.deleted.append(doc_id)
        report.seconds = time.perf_counter() - start
        span.set_attribute("added", len(report.added))
        span.set_attribute("updated", len(report.updated))
        span.set_attribute("deleted", len(report.deleted))
        span.set_attribute("texts_embedded", report.texts_embedded)
    save = getattr(store, "save", None)
    if callable(save) and getattr(store, "persist_dir", None) is not None:
        save()
    return report


__all__ = ["IngestReport", "ingest", "build_records"]
