# path: book/projects/p3-rag-assistant/rag_assistant/ingestion/handlers.py
"""Worker handlers: `ingest.upsert` and `ingest.purge`.

Both are idempotent, because the queue (reliability.JobQueue) delivers at least once:

- upsert compares the document fingerprint and target versions with the registry and does
  nothing when they match; every index write is "replace this document", so a retry after a
  crash between two writes converges to the same state.
- purge deletes from every index version, every cache, the embedding cache, the contextual
  prefix cache and the blob store, then marks the tombstone purged. Running it twice deletes
  nothing the second time.

Ordering guards (sequence numbers from the registry):

- an upsert whose seq is older than the document's tombstone is skipped (no resurrection);
- an upsert older than the version already indexed is skipped (no regression by a slow retry);
- a purge older than a later re-creation is skipped (an intentional re-upload wins).

The registry write is the commit point: indexes are written first, then the BM25 snapshot is
published, the registry last, so a crash in between leaves the registry describing the old
state and the retry redoes the work. Publishing the snapshot before the generation bump means
an API replica that sees the new generation can always load the new BM25; the reverse order
would let it cache old content under the new stamp.
"""
from __future__ import annotations

import threading
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import Any

from aie_core.observability import NoopTracer, Tracer
from ragkit.pipeline import diff_chunks
from ragkit.retrieval import indexed_text
from reliability import Job, JobContext, PermanentJobError

from ..caching.caches import CacheSet, ForgettableEmbeddings
from ..domain.keys import generation_scopes, partition_for_doc
from ..domain.models import ChangeKind, DocRecord, IngestOutcome, IngestPayload, now
from ..observability.metrics import Metrics
from ..retrieval.index import IndexSet
from .processing import DocumentProcessor
from .registry import DocumentRegistry
from .sources import BlobConnector, SourceConnector, SourceNotFound


@dataclass
class IngestionDeps:
    registry: DocumentRegistry
    index_set: IndexSet
    connectors: dict[str, SourceConnector]
    processor: DocumentProcessor
    embeddings: ForgettableEmbeddings
    caches: list[CacheSet]
    blobs: BlobConnector
    tenancy_mode: str
    metrics: Metrics = field(default_factory=Metrics)
    tracer: Tracer = field(default_factory=NoopTracer)
    context_cache: MutableMapping[str, str] | None = None  # ContextualEnricher cache, if enabled


class IngestHandlers:
    def __init__(self, deps: IngestionDeps) -> None:
        self.d = deps
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _lock(self, doc_id: str) -> threading.Lock:
        # Serializes jobs for one document inside this process. Across processes, route a
        # document's jobs to one partition or take a database row lock (see Chapter 15).
        with self._locks_guard:
            return self._locks.setdefault(doc_id, threading.Lock())

    def handlers(self) -> dict[str, Any]:
        return {"ingest.upsert": self.upsert, "ingest.purge": self.purge}

    # ------------------------------------------------------------------ upsert
    def upsert(self, job: Job, ctx: JobContext | None = None) -> dict[str, Any]:
        p = IngestPayload.model_validate(job.payload)
        with self._lock(p.doc_id), self.d.tracer.span("ingest.upsert", doc_id=p.doc_id, job_id=job.id,
                                                      attempt=job.attempts, reason=p.reason) as span:
            outcome = self._upsert(p, ctx)
            span.set_attribute("change", outcome.change)
            span.set_attribute("chunks.added", outcome.added)
            span.set_attribute("chunks.removed", outcome.removed)
            self.d.metrics.inc("rag_ingest_jobs_total", change=outcome.change)
            if outcome.freshness_lag_s is not None:
                self.d.metrics.observe("rag_freshness_lag_s", outcome.freshness_lag_s)
            return outcome.model_dump()

    def _upsert(self, p: IngestPayload, ctx: JobContext | None) -> IngestOutcome:
        d = self.d
        rec = d.registry.get(p.doc_id)
        if rec is not None and rec.deleted_seq is not None and p.seq < rec.deleted_seq:
            return IngestOutcome(doc_id=p.doc_id, change="skipped_tombstone")
        if rec is not None and rec.status == "active" and p.seq < rec.submitted_seq and p.reason != "reindex":
            return IngestOutcome(doc_id=p.doc_id, change="skipped_stale")
        connector = d.connectors.get(p.connector)
        if connector is None:
            raise PermanentJobError(f"unknown connector {p.connector!r}")
        try:
            raw = connector.read(p.uri)
        except SourceNotFound as exc:  # the source no longer has it; deletion has its own path
            raise PermanentJobError(f"source item {p.connector}:{p.uri} is gone") from exc
        doc = d.processor.parse(raw, p.uri)
        if doc.id != p.doc_id:
            raise PermanentJobError(f"{p.uri} now declares id {doc.id!r}, job was for {p.doc_id!r}")
        fingerprint = d.processor.fingerprint(doc)
        targets = p.target_versions or d.index_set.writable_versions()
        active = rec is not None and rec.status == "active"
        if active and rec.fingerprint == fingerprint and set(targets) <= set(rec.index_versions):  # type: ignore[union-attr]
            return IngestOutcome(doc_id=doc.id, change="unchanged", unchanged=len(rec.chunk_ids),  # type: ignore[union-attr]
                                 versions=rec.index_versions)  # type: ignore[union-attr]

        if not (active and rec.fingerprint == fingerprint):  # type: ignore[union-attr]
            # The content changed since the job was planned (e.g. a reindex job reading a newer
            # source): a partial write would leave the active version stale, so write them all.
            targets = sorted(set(targets) | set(d.index_set.writable_versions()))
        chunks = d.processor.chunk(doc)
        prev_ids = rec.chunk_ids if active else []  # type: ignore[union-attr]
        diff = diff_chunks(prev_ids, chunks)
        prev_ctx = rec.context_keys if active else {}  # type: ignore[union-attr]
        new_ctx = {c.id: str(c.metadata["context_key"]) for c in chunks if c.metadata.get("context_key")}
        reembedded = sum(1 for cid in diff.unchanged if prev_ctx.get(cid) != new_ctx.get(cid))
        change = _classify(rec if active else None, doc.content_hash, doc.tenant, doc.acl_groups, fingerprint)

        partition = partition_for_doc(doc.tenant, d.tenancy_mode)
        old_partition = partition_for_doc(rec.tenant, d.tenancy_mode) if active else partition  # type: ignore[union-attr]
        if ctx is not None:
            ctx.checkpoint()  # do not start index writes after losing the lease or during shutdown
        for version in targets:
            if old_partition != partition:  # tenant moved: remove from the partition it left
                d.index_set.get(version, old_partition).delete(doc.id)
            d.index_set.get(version, partition).write(doc.id, chunks)
        d.embeddings.track(doc.id, [indexed_text(c) for c in chunks])
        if d.context_cache is not None:
            for cid in diff.removed:
                if cid in prev_ctx:
                    d.context_cache.pop(prev_ctx[cid], None)

        same_content = active and rec.fingerprint == fingerprint  # type: ignore[union-attr]
        versions = sorted(set(targets) | (set(rec.index_versions) if same_content else set()))  # type: ignore[union-attr]
        indexed_at = now()
        meta = doc.metadata
        new_rec = DocRecord(
            doc_id=doc.id, tenant=doc.tenant or "shared", acl_groups=list(doc.acl_groups), version=doc.version,
            title=doc.title, content_hash=doc.content_hash, fingerprint=fingerprint, connector=p.connector,
            uri=p.uri, chunk_ids=[c.id for c in chunks], context_keys=new_ctx, index_versions=versions,
            status="active", submitted_seq=max(p.seq, rec.submitted_seq if active else 0),  # type: ignore[union-attr]
            deleted_seq=rec.deleted_seq if rec else None, submitted_at=p.submitted_at, indexed_at=indexed_at,
            authority=int(meta.get("authority", 1)), effective_date=meta.get("effective_date"),
            supersedes=list(meta.get("supersedes") or []),
        )
        d.index_set.save_snapshots()  # before the commit: replicas never see a generation without its BM25
        d.registry.put(new_rec)  # commit point
        scopes = generation_scopes(doc.tenant) + (generation_scopes(rec.tenant) if rec else [])
        d.registry.bump(scopes)
        for cache in d.caches:
            cache.invalidate_docs([doc.id])
        return IngestOutcome(doc_id=doc.id, change="reindex" if p.reason == "reindex" and same_content else change,
                             added=len(diff.added), unchanged=len(diff.unchanged), removed=len(diff.removed),
                             reembedded=reembedded, versions=versions, freshness_lag_s=new_rec.freshness_lag_s)

    # ------------------------------------------------------------------ purge
    def purge(self, job: Job, ctx: JobContext | None = None) -> dict[str, Any]:
        p = IngestPayload.model_validate(job.payload)
        with self._lock(p.doc_id), self.d.tracer.span("ingest.purge", doc_id=p.doc_id, job_id=job.id) as span:
            result = self._purge(p)
            span.set_attribute("purged", result.get("purged", False))
            self.d.metrics.inc("rag_ingest_jobs_total", change="purged" if result.get("purged") else "skipped")
            return result

    def _purge(self, p: IngestPayload) -> dict[str, Any]:
        d = self.d
        rec = d.registry.get(p.doc_id)
        if rec is not None and rec.status == "active" and rec.submitted_seq > p.seq:
            return {"doc_id": p.doc_id, "purged": False, "reason": "re-created after this delete"}
        removed: dict[str, tuple[int, int]] = {}
        for ix in d.index_set.existing():
            lexical, dense = ix.delete(p.doc_id)
            if lexical or dense:
                removed[f"{ix.version}/{ix.partition}"] = (lexical, dense)
        vectors_forgotten = d.embeddings.forget_doc(p.doc_id)
        blobs = d.blobs.delete_doc(p.doc_id)
        contexts = 0
        if d.context_cache is not None and rec is not None:
            for key in rec.context_keys.values():
                contexts += d.context_cache.pop(key, None) is not None
        cache_entries = sum(c.invalidate_docs([p.doc_id]) for c in d.caches)
        d.index_set.save_snapshots()
        if rec is not None:
            rec = rec.model_copy(update={"status": "deleted", "purged_at": now(), "chunk_ids": [],
                                         "context_keys": {}, "index_versions": [],
                                         "deleted_seq": max(rec.deleted_seq or 0, p.seq)})
            d.registry.put(rec)
            d.registry.bump(generation_scopes(rec.tenant))
        return {"doc_id": p.doc_id, "purged": True, "indexes": removed, "vectors_forgotten": vectors_forgotten,
                "blobs": blobs, "context_prefixes": contexts, "cache_entries": cache_entries}


def _classify(rec: DocRecord | None, content_hash: str, tenant: str | None, acl: list[str],
              fingerprint: str) -> ChangeKind:
    if rec is None:
        return "new"
    if rec.content_hash != content_hash:
        return "content"
    if rec.tenant != (tenant or "shared") or sorted(rec.acl_groups) != sorted(acl):
        return "acl_only"
    return "metadata" if rec.fingerprint != fingerprint else "unchanged"


__all__ = ["IngestHandlers", "IngestionDeps"]
