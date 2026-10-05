# path: book/projects/p3-rag-assistant/rag_assistant/ingestion/service.py
"""IngestionService: the producer side of the indexing pipeline.

It never writes an index. It validates, decides, and enqueues:

- `submit_upload` (admin API) stores the original in the blob store and enqueues an upsert;
- `submit_uri` (connectors, full syncs) enqueues an upsert unless the document is tombstoned;
- `delete` makes a document invisible *now* (tombstone + cache purge + generation bump) and
  enqueues the physical purge;
- `sync` reconciles a whole connector: new and changed items are enqueued, items that vanished
  from the source are deleted;
- `start_reindex` / `promote` / `rollback` run a blue/green index rebuild.

Idempotency key of an upsert = doc id + fingerprint + target versions + tombstone sequence.
Resubmitting the same content is free; re-adding a document after deleting it is a new job,
because the tombstone sequence changed.
"""
from __future__ import annotations

import argparse
from typing import Any

from reliability import JobQueue

from ..domain.keys import generation_scopes
from ..domain.models import IngestPayload, now
from ..ingestion.handlers import IngestionDeps
from ..retrieval.index import IndexSet
from .sources import BlobConnector, SourceNotFound


class ReindexIncomplete(RuntimeError):
    pass


class IngestionService:
    def __init__(self, deps: IngestionDeps, queue: JobQueue, *, max_inline_bytes: int = 1_000_000) -> None:
        self.d = deps
        self.queue = queue
        self.max_inline_bytes = max_inline_bytes

    @property
    def index_set(self) -> IndexSet:
        return self.d.index_set

    # ------------------------------------------------------------------ submissions
    def _enqueue_upsert(self, doc_id: str, tenant: str, connector: str, uri: str, content_hash: str,
                        fingerprint: str, *, targets: list[str] | None = None, reason: str = "change") -> dict[str, Any]:
        rec = self.d.registry.get(doc_id)
        tomb = (rec.deleted_seq or 0) if rec else 0
        seq = self.d.registry.next_seq()
        payload = IngestPayload(op="upsert", doc_id=doc_id, connector=connector, uri=uri, content_hash=content_hash,
                                seq=seq, submitted_at=now(), target_versions=targets or [], reason=reason)
        key = f"upsert:{doc_id}:{fingerprint}:{','.join(targets or ['*'])}:{tomb}"
        job = self.queue.enqueue("ingest.upsert", payload.model_dump(mode="json"), idempotency_key=key,
                                 tenant_id=tenant)
        return {"doc_id": doc_id, "job_id": job.id, "state": job.state.value, "deduplicated": job.payload["seq"] != seq}

    def submit_upload(self, content: bytes, filename: str) -> dict[str, Any]:
        """Validate now (so the admin gets a 422, not a dead letter), store, enqueue."""
        if len(content) > self.max_inline_bytes:
            raise ValueError(f"upload is {len(content)} bytes; the limit is {self.max_inline_bytes}")
        doc = self.d.processor.parse(content, filename)
        uri = BlobConnector.uri_for(doc.id, filename, content)
        self.d.blobs.put(uri, content)
        out = self._enqueue_upsert(doc.id, doc.tenant or "shared", "blob", uri, doc.content_hash,
                                   self.d.processor.fingerprint(doc))
        return {**out, "tenant": doc.tenant}

    def peek_tenant(self, content: bytes, filename: str) -> str:
        return self.d.processor.parse(content, filename).tenant or "shared"

    def submit_uri(self, connector: str, uri: str, *, force: bool = False) -> dict[str, Any]:
        raw = self.d.connectors[connector].read(uri)
        doc = self.d.processor.parse(raw, uri)
        rec = self.d.registry.get(doc.id)
        if rec is not None and rec.status != "active" and not force:
            # The tombstone is the suppression list: a sync that still sees the file must not
            # bring a deleted document back. Re-adding is an explicit decision (force or upload).
            return {"doc_id": doc.id, "suppressed": True}
        return self._enqueue_upsert(doc.id, doc.tenant or "shared", connector, uri, doc.content_hash,
                                    self.d.processor.fingerprint(doc))

    def delete(self, doc_id: str) -> dict[str, Any]:
        rec = self.d.registry.get(doc_id)
        if rec is None or rec.status == "deleted":
            return {"doc_id": doc_id, "status": "absent"}
        seq = self.d.registry.next_seq()
        if rec.status == "active":
            rec = rec.model_copy(update={"status": "deleting", "deleted_seq": seq, "deleted_at": now()})
            self.d.registry.put(rec)  # invisible from this point: retrieval filters "deleting"
            self.d.registry.bump(generation_scopes(rec.tenant))
            for cache in self.d.caches:
                cache.invalidate_docs([doc_id])
        payload = IngestPayload(op="purge", doc_id=doc_id, seq=rec.deleted_seq or seq, submitted_at=now(),
                                reason="delete")
        job = self.queue.enqueue("ingest.purge", payload.model_dump(mode="json"),
                                 idempotency_key=f"purge:{doc_id}:{rec.deleted_seq}", tenant_id=rec.tenant)
        return {"doc_id": doc_id, "status": "deleting", "job_id": job.id}

    def sync(self, connector: str = "folder") -> dict[str, Any]:
        """Full reconciliation of one connector against the registry."""
        seen: set[str] = set()
        submitted, suppressed, errors = [], [], []
        for uri in self.d.connectors[connector].list():
            try:
                out = self.submit_uri(connector, uri)
            except (SourceNotFound, ValueError) as exc:
                errors.append({"uri": uri, "error": str(exc)})
                continue
            except Exception as exc:  # InvalidDocument and friends: report, keep syncing
                errors.append({"uri": uri, "error": f"{type(exc).__name__}: {exc}"})
                continue
            seen.add(out["doc_id"])
            (suppressed if out.get("suppressed") else submitted).append(out["doc_id"])
        vanished = [r.doc_id for r in self.d.registry.all("active") if r.connector == connector and r.doc_id not in seen]
        for doc_id in vanished:
            self.delete(doc_id)  # deletion at the source propagates
        return {"submitted": submitted, "suppressed": suppressed, "deleted": vanished, "errors": errors}

    # ------------------------------------------------------------------ blue/green
    def start_reindex(self, new_version: str) -> int:
        self.index_set.begin_build(new_version)
        n = 0
        for rec in self.d.registry.all("active"):
            self._enqueue_upsert(rec.doc_id, rec.tenant, rec.connector, rec.uri, rec.content_hash, rec.fingerprint,
                                 targets=[new_version], reason="reindex")
            n += 1
        return n

    def reindex_progress(self) -> dict[str, int]:
        building = self.index_set.building_version
        active = self.d.registry.all("active")
        done = sum(1 for r in active if building and building in r.index_versions)
        return {"total": len(active), "done": done}

    def promote(self) -> str:
        progress = self.reindex_progress()
        if progress["done"] < progress["total"]:
            raise ReindexIncomplete(f"{progress['done']}/{progress['total']} documents in the new version")
        version = self.index_set.promote()
        self._bump_all()
        return version

    def rollback(self) -> str:
        version = self.index_set.rollback()
        self._bump_all()
        return version

    def _bump_all(self) -> None:
        scopes = {r.tenant for r in self.d.registry.all()} | {"shared"}
        self.d.registry.bump(scopes)
        for cache in self.d.caches:
            cache.sweep(lambda stamp: True)


def main(argv: list[str] | None = None) -> int:
    """`rag-assistant-sync`: enqueue a full sync of the docs folder (run the worker to process it)."""
    from ..config import AssistantSettings
    from ..wiring import build_container

    ap = argparse.ArgumentParser(description="Enqueue a sync of the configured docs folder")
    ap.add_argument("--drain", action="store_true", help="also process the queue in this process")
    args = ap.parse_args(argv)
    container = build_container(AssistantSettings())
    print(container.ingestion.sync("folder"))
    if args.drain:
        print({"processed": container.drain()})
    return 0


__all__ = ["IngestionService", "ReindexIncomplete", "main"]
