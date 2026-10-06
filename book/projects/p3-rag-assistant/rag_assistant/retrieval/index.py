# path: book/projects/p3-rag-assistant/rag_assistant/retrieval/index.py
"""Index versions and partitions: which BM25 index and which vector namespace hold what.

    IndexSet
      version "v1" (active)      partition "all"      -> SearchIndex(BM25Index, DenseRetriever ns northwind:model:v1)
      version "v2" (building)    partition "all"      -> SearchIndex(...                          ns northwind:model:v2)

In namespace tenancy mode each version has one partition per tenant plus "shared", and a
query opens only the partitions its principal may read. In shared mode there is one
partition and isolation comes from ACL filters applied before scoring (ragkit BM25Index and
DenseRetriever both pre-filter by tenant and groups).

Blue/green: a reindex builds version N+1 next to N while N keeps serving. New writes go to
both (dual write). Promotion is one registry write; the old version stays for rollback until
it is dropped. Nothing here re-implements a store: BM25Index and DenseRetriever come from
ragkit, the vector store from semsearch (NumpyVectorStore or PgVectorStore).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Callable

from ragkit.documents import Chunk
from ragkit.retrieval import BM25Index, DenseRetriever, Principal

from ..ingestion.registry import DocumentRegistry

ACTIVE, BUILDING, PREVIOUS, SNAPSHOT_GEN = "active_version", "building_version", "previous_version", "snapshot_gen"


class SearchIndex:
    """One (version, partition): a lexical index and a dense namespace that always change together."""

    def __init__(self, version: str, partition: str, bm25: BM25Index, dense: DenseRetriever) -> None:
        self.version = version
        self.partition = partition
        self.bm25 = bm25
        self.dense = dense
        self.docs: set[str] = set()

    def write(self, doc_id: str, chunks: list[Chunk]) -> int:
        """Replace everything this index holds for doc_id. Idempotent: same input, same state."""
        self.bm25.replace_document(doc_id, chunks)
        if chunks:
            self.dense.index(chunks)
            self.docs.add(doc_id)
        else:
            self.dense.delete_document(doc_id)
            self.docs.discard(doc_id)
        return len(chunks)

    def delete(self, doc_id: str) -> tuple[int, int]:
        lexical = self.bm25.delete_document(doc_id)
        dense = self.dense.delete_document(doc_id)
        self.docs.discard(doc_id)
        return lexical, dense

    def dense_doc_ids(self) -> set[str]:
        return set(self.dense.store.doc_versions(self.dense.namespace))

    def holds(self, doc_id: str) -> bool:
        """True if any store of this index still has data for the document."""
        return doc_id in _bm25_doc_ids(self.bm25) or doc_id in self.dense_doc_ids()

    def get_chunks(self, chunk_ids: list[str], principal: Principal) -> list[Chunk]:
        """Chunks by id, ACL-checked for the principal (the lexical index holds every chunk it serves)."""
        return self.bm25.get_chunks(chunk_ids, principal)

    @property
    def chunk_count(self) -> int:
        return len(self.bm25)


def _bm25_doc_ids(index: BM25Index) -> set[str]:
    """Doc ids present in a BM25 index (reconciliation and audits)."""
    return index.doc_ids()


class IndexSet:
    def __init__(self, registry: DocumentRegistry, *, make_dense: Callable[[str, str], DenseRetriever],
                 default_version: str, tenancy_mode: str, snapshot_dir: Path | None = None) -> None:
        self.registry = registry
        self.make_dense = make_dense
        self.tenancy_mode = tenancy_mode
        self.snapshot_dir = snapshot_dir
        self._indexes: dict[tuple[str, str], SearchIndex] = {}
        self._lock = threading.RLock()
        self._loaded_snapshot: str | None = None
        if registry.get_state(ACTIVE) is None:
            registry.set_state(ACTIVE, default_version)

    # ------------------------------------------------------------------ versions
    @property
    def active_version(self) -> str:
        return self.registry.get_state(ACTIVE) or "v1"

    @property
    def building_version(self) -> str | None:
        return self.registry.get_state(BUILDING)

    @property
    def previous_version(self) -> str | None:
        return self.registry.get_state(PREVIOUS)

    def writable_versions(self) -> list[str]:
        versions = [self.active_version]
        if self.building_version and self.building_version not in versions:
            versions.append(self.building_version)
        return versions

    def get(self, version: str, partition: str) -> SearchIndex:
        with self._lock:
            key = (version, partition)
            if key not in self._indexes:
                self._indexes[key] = SearchIndex(version, partition, BM25Index(), self.make_dense(version, partition))
            return self._indexes[key]

    def existing(self, version: str | None = None) -> list[SearchIndex]:
        with self._lock:
            return [ix for (v, _), ix in self._indexes.items() if version is None or v == version]

    def get_chunks(self, chunk_ids: list[str], principal: Principal, *, version: str | None = None) -> list[Chunk]:
        """Look chunks up by id in one version (default: active), in the order asked.

        Same guarantees as retrieval: chunks the principal may not read, and chunks of documents
        that are not `active` in the registry (deleting, deleted), are omitted. Use it to resolve
        citations or re-display stored answers; it never widens what a user can see.
        """
        wanted = list(chunk_ids)
        found: dict[str, Chunk] = {}
        for ix in self.existing(version or self.active_version):
            for c in ix.get_chunks(wanted, principal):
                found.setdefault(c.id, c)
        live: dict[str, bool] = {}
        for c in found.values():
            if c.doc_id not in live:
                rec = self.registry.get(c.doc_id)
                live[c.doc_id] = rec is not None and rec.status == "active"
        return [found[cid] for cid in wanted if cid in found and live[found[cid].doc_id]]

    def begin_build(self, version: str) -> None:
        if version == self.active_version:
            raise ValueError(f"{version} is already active")
        self.registry.set_state(BUILDING, version)

    def promote(self) -> str:
        building = self.building_version
        if not building:
            raise ValueError("no index version is being built")
        old = self.active_version
        self.registry.set_state(PREVIOUS, old)
        self.registry.set_state(ACTIVE, building)
        self.registry.set_state(BUILDING, None)
        return building

    def rollback(self) -> str:
        previous = self.previous_version
        if not previous:
            raise ValueError("no previous version to roll back to")
        self.registry.set_state(PREVIOUS, self.active_version)
        self.registry.set_state(ACTIVE, previous)
        return previous

    def drop_version(self, version: str) -> int:
        if version in self.writable_versions():
            raise ValueError(f"refusing to drop writable version {version}")
        dropped = 0
        with self._lock:
            for key in [k for k in self._indexes if k[0] == version]:
                ix = self._indexes.pop(key)
                for doc_id in list(ix.dense_doc_ids() | ix.docs):
                    ix.delete(doc_id)
                    dropped += 1
        if self.previous_version == version:
            self.registry.set_state(PREVIOUS, None)
        return dropped

    # ------------------------------------------------------------------ snapshots (worker -> API)
    def save_snapshots(self) -> None:
        """Persist BM25 indexes so API replicas can load what the worker built. Vectors live in
        the vector store already (pgvector in Compose), so only the lexical side is written."""
        if self.snapshot_dir is None:
            return
        with self._lock:
            for (version, partition), ix in self._indexes.items():
                folder = self.snapshot_dir / version
                folder.mkdir(parents=True, exist_ok=True)
                ix.bm25.save(folder / f"{partition}.bm25.json")
                tmp = folder / f"{partition}.docs.json.tmp"
                tmp.write_text(json.dumps(sorted(ix.docs)), encoding="utf-8")
                tmp.replace(folder / f"{partition}.docs.json")
        gen = str(int(self.registry.get_state(SNAPSHOT_GEN) or 0) + 1)
        self.registry.set_state(SNAPSHOT_GEN, gen)
        self._loaded_snapshot = gen

    def refresh(self) -> bool:
        """Reload BM25 snapshots if the worker published newer ones. Returns True if reloaded."""
        if self.snapshot_dir is None:
            return False
        gen = self.registry.get_state(SNAPSHOT_GEN)
        if gen is None or gen == self._loaded_snapshot:
            return False
        with self._lock:
            for folder in sorted(p for p in self.snapshot_dir.iterdir() if p.is_dir()):
                for f in folder.glob("*.bm25.json"):
                    partition = f.name[: -len(".bm25.json")]
                    ix = self.get(folder.name, partition)
                    ix.bm25 = BM25Index.load(f)
                    manifest = folder / f"{partition}.docs.json"
                    ix.docs = set(json.loads(manifest.read_text(encoding="utf-8"))) if manifest.exists() else set()
            self._loaded_snapshot = gen
        self.reconcile()
        return True

    # ------------------------------------------------------------------ reconciliation
    def reconcile(self) -> dict[str, list[str]]:
        """Remove from every index any document the registry lists as deleting or deleted.

        This is the backstop against resurrection: an old snapshot, a restored backup, or a
        crashed purge can bring deleted data back into an index; reconciling against the
        system of record takes it out again before it can be served.
        """
        # Only documents the registry knows as not active: one it does not know yet may be in
        # flight (the worker publishes its snapshot before the registry commit).
        inactive = {r.doc_id for r in self.registry.all() if r.status != "active"}
        removed: dict[str, list[str]] = {}
        for ix in self.existing():
            present = ix.docs | ix.dense_doc_ids() | _bm25_doc_ids(ix.bm25)
            stray = sorted(present & inactive)
            for doc_id in stray:
                ix.delete(doc_id)
            if stray:
                removed[f"{ix.version}/{ix.partition}"] = stray
        return removed

    def stats(self) -> dict[str, Any]:
        return {f"{ix.version}/{ix.partition}": ix.chunk_count for ix in self.existing()}


__all__ = ["IndexSet", "SearchIndex", "ACTIVE", "BUILDING", "PREVIOUS", "SNAPSHOT_GEN"]
