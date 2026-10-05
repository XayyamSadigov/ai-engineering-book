# path: book/projects/p2-semantic-search/semsearch/adapters/numpy_store.py
"""Exact vector search on NumPy with metadata filters, namespaces, and file persistence.

Design:
- One `_Namespace` per namespace: a float32 matrix of L2-normalized vectors plus a parallel
  list of records. Cosine similarity is then a single matrix-vector product.
- Filtering happens *before* scoring (pre-filter): build a boolean mask from metadata, score
  only the allowed rows. With exact search this is both correct and cheap, which is the main
  argument for exact search at small scale.
- Writes rebuild the namespace's matrix lazily on the next search. That makes bulk ingestion
  O(n) instead of O(n^2) and keeps reads simple. A lock serializes writers and the rebuild.
- Persistence: `<dir>/<namespace>.npz` holds vectors, `<dir>/<namespace>.json` holds records
  without vectors plus the dimensions, so files can be inspected and diffed.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from ..domain.filters import matches
from ..domain.models import DocVersion, SearchFilter, SearchHit, Vector, VectorRecord
from .base import check_dimensions, check_document


def _unit(v: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(arr))
    return arr / n if n > 0 else arr


@dataclass
class _Namespace:
    records: dict[str, VectorRecord] = field(default_factory=dict)
    # materialized view, rebuilt when dirty
    ids: list[str] = field(default_factory=list)
    matrix: np.ndarray | None = None
    dirty: bool = True

    def materialize(self, dimensions: int) -> None:
        if not self.dirty:
            return
        self.ids = list(self.records.keys())
        if self.ids:
            self.matrix = np.vstack([_unit(self.records[i].vector) for i in self.ids]).astype(np.float32)
        else:
            self.matrix = np.zeros((0, dimensions), dtype=np.float32)
        self.dirty = False


class NumpyVectorStore:
    """Exact cosine search. Good to roughly a million vectors per namespace on one machine,
    as long as the matrix fits in RAM (n * d * 4 bytes) and the query rate is modest."""

    def __init__(self, dimensions: int, persist_dir: str | Path | None = None) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self._ns: dict[str, _Namespace] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ writes
    def upsert(self, records: Sequence[VectorRecord]) -> int:
        check_dimensions(records, self.dimensions)
        with self._lock:
            for r in records:
                ns = self._ns.setdefault(r.namespace, _Namespace())
                ns.records[r.id] = r
                ns.dirty = True
        return len(records)

    def replace_document(self, namespace: str, doc_id: str, doc_version: str, records: Sequence[VectorRecord]) -> int:
        check_dimensions(records, self.dimensions)
        check_document(namespace, doc_id, doc_version, records)
        with self._lock:  # the lock makes delete+insert one step for readers
            ns = self._ns.setdefault(namespace, _Namespace())
            stale = [rid for rid, r in ns.records.items() if r.doc_id == doc_id]
            for rid in stale:
                del ns.records[rid]
            for r in records:
                ns.records[r.id] = r
            ns.dirty = True
        return len(records)

    def delete_document(self, namespace: str, doc_id: str) -> int:
        with self._lock:
            ns = self._ns.get(namespace)
            if ns is None:
                return 0
            stale = [rid for rid, r in ns.records.items() if r.doc_id == doc_id]
            for rid in stale:
                del ns.records[rid]
            ns.dirty = ns.dirty or bool(stale)
            return len(stale)

    # ------------------------------------------------------------------ reads
    def search(
        self,
        namespace: str,
        query: Vector,
        k: int,
        flt: SearchFilter | None = None,
        *,
        exact: bool = False,  # always exact; accepted for protocol compatibility
    ) -> list[SearchHit]:
        if len(query) != self.dimensions:
            raise ValueError(f"query has {len(query)} dimensions, store expects {self.dimensions}")
        if k <= 0:
            return []
        with self._lock:
            ns = self._ns.get(namespace)
            if ns is None or not ns.records:
                return []
            ns.materialize(self.dimensions)
            ids, matrix, records = ns.ids, ns.matrix, ns.records
            assert matrix is not None
            if flt is None or flt.is_empty():
                rows = np.arange(len(ids))
            else:
                rows = np.fromiter((i for i, rid in enumerate(ids) if matches(records[rid], flt)), dtype=np.int64)
            if rows.size == 0:
                return []
            scores = matrix[rows] @ _unit(query)
            kk = min(k, rows.size)
            top = np.argpartition(-scores, kk - 1)[:kk]
            top = top[np.argsort(-scores[top], kind="stable")]
            return [self._hit(records[ids[rows[t]]], float(scores[t])) for t in top]

    @staticmethod
    def _hit(r: VectorRecord, score: float) -> SearchHit:
        return SearchHit(
            id=r.id,
            doc_id=r.doc_id,
            doc_version=r.doc_version,
            ordinal=r.ordinal,
            tenant=r.tenant,
            score=score,
            text=r.text,
            tags=r.tags,
            metadata=r.metadata,
        )

    def doc_versions(self, namespace: str) -> dict[str, DocVersion]:
        with self._lock:
            ns = self._ns.get(namespace)
            if ns is None:
                return {}
            out: dict[str, DocVersion] = {}
            for r in ns.records.values():
                cur = out.get(r.doc_id)
                out[r.doc_id] = DocVersion(doc_id=r.doc_id, doc_version=r.doc_version, chunks=(cur.chunks if cur else 0) + 1)
            return out

    def count(self, namespace: str | None = None) -> int:
        with self._lock:
            if namespace is not None:
                ns = self._ns.get(namespace)
                return len(ns.records) if ns else 0
            return sum(len(ns.records) for ns in self._ns.values())

    def namespaces(self) -> list[str]:
        with self._lock:
            return sorted(name for name, ns in self._ns.items() if ns.records)

    # ------------------------------------------------------------------ persistence
    @staticmethod
    def _file_stem(namespace: str) -> str:
        return namespace.replace(":", "__")

    def save(self, directory: str | Path | None = None) -> Path:
        target = Path(directory) if directory else self.persist_dir
        if target is None:
            raise ValueError("no directory given and no persist_dir configured")
        target.mkdir(parents=True, exist_ok=True)
        with self._lock:
            for name, ns in self._ns.items():
                ns.materialize(self.dimensions)
                stem = self._file_stem(name)
                vec_tmp = target / f"{stem}.tmp.npz"
                meta_tmp = target / f"{stem}.json.tmp"
                np.savez_compressed(vec_tmp, ids=np.array(ns.ids, dtype=np.str_), vectors=ns.matrix)
                meta = {
                    "namespace": name,
                    "dimensions": self.dimensions,
                    "records": [r.model_dump(exclude={"vector"}) for r in (ns.records[i] for i in ns.ids)],
                }
                meta_tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
                # write-then-rename: a crash mid-save never leaves a half-written index
                vec_tmp.replace(target / f"{stem}.npz")
                meta_tmp.replace(target / f"{stem}.json")
        return target

    @classmethod
    def load(cls, directory: str | Path, dimensions: int | None = None) -> "NumpyVectorStore":
        directory = Path(directory)
        metas = sorted(directory.glob("*.json"))
        if dimensions is None:
            if not metas:
                raise FileNotFoundError(f"no index files in {directory}")
            dimensions = int(json.loads(metas[0].read_text(encoding="utf-8"))["dimensions"])
        store = cls(dimensions, persist_dir=directory)
        for meta_path in metas:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if int(meta["dimensions"]) != dimensions:
                raise ValueError(f"{meta_path.name} has dimensions {meta['dimensions']}, expected {dimensions}")
            with np.load(meta_path.with_suffix(".npz"), allow_pickle=False) as data:  # never unpickle index files
                ids = [str(i) for i in data["ids"]]
                vectors = data["vectors"]
            by_id = {rec["id"]: rec for rec in meta["records"]}
            records = [VectorRecord(**by_id[rid], vector=vectors[row].tolist()) for row, rid in enumerate(ids)]
            store.upsert(records)
        return store


__all__ = ["NumpyVectorStore"]
