# path: book/projects/p2-semantic-search/semsearch/adapters/base.py
"""The storage contract every vector store adapter implements.

The protocol is deliberately small. It names the operations an application actually needs
(write a document's chunks atomically, delete a document, filtered top-k search, report what
is indexed) and nothing a particular engine happens to offer.
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ..domain.models import DocVersion, SearchFilter, SearchHit, Vector, VectorRecord


class DimensionMismatchError(ValueError):
    """A vector's length differs from the store's configured dimensions."""


@runtime_checkable
class VectorStore(Protocol):
    dimensions: int

    def upsert(self, records: Sequence[VectorRecord]) -> int:
        """Insert or replace records by id. Returns the number written."""
        ...

    def replace_document(self, namespace: str, doc_id: str, doc_version: str, records: Sequence[VectorRecord]) -> int:
        """Atomically make `records` the only chunks of `doc_id` in `namespace`.

        Chunks of any other version of the document are removed in the same operation, so a
        reader sees either the old version or the new one, never neither and never both.
        """
        ...

    def delete_document(self, namespace: str, doc_id: str) -> int:
        """Remove every chunk of a document. Returns the number removed."""
        ...

    def search(
        self,
        namespace: str,
        query: Vector,
        k: int,
        flt: SearchFilter | None = None,
        *,
        exact: bool = False,
    ) -> list[SearchHit]:
        """Top-k by cosine similarity among records that satisfy `flt`.

        `exact=True` forces a brute-force scan; it exists so that an approximate index can
        be measured against ground truth. Adapters that are always exact ignore it.
        """
        ...

    def doc_versions(self, namespace: str) -> dict[str, DocVersion]:
        """doc_id -> what is currently indexed. Drives incremental ingestion and pruning."""
        ...

    def count(self, namespace: str | None = None) -> int: ...

    def namespaces(self) -> list[str]: ...


def check_dimensions(records: Sequence[VectorRecord], dimensions: int) -> None:
    for r in records:
        if len(r.vector) != dimensions:
            raise DimensionMismatchError(
                f"record {r.id} has {len(r.vector)} dimensions, store expects {dimensions}"
            )


def check_document(namespace: str, doc_id: str, doc_version: str, records: Sequence[VectorRecord]) -> None:
    for r in records:
        if (r.namespace, r.doc_id, r.doc_version) != (namespace, doc_id, doc_version):
            raise ValueError(f"record {r.id} does not belong to {namespace}/{doc_id}@{doc_version}")


__all__ = ["VectorStore", "DimensionMismatchError", "check_dimensions", "check_document"]
