# path: book/projects/p2-semantic-search/tests/conftest.py
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Callable, Iterator

import pytest
from aie_core.embeddings import FakeEmbeddings

from semsearch.adapters.base import VectorStore
from semsearch.adapters.numpy_store import NumpyVectorStore
from semsearch.domain.models import VectorRecord, chunk_id

VOCAB = [
    "refund", "policy", "deadline", "vpn", "error", "laptop", "pto", "carryover",
    "incident", "sev1", "returns", "api", "tracking", "webhook", "failover", "stipend",
]
NS = "test:fake:v1"
SHARED_DOCS = Path(__file__).resolve().parents[2] / "shared-data" / "docs"


@pytest.fixture
def vocab_embedder() -> FakeEmbeddings:
    return FakeEmbeddings(vocabulary=VOCAB)


def make_record(
    embedder: FakeEmbeddings,
    doc_id: str,
    text: str,
    *,
    ordinal: int = 0,
    version: str = "1",
    tenant: str = "shared",
    acl: tuple[str, ...] = ("all",),
    tags: tuple[str, ...] = (),
    namespace: str = NS,
) -> VectorRecord:
    return VectorRecord(
        id=chunk_id(doc_id, version, ordinal),
        namespace=namespace,
        tenant=tenant,
        doc_id=doc_id,
        doc_version=version,
        ordinal=ordinal,
        text=text,
        vector=embedder.embed_query(text),
        acl_groups=acl,
        tags=tags,
        metadata={"title": doc_id},
        embedding_model=embedder.model,
    )


def _pg_factory(dims: int) -> VectorStore:
    from semsearch.adapters.pg_store import PgVectorStore

    table = f"t_{uuid.uuid4().hex[:10]}"
    return PgVectorStore(os.environ["DATABASE_URL"], dims, table=table, ef_search=64)


STORE_PARAMS = [
    pytest.param("numpy", id="numpy"),
    pytest.param(
        "pgvector",
        id="pgvector",
        marks=[
            pytest.mark.integration,
            pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set"),
        ],
    ),
]


@pytest.fixture(params=STORE_PARAMS)
def store_factory(request: pytest.FixtureRequest) -> Iterator[Callable[[int], VectorStore]]:
    created: list[VectorStore] = []

    def factory(dims: int) -> VectorStore:
        store: VectorStore = NumpyVectorStore(dims) if request.param == "numpy" else _pg_factory(dims)
        created.append(store)
        return store

    yield factory
    for s in created:
        if request.param == "pgvector":
            s._conn.execute(f"DROP TABLE IF EXISTS {s.table}")  # type: ignore[attr-defined]
            s.close()  # type: ignore[attr-defined]
