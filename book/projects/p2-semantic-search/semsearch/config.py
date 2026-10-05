# path: book/projects/p2-semantic-search/semsearch/config.py
"""Environment-driven settings and the factories that turn them into an embedder and a store."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from aie_core.embeddings import EmbeddingClient, FakeEmbeddings
from aie_core.settings import Settings as CoreSettings
from aie_core.settings import make_embedding_client
from pydantic_settings import BaseSettings, SettingsConfigDict

from .adapters.base import VectorStore
from .adapters.numpy_store import NumpyVectorStore
from .domain.models import make_namespace

PROJECT_DIR = Path(__file__).resolve().parent.parent


class SearchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    embedding_dimensions: int = 256  # used by the fake embedder; real models report their own
    vector_backend: Literal["numpy", "pgvector"] = "numpy"
    index_dir: Path = Path(".index")
    database_url: str | None = None
    pg_table: str = "chunks"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    hnsw_ef_search: int = 100
    hnsw_iterative_scan: Literal["off", "strict_order", "relaxed_order"] = "relaxed_order"
    pg_statement_timeout_ms: int | None = None  # per-query cap on pgvector reads; unset = server default
    index_name: str = "knowledge"
    index_version: str = "v1"
    docs_dir: Path = PROJECT_DIR.parent / "shared-data" / "docs"
    chunk_max_chars: int = 1200
    search_default_k: int = 8
    search_max_k: int = 50


def make_embedder(core: CoreSettings | None = None, search: SearchSettings | None = None) -> EmbeddingClient:
    core = core or CoreSettings()
    search = search or SearchSettings()
    if core.embedding_provider == "fake":
        # aie_core's factory uses the fake's default width; we want it configurable.
        return FakeEmbeddings(dimensions=search.embedding_dimensions, model=core.embedding_model)
    return make_embedding_client(core)


def embedder_dimensions(embedder: EmbeddingClient) -> int:
    """Real clients may learn their width from the first response; probe once if needed."""
    if not embedder.dimensions:
        embedder.embed_query("dimension probe")
    return int(embedder.dimensions)


def namespace_for(embedder: EmbeddingClient, search: SearchSettings | None = None) -> str:
    search = search or SearchSettings()
    return make_namespace(search.index_name, embedder.model, search.index_version)


def make_store(dimensions: int, search: SearchSettings | None = None) -> VectorStore:
    search = search or SearchSettings()
    if search.vector_backend == "pgvector":
        if not search.database_url:
            raise ValueError("VECTOR_BACKEND=pgvector requires DATABASE_URL")
        from .adapters.pg_store import PgVectorStore

        return PgVectorStore(
            search.database_url,
            dimensions,
            table=search.pg_table,
            m=search.hnsw_m,
            ef_construction=search.hnsw_ef_construction,
            ef_search=search.hnsw_ef_search,
            iterative_scan=search.hnsw_iterative_scan,
            statement_timeout_ms=search.pg_statement_timeout_ms,
        )
    if (search.index_dir / "").exists() and any(search.index_dir.glob("*.json")):
        return NumpyVectorStore.load(search.index_dir, dimensions)
    return NumpyVectorStore(dimensions, persist_dir=search.index_dir)


__all__ = ["SearchSettings", "make_embedder", "embedder_dimensions", "namespace_for", "make_store"]
