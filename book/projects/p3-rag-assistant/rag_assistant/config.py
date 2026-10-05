# path: book/projects/p3-rag-assistant/rag_assistant/config.py
"""All configuration for the assistant, from environment variables (prefix RAG_).

Provider settings (LLM_PROVIDER, LLM_MODEL, EMBEDDING_PROVIDER, API keys, TRACE_SINK) are read
by aie_core.Settings and are deliberately not duplicated here. Every number below is an
illustrative default chosen for the Northwind corpus; Chapter 15 explains where each comes from.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DOCS_DIR = PROJECT_DIR.parent / "shared-data" / "docs"
DEFAULT_GOLD_PATH = PROJECT_DIR.parent / "shared-data" / "eval" / "retrieval_gold.jsonl"
DEFAULT_AUTHORITY_RULES = Path(__file__).resolve().parent / "data" / "authority_rules.json"
DEV_AUTH_SECRET = "dev-only-change-me"


class AssistantSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # --- sources and storage --------------------------------------------------------------
    docs_dir: Path = DEFAULT_DOCS_DIR
    blob_dir: Path | None = None  # uploaded originals; None keeps them in memory (tests)
    snapshot_dir: Path | None = None  # BM25 snapshots shared by worker and API; None = in-process only
    vector_backend: Literal["numpy", "pgvector"] = "numpy"
    registry_backend: Literal["memory", "sql"] = "memory"
    registry_url: str | None = None  # SQLAlchemy URL; falls back to DATABASE_URL for sql
    queue_backend: Literal["memory", "redis"] = "memory"
    cache_backend: Literal["memory", "redis"] = "memory"  # embedding cache store
    max_inline_bytes: int = 1_000_000  # larger uploads belong in object storage

    # --- index identity -------------------------------------------------------------------
    index_name: str = "northwind"
    index_version: str = "v1"  # initial active version; blue/green reindex creates the next one
    tenancy_mode: Literal["shared", "namespace"] = "shared"
    chunk_max_tokens: int = 300
    pipeline_version: str = "p3-ingest-1"  # bump when parse/normalize/annotate logic changes
    authority_rules_path: Path = DEFAULT_AUTHORITY_RULES
    contextual_enrichment: bool = False

    # --- retrieval ------------------------------------------------------------------------
    candidate_k: int = 40
    rerank_k: int = 20
    final_k: int = 8
    reranker: Literal["lexical", "cross_encoder", "none"] = "lexical"
    authority_weight: float = 0.05
    parallel_retrieval: bool = True
    retrieve_timeout_s: float | None = 0.3  # first-stage budget; a slow retriever is skipped (degraded)
    rerank_timeout_s: float | None = 0.25  # past this, keep the fused order (degraded "rerank:TimeoutError")
    pg_statement_timeout_ms: int | None = 250  # vector store reads are cancelled server-side after this

    # --- generation and packing -----------------------------------------------------------
    evidence_token_budget: int = 2500
    max_evidence_blocks: int = 6
    generation_max_tokens: int = 700

    # --- budgets, reliability -------------------------------------------------------------
    request_deadline_s: float = 8.0
    min_generation_s: float = 1.5  # below this much remaining budget, return sources only
    breaker_min_calls: int = 5
    breaker_open_s: float = 30.0
    admission_capacity: int = 32
    tenant_requests_per_minute: float = 600.0

    # --- caching --------------------------------------------------------------------------
    retrieval_cache: bool = True
    retrieval_cache_ttl_s: float = 300.0
    answer_cache: bool = True
    answer_cache_ttl_s: float = 600.0
    cache_max_entries: int = 10_000
    guard_policy_version: str = "rag-guard-1"

    # --- API and auth ---------------------------------------------------------------------
    auth_secret: str = Field(default=DEV_AUTH_SECRET, repr=False)
    # The API refuses to start with the published dev secret unless this is set; Docker Compose sets
    # it for local development only. Anyone who has read this file can mint tokens for that secret.
    allow_dev_auth_secret: bool = False
    admin_group: str = "rag-admin"
    inline_ingest: bool = False  # process ingestion jobs inside the API process (dev only)
    sync_on_startup: bool = False  # enqueue (and, with inline_ingest, process) a docs sync at startup

    # --- worker ---------------------------------------------------------------------------
    worker_visibility_timeout_s: float = 120.0
    worker_poll_interval_s: float = 1.0

    # --- freshness SLO --------------------------------------------------------------------
    freshness_slo_s: float = 300.0  # a change should be searchable within this many seconds


__all__ = ["AssistantSettings", "DEV_AUTH_SECRET", "DEFAULT_DOCS_DIR", "DEFAULT_GOLD_PATH", "DEFAULT_AUTHORITY_RULES", "PROJECT_DIR"]
