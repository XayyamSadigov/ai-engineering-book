# path: book/projects/ragkit/ragkit/retrieval/settings.py
"""Environment-driven defaults for the retrieval funnel (RAGKIT_RETRIEVAL_* variables)."""
from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAGKIT_RETRIEVAL_", env_file=".env", extra="ignore")

    candidate_k: int = 50  # per first-stage retriever, per search query
    rerank_k: int = 20  # fused candidates handed to the reranker
    final_k: int = 8  # hits returned to the generator
    fusion: Literal["rrf", "weighted"] = "rrf"
    rrf_k: int = 60
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    cross_encoder_model: str | None = None  # a local cross-encoder name or path; unset = lexical fallback
    rerank_batch_size: int = 8  # candidates per LLM reranking call
    parallel: bool = True
    retrieve_timeout_s: float | None = None  # deadline for the first-stage phase; unset = wait for all
    diversity: Literal["none", "mmr"] = "none"  # "mmr" adds an MMRDiversifier after the reranker
    mmr_lambda: float = 0.7  # 1.0 = pure relevance; lower values push harder against near-copies
    diversify_pool_k: int | None = None  # pool the diversifier chooses from; unset = min(2 * final_k, rerank_k)
    rerank_timeout_s: float | None = None  # deadline for the reranker; on expiry keep the fused order


__all__ = ["RetrievalSettings"]
