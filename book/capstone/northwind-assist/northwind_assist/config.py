# path: book/capstone/northwind-assist/northwind_assist/config.py
"""Capstone settings. Every behavior-relevant knob is an NA_* environment variable.

Model provider settings (LLM_PROVIDER, LLM_MODEL, API keys, EMBEDDING_*) stay in aie_core's
Settings; tracing uses the Chapter 31 variables (TRACE_SINK, OTEL_*, CAPTURE_*). Settings are
validated once at startup (Chapter 32): a bad combination fails the boot, not the first request.
"""
from __future__ import annotations

from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

TENANTS = ("retail", "logistics")


DEV_JWT_SECRET = "dev-only-secret-change-me-0123456789abcdef"    # published in this repository


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NA_", env_file=".env", extra="ignore")

    environment: Literal["dev", "test", "staging", "prod"] = "dev"
    app_version: str = "1.0.0"
    git_sha: str = "local"

    # ------------------------------------------------------------------ auth (JWT)
    auth_mode: Literal["hs256", "rs256", "jwks"] = "hs256"
    jwt_secret: SecretStr = SecretStr(DEV_JWT_SECRET)
    jwt_public_key: str | None = None          # PEM, for rs256
    jwks_url: str | None = None                # for jwks (keys rotate at the IdP)
    jwt_issuer: str = "https://sso.northwind.example"
    jwt_audience: str = "northwind-assist"
    jwt_leeway_s: int = 30
    dev_login: bool = True                     # /v1/auth/dev-token, only with hs256 outside prod
    allowed_tenants: list[str] = Field(default_factory=lambda: list(TENANTS))

    # ------------------------------------------------------------------ models
    model_map: dict[str, str] = Field(default_factory=dict)   # catalog alias -> provider model id
    request_deadline_s: float = 8.0            # p95 completion target (illustrative)
    ttft_target_s: float = 2.0
    breaker_min_calls: int = 10
    breaker_failure_rate: float = 0.5
    breaker_open_s: float = 30.0

    # ------------------------------------------------------------------ retrieval
    index_name: str = "northwind"
    chunk_max_tokens: int = 200
    candidate_k: int = 30
    rerank_k: int = 12
    final_k: int = 6
    rag_enforce_acl: bool = True               # False simulates an ingestion bug that drops ACLs
    knowledge_backend: Literal["local", "p3"] = "local"   # p3: Project 3 registry, queue, worker, pgvector
    knowledge_sync_on_start: bool = True       # p3 backend: sync and drain in-process (dev); Compose uses the worker
    extra_docs_dir: str | None = None          # additional Markdown documents (eval red team)

    # ------------------------------------------------------------------ caches
    retrieval_cache_ttl_s: float = 900.0
    answer_cache_ttl_s: float = 3600.0
    answer_cache: bool = True

    # ------------------------------------------------------------------ tools and agents
    idempotency_db: str = ":memory:"
    audit_log_path: str | None = None
    approval_ttl_s: float = 900.0
    four_eyes: bool = True                     # requester may never approve their own action
    agent_max_steps: int = 6
    agent_max_tool_calls: int = 6
    agent_max_cost_usd: float = 0.25
    agent_deadline_s: float = 30.0
    event_dir: str | None = None               # JSONL agent event logs; None keeps them in memory

    # ------------------------------------------------------------------ memory
    memory_db: str | None = None               # SQLite file; None uses the in-memory store
    memory_secret: SecretStr = SecretStr("dev-only-memory-fingerprint-key")

    # ------------------------------------------------------------------ reliability and cost
    admission_capacity: int = 32
    admission_max_queue: int = 32
    tenant_requests_per_minute: float = 600.0
    tenant_daily_budget_usd: dict[str, float] = Field(
        default_factory=lambda: {"retail": 25.0, "logistics": 25.0})   # illustrative
    cost_alert_thresholds: list[float] = Field(default_factory=lambda: [0.5, 0.8, 1.0])
    cost_ledger_path: str | None = None
    flags_path: str | None = None

    # ------------------------------------------------------------------ guardrails
    canaries: list[str] = Field(default_factory=lambda: ["NW-CANARY-5e1f0c2a9b7d"])
    allowed_hosts: list[str] = Field(default_factory=lambda: ["intranet.northwind.example",
                                                              "docs.northwind.example"])
    allowed_mail_domains: list[str] = Field(default_factory=lambda: ["northwind.example"])

    @model_validator(mode="after")
    def _check(self) -> Settings:
        if self.environment in ("staging", "prod"):
            # Staging is reachable by more people than a laptop: a secret printed in the repository and
            # a login stub would let anyone mint an admin token for any tenant.
            if self.auth_mode == "hs256" and self.jwt_secret.get_secret_value() == DEV_JWT_SECRET:
                raise ValueError("NA_JWT_SECRET is the published development secret; set a real one")
            if self.dev_login:
                raise ValueError("NA_DEV_LOGIN must be false outside dev and test")
        if self.environment == "prod":
            if self.auth_mode == "hs256":
                raise ValueError("NA_AUTH_MODE=hs256 is for development; use rs256 or jwks in prod")
            if not self.rag_enforce_acl:
                raise ValueError("NA_RAG_ENFORCE_ACL=false is only allowed in eval and test runs")
        if self.auth_mode == "rs256" and not self.jwt_public_key:
            raise ValueError("NA_AUTH_MODE=rs256 needs NA_JWT_PUBLIC_KEY")
        if self.auth_mode == "jwks" and not self.jwks_url:
            raise ValueError("NA_AUTH_MODE=jwks needs NA_JWKS_URL")
        if not (0 < self.final_k <= self.rerank_k <= self.candidate_k):
            raise ValueError("need 0 < final_k <= rerank_k <= candidate_k")
        return self

    def behavior_config(self) -> dict[str, str]:
        """The settings that change answers; recorded in the version manifest (Chapter 32)."""
        keys = ("chunk_max_tokens", "candidate_k", "rerank_k", "final_k", "rag_enforce_acl",
                "agent_max_steps", "agent_max_tool_calls", "four_eyes")
        return {k: str(getattr(self, k)) for k in keys}


__all__ = ["Settings", "TENANTS"]
