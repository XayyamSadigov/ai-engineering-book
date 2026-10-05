# path: book/projects/p1-extraction-api/extraction_api/config.py
"""Project settings (EXTRACT_* variables). Model and tracing settings come from aie_core."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiKey(BaseModel):
    """One caller credential. The key itself is a secret; role and tenant are what it grants."""

    key: SecretStr
    principal: str = Field(min_length=1, max_length=100)    # recorded as the reviewer, logged on spans
    role: Literal["submitter", "reviewer", "admin"] = "submitter"
    tenant: str | None = None                                # None: may act for any tenant (admin tooling)


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EXTRACT_", env_file=".env", extra="ignore")

    accept_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    classify_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    classify_samples: int = Field(default=1, ge=1, le=9)
    max_schema_repairs: int = Field(default=2, ge=0, le=5)
    max_rule_repairs: int = Field(default=1, ge=0, le=3)
    require_po: bool = True
    dollar_means: str = "USD"
    max_document_chars: int = 50_000
    max_request_bytes: int = 8_000_000   # rejected with 413 before the body is parsed
    max_batch_size: int = 100
    batch_concurrency: int = 8
    document_deadline_s: float = Field(default=120.0, gt=0)   # wall-clock budget for all calls of one document
    call_timeout_s: float = Field(default=45.0, gt=0)         # per model call, capped by the remaining deadline
    # JSON list of ApiKey objects. Empty means authentication is OFF (local development only).
    api_keys: list[ApiKey] = Field(default_factory=list)
    review_backend: Literal["memory", "sqlite"] = "memory"
    review_db_path: str = "data/review.db"
    shared_data_dir: str | None = None  # where ReplayLLM finds sample data when LLM_PROVIDER=fake


__all__ = ["ApiKey", "AppSettings"]
