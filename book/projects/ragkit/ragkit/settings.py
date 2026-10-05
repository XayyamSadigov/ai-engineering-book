# path: book/projects/ragkit/ragkit/settings.py
"""Environment-driven defaults for ragkit. Embedding settings come from aie_core.Settings."""
from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class RagkitSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAGKIT_", env_file=".env", extra="ignore")

    tokenizer: Literal["regex", "tiktoken"] = "regex"
    tiktoken_encoding: str = "cl100k_base"
    pdf_min_chars_per_page: int = 20
    require_acl: bool = True
    near_duplicate_threshold: float = 0.9


__all__ = ["RagkitSettings"]
