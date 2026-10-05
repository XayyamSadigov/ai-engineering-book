# path: book/projects/p4-support-assistant/support_assistant/config.py
"""Environment-driven settings for Project 4. LLM settings come from aie_core.Settings."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_DIR = Path(__file__).resolve().parents[1]


class AssistantSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="P4_", env_file=".env", extra="ignore")

    shared_data_dir: Path = PROJECT_DIR.parent / "shared-data"
    extra_tickets_path: Path | None = PROJECT_DIR / "data" / "injected_tickets.jsonl"
    # Recipient allowlist for send_reply: internal domain plus contracted customer domains.
    allowed_recipient_domains: list[str] = ["northwind.example"]
    allowed_recipients: list[str] = []
    approval_ttl_s: float = 900.0
    four_eyes: bool = False                 # true: requester may not approve their own send
    idempotency_db: str = ":memory:"        # a file path makes duplicate suppression survive restarts
    audit_log_path: str | None = None       # JSONL audit trail; None keeps events in memory
    max_rounds: int = 6
    tool_max_attempts: int = 3
    lookup_rate_per_minute: int = 20
    send_rate_per_hour: int = 30
    demo_llm: bool = True                   # with LLM_PROVIDER=fake, use the keyword-driven demo model


__all__ = ["AssistantSettings", "PROJECT_DIR"]
