# path: book/projects/p5-incident-agent/incident_agent/config.py
"""Settings for Project 5. Model selection comes from aie_core (LLM_PROVIDER, LLM_MODEL, ...);
everything specific to the incident agent uses the P5_ prefix."""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_DIR = Path(__file__).resolve().parents[1]


class P5Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="P5_", extra="ignore")

    shared_data_dir: Path = PROJECT_DIR.parent / "shared-data"
    data_dir: Path = PROJECT_DIR / "data"
    state_dir: Path = Path(".p5-state")
    channel: str = "#incidents"
    max_plan_steps: int = Field(default=8, ge=1, le=12)
    max_replans: int = Field(default=2, ge=0)
    max_revisions: int = Field(default=3, ge=1)
    min_improvement: float = 0.05
    judge_pass_score: int = Field(default=4, ge=1, le=5)
    judge_model: str | None = None
    step_max_steps: int = 3
    step_max_tool_calls: int = 2
    max_llm_calls: int = 60
    chunk_tokens: int = 250


# Who may use the service, with the groups the server trusts. In production this comes from
# your identity provider; it is never taken from the request body.
DIRECTORY: dict[str, dict[str, object]] = {
    "oncall-logistics": {"user": "oncall-logistics", "tenant": "logistics", "groups": ["all", "it-oncall"]},
    "ic-logistics": {"user": "ic-logistics", "tenant": "logistics", "groups": ["all", "it-oncall", "managers"]},
    "analyst-logistics": {"user": "analyst-logistics", "tenant": "logistics", "groups": ["all"]},
    "oncall-retail": {"user": "oncall-retail", "tenant": "retail", "groups": ["all", "it-oncall"]},
}

__all__ = ["P5Settings", "DIRECTORY", "PROJECT_DIR"]
