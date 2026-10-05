# path: book/projects/examples/ch32/northwind_triage/config.py
"""Typed configuration, validated once at startup.

Two sources: aie_core.Settings (LLM_PROVIDER, LLM_MODEL, keys, tracing; shared by every
chapter) and TRIAGE_* variables for this application. Rules that depend on the environment
(prod must not run the fake provider, must have a key, must know its git sha) are enforced
here so a misconfigured deployment fails at boot, not on the first customer request.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from aie_core.settings import Settings as LLMSettings
from pydantic import Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["dev", "test", "staging", "prod"]


class ConfigError(RuntimeError):
    """Raised at startup with every problem found, not just the first."""


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRIAGE_", env_file=".env", extra="ignore",
                                      frozen=True)

    environment: Environment = "dev"
    app_version: str = "0.0.0-dev"
    git_sha: str = "unknown"

    prompt_id: str = "triage.classify"
    prompt_control_version: str = "1.0.0"
    prompt_treatment_version: str | None = None
    model_candidate: str | None = None            # control model comes from LLM_MODEL

    flags_path: Path | None = None
    prompt_dir: Path | None = None
    classifier_timeout_s: float = Field(default=8.0, gt=0, le=60)

    dataset_version: str | None = None            # recorded in the manifest when known
    evaluator_version: str | None = None

    llm: LLMSettings = Field(default_factory=LLMSettings)

    @model_validator(mode="after")
    def _environment_rules(self) -> "AppSettings":
        problems: list[str] = []
        deployed = self.environment in {"staging", "prod"}
        if deployed and self.llm.llm_provider == "fake":
            problems.append("LLM_PROVIDER=fake is not allowed in staging/prod")
        if deployed and self.git_sha == "unknown":
            problems.append("TRIAGE_GIT_SHA must be set in staging/prod (it goes into every trace)")
        if self.llm.llm_provider == "openai" and self.llm.openai_api_key is None and not self.llm.llm_base_url:
            problems.append("LLM_PROVIDER=openai needs OPENAI_API_KEY (or LLM_BASE_URL for a local server)")
        if self.llm.llm_provider == "anthropic" and self.llm.anthropic_api_key is None:
            problems.append("LLM_PROVIDER=anthropic needs ANTHROPIC_API_KEY")
        if self.environment == "prod" and self.flags_path is None:
            problems.append("TRIAGE_FLAGS_PATH is required in prod (no implicit 100% rollouts)")
        if self.prompt_treatment_version == self.prompt_control_version:
            problems.append("prompt treatment version equals control; the experiment measures nothing")
        if problems:
            raise ValueError("; ".join(problems))
        return self


def load_settings(**overrides: object) -> AppSettings:
    """The only way the app reads configuration. Converts validation noise into one
    readable ConfigError and never echoes secret values."""
    try:
        return AppSettings(**overrides)  # type: ignore[arg-type]
    except ValidationError as exc:
        lines = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"]) or "settings"
            lines.append(f"{loc}: {err['msg']}")
        raise ConfigError("invalid configuration:\n  " + "\n  ".join(lines)) from None
