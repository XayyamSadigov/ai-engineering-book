# path: book/projects/examples/ch32/tests/test_config.py
from __future__ import annotations

import pytest

from northwind_triage.config import ConfigError, load_settings


def test_dev_defaults_run_offline() -> None:
    s = load_settings()
    assert (s.environment, s.llm.llm_provider) == ("dev", "fake")


def test_environment_variables_are_read(monkeypatch) -> None:
    monkeypatch.setenv("TRIAGE_PROMPT_CONTROL_VERSION", "1.1.0")
    monkeypatch.setenv("LLM_MODEL", "model-x")
    s = load_settings()
    assert s.prompt_control_version == "1.1.0"
    assert s.llm.llm_model == "model-x"


def test_prod_refuses_fake_provider_and_reports_every_problem(monkeypatch) -> None:
    monkeypatch.setenv("TRIAGE_ENVIRONMENT", "prod")
    with pytest.raises(ConfigError) as exc:
        load_settings()
    msg = str(exc.value)
    assert "LLM_PROVIDER=fake" in msg and "TRIAGE_GIT_SHA" in msg and "TRIAGE_FLAGS_PATH" in msg


def test_missing_key_is_a_startup_error(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        load_settings()


def test_secrets_never_appear_in_repr_or_errors(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live-SUPERSECRET")
    s = load_settings(environment="staging", git_sha="abc", flags_path=tmp_path / "f.json")
    assert "SUPERSECRET" not in repr(s) and "SUPERSECRET" not in str(s.model_dump())
    monkeypatch.setenv("TRIAGE_CLASSIFIER_TIMEOUT_S", "-1")
    with pytest.raises(ConfigError) as exc:
        load_settings()
    assert "SUPERSECRET" not in str(exc.value)


def test_pointless_experiment_is_rejected() -> None:
    with pytest.raises(ConfigError, match="measures nothing"):
        load_settings(prompt_treatment_version="1.0.0")


def test_settings_are_immutable() -> None:
    s = load_settings()
    with pytest.raises(Exception):
        s.environment = "prod"  # type: ignore[misc]
