# path: book/projects/examples/ch32/tests/test_triage_service.py
"""Use-case tests with FakeLLM through the real composition root."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from aie_core.llm.errors import RateLimitError
from aie_core.llm.providers import FakeLLM
from fastapi.testclient import TestClient

from northwind_triage.adapters.http_api import create_app
from northwind_triage.composition import build_service
from northwind_triage.config import ConfigError, load_settings

GOOD = {"category": "pos_payments", "priority": "P2", "confidence": 0.9, "rationale": "cards"}


def write_flags(tmp_path: Path, prompt_pct: float = 0, model_pct: float = 0, kill: bool = False) -> Path:
    path = tmp_path / "flags.json"
    path.write_text(json.dumps({
        "triage.prompt": {"allocations": [{"variant": "treatment", "percent": prompt_pct},
                                          {"variant": "control", "percent": 100 - prompt_pct}],
                          "kill_switch": kill},
        "triage.model": {"allocations": [{"variant": "candidate", "percent": model_pct},
                                         {"variant": "control", "percent": 100 - model_pct}]},
    }))
    return path


def service(tmp_path, llm, tracer, **overrides):
    settings = load_settings(environment="test", prompt_treatment_version="1.1.0",
                             model_candidate="model-b", flags_path=write_flags(tmp_path, **overrides.pop("flags", {})),
                             **overrides)
    return build_service(settings, llm_client=llm, tracer=tracer)


def test_happy_path_applies_business_rules(tmp_path, tracer, ticket) -> None:
    llm = FakeLLM(responses=[GOOD])
    result = service(tmp_path, llm, tracer).triage(ticket, unit_id="u-1")
    assert result.outcome == "ok"
    assert result.triage.priority.value == "P2"           # model said P2; floor for POS is P3
    assert llm.last_request.temperature == 0.0
    assert llm.last_request.response_schema is not None


def test_every_span_carries_the_version_manifest(tmp_path, tracer, ticket) -> None:
    result = service(tmp_path, FakeLLM(responses=[GOOD]), tracer).triage(ticket, unit_id="u-1")
    span = tracer.find("triage.request")[0]
    assert span.attributes["version.fingerprint"] == result.manifest.fingerprint()
    assert span.attributes["version.prompts.triage.classify"].startswith("1.0.0#")
    assert span.attributes["version.tool_schemas.create_ticket"].startswith("2.0.0#")
    assert span.attributes["version.flags.triage.prompt"] == "control"
    assert span.attributes["llm.served_model"] == "fake-model"


def test_flags_select_prompt_and_model(tmp_path, tracer, ticket) -> None:
    llm = FakeLLM(handler=lambda req: GOOD)
    svc = service(tmp_path, llm, tracer, flags={"prompt_pct": 100, "model_pct": 100})
    result = svc.triage(ticket, unit_id="u-1")
    assert result.manifest.prompts["triage.classify"].startswith("1.1.0#")
    assert llm.last_request.model == "model-b"
    assert "Suspected phishing" in llm.last_request.messages[0].text   # 1.1.0 wording


def test_kill_switch_returns_everyone_to_control(tmp_path, tracer, ticket) -> None:
    llm = FakeLLM(handler=lambda req: GOOD)
    svc = service(tmp_path, llm, tracer, flags={"prompt_pct": 100, "kill": True})
    assert svc.triage(ticket, unit_id="u-1").manifest.flags["triage.prompt"] == "control"


def test_unparseable_output_goes_to_a_human(tmp_path, tracer, ticket) -> None:
    result = service(tmp_path, FakeLLM(responses=["I think it's about payments."]), tracer).triage(ticket, "u-1")
    assert (result.outcome, result.triage.route) == ("parse_error", "human_review")
    assert tracer.find("triage.request")[0].attributes["triage.outcome"] == "parse_error"


def test_provider_errors_stop_at_the_adapter(tmp_path, tracer, ticket) -> None:
    llm = FakeLLM(responses=[RateLimitError("slow down", provider="fake")])
    result = service(tmp_path, llm, tracer).triage(ticket, "u-1")
    assert result.outcome == "classifier_unavailable"
    assert result.triage.route == "human_review"


def test_flag_pointing_at_unconfigured_variant_fails_at_startup(tmp_path, tracer) -> None:
    settings = load_settings(environment="test", flags_path=write_flags(tmp_path, prompt_pct=10))
    with pytest.raises(ConfigError, match="no configured value"):
        build_service(settings, llm_client=FakeLLM(), tracer=tracer)


def test_missing_prompt_version_fails_at_startup(tmp_path, tracer) -> None:
    with pytest.raises(ConfigError, match="9.9.9"):
        build_service(load_settings(environment="test", prompt_control_version="9.9.9"),
                      llm_client=FakeLLM(), tracer=tracer)


def test_http_adapter_returns_fingerprint(tmp_path, tracer, ticket) -> None:
    svc = service(tmp_path, FakeLLM(handler=lambda req: GOOD), tracer)
    client = TestClient(create_app(svc))
    r = client.post("/v1/triage", json={"ticket": ticket.model_dump()}, headers={"x-user-id": "u-1"})
    assert r.status_code == 200
    assert r.json()["outcome"] == "ok"
    assert len(r.json()["version_fingerprint"]) == 16
    v = client.get("/v1/version").json()
    assert v["manifest"]["app"] == "northwind-triage"
