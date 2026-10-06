# path: book/projects/examples/ch32/tests/test_hardening.py
"""Edge cases behind the chapter's guarantees: re-recording, manifests, baselines, canary inputs."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from northwind_triage.eval_gate import run
from northwind_triage.experiments import ArmStats
from northwind_triage.record_replay import RecordReplayTransport
from northwind_triage.version_manifest import VersionManifest

from .conftest import FakeOpenAIUpstream
from .test_record_replay import make_classifier


def test_re_recording_replaces_the_old_exchange(tmp_path: Path) -> None:
    cassette = tmp_path / "c.json"
    make_classifier(RecordReplayTransport(cassette, "record", FakeOpenAIUpstream('{"old": 1}').transport())) \
        .classify(system="s", user="u", model="model-a")
    make_classifier(RecordReplayTransport(cassette, "record", FakeOpenAIUpstream('{"new": 2}').transport())) \
        .classify(system="s", user="u", model="model-a")
    replayed = make_classifier(RecordReplayTransport(cassette, "replay")).classify(system="s", user="u", model="model-a")
    assert "new" in replayed.text


def test_semconv_names_the_classifier_model() -> None:
    m = VersionManifest(app="a", app_version="1", git_sha="x", environment="test",
                        models={"provider": "openai", "classifier": "model-a"})
    assert m.as_semconv_attributes()["llm.model"] == "model-a"


def test_a_missing_baseline_fails_closed(tmp_path: Path) -> None:
    assert run(["--baseline", str(tmp_path / "none.json")]) == 2


def test_the_eval_ignores_flag_settings_in_the_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TRIAGE_PROMPT_TREATMENT_VERSION", "1.1.0")
    report = tmp_path / "r.json"
    assert run(["--prompt-version", "1.0.0", "--report", str(report)]) == 0
    assert '"1.1.0"' not in report.read_text()


def test_canary_counts_must_fit_their_requests() -> None:
    with pytest.raises(ValidationError):
        ArmStats(requests=500, task_successes=900, errors=0, safety_violations=0, p95_latency_ms=1, cost_per_request=0)
