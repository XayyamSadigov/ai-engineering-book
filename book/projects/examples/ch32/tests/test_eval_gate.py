# path: book/projects/examples/ch32/tests/test_eval_gate.py
from __future__ import annotations

import json
import shutil
from pathlib import Path

from aie_core.llm.providers import FakeLLM

from northwind_triage.eval_gate import DEFAULT_BASELINE, DEFAULT_DATASET, run


def test_committed_baseline_passes(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    assert run(["--prompt-version", "1.1.0", "--report", str(report)]) == 0
    data = json.loads(report.read_text())
    assert data["failures"] == []
    assert data["manifest"]["prompts"]["triage.classify"].startswith("1.1.0#")
    assert data["manifest"]["datasets"]["triage_golden"].startswith("golden-60#")


def test_regressed_model_fails_the_gate(tmp_path: Path) -> None:
    broken = FakeLLM(handler=lambda req: {"category": "hardware", "priority": "P4", "confidence": 0.9})
    report = tmp_path / "report.json"
    assert run(["--report", str(report)], llm_client=broken) == 1
    failures = json.loads(report.read_text())["failures"]
    assert any("critical slice recall.security_report" in f for f in failures)


def test_unparseable_outputs_fail_on_error_rate(tmp_path: Path) -> None:
    chatty = FakeLLM(handler=lambda req: "Sure! This looks like a hardware issue.")
    report = tmp_path / "report.json"
    assert run(["--report", str(report)], llm_client=chatty) == 1
    assert any("error_rate" in f for f in json.loads(report.read_text())["failures"])


def test_changed_dataset_makes_baseline_incomparable(tmp_path: Path) -> None:
    dataset = tmp_path / "golden.jsonl"
    lines = DEFAULT_DATASET.read_text().splitlines()[:-1]  # drop one case
    dataset.write_text("\n".join(lines) + "\n")
    baseline = tmp_path / "baseline.json"
    shutil.copy(DEFAULT_BASELINE, baseline)
    assert run(["--dataset", str(dataset), "--baseline", str(baseline)]) == 2


def test_two_simultaneous_changes_lose_attribution(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    doc = json.loads(DEFAULT_BASELINE.read_text())
    doc["manifest"]["models"]["classifier"] = "some-other-model"   # pretend baseline used another model
    baseline.write_text(json.dumps(doc))
    report = tmp_path / "r.json"
    # Prompt 1.1.0 AND a different model versus baseline: two components changed.
    code = run(["--prompt-version", "1.1.0", "--baseline", str(baseline), "--report", str(report),
                "--strict-attribution"])
    assert code == 1
    assert any("attribution is lost" in f for f in json.loads(report.read_text())["failures"])
