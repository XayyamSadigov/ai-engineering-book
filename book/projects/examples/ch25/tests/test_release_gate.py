# path: book/projects/examples/ch25/tests/test_release_gate.py
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

import release_gate
import run_suite

CH25 = Path(__file__).resolve().parents[1]
GATES = CH25 / "ci" / "gates.toml"
BASELINES = CH25 / "ci" / "baselines"


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict[str, Path]:
    out = {}
    for system in ("candidate", "regressed"):
        d = tmp_path_factory.mktemp(system) / "runs"
        assert run_suite.main(["--system", system, "--out", str(d)]) == 0
        out[system] = d
    return out


def _gate(runs_dir: Path, out: Path, config: Path = GATES) -> int:
    return release_gate.main(["--config", str(config), "--runs", str(runs_dir), "--baselines", str(BASELINES),
                              "--out", str(out)])


def test_candidate_passes_and_writes_artifacts(runs, tmp_path, monkeypatch) -> None:
    step_summary = tmp_path / "step_summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(step_summary))
    assert _gate(runs["candidate"], tmp_path / "out") == release_gate.EXIT_PASS
    summary = (tmp_path / "out" / "summary.md").read_text()
    assert "PASS" in summary and step_summary.read_text() == summary
    assert sorted(p.name for p in (tmp_path / "out" / "reports").iterdir()) == [
        "agent.md", "classification.md", "extraction.md", "tools.md"]


def test_regressed_fails_and_summary_names_the_cases(runs, tmp_path) -> None:
    assert _gate(runs["regressed"], tmp_path) == release_gate.EXIT_FAIL
    summary = (tmp_path / "summary.md").read_text()
    assert "critical_fields_exact all pass" in summary and "INV-006" in summary
    assert "critical [critical] traj_success" in summary and "AG-002" in summary
    assert "| classification | PASS |" in summary


def test_missing_run_is_an_error_not_a_pass(runs, tmp_path) -> None:
    partial = tmp_path / "runs"
    shutil.copytree(runs["candidate"], partial)
    (partial / "agent.json").unlink()
    assert _gate(partial, tmp_path / "out") == release_gate.EXIT_ERROR
    assert "cannot read run" in (tmp_path / "out" / "summary.md").read_text()


def test_missing_baselines_block_instead_of_skipping_regression_rules(runs, tmp_path) -> None:
    # A failed artifact download leaves an empty baselines folder. Without require_baseline the
    # regression and slice rules would silently not run and the regressed system could pass.
    empty = tmp_path / "no-baselines"
    empty.mkdir()
    code = release_gate.main(["--config", str(GATES), "--runs", str(runs["candidate"]), "--baselines", str(empty),
                              "--out", str(tmp_path / "out")])
    assert code == release_gate.EXIT_FAIL
    assert "baseline present" in (tmp_path / "out" / "summary.md").read_text()


def test_bad_config_is_an_error(runs, tmp_path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("name = 'x'\n[[suites]]\nname = 'agent'\n[suites.gate]\nmin_cases = 'many'\n")
    assert _gate(runs["candidate"], tmp_path / "out", bad) == release_gate.EXIT_ERROR


def test_edited_dataset_breaks_the_pin(runs, tmp_path) -> None:
    cfg = GATES.read_text().replace('pinned_dataset_hash = "5785c798ec86"', 'pinned_dataset_hash = "000000000000"')
    edited = tmp_path / "gates.toml"
    edited.write_text(cfg)
    assert _gate(runs["candidate"], tmp_path / "out", edited) == release_gate.EXIT_FAIL
    assert "dataset pinned" in (tmp_path / "out" / "summary.md").read_text()


def test_aggregate_rules_fail_when_macro_f1_is_too_low(runs, tmp_path) -> None:
    cfg = GATES.read_text().replace("metric = \"macro_f1\"\nmin = 0.85", "metric = \"macro_f1\"\nmin = 0.99")
    strict = tmp_path / "gates.toml"
    strict.write_text(cfg)
    assert _gate(runs["candidate"], tmp_path / "out", strict) == release_gate.EXIT_FAIL
    assert "aggregate macro_f1" in (tmp_path / "out" / "summary.md").read_text()


@pytest.mark.parametrize("path", ["ci/github/eval-gate.yml", "ci/gitlab/.gitlab-ci.yml"])
def test_ci_workflows_parse_and_call_the_gate(path) -> None:
    text = (CH25 / path).read_text()
    doc = yaml.safe_load(text)
    assert isinstance(doc, dict)
    for needle in ("release_gate.py", "--eval-suite fast", "eval-out"):
        assert needle in text, needle
