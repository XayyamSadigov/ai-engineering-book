# path: book/projects/examples/ch32/tests/test_ci_files.py
"""The pipelines are code too: parse them and check the stage order the chapter promises."""
from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
CI = Path(__file__).resolve().parent.parent / "ci"


def test_github_workflow_orders_the_gates() -> None:
    wf = yaml.safe_load((CI / "github-actions.yml").read_text())
    jobs = wf["jobs"]
    assert set(jobs) >= {"lint", "typecheck", "unit", "offline-eval", "build", "canary", "promote", "nightly-drift"}
    assert set(jobs["unit"]["needs"]) == {"lint", "typecheck"}
    assert jobs["offline-eval"]["needs"] == "unit"
    assert jobs["build"]["needs"] == "offline-eval"
    assert jobs["canary"]["needs"] == "build"
    assert jobs["promote"]["environment"] == "production"
    assert jobs["unit"]["env"]["CASSETTE_MODE"] == "replay"
    # yaml parses the bare key `on` as True
    assert "schedule" in (wf.get("on") or wf.get(True))


def test_gitlab_pipeline_orders_the_gates() -> None:
    p = yaml.safe_load((CI / "gitlab-ci.yml").read_text())
    assert p["stages"] == ["lint", "test", "eval", "build", "canary", "promote"]
    assert p["offline-eval"]["needs"] == ["unit"]
    assert p["build"]["needs"] == ["offline-eval"]
    assert p["promote"]["rules"][0]["when"] == "manual"
    assert p["unit"]["variables"]["CASSETTE_MODE"] == "replay"
