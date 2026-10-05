# path: book/projects/examples/ch25/ci/pytest_evalplugin.py
"""pytest plugin that makes evaluation suites first-class, opt-in test targets.

Markers:
    eval_fast   minutes, deterministic or cheap; runs on every merge request
    eval_full   judged, repeated, expensive; nightly and on release candidates

Options:
    --eval-suite {none,fast,full}   which eval tests run (default none: plain `pytest` stays fast)
    --eval-system NAME              stand-in system under test (default $EVAL_SYSTEM or candidate)
    --eval-out DIR                  where eval tests write Run JSON for the release gate

Eval tests check that the harness is healthy (cases ran, no target or evaluator errors) and
write the Run artifacts. The release gate script, not pytest, decides whether to ship, so the
decision logic lives in one reviewed config file instead of being scattered across asserts.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

LEVELS = {"none": set(), "fast": {"eval_fast"}, "full": {"eval_fast", "eval_full"}}


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("eval", "AI evaluation suites")
    group.addoption("--eval-suite", choices=sorted(LEVELS), default="none", help="which eval tests to run")
    group.addoption("--eval-system", default=os.environ.get("EVAL_SYSTEM", "candidate"),
                    help="system under evaluation (stand-in name)")
    group.addoption("--eval-out", default=None, help="directory for Run JSON artifacts")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "eval_fast: fast evaluation suite, runs on every merge request")
    config.addinivalue_line("markers", "eval_full: full evaluation suite, nightly and release candidates")
    config._eval_written = []  # type: ignore[attr-defined]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    enabled = LEVELS[config.getoption("--eval-suite")]
    for item in items:
        for marker in ("eval_fast", "eval_full"):
            if item.get_closest_marker(marker) and marker not in enabled:
                item.add_marker(pytest.mark.skip(reason=f"{marker} not selected (use --eval-suite)"))


@pytest.fixture(scope="session")
def eval_system(pytestconfig: pytest.Config) -> str:
    return pytestconfig.getoption("--eval-system")


@pytest.fixture(scope="session")
def eval_out(pytestconfig: pytest.Config, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = pytestconfig.getoption("--eval-out")
    path = Path(out) if out else tmp_path_factory.mktemp("eval-out")
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def record_run(pytestconfig: pytest.Config, eval_out: Path):
    """Save a Run as <eval_out>/<name>.json and remember it for the terminal summary."""

    def save(name: str, run) -> Path:  # noqa: ANN001 - evalkit.Run
        path = run.save_json(eval_out / f"{name}.json")
        pytestconfig._eval_written.append(path)  # type: ignore[attr-defined]
        return path

    return save


def pytest_terminal_summary(terminalreporter, exitstatus: int, config: pytest.Config) -> None:  # noqa: ANN001
    written = getattr(config, "_eval_written", [])
    if written:
        terminalreporter.section("evaluation runs written")
        for p in written:
            terminalreporter.write_line(str(p))
