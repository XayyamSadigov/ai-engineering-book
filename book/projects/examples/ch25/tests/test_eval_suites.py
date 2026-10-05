# path: book/projects/examples/ch25/tests/test_eval_suites.py
"""Evaluation suites as opt-in pytest targets.

    pytest                                      # unit tests only; these are skipped
    pytest --eval-suite fast --eval-out eval-out/runs
    pytest --eval-suite full                    # adds repeated trials

The tests assert harness health and write Run JSON; ci/release_gate.py makes the decision.
"""
from __future__ import annotations

import pytest

from taskevals.suites import SUITES, run_suite
from taskevals.trajectory import pass_all_k


@pytest.mark.eval_fast
@pytest.mark.parametrize("suite", list(SUITES))
def test_fast_suite_runs_cleanly(suite, eval_system, record_run) -> None:
    run = run_suite(suite, eval_system)
    record_run(suite, run)
    assert len(run.by_case()) == len(SUITES[suite].build_dataset())
    assert run.error_rate == 0.0, [r.error for r in run.errors][:3]
    assert run.evaluator_error_count == 0


@pytest.mark.eval_full
def test_agent_suite_is_reliable_across_trials(eval_system, record_run) -> None:
    from evalkit import RunVersions, run_target

    suite = SUITES["agent"]
    run = run_target(suite.make_target(eval_system), suite.build_dataset(), evaluators=suite.make_evaluators(),
                     versions=RunVersions(target="northwind-agent-trials", extra={"system": eval_system}),
                     repeats=3, concurrency=4)
    record_run("agent_trials", run)
    assert run.flaky_cases("traj_success") == []
    assert pass_all_k(run, "traj_success") >= 0.75
