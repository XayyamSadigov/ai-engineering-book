# path: book/projects/evalkit/tests/test_runner.py
import asyncio
import threading
import time

import pytest
from aie_core.llm.types import Completion, Message, Usage
from aie_core.observability import InMemoryTracer

from evalkit import (
    Dataset,
    EvalCase,
    FunctionEvaluator,
    Run,
    RunVersions,
    Score,
    TargetResult,
    evaluator,
    run_target,
    score_run,
)


@pytest.fixture
def ds() -> Dataset:
    return Dataset(
        [EvalCase(id=f"c{i}", input=i, expected=i * 2, tags=["small" if i < 5 else "large"]) for i in range(10)],
        name="double",
        version="1",
    )


correct = FunctionEvaluator("correct", lambda c, o: o == c.expected)


def test_runs_every_case_in_dataset_order_with_bounded_concurrency(ds):
    active, peak, lock = 0, 0, threading.Lock()

    def target(case):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01 * (10 - case.input) / 10)  # finish out of order
        with lock:
            active -= 1
        return case.input * 2

    run = run_target(target, ds, evaluators=[correct], concurrency=3,
                     versions=RunVersions(target="doubler", prompt="p1", model="m1"))
    assert [r.case_id for r in run.results] == ds.ids
    assert peak <= 3
    assert run.mean("correct") == 1.0 and run.pass_rate("correct") == 1.0
    assert run.versions.dataset == ds.fingerprint
    assert run.versions.evaluators == {"correct": "1"}
    assert run.dataset_hash == ds.content_hash


def test_target_errors_count_as_failures_not_missing(ds):
    def target(case):
        if case.input == 3:
            raise RuntimeError("provider timeout")
        return case.input * 2

    run = run_target(target, ds, evaluators=[correct])
    assert run.error_rate == pytest.approx(0.1)
    failed = next(r for r in run.results if r.case_id == "c3")
    assert failed.error_type == "RuntimeError" and failed.scores["correct"] == 0.0
    assert failed.details["correct"] == "target_error"
    assert run.mean("correct") == pytest.approx(0.9)  # denominator still 10


def test_evaluator_crash_is_recorded_not_scored_zero(ds):
    @evaluator("fragile")
    def fragile(case, output):
        if case.input == 4:
            raise KeyError("missing field")
        return 1.0

    run = run_target(lambda c: c.input * 2, ds, evaluators=[fragile])
    assert run.evaluator_error_count == 1
    row = next(r for r in run.results if r.case_id == "c4")
    assert row.scores["fragile"] is None and "KeyError" in row.evaluator_errors["fragile"]
    assert len(run.case_scores("fragile")) == 9


def test_completion_and_target_result_are_unpacked(ds):
    def target(case):
        if case.input % 2:
            return TargetResult(output=case.input * 2, cost_usd=0.01, input_tokens=5, trace_id="t-abc")
        return Completion(message=Message.assistant(str(case.input * 2)), usage=Usage(input_tokens=7, output_tokens=2),
                          model="m", provider="fake", raw={"cost_usd": 0.002})

    run = run_target(target, ds)
    odd = next(r for r in run.results if r.case_id == "c1")
    even = next(r for r in run.results if r.case_id == "c2")
    assert odd.trace_id == "t-abc" and odd.cost_usd == 0.01
    assert even.output == "4" and even.input_tokens == 7 and even.cost_usd == 0.002
    assert even.metadata["model"] == "m"
    assert run.total_cost_usd == pytest.approx(5 * 0.01 + 5 * 0.002)


def test_spans_are_emitted_and_linked_by_trace_id(ds):
    tracer = InMemoryTracer()
    run = run_target(lambda c: c.input, ds, tracer=tracer)
    spans = tracer.find("eval.case")
    assert len(spans) == 10
    assert {s.span_id for s in spans} == {r.trace_id for r in run.results}
    assert all(s.attributes["run_id"] == run.run_id for s in spans)


def test_async_target_runs_under_semaphore(ds):
    active, peak = 0, 0

    async def target(case):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.005)
        active -= 1
        return case.input * 2

    run = run_target(target, ds, evaluators=[correct], concurrency=4)
    assert peak <= 4 and run.mean("correct") == 1.0
    assert [r.case_id for r in run.results] == ds.ids


def test_repeats_expose_flaky_cases(ds):
    calls: dict[str, int] = {}
    lock = threading.Lock()

    def target(case):
        with lock:
            calls[case.id] = calls.get(case.id, 0) + 1
            n = calls[case.id]
        if case.id == "c7" and n % 2 == 0:
            return -1  # wrong every other call: nondeterministic behavior
        return case.input * 2

    run = run_target(target, ds, evaluators=[correct], repeats=4, concurrency=1)
    assert len(run.results) == 40
    assert run.flaky_cases("correct") == ["c7"]
    assert run.case_scores("correct")["c7"] == pytest.approx(0.5)


def test_score_run_rescoring_does_not_call_target_and_checks_dataset(ds):
    calls = 0

    def target(case):
        nonlocal calls
        calls += 1
        return case.input * 2

    run = run_target(target, ds, evaluators=[correct])
    close = FunctionEvaluator("close", lambda c, o: Score(name="close", value=1 - abs(o - c.expected) / 10, passed=True),
                              version="2")
    rescored = score_run(run, ds, [close])
    assert calls == 10
    assert set(rescored.metric_names()) == {"correct", "close"}
    assert rescored.versions.evaluators == {"correct": "1", "close": "2"}
    other = Dataset([EvalCase(id="c0", input=0, expected=1)], name="other")
    with pytest.raises(ValueError, match="dataset hash"):
        score_run(run, other, [close])


def test_run_roundtrip_and_latency_percentiles(ds, tmp_path):
    run = run_target(lambda c: c.input * 2, ds, evaluators=[correct])
    for i, r in enumerate(run.results):
        r.latency_ms = float(i + 1) * 100  # 100..1000
    assert run.latency_percentile(50) == 500 and run.latency_percentile(95) == 1000
    loaded = Run.load_json(run.save_json(tmp_path / "run.json"))
    assert loaded.case_scores("correct") == run.case_scores("correct")
    assert loaded.versions == run.versions


def test_invalid_arguments(ds):
    with pytest.raises(ValueError):
        run_target(lambda c: c, ds, concurrency=0)
