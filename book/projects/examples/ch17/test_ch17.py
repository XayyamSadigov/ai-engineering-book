# path: book/projects/examples/ch17/test_ch17.py
"""Offline tests for the Chapter 17 workflow engine, patterns, and the two
triage implementations. No network, no API keys: the model is a FakeModel.
"""
from __future__ import annotations

import asyncio
import json
import time

import pytest
from pydantic import BaseModel

from patterns import branch, fallback, fan_out, map_reduce, retry, sequence
from triage_domain import FakeModel, TriageState
from triage_graph import build_triage_graph, route_after_validate
from triage_pipeline import PipelineMetrics, run_pipeline
from workflow_engine import (END, FatalError, Graph, InMemoryCheckpointer, RetryPolicy, StaleHandleError,
                             StepValidationError, TransientError)

OK = json.dumps({"verdict": "ok", "reasons": []})
FIXABLE = json.dumps({"verdict": "fixable", "reasons": ["tone too formal"]})
ESCALATE = json.dumps({"verdict": "escalate", "reasons": ["policy conflict"]})


def shipping_ticket() -> TriageState:
    return TriageState(ticket_id="T-1", tenant="logistics", text="My parcel is 9 days late.")


def refund_ticket() -> TriageState:
    return TriageState(ticket_id="T-2", tenant="retail", text="I want my money back for a broken kettle.")


class Sent:
    """Sender fake with at-most-once delivery per idempotency key."""

    def __init__(self, fail_after_delivery: int = 0) -> None:
        self.messages: list[tuple[str, str]] = []
        self.keys: list[str] = []
        self._delivered: set[str] = set()
        self._fail_after_delivery = fail_after_delivery

    def __call__(self, ticket_id: str, body: str, key: str) -> None:
        self.keys.append(key)
        if key not in self._delivered:
            self._delivered.add(key)
            self.messages.append((ticket_id, body))
        if self._fail_after_delivery > 0:   # the reply went out, then the call timed out
            self._fail_after_delivery -= 1
            raise TransientError("send timed out after delivery")


# --- patterns ---------------------------------------------------------------
def test_sequence_feeds_output_forward() -> None:
    run = sequence(lambda x: x + 1, lambda x: x * 10, str)
    assert run(1) == "20"


def test_branch_routes_on_decision() -> None:
    run = branch(lambda x: "even" if x % 2 == 0 else "odd",
                 {"even": lambda x: x // 2, "odd": lambda x: 3 * x + 1})
    assert run(4) == 2 and run(3) == 10


def test_fan_out_runs_concurrently_and_keeps_order() -> None:
    async def job(i: int, delay: float) -> int:
        await asyncio.sleep(delay)
        return i

    async def boom() -> int:
        raise TransientError("provider down")

    t0 = time.perf_counter()
    results = asyncio.run(fan_out([lambda: job(0, 0.05), lambda: job(1, 0.05), boom, lambda: job(3, 0.05)]))
    elapsed = time.perf_counter() - t0
    assert results[0] == 0 and results[1] == 1 and results[3] == 3
    assert isinstance(results[2], TransientError)
    assert elapsed < 0.12, "branches must overlap, not serialize"


def test_map_reduce_counts_words_across_documents() -> None:
    docs = ["alpha beta", "gamma", "delta epsilon zeta"]

    async def count(doc: str) -> int:
        return len(doc.split())

    assert asyncio.run(map_reduce(docs, count, sum)) == 6


def test_retry_stops_at_attempts_and_raises_last_error() -> None:
    calls: list[int] = []

    def flaky() -> str:
        calls.append(1)
        raise TransientError("429")

    with pytest.raises(TransientError):
        retry(flaky, attempts=3)
    assert len(calls) == 3


def test_fallback_uses_secondary_only_on_listed_errors() -> None:
    def primary_down() -> str:
        raise TransientError("primary provider unavailable")

    def primary_buggy() -> str:
        raise ValueError("bug in our code")

    assert fallback(primary_down, lambda: "cheap model answer") == "cheap model answer"
    with pytest.raises(ValueError):
        fallback(primary_buggy, lambda: "cheap model answer", on=(TransientError,))


# --- engine -----------------------------------------------------------------
class Counter(BaseModel):
    n: int = 0
    path: list[str] = []


def test_engine_sequence_and_branch() -> None:
    g: Graph[Counter] = Graph(Counter)
    g.add_node("inc", lambda s: s.model_copy(update={"n": s.n + 1, "path": s.path + ["inc"]}))
    g.add_node("double", lambda s: s.model_copy(update={"n": s.n * 2, "path": s.path + ["double"]}))
    g.add_node("done", lambda s: s.model_copy(update={"path": s.path + ["done"]}))
    g.add_router("inc", lambda s: "double" if s.n < 4 else "done")
    g.add_edge("double", "inc")
    g.add_edge("done", END)
    result = g.run(Counter())
    assert result.status == "completed"
    assert result.state.n == 7  # 1 -> 2 -> 3 -> 6 -> 7
    assert result.state.path == ["inc", "double", "inc", "double", "inc", "done"]


def test_engine_retry_exhaustion_records_attempts_and_fails_run() -> None:
    attempts: list[int] = []

    def flaky(s: Counter) -> Counter:
        attempts.append(1)
        raise TransientError("timeout")

    cps = InMemoryCheckpointer()
    g: Graph[Counter] = Graph(Counter, checkpointer=cps)
    g.add_node("flaky", flaky, retry=RetryPolicy(max_attempts=3))
    result = g.run(Counter(), run_id="r1")
    assert result.status == "failed" and len(attempts) == 3
    assert result.trace[0].attempts == 3 and "TransientError" in (result.error or "")
    assert cps.latest("r1") is not None and cps.latest("r1").status == "failed"


def test_engine_does_not_retry_validation_errors() -> None:
    attempts: list[int] = []

    def bad(s: Counter) -> Counter:
        attempts.append(1)
        raise StepValidationError("schema mismatch")

    g: Graph[Counter] = Graph(Counter)
    g.add_node("bad", bad, retry=RetryPolicy(max_attempts=5))
    assert g.run(Counter()).status == "failed" and len(attempts) == 1


def test_engine_max_steps_guards_against_cycles() -> None:
    g: Graph[Counter] = Graph(Counter, max_steps=10)
    g.add_node("loop", lambda s: s)
    g.add_edge("loop", "loop")
    result = g.run(Counter())
    assert result.status == "failed" and result.error == "max_steps exceeded"


# --- triage pipeline ----------------------------------------------------------
def test_pipeline_sequence_sends_shipping_reply_without_approval() -> None:
    model = FakeModel({"classify": ["shipping"], "draft": ["We will re-ship at no cost."], "validate": [OK]})
    sent = Sent()
    metrics = PipelineMetrics()
    state = run_pipeline(shipping_ticket(), model, approver=lambda s: True, sender=sent, metrics=metrics)
    assert state.outcome == "sent" and sent.messages == [("T-1", "We will re-ship at no cost.")]
    assert [c.task for c in model.calls] == ["classify", "draft", "validate"]
    assert state.approved is None, "shipping replies do not need a human"
    assert metrics.orchestration_overhead_ms >= 0 and set(metrics.step_ms) >= {"classify", "draft", "validate", "send"}


def test_pipeline_redrafts_once_then_escalates() -> None:
    model = FakeModel({"classify": ["shipping"], "draft": ["I guarantee delivery.", "I promise delivery."],
                       "validate": [OK, OK]})
    state = run_pipeline(shipping_ticket(), model, approver=lambda s: True, sender=Sent())
    assert state.outcome == "escalated" and state.draft_attempts == 2
    assert state.report is not None and state.report.verdict == "fixable"


def test_pipeline_requires_synchronous_approval_for_refunds() -> None:
    model = FakeModel({"classify": ["refund"], "draft": ["Refund approved within 30 days."], "validate": [OK]})
    seen: list[str] = []

    def approver(s: TriageState) -> bool:
        seen.append(s.ticket_id)
        return False

    state = run_pipeline(refund_ticket(), model, approver=approver, sender=Sent())
    assert seen == ["T-2"] and state.outcome == "escalated" and state.approved is False


# --- triage graph -----------------------------------------------------------
def test_graph_branch_rules_are_testable_without_running_steps() -> None:
    s = refund_ticket()
    s.category = "refund"; s.draft_attempts = 1
    from triage_domain import ValidationReport
    s.report = ValidationReport(verdict="ok")
    assert route_after_validate(s) == "approval"
    s.report = ValidationReport(verdict="fixable")
    assert route_after_validate(s) == "draft"
    s.draft_attempts = 2
    assert route_after_validate(s) == "escalate"
    s.category = "shipping"; s.report = ValidationReport(verdict="ok")
    assert route_after_validate(s) == "send"


def test_graph_pauses_for_approval_and_resumes() -> None:
    model = FakeModel({"classify": ["refund"], "draft": ["Refund issued within 30 days."], "validate": [OK]})
    sent = Sent()
    cps = InMemoryCheckpointer()
    g = build_triage_graph(model, sent, checkpointer=cps)

    paused = g.run(refund_ticket(), run_id="run-approve")
    assert paused.status == "paused" and paused.handle is not None and paused.handle.node == "approval"
    assert sent.messages == [], "nothing is sent while a human has not decided"
    assert cps.latest("run-approve").status == "paused"

    # ...minutes or days later, in a different process:
    g2 = build_triage_graph(model, sent, checkpointer=cps)
    done = g2.resume(paused.handle, {"approved": True, "note": "ok per policy"})
    assert done.status == "completed" and done.state.outcome == "sent"
    assert done.state.approval_note == "ok per policy" and len(sent.messages) == 1
    assert [c.task for c in model.calls] == ["classify", "draft", "validate"], "resume re-runs no model step"


def test_graph_rejection_routes_to_escalate() -> None:
    model = FakeModel({"classify": ["account"], "draft": ["Your email is set to a@b.c"], "validate": [OK, OK]})
    sent = Sent()
    g = build_triage_graph(model, sent)
    result = g.run(TriageState(ticket_id="T-3", tenant="retail", text="change my email"))
    # first draft leaked data -> fixable -> redraft (same leaky script) -> escalate before any human
    assert result.status == "completed" and result.state.outcome == "escalated"
    assert result.state.draft_attempts == 2 and sent.messages == []


def test_graph_retries_transient_model_errors_per_node() -> None:
    inner = FakeModel({"classify": ["shipping"], "draft": ["Re-shipping now."], "validate": [OK]})
    failures = {"n": 2}

    def flaky(task: str, prompt: str) -> str:
        if task == "draft" and failures["n"] > 0:
            failures["n"] -= 1
            raise TransientError("rate limited")
        return inner(task, prompt)

    g = build_triage_graph(flaky, Sent(), model_retry=RetryPolicy(max_attempts=3))
    result = g.run(shipping_ticket())
    assert result.status == "completed" and result.state.outcome == "sent"
    draft_record = next(r for r in result.trace if r.node == "draft")
    assert draft_record.attempts == 3


def test_checkpoint_replay_and_resume_after_crash() -> None:
    model = FakeModel({"classify": ["shipping"], "draft": ["Re-shipping now."], "validate": [OK]})
    sent = Sent()
    cps = InMemoryCheckpointer()
    crash = {"armed": True}

    def crashing_draft(task: str, prompt: str) -> str:
        if task == "draft" and crash["armed"]:
            crash["armed"] = False
            raise FatalError("process killed mid-step")
        return model(task, prompt)

    g = build_triage_graph(crashing_draft, sent, checkpointer=cps)
    first = g.run(shipping_ticket(), run_id="run-crash")
    assert first.status == "failed" and "FatalError" in (first.error or "")
    assert [n for n, _ in g.replay("run-crash")] == ["classify", "retrieve_policy", "draft"]
    assert cps.latest("run-crash").next_node == "draft", "failed checkpoint points at the step to redo"

    # New process: same checkpointer, fixed model. Continue from the last durable state.
    g2 = build_triage_graph(model, sent, checkpointer=cps)
    second = g2.resume_from_checkpoint("run-crash")
    assert second.status == "completed" and second.state.outcome == "sent"
    assert [c.task for c in model.calls] == ["classify", "draft", "validate"], "classify was not re-run"
    nodes = [n for n, _ in g2.replay("run-crash")]
    assert nodes == ["classify", "retrieve_policy", "draft", "draft", "validate", "send"]
    states = [s for _, s in g2.replay("run-crash")]
    assert states[0].category == "shipping" and states[0].draft is None
    assert states[-1].sent is True


def test_resume_refuses_a_stale_approval_handle() -> None:
    model = FakeModel({"classify": ["refund"], "draft": ["Refund issued within 30 days."], "validate": [OK]})
    sent, cps = Sent(), InMemoryCheckpointer()
    g = build_triage_graph(model, sent, checkpointer=cps)
    paused = g.run(refund_ticket(), run_id="run-stale")
    assert paused.handle is not None and paused.handle.state_hash is not None
    cps.latest("run-stale").state["draft"] = "Refund guaranteed, no receipt needed."  # changed after review
    with pytest.raises(StaleHandleError):
        g.resume(paused.handle, {"approved": True})
    assert sent.messages == [], "the human approved a draft that is no longer the one to send"


def test_send_reexecuted_after_ambiguous_failure_delivers_once() -> None:
    model = FakeModel({"classify": ["shipping"], "draft": ["Re-shipping now."], "validate": [OK]})
    sent, cps = Sent(fail_after_delivery=1), InMemoryCheckpointer()
    g = build_triage_graph(model, sent, checkpointer=cps)
    first = g.run(shipping_ticket(), run_id="run-dup")
    assert first.status == "failed" and "send" in (first.error or "")
    second = g.resume_from_checkpoint("run-dup")
    assert second.status == "completed"
    assert len(sent.keys) == 2 and sent.keys[0] == sent.keys[1] == "run-dup:send:0"
    assert len(sent.messages) == 1, "same key, so the second execution was suppressed"


def test_failed_attempt_does_not_leak_partial_state() -> None:
    class Counter(BaseModel):
        n: int = 0

    calls = {"k": 0}

    def flaky(s: Counter) -> Counter:
        s.n += 1                      # mutate first ...
        calls["k"] += 1
        if calls["k"] == 1:
            raise TransientError("boom")   # ... then fail
        return s

    g: Graph[Counter] = Graph(Counter)
    g.add_node("inc", flaky, retry=RetryPolicy(max_attempts=2))
    assert g.run(Counter()).state.n == 1


def test_tracer_gets_one_span_per_node() -> None:
    class Span:
        def __init__(self, name: str, attrs: dict) -> None:
            self.name, self.attributes = name, dict(attrs)

        def set_attribute(self, k: str, v: object) -> None:
            self.attributes[k] = v

    class Tracer:
        def __init__(self) -> None:
            self.spans: list[Span] = []

        def span(self, name: str, **attrs: object):
            import contextlib

            @contextlib.contextmanager
            def cm():
                sp = Span(name, attrs)
                self.spans.append(sp)
                yield sp
            return cm()

    model = FakeModel({"classify": ["shipping"], "draft": ["Re-shipping now."], "validate": [OK]})
    tracer = Tracer()
    build_triage_graph(model, Sent(), tracer=tracer).run(shipping_ticket(), run_id="run-trace")
    assert [s.attributes["node"] for s in tracer.spans] == ["classify", "retrieve_policy", "draft", "validate", "send"]
    assert all(s.name == "workflow.node" and s.attributes["workflow.status"] == "ok" for s in tracer.spans)
    assert tracer.spans[3].attributes["workflow.next_node"] == "send"
