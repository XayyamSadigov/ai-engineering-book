# path: book/projects/guardrails/tests/test_pipeline.py
from __future__ import annotations

import pytest
from aie_core.observability import InMemoryTracer

from guardrails import Action, BaseCheck, FailMode, GuardrailPipeline, Stage, Verdict


class Fixed(BaseCheck):
    def __init__(self, name, verdict, stages=frozenset({Stage.INPUT}), fail_mode=FailMode.CLOSED):
        self.name, self._v, self.stages, self.fail_mode = name, verdict, stages, fail_mode
        self.seen: list[str] = []

    def evaluate(self, subject, ctx):
        self.seen.append(subject.text)
        if isinstance(self._v, Exception):
            raise self._v
        return self._v(subject.text) if callable(self._v) else self._v


def test_redactions_thread_to_later_checks(ctx):
    a = Fixed("a", lambda t: Verdict.redact(t.replace("secret", "[x]"), "masked"))
    b = Fixed("b", Verdict.allow())
    res = GuardrailPipeline([a, b]).check_input("my secret", ctx)
    assert res.text == "my [x]" and b.seen == ["my [x]"]
    assert res.action is Action.REDACT and res.allowed


def test_block_short_circuits_and_reports(ctx):
    a = Fixed("a", Verdict.block("no"))
    b = Fixed("b", Verdict.allow())
    res = GuardrailPipeline([a, b]).check_input("x", ctx)
    assert not res.allowed and res.blocked_by == "a" and b.seen == []


def test_flags_do_not_block(ctx):
    res = GuardrailPipeline([Fixed("a", Verdict.flag("odd", 0.6))]).check_input("x", ctx)
    assert res.allowed and res.action is Action.FLAG and len(res.flags) == 1


def test_fail_closed_blocks_on_exception(ctx):
    res = GuardrailPipeline([Fixed("boom", RuntimeError("down"))]).check_input("x", ctx)
    assert not res.allowed and res.errors and res.blocked_by == "boom"


def test_fail_open_flags_on_exception(ctx):
    c = Fixed("boom", RuntimeError("down"), fail_mode=FailMode.OPEN)
    res = GuardrailPipeline([c]).check_input("x", ctx)
    assert res.allowed and res.action is Action.FLAG and res.errors[0].error


def test_wrong_return_type_is_treated_as_failure(ctx):
    res = GuardrailPipeline([Fixed("bad", "allow")]).check_input("x", ctx)
    assert not res.allowed


def test_stage_routing(ctx):
    out_only = Fixed("o", Verdict.block("no"), stages=frozenset({Stage.OUTPUT}))
    pipe = GuardrailPipeline([out_only])
    assert pipe.check_input("x", ctx).allowed
    assert not pipe.check_output("x", ctx).allowed


def test_non_check_rejected():
    with pytest.raises(TypeError):
        GuardrailPipeline([object()])  # type: ignore[list-item]


def test_traces_carry_decisions_not_raw_text(ctx):
    tracer = InMemoryTracer()
    pipe = GuardrailPipeline([Fixed("a", Verdict.flag("odd", 0.7))], tracer=tracer)
    pipe.check_input("salary of employee 4021 is 99000", ctx)
    stage_span = tracer.find("guardrail.input")[0]
    check_span = tracer.find("guardrail.check")[0]
    assert stage_span.attributes["guardrail.action"] == "flag"
    assert check_span.attributes["guardrail.score"] == 0.7
    dumped = repr([s.to_dict() for s in tracer.spans])
    assert "99000" not in dumped and "4021" not in dumped
