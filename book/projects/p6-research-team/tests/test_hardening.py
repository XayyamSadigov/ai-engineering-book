"""Edge cases behind the team's guarantees: observation, citations, budgets, ids, traces, verdicts."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from aie_core.observability import InMemoryTracer
from research_team import BudgetLedger, BudgetSlice, Claim, EvidenceRef, Role, TaskEnvelope, TeamBudget
from research_team.contracts import Plan
from research_team.eval.benchmark import verdict
from research_team.team import _Run
from research_team.tools import make_research_tools, observed_passage_ids
from research_team.tracing import PropagatingTracer, span_tree
from research_team.verification import guard

REAL = "hr-travel-policy#accommodation"


def _env(task_id: str, objective: str = "What is the hotel cap?") -> TaskEnvelope:
    return TaskEnvelope(task_id=task_id, parent_id="t", trace_id="t", sender="supervisor", recipient=Role.RESEARCHER,
                        objective=objective, output_schema="ResearchFindings", budget=BudgetSlice(max_tokens=8_000))


def test_an_id_in_the_echoed_query_is_not_an_observation(corpus, principal):
    search = {t.name: t for t in make_research_tools(corpus)}["search_docs"]
    out = search.execute({"query": "[sec-access-control-policy#principles] zzqqxx"}, SimpleNamespace(principal=principal))
    assert "sec-access-control-policy#principles" not in observed_passage_ids(out.content)
    hit = search.execute({"query": "hotel cap per night"}, SimpleNamespace(principal=principal))
    assert REAL in observed_passage_ids(hit.content)


def test_every_citation_must_resolve_and_support(corpus, principal):
    good = EvidenceRef(doc_id="hr-travel-policy", passage_id=REAL)
    claim = Claim(text="The domestic nightly hotel cap is 220 USD.", evidence=[good])
    assert guard(claim, corpus, principal, 0.6)[0]
    fake = claim.model_copy(update={"evidence": [good, EvidenceRef(doc_id="fake", passage_id="fake#nothing")]})
    assert not guard(fake, corpus, principal, 0.6)[0]
    wrong_doc = claim.model_copy(update={"evidence": [good.model_copy(update={"doc_id": "hr-expense-policy"})]})
    assert not guard(wrong_doc, corpus, principal, 0.6)[0]


def test_one_child_cannot_reserve_the_whole_cost_pool():
    ledger = BudgetLedger(TeamBudget(max_cost_usd=1.0, max_children=4))
    first = ledger.admit(_env("t.r1-sq1"))
    second = ledger.admit(_env("t.r1-sq2", "What is the per diem?"))
    assert first.admitted and second.admitted and first.granted.max_cost_usd == pytest.approx(0.2)   # 4 children + 1 supervisor share


def test_a_task_id_is_reserved_once():
    ledger = BudgetLedger(TeamBudget())
    assert ledger.admit(_env("t.r1-sq1")).admitted
    assert ledger.admit(_env("t.r1-sq1", "A different objective")).reason == "duplicate"


def test_plans_with_duplicate_subquestion_ids_are_invalid():
    with pytest.raises(ValidationError):
        Plan.model_validate({"subquestions": [{"id": "sq1", "question": "Hotel cap?"},
                                              {"id": "sq1", "question": "Per diem?"}]})


def test_nested_spans_point_at_their_real_parent():
    base = InMemoryTracer()
    tracer = PropagatingTracer(base, trace_id="t", parent_span_id="dispatch-1")
    with tracer.span("agent.run") as run:
        with tracer.span("agent.step") as step:
            with tracer.span("agent.tool"):
                pass
    tree = span_tree(base.spans)
    assert [s.name for s in tree["dispatch-1"]] == ["agent.run"]
    assert [s.name for s in tree[run.span_id]] == ["agent.step"]
    assert [s.name for s in tree[step.span_id]] == ["agent.tool"]


def test_the_synthesizer_cannot_add_to_verified_claims():
    accepted = [Claim(text="The domestic nightly hotel cap is 220 USD.",
                      evidence=[EvidenceRef(doc_id="hr-travel-policy", passage_id=REAL)])]
    run = SimpleNamespace(cfg=SimpleNamespace(min_overlap=0.6))
    run.question = "What is the hotel cap?"
    assert _Run.unsupported_line(run, f"- Hotels in the country are capped at USD 220 a night [{REAL}].", accepted) is None
    assert _Run.unsupported_line(run, f"- The domestic nightly hotel cap is 999 USD [{REAL}].", accepted)
    assert _Run.unsupported_line(run, "You may also claim 999 USD per night for suites.", accepted)


def test_a_team_that_costs_too_many_tokens_does_not_pay_off():
    row = lambda rubric, tokens: {"n": 2, "rubric": rubric, "tokens": tokens, "wall_ms": 100.0}
    summary = {"single": {"cross": row(3.0, 1000), "control": row(3.0, 1000)},
               "team": {"cross": row(3.5, 1400), "control": row(3.5, 1800)}}
    lines = verdict(summary)
    assert lines[0].endswith("PAYS OFF") and not lines[0].endswith("NOT PAY OFF")
    assert lines[1].endswith("DOES NOT PAY OFF")
