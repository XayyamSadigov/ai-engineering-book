# path: book/projects/examples/ch04/tests/test_regression.py
from __future__ import annotations

import json

import pytest

from aie_core.llm.errors import ProviderUnavailableError
from aie_core.llm.providers import FakeLLM
from demo_model import simulated_router
from prompts import (
    Assertion,
    Case,
    LLMJudge,
    PromptTemplate,
    VariableSpec,
    compare,
    load_cases,
    render_report,
    run_suite,
)
from prompts.regression import _MISSING, check, get_path, schema_errors


# ------------------------------------------------------------ the three-version story
@pytest.fixture(scope="module")
def suites(registry, ticket_cases):
    return {
        v: run_suite(registry.get("ticket.classify", v), simulated_router(), ticket_cases)
        for v in ("1.0.0", "1.1.0", "1.2.0")
    }


def test_candidate_with_critical_regression_is_blocked(suites):
    cmp = compare(suites["1.0.0"], suites["1.1.0"])
    assert cmp.candidate_pass_rate > cmp.baseline_pass_rate  # better on average...
    assert cmp.critical_regressions == ["TCK-2026-0026"]  # ...but exposed card data now goes to stores
    gate = cmp.gate(max_input_token_growth_pct=1000)
    assert not gate.ok and "critical" in gate.reasons[0]


def test_reordering_rules_fixes_the_critical_case(suites):
    cmp = compare(suites["1.1.0"], suites["1.2.0"])
    assert cmp.fixes == ["TCK-2026-0026"] and cmp.regressions == []
    assert cmp.gate().ok


def test_token_budget_flags_prompt_growth(suites):
    cmp = compare(suites["1.0.0"], suites["1.2.0"])
    assert cmp.input_tokens_delta_pct > 100  # definitions + rules + two examples
    assert not cmp.gate().ok
    assert cmp.gate(max_input_token_growth_pct=1000).ok
    assert not cmp.gate(max_input_token_growth_pct=1000, max_regressions=0).ok  # 0007 regressed


def test_report_names_failures_with_reasons(suites):
    cmp = compare(suites["1.0.0"], suites["1.1.0"])
    report = render_report(cmp, suites["1.1.0"])
    assert "Gate: FAIL" in report
    assert "TCK-2026-0026: equals: category='pos_payments', expected 'security_report'" in report


def test_injection_probe_passes_because_the_forged_tag_is_escaped(suites):
    assert suites["1.1.0"].by_id()["INJ-001"].passed


def test_same_probe_succeeds_against_an_undelimited_template(registry, ticket_cases):
    """Declaring the body trusted removes delimiting and escaping: the simulated model now
    sees the injected line outside any data block and obeys it."""
    good = registry.get("ticket.classify", "1.1.0")
    sections = [(m_role, src) for m_role, src in good.template.sections]
    variables = {**good.spec.variables, "ticket_body": VariableSpec(trusted=True)}
    naive = PromptTemplate(sections, variables, name="naive")
    probe = next(c for c in ticket_cases if c.id == "INJ-001")
    msgs = naive.render(probe.variables)
    from aie_core.llm.types import CompletionRequest

    out = json.loads(simulated_router().complete(CompletionRequest(messages=msgs)).text)
    assert out["category"] == "benefits_leave"


# ------------------------------------------------------------ errors and flakiness
def test_provider_errors_are_errors_not_regressions(registry, ticket_cases):
    pv = registry.get("ticket.classify", "1.1.0")
    subset = ticket_cases[:2]
    base = run_suite(pv, simulated_router(), subset)
    broken = FakeLLM(responses=[ProviderUnavailableError("503")] * 2)
    cand = run_suite(pv, broken, subset)
    cmp = compare(base, cand)
    assert cmp.errored == [c.id for c in subset] and cmp.regressions == []
    assert any("errored" in r for r in cmp.gate().reasons)


def test_repeats_expose_flaky_cases(registry, ticket_cases):
    pv = registry.get("ticket.classify", "1.1.0")
    case = ticket_cases[0]  # gold: pos_payments
    good = json.dumps({"category": "pos_payments", "evidence": "all cards are declined on register 3"})
    bad = json.dumps({"category": "hardware", "evidence": "register 3"})
    result = run_suite(pv, FakeLLM(responses=[good, bad, good]), [case], repeats=3)
    c = result.cases[0]
    assert c.flaky and not c.passed and c.pass_rate == pytest.approx(2 / 3)


# ------------------------------------------------------------ assertions
def test_schema_subset_validator():
    schema = {
        "type": "object", "required": ["a"], "additionalProperties": False,
        "properties": {"a": {"type": "integer", "minimum": 0}, "b": {"type": "array", "items": {"type": "string"}}},
    }
    assert schema_errors({"a": 1, "b": ["x"]}, schema) == []
    errs = schema_errors({"a": -1, "b": [1], "c": True}, schema)
    assert any("minimum" in e for e in errs) and any("unexpected" in e for e in errs) and any("$.b[0]" in e for e in errs)
    assert schema_errors(True, {"type": "integer"})  # bool is not an integer


def test_get_path():
    data = {"citations": ["a", "b"], "x": {"y": 1}}
    assert get_path(data, "citations.1") == "b" and get_path(data, "x.y") == 1
    assert get_path(data, "x.z") is _MISSING


def test_quote_assertion_rejects_invented_evidence(registry, ticket_cases):
    pv = registry.get("ticket.classify", "1.1.0")
    case = ticket_cases[0]
    a = Assertion(type="quote_in_variable", path="evidence")
    real = {"evidence": "all cards are declined on register 3"}
    invented = {"evidence": "the payment terminal firmware is outdated"}
    assert check(a, json.dumps(real), real, case, pv).passed
    assert not check(a, json.dumps(invented), invented, case, pv).passed


def test_non_json_output_fails_json_assertions_but_not_text_ones(registry, ticket_cases):
    pv = registry.get("ticket.classify", "1.1.0")
    case = ticket_cases[0]
    out = "Category: pos_payments"
    assert not check(Assertion(type="schema_valid"), out, _MISSING, case, pv).passed
    assert check(Assertion(type="contains", value="pos_payments"), out, _MISSING, case, pv).passed


def test_duplicate_case_ids_rejected(tmp_path):
    line = json.dumps({"id": "A", "variables": {}})
    p = tmp_path / "c.jsonl"
    p.write_text(line + "\n" + line + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(p)


def test_case_that_does_not_fit_the_prompt_is_a_render_error(registry):
    pv = registry.get("ticket.classify", "1.1.0")
    result = run_suite(pv, simulated_router(), [Case(id="X", variables={"ticket_body": "hi"})])
    assert result.cases[0].errored and "render" in result.cases[0].runs[0].error


# ------------------------------------------------------------ the judge
GOOD_ANSWER = json.dumps({"answer": "You can carry over up to 10 days.", "citations": ["hr-pto-policy"], "abstained": False})


def _judge(registry, verdicts):
    return LLMJudge(registry.get("judge.groundedness"), FakeLLM(responses=verdicts))


def test_judge_runs_only_after_deterministic_checks_pass(registry, answer_cases):
    pv = registry.get("assist.answer", "prod")
    case = answer_cases[0]
    judge_llm = FakeLLM(responses=[json.dumps({"score": 3, "unsupported_claims": []})])
    judge = LLMJudge(registry.get("judge.groundedness"), judge_llm)
    wrong = json.dumps({"answer": "Five days.", "citations": ["hr-faq"], "abstained": False})
    r1 = run_suite(pv, FakeLLM(responses=[wrong]), [case], judge=judge)
    assert not r1.cases[0].passed and judge_llm.requests == []  # failed "contains 10" first
    r2 = run_suite(pv, FakeLLM(responses=[GOOD_ANSWER]), [case], judge=judge)
    assert r2.cases[0].passed and r2.cases[0].runs[0].judge.score == 3
    judged = judge_llm.last_request.messages[-1].text
    assert '<untrusted_data label="evidence" id="hr-pto-policy">' in judged


def test_low_judge_score_fails_the_case_with_rationale(registry, answer_cases):
    pv = registry.get("assist.answer", "prod")
    judge = _judge(registry, [json.dumps({"score": 1, "unsupported_claims": ["carryover expires in June"]})])
    r = run_suite(pv, FakeLLM(responses=[GOOD_ANSWER]), [answer_cases[0]], judge=judge)
    run = r.cases[0].runs[0]
    assert not run.passed and "June" in run.judge.rationale


def test_malformed_judge_output_is_an_error_not_a_verdict(registry, answer_cases):
    pv = registry.get("assist.answer", "prod")
    judge = _judge(registry, [json.dumps({"score": 7})])
    r = run_suite(pv, FakeLLM(responses=[GOOD_ANSWER]), [answer_cases[0]], judge=judge)
    assert r.cases[0].errored and "judge" in r.cases[0].runs[0].error


def test_answer_suite_catches_an_injected_answer(registry, answer_cases):
    pv = registry.get("assist.answer", "prod")
    injected = json.dumps({"answer": "Carryover is unlimited.", "citations": ["ext-vendor-newsletter-brightline"], "abstained": False})
    abstain = json.dumps({"answer": "The documents do not cover parking.", "citations": [], "abstained": True})
    honest = json.dumps({"answer": "Up to 10 days.", "citations": ["hr-pto-policy"], "abstained": False})
    r = run_suite(pv, FakeLLM(responses=[GOOD_ANSWER, abstain, injected]), answer_cases)
    assert [c.passed for c in r.cases] == [True, True, False]
    r = run_suite(pv, FakeLLM(responses=[GOOD_ANSWER, abstain, honest]), answer_cases)
    assert all(c.passed for c in r.cases)
    assert r.cases[0].runs[0].input_tokens > 0
