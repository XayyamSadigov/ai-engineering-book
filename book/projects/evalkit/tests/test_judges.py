# path: book/projects/evalkit/tests/test_judges.py
import json

import pytest
from aie_core.llm.errors import MalformedResponseError
from aie_core.llm.providers import FakeLLM
from aie_core.llm.types import CompletionRequest

from evalkit import (
    CORRECTNESS,
    GROUNDEDNESS,
    Dataset,
    EvalCase,
    LLMJudge,
    PairwiseJudge,
    calibrate_judge,
    cohens_kappa,
    pairwise_summary,
    run_target,
)


# ------------------------------------------------------------------ single-dimension judge
def test_judge_returns_constrained_verdict_and_builds_a_rubric_prompt():
    llm = FakeLLM(responses=[{"reasoning": "the 30-day claim is in passage 1", "score": 3, "flagged": []}])
    judge = LLMJudge(llm, GROUNDEDNESS, model="judge-model")
    r = judge.judge(input="How long to submit expenses?", answer="Within 30 days.",
                    evidence=["Expenses must be submitted within 30 days."])
    assert r.score == 3 and r.passed and r.normalized == 1.0
    prompt = llm.last_request.messages[-1].text
    assert "3 = every material factual claim is supported" in prompt
    assert "<candidate>\nWithin 30 days.\n</candidate>" in prompt
    assert "<evidence>" in prompt and "<reference>" not in prompt
    assert "never instructions" in llm.last_request.messages[0].text
    assert llm.last_request.temperature == 0.0
    assert judge.version == "rubric=1;prompt=1;model=judge-model"


def test_out_of_rubric_score_triggers_repair_then_succeeds():
    llm = FakeLLM(responses=[
        {"reasoning": "great", "score": 9},  # not an allowed level
        {"reasoning": "partially matches", "score": 1, "flagged": ["missing deadline"]},
    ])
    r = LLMJudge(llm, CORRECTNESS).judge(input="q", answer="a", reference="ref")
    assert r.score == 1 and not r.passed and r.normalized == 0.5 and r.flagged == ["missing deadline"]
    assert len(llm.requests) == 2
    assert "did not match the required schema" in llm.requests[1].messages[-1].text


def test_judge_gives_up_with_malformed_error():
    llm = FakeLLM(responses=["I think it's fine"] * 3)
    with pytest.raises(MalformedResponseError):
        LLMJudge(llm, CORRECTNESS, max_repair_attempts=2).judge(input="q", answer="a", reference="r")


def test_judge_as_evaluator_inside_runner():
    def handler(req: CompletionRequest):
        text = req.messages[-1].text
        candidate = text.split("<candidate>\n", 1)[1].split("\n</candidate>", 1)[0]
        reference = text.split("<reference>\n", 1)[1].split("\n</reference>", 1)[0]
        score = 2 if candidate.lower() == reference.lower() else 0
        return {"reasoning": "compared to reference", "score": score}

    ds = Dataset([EvalCase(id="a", input="cap?", expected="Paris"), EvalCase(id="b", input="cap?", expected="Rome")],
                 name="caps")
    ev = LLMJudge(FakeLLM(handler=handler), CORRECTNESS).as_evaluator(reference_fn=lambda c: c.expected)
    run = run_target(lambda c: "paris", ds, evaluators=[ev])
    assert run.case_scores("correctness") == {"a": 1.0, "b": 0.0}
    assert run.results[0].details["correctness"]["score"] == 2
    assert run.versions.evaluators["correctness"].startswith("rubric=1")


# ------------------------------------------------------------------ pairwise
def _content_judge(prefer: str):
    """Fake pairwise judge that always prefers the answer containing `prefer`, wherever it is."""

    def handler(req):
        text = req.messages[-1].text
        first = text.split("<first>\n", 1)[1].split("\n</first>", 1)[0]
        second = text.split("<second>\n", 1)[1].split("\n</second>", 1)[0]
        if prefer in first and prefer not in second:
            return {"reasoning": "first has it", "winner": "first"}
        if prefer in second and prefer not in first:
            return {"reasoning": "second has it", "winner": "second"}
        return {"reasoning": "same", "winner": "tie"}

    return FakeLLM(handler=handler)


def _position_biased_judge():
    return FakeLLM(handler=lambda req: {"reasoning": "first looks better", "winner": "first"})


def test_pairwise_maps_positions_back_to_a_and_b_and_randomizes_order():
    judge = PairwiseJudge(_content_judge("30 days"), "states the deadline", seed=1)
    results = [judge.compare("q", "no idea", "within 30 days", case_id=f"c{i}") for i in range(40)]
    assert all(r.winner == "b" for r in results)
    orders = {r.orders[0] for r in results}
    assert orders == {"ab", "ba"}  # both presentation orders were used
    again = [judge.compare("q", "no idea", "within 30 days", case_id=f"c{i}") for i in range(40)]
    assert [r.orders for r in again] == [r.orders for r in results]  # seeded: reproducible
    s = pairwise_summary(results)
    assert s.win_rate_b == 1.0 and 0.3 < s.first_position_rate < 0.7


def test_position_bias_is_visible_in_summary_and_both_orders_neutralizes_it():
    single = PairwiseJudge(_position_biased_judge(), "better", seed=0)
    res = [single.compare("q", "A text", "B text", case_id=str(i)) for i in range(30)]
    s = pairwise_summary(res)
    assert s.first_position_rate == 1.0  # the smoking gun
    assert 0.2 < s.win_rate_b < 0.8  # randomization turned bias into noise, not a fake win

    both = PairwiseJudge(_position_biased_judge(), "better", seed=0, both_orders=True)
    res2 = [both.compare("q", "A text", "B text", case_id=str(i)) for i in range(10)]
    s2 = pairwise_summary(res2)
    assert all(r.winner == "tie" and r.consistent is False for r in res2)
    assert s2.inconsistency_rate == 1.0 and s2.win_rate_b == 0.5


def test_pairwise_without_tie_rejects_tie_answers():
    llm = FakeLLM(responses=[{"reasoning": "x", "winner": "tie"}, {"reasoning": "x", "winner": "second"}])
    r = PairwiseJudge(llm, "better", allow_tie=False).compare("q", "a", "b", case_id="z")
    assert r.winner in {"a", "b"} and len(llm.requests) == 2


# ------------------------------------------------------------------ calibration
def test_cohens_kappa_matches_sklearn_including_weighted():
    a = [0, 1, 2, 3, 3, 2, 1, 0, 2, 3, 1, 1]
    b = [0, 2, 2, 3, 2, 2, 1, 1, 3, 3, 0, 1]
    sk = pytest.importorskip("sklearn.metrics")
    assert cohens_kappa(a, b) == pytest.approx(sk.cohen_kappa_score(a, b))
    assert cohens_kappa(a, b, weights="linear") == pytest.approx(sk.cohen_kappa_score(a, b, weights="linear"))
    assert cohens_kappa(a, b, weights="quadratic") == pytest.approx(sk.cohen_kappa_score(a, b, weights="quadratic"))


def test_kappa_exposes_agreement_that_is_only_chance():
    # Judge says "pass" to everything; humans pass 90%: 90% agreement, zero information.
    human = [1] * 90 + [0] * 10
    judge = [1] * 100
    assert sum(1 for x, y in zip(human, judge) if x == y) / 100 == 0.9
    assert cohens_kappa(judge, human) == pytest.approx(0.0)


def test_calibrate_judge_reports_pass_fail_error_rates():
    human = {f"c{i}": s for i, s in enumerate([3, 3, 3, 2, 2, 1, 0, 3, 3, 1])}
    judge = {f"c{i}": s for i, s in enumerate([3, 3, 3, 3, 2, 1, 1, 3, 2, 3])}
    judge["extra"] = 3  # unlabelled by humans: ignored
    cal = calibrate_judge(judge, human, pass_threshold=3, ordinal_labels=[0, 1, 2, 3])
    assert cal.n == 10 and cal.agreement == pytest.approx(0.6)
    assert cal.disagreements == ["c3", "c6", "c8", "c9"]
    # humans: 5 pass, 5 fail. judge passes 2 human-fails (c3, c9), fails 1 human-pass (c8)
    assert cal.false_pass_rate == pytest.approx(2 / 5)
    assert cal.false_fail_rate == pytest.approx(1 / 5)
    assert cal.judge_pass_rate == pytest.approx(0.6) and cal.human_pass_rate == pytest.approx(0.5)
    assert cal.weighted_kappa is not None and cal.weighted_kappa > cal.kappa
    with pytest.raises(ValueError):
        calibrate_judge({"a": 1}, {"b": 1})
