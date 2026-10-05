# path: book/projects/examples/ch25/tests/test_text_tasks.py
"""Summarization, RAG answers, prompts, and tool use."""
from __future__ import annotations

from aie_core import FakeLLM
from evalkit import EvalCase
from evalkit.runner import coerce_scores

from taskevals.prompts import PromptContractEvaluator, reply_judge
from taskevals.rag import RagAnswerEvaluator, claim_support, judge_evaluators
from taskevals.summarization import FAITHFULNESS, SummaryEvaluator, compression_ratio
from taskevals.tools import ToolUseEvaluator

INCIDENT = (
    "On 12 November 2025 between 08:05 and 09:40, card payments failed at 37 retail stores. "
    "The root cause was an expired TLS certificate on the PayBridge adapter, which the store servers "
    "use to reach the payment provider. Cash payments were unaffected. The certificate was renewed at 09:32 "
    "and the adapters were restarted by 09:40. Monitoring did not alert because the health check used plain HTTP. "
    "Action items: alert on certificate expiry 30 days ahead, except for internal test environments, and switch "
    "the health check to HTTPS. Owner: Retail Platform team. Incident id INC-2025-11-POS."
) * 2

SUMMARY_CASE = EvalCase(
    id="S1", input={"source": INCIDENT},
    expected={"key_facts": [["expired TLS certificate", "expired certificate"], ["37 retail stores", "37 stores"],
                            ["health check used plain HTTP", "health check"], ["30 days"]],
              "must_keep": ["except for internal test environments"], "compression": [0.05, 0.6]},
)


def _scores(ev, case, out):
    return {s.name: s for s in ev(case, out)}


def test_good_summary_passes_all_summary_metrics() -> None:
    summary = ("Card payments failed at 37 retail stores because of an expired TLS certificate on the PayBridge "
               "adapter. The health check used plain HTTP, so monitoring did not alert. Alert on certificate expiry "
               "30 days ahead, except for internal test environments.")
    s = _scores(SummaryEvaluator(), SUMMARY_CASE, summary)
    assert all(x.passed for x in s.values()), {k: (v.value, v.detail) for k, v in s.items()}


def test_invented_number_breaks_faithfulness_and_dropped_qualifier_is_flagged() -> None:
    summary = ("Card payments failed at 52 retail stores because of an expired TLS certificate. The health check "
               "used plain HTTP. Alert on certificate expiry 30 days ahead.")
    s = _scores(SummaryEvaluator(), SUMMARY_CASE, summary)
    assert not s["summary_faithfulness"].passed and "52" in s["summary_faithfulness"].detail["unsupported"][0]
    assert not s["summary_qualifiers"].passed
    assert not s["summary_coverage"].passed  # "37 stores" is gone


def test_copying_the_source_is_not_a_summary() -> None:
    s = _scores(SummaryEvaluator(), SUMMARY_CASE, INCIDENT)
    assert s["summary_coverage"].passed and s["summary_faithfulness"].passed
    assert not s["summary_compression_ok"].passed and compression_ratio(INCIDENT, INCIDENT) == 1.0


def test_faithfulness_rubric_is_a_valid_evalkit_rubric() -> None:
    assert FAITHFULNESS.scores == [0, 1, 2] and FAITHFULNESS.pass_threshold == 2


EVIDENCE = [
    {"id": "hr-pto-policy", "text": "From 1 January 2026, employees may carry over up to 10 unused PTO days into "
                                    "the next calendar year. Carried-over days must be used by 31 March."},
    {"id": "hr-faq", "text": "You can carry over up to 5 days of PTO."},
]
RAG_CASE = EvalCase(id="R1", input={"question": "How many PTO days can I carry over into next year?",
                                    "evidence": EVIDENCE},
                    expected={"answerable": True, "required_sources": ["hr-pto-policy"]})


def test_supported_answer_with_valid_citation_passes() -> None:
    out = {"answer": "You can carry over up to 10 unused PTO days into the next year. They must be used by "
                     "31 March. [hr-pto-policy]", "citations": ["hr-pto-policy"]}
    s = _scores(RagAnswerEvaluator(), RAG_CASE, out)
    assert all(x.passed for x in s.values()), {k: (v.value, v.detail) for k, v in s.items()}


def test_invented_deadline_is_unsupported_and_wrong_citation_is_flagged() -> None:
    out = {"answer": "You can carry over up to 10 unused PTO days. They must be used by 30 June.",
           "citations": ["hr-travel-policy"]}
    s = _scores(RagAnswerEvaluator(), RAG_CASE, out)
    assert s["rag_faithfulness"].value == 0.5
    assert s["rag_faithfulness"].detail[0]["unsupported_atoms"] == ["30"]
    assert not s["rag_citations_valid"].passed


def test_abstention_is_correct_only_when_evidence_cannot_answer() -> None:
    unanswerable = EvalCase(id="R2", input={"question": "What is the CEO's salary?", "evidence": EVIDENCE},
                            expected={"answerable": False})
    abstain = {"answer": "INSUFFICIENT EVIDENCE: the documents do not say.", "citations": []}
    assert _scores(RagAnswerEvaluator(), unanswerable, abstain)["rag_abstention_correct"].passed
    assert not _scores(RagAnswerEvaluator(), RAG_CASE, abstain)["rag_abstention_correct"].passed


def test_claim_support_picks_the_passage_that_contains_the_numbers() -> None:
    claims = claim_support("Employees may carry over up to 10 unused PTO days.", EVIDENCE)
    assert claims[0]["passage"] == "hr-pto-policy" and claims[0]["supported"]


def test_rag_judges_delimit_evidence_and_parse_verdicts() -> None:
    llm = FakeLLM(handler=lambda req: {"reasoning": "claims supported", "score": 3 if "groundedness" in
                                       req.messages[-1].text else 2, "flagged": []})
    out = {"answer": "Up to 10 days [hr-pto-policy]", "citations": ["hr-pto-policy"]}
    scores = [s for ev in judge_evaluators(llm) for s in coerce_scores(ev.name, ev(RAG_CASE, out))]
    assert all(s.passed for s in scores)
    assert "<evidence>" in llm.requests[0].messages[-1].text and "[hr-pto-policy]" in llm.requests[0].messages[-1].text


def test_prompt_contract_and_reply_judge() -> None:
    case = EvalCase(id="P1", input={"request": "How do I reset my password?"},
                    expected={"required": ["id.northwind.example/reset"], "forbidden": ["password:"],
                              "max_chars": 300})
    good = "Reset it at id.northwind.example/reset, then update your saved keychain entry."
    bad = "Your new password: hunter2"
    assert all(s.passed for s in PromptContractEvaluator()(case, good))
    failed = {s.name for s in PromptContractEvaluator()(case, bad) if not s.passed}
    assert failed == {"prompt_required", "prompt_forbidden"}
    judge = reply_judge(FakeLLM(handler=lambda req: {"reasoning": "concrete step", "score": 2, "flagged": []}))
    assert coerce_scores(judge.name, judge(case, good))[0].passed


def test_tool_use_selection_hallucination_and_arguments() -> None:
    case = EvalCase(id="T", input={"request": "urgent laptop", "tools": ["create_ticket"]},
                    expected={"tool": "create_ticket", "arguments": {"priority": "P1"}})
    ev = ToolUseEvaluator()
    ok = [{"id": "1", "name": "create_ticket",
           "arguments": {"subject": "Laptop broken", "body": "Urgent: laptop will not boot.", "category": "hardware",
                         "priority": "P1"}}]
    assert all(s.passed for s in ev(case, ok))
    invalid = [{"id": "1", "name": "create_ticket",
                "arguments": {"subject": "Laptop broken", "body": "Urgent: laptop will not boot.", "category": "hardware",
                              "priority": "high"}}]
    s = _scores(ev, case, invalid)
    assert not s["tool_args_valid"].passed and not s["tool_args_correct"].passed
    hallucinated = [{"id": "1", "name": "close_ticket", "arguments": {}}]
    s = _scores(ev, case, hallucinated)
    assert not s["tool_known"].passed and not s["tool_selected_correctly"].passed
    no_tool_case = EvalCase(id="N", input={"request": "thanks", "tools": ["create_ticket"]}, expected={"tool": None})
    assert all(x.passed for x in ev(no_tool_case, "You're welcome."))
