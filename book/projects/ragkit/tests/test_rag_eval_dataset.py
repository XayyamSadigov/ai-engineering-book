# path: book/projects/ragkit/tests/test_rag_eval_dataset.py
from __future__ import annotations

import json

from aie_core.llm.providers import FakeLLM
from evalkit import check_leakage
from rag_eval_fixtures import chunk

from ragkit.eval.rag_dataset import (
    DEFAULT_GOLD_PATH,
    difficulty_of,
    expectation,
    load_gold_dataset,
    rag_input,
    synthesize_questions,
    synthetic_dataset,
)


def test_gold_converts_every_row_with_principal_and_rubric():
    ds = load_gold_dataset()
    assert len(ds) == sum(1 for line in DEFAULT_GOLD_PATH.read_text().splitlines() if line.strip())
    c = ds.get("RQ-001")
    exp = expectation(c)
    assert exp.required_doc_ids == ["hr-pto-policy"] and exp.acceptable_doc_ids == ["hr-faq"]
    assert exp.grades() == {"hr-pto-policy": 2, "hr-faq": 1}
    assert c.rubric and "conflicting-versions" in c.tags
    assert rag_input(ds.get("RQ-027")).principal.tenant == "retail"


def test_forbidden_rows_are_inverted():
    ds = load_gold_dataset()
    exp = expectation(ds.get("RQ-020"))
    assert exp.required_doc_ids == []
    assert exp.forbidden_doc_ids == ["it-incident-response-runbook"]
    assert exp.expect_abstain and not exp.answerable
    assert "it-oncall" not in rag_input(ds.get("RQ-020")).principal.groups


def test_explicit_forbidden_ids_express_answer_but_never_touch():
    """The corrected form of RQ-037: answer from the public FAQ, never retrieve the restricted policy."""
    from ragkit.eval.rag_dataset import gold_row_to_case

    row = {"id": "RQ-037-v2", "question": "Within how many hours must a leaver's access be disabled?",
           "required_doc_ids": ["hr-faq"], "forbidden_doc_ids": ["sec-access-control-policy"],
           "answer_rubric": ["Access is removed within 24 hours of the last working day"],
           "user_groups": ["all"], "tenant": "shared", "tags": ["exact-fact", "restricted-neighbor"]}
    exp = expectation(gold_row_to_case(row))
    assert exp.required_doc_ids == ["hr-faq"] and exp.forbidden_doc_ids == ["sec-access-control-policy"]
    assert not exp.expect_abstain
    # rows without the field convert exactly as before
    assert expectation(load_gold_dataset().get("RQ-037")).forbidden_doc_ids == ["sec-access-control-policy"]


def test_gold_split_by_anchor_document_has_no_group_leakage():
    ds = load_gold_dataset()
    dev, holdout = ds.split(0.3, group_by="anchor_doc", seed="ch14")
    assert len(dev) + len(holdout) == len(ds)
    assert check_leakage(dev, holdout, group_by="anchor_doc").clean


PASSAGE = ("Employees may carry over up to 10 unused PTO days into the next calendar year. "
           "Carried-over days must be used by 31 March.")


def _handler(req):
    text = req.messages[-1].content
    if "Write 2 questions" in text:
        return json.dumps({"questions": [
            {"question": "How many vacation days can I bring into next year?", "answer": "10",
             "evidence_quote": "Employees may carry over up to 10 unused PTO days into the next calendar year."},
            {"question": "How many unused PTO days carry over to the next calendar year?", "answer": "10",
             "evidence_quote": "Employees may carry over up to 10 unused PTO days into the next calendar year."},
            {"question": "What is the parental leave policy for adoptive parents?", "answer": "16 weeks",
             "evidence_quote": "Adoptive parents receive 16 weeks."},
        ]})
    if "Write 1 questions" in text:
        return json.dumps({"questions": [
            {"question": "When do I have to use the days I carried over?", "answer": "31 March",
             "evidence_quote": "Carried-over days must be used by 31 March."}]})
    if "answerable" in text:
        return json.dumps({"answerable": "carried over" not in text.lower(), "answer": "x"})
    raise AssertionError(text)


def test_synthesis_filters_ungrounded_and_tags_difficulty():
    c = chunk("hr-pto-policy", text=PASSAGE)
    rep = synthesize_questions([c], FakeLLM(handler=_handler), per_chunk=2)
    # third question was cut by per_chunk; of the two kept candidates, both grounded and answerable
    assert len(rep.kept) == 2
    tags = {k.question: k.difficulty for k in rep.kept}
    assert tags["How many vacation days can I bring into next year?"] == "paraphrase"
    assert tags["How many unused PTO days carry over to the next calendar year?"] == "lexical"


def test_synthesis_drops_unanswerable_duplicates_and_gold_leaks():
    gold = load_gold_dataset()
    c = chunk("hr-pto-policy", text=PASSAGE)

    def handler(req):
        text = req.messages[-1].content
        if "Write 3 questions" in text:
            return json.dumps({"questions": [
                {"question": "How many unused PTO days can I carry over into next year, and by when?",  # ~ RQ-001
                 "answer": "10", "evidence_quote": "Carried-over days must be used by 31 March."},
                {"question": "Up to how many vacation days roll into January?", "answer": "10",
                 "evidence_quote": "Employees may carry over up to 10 unused PTO days into the next calendar year."},
                {"question": "Up to how many vacation days roll into January then?", "answer": "10",
                 "evidence_quote": "Employees may carry over up to 10 unused PTO days into the next calendar year."},
            ]})
        return json.dumps({"answerable": True, "answer": "10"})

    rep = synthesize_questions([c], FakeLLM(handler=handler), gold=gold, per_chunk=3)
    assert rep.drop_counts() == {"gold-leak": 1, "duplicate": 1}
    assert len(rep.kept) == 1


def test_synthesis_reports_unanswerable_and_ungrounded():
    c = chunk("hr-pto-policy", text=PASSAGE)

    def handler(req):
        text = req.messages[-1].content
        if "Write 2 questions" in text:
            return json.dumps({"questions": [
                {"question": "Does the policy apply to contractors in Germany?", "answer": "no",
                 "evidence_quote": "Employees may carry over up to 10 unused PTO days into the next calendar year."},
                {"question": "What is the stipend for home office equipment?", "answer": "400 USD",
                 "evidence_quote": "The stipend is 400 USD."},
            ]})
        return json.dumps({"answerable": False})

    rep = synthesize_questions([c], FakeLLM(handler=handler), per_chunk=2)
    assert rep.drop_counts() == {"unanswerable": 1, "ungrounded": 1}


def test_synthetic_dataset_uses_chunk_acl_as_principal():
    c = chunk("it-incident-response-runbook", text=PASSAGE, groups=["it-oncall"])
    rep = synthesize_questions([c], FakeLLM(handler=_handler), per_chunk=2)
    ds = synthetic_dataset(rep.kept)
    first = ds.cases[0]
    assert set(rag_input(first).principal.groups) == {"all", "it-oncall"}
    assert expectation(first).required_doc_ids == ["it-incident-response-runbook"]
    assert "synthetic" in first.tags and any(t.startswith("difficulty:") for t in first.tags)


def test_difficulty_reasoning_keyword():
    assert difficulty_of("Why is the FAQ different from the policy?", PASSAGE)[0] == "reasoning"
