# path: book/projects/ragkit/tests/test_rag_eval_metrics.py
from __future__ import annotations

import math

import pytest
from rag_eval_fixtures import EMPLOYEE, case, chunk, output

from ragkit.eval.rag_dataset import RagExpectation, RagOutput
from ragkit.eval.rag_metrics import (
    abstention_cell,
    answer_scores,
    citation_scores,
    context_relevance,
    hit_at_k,
    leak_report,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    retrieval_scores,
)
from ragkit.retrieval.types import Principal


def test_worked_example_two_relevant_one_found_at_rank_two():
    ranked = ["x", "rel1", "y", "z", "w"]
    required = ["rel1", "rel2"]
    assert recall_at_k(ranked, required, 5) == 0.5
    assert precision_at_k(ranked, required, 5) == 0.2
    assert hit_at_k(ranked, required, 5) == 1.0
    assert hit_at_k(ranked, required, 1) == 0.0
    assert reciprocal_rank(ranked, required) == 0.5


def test_precision_divides_by_k_not_by_returned():
    assert precision_at_k(["rel"], ["rel"], 5) == 0.2


def test_recall_undefined_without_required():
    with pytest.raises(ValueError):
        recall_at_k(["a"], [], 5)


def test_ndcg_graded_relevance_by_hand():
    grades = {"req": 2, "acc": 1}
    dcg = (2**1 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3)
    idcg = (2**2 - 1) / math.log2(2) + (2**1 - 1) / math.log2(3)
    assert ndcg_at_k(["acc", "req"], grades, 10) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["req", "acc"], grades, 10) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], grades, 10) == 0.0


def test_ndcg_duplicates_earn_nothing():
    assert ndcg_at_k(["req", "req"], {"req": 2, "acc": 1}, 2) < 1.0


def test_retrieval_scores_dedupe_docs_but_precision_counts_chunks():
    hits = [chunk("pto", 0), chunk("pto", 1), chunk("faq", 0), chunk("other", 0)]
    c = case(required=["pto"], acceptable=["faq"])
    scores = {s.name: s.value for s in retrieval_scores(c, output(hits, packed=2), ks=(3,), ndcg_k=10)}
    assert scores["hit@3"] == 1.0
    assert scores["recall@3"] == 1.0
    assert scores["precision@3"] == 1.0  # pto, pto, faq: all relevant chunks
    assert scores["mrr"] == 1.0
    assert scores["context_relevance"] == 1.0
    assert scores["evidence_packed"] == 1.0
    assert scores["no_permission_leak"] == 1.0


def test_forbidden_case_scores_only_the_leak_check():
    c = case(forbidden=["secret"], abstain=True, tags=["forbidden-doc", "abstain"])
    names = [s.name for s in retrieval_scores(c, output([chunk("faq")], abstained=True))]
    assert names == ["no_permission_leak"]


def test_forbidden_doc_retrieved_is_a_leak_even_if_not_packed():
    c = case(forbidden=["secret"], abstain=True)
    hits = [chunk("faq"), chunk("secret")]
    rep = leak_report(output(hits, packed=1, abstained=True), RagExpectation.model_validate(c.expected), EMPLOYEE)
    assert rep.leaked and rep.forbidden_retrieved == ["secret"] and rep.forbidden_packed == []


def test_acl_violation_detected_without_any_gold_label():
    restricted = chunk("runbook", groups=["it-oncall"])
    other_tenant = chunk("pos", tenant="retail")
    rep = leak_report(output([restricted, other_tenant]), RagExpectation(required_doc_ids=["x"]),
                      Principal(user_id="u", tenant="logistics", groups=["all"]))
    assert set(rep.acl_violations) == {restricted.id, other_tenant.id}
    assert rep.leaked_doc_ids == ["runbook", "pos"]


def test_context_relevance_counts_packed_chunks():
    assert context_relevance(["pto", "faq", "noise", "noise"], {"pto", "faq"}) == 0.5
    assert context_relevance([], {"pto"}) == 0.0


def test_citation_precision_recall_and_validity():
    hits = [chunk("pto"), chunk("faq"), chunk("noise")]
    exp = RagExpectation(required_doc_ids=["pto", "travel"], acceptable_doc_ids=["faq"])
    out = output(hits, packed=2, cited=["pto:c0", "noise:c0"])
    cs = citation_scores(out, exp)
    assert cs.precision == 0.5  # pto relevant, noise not
    assert cs.recall == 0.5  # pto cited, travel not
    assert not cs.valid and cs.invalid_chunk_ids == ["noise:c0"]  # noise was retrieved but never packed


@pytest.mark.parametrize(
    "abstained,expected,cell",
    [(False, False, "answered"), (True, True, "correct-abstain"), (True, False, "false-abstain"), (False, True, "false-answer")],
)
def test_abstention_cells(abstained, expected, cell):
    assert abstention_cell(abstained, expected) == cell


def test_answer_scores_skip_citations_for_abstentions():
    c = case(required=["pto"])
    names = [s.name for s in answer_scores(c, output([chunk("pto")], abstained=True))]
    assert names == ["abstention_correct"]


def test_output_roundtrips_through_json():
    out = output([chunk("pto"), chunk("faq")], packed=1)
    again = RagOutput.coerce(out.model_dump(mode="json"))
    assert again.cited_doc_ids == ["pto"] and again.retrieved_doc_ids == ["pto", "faq"]


def test_from_grounded_qa_adapter_duck_typed():
    from types import SimpleNamespace as NS

    from ragkit.eval.rag_dataset import from_grounded_qa

    hits = [chunk("pto"), chunk("faq")]
    res = output(hits).retrieval
    qa = NS(packed=NS(blocks=[NS(chunk_ids=["pto:c0"], truncated=True)],
                      notes=[NS(kind="dropped_budget", chunk_ids=["faq:c0"])]),
            envelope=NS(text="Ten days carry over [E1].", action="answer", status="answered",
                        citations=[NS(chunk_ids=["pto:c0"])]))
    out = from_grounded_qa(qa, res)
    assert out.answer == "Ten days carry over."
    assert out.packed_chunk_ids == ["pto:c0"] and out.cited_doc_ids == ["pto"] and not out.abstained
    assert out.metadata["truncated_chunk_ids"] == ["pto:c0"]
    assert out.metadata["pack_notes"] == {"dropped_budget": ["faq:c0"]}
    qa.envelope.action = "escalate"
    esc = from_grounded_qa(qa, res)
    assert not esc.abstained and esc.metadata["escalated"]
    qa.envelope.action = "abstain"
    ab = from_grounded_qa(qa, res)
    assert ab.abstained and ab.answer == "" and ab.metadata["abstain_message"] == "Ten days carry over [E1]."
