# path: book/projects/examples/ch10/test_ch10.py
"""Offline tests for the Chapter 10 minimal RAG and its failure catalogue."""
from __future__ import annotations

import pytest
from aie_core.embeddings import FakeEmbeddings
from aie_core.llm.providers import FakeLLM

import failure_modes as fm
from minimal_rag import (
    FIXTURES_DIR, Hit, InMemoryIndex, MinimalRAG, build_default, build_request, chunk_fixed,
    load_docs, pack_evidence, validate_citations,
)

VOCAB = ["refund", "policy", "deadline", "vpn", "token", "reset", "carryover", "days", "laptop", "salary"]


def doc(doc_id: str, body: str, acl: tuple[str, ...] = ("all",)):
    return fm.make_doc(doc_id, body, acl=acl)


# --------------------------------------------------------------------------- chunking
def test_fixed_chunks_cover_text_with_overlap():
    body = "".join(chr(97 + i % 26) for i in range(1000))
    chunks = chunk_fixed(doc("d", body), size=300, overlap=50)
    assert [c.id for c in chunks] == ["d#c0", "d#c1", "d#c2", "d#c3"]
    assert chunks[0].text[-50:] == chunks[1].text[:50]          # overlap is exact
    assert chunks[-1].text.endswith(body[-10:])                  # nothing dropped at the tail
    assert all(len(c.text) <= 300 for c in chunks)


def test_chunk_ids_are_stable_and_carry_metadata():
    d = doc("hr-x", "x" * 500, acl=("hr",))
    a, b = chunk_fixed(d, 200, 20), chunk_fixed(d, 200, 20)
    assert [c.id for c in a] == [c.id for c in b]
    assert a[0].acl_groups == ("hr",) and a[0].version == "1.0"


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_fixed(doc("d", "abc"), size=10, overlap=10)


# --------------------------------------------------------------------------- index
def test_shared_corpus_ingests_with_unique_chunk_ids():
    rag = MinimalRAG(FakeLLM(), FakeEmbeddings(dimensions=256))
    n = rag.ingest(load_docs())
    ids = [c.id for c in rag.index.chunks]
    assert n == len(ids) > 24 and len(set(ids)) == len(ids)


def test_search_ranks_by_similarity_with_vocabulary_embeddings():
    index = InMemoryIndex(FakeEmbeddings(vocabulary=VOCAB))
    index.add(chunk_fixed(doc("vpn", "vpn token reset"), 100, 10)
              + chunk_fixed(doc("refund", "refund policy deadline days"), 100, 10))
    hits = index.search("refund deadline", k=2)
    assert [h.chunk.doc_id for h in hits] == ["refund", "vpn"]
    assert hits[0].score > hits[1].score


def test_acl_filter_happens_inside_retrieval():
    index = InMemoryIndex(FakeEmbeddings(vocabulary=VOCAB))
    index.add(chunk_fixed(doc("bands", "salary salary days", acl=("hr",)), 100, 10)
              + chunk_fixed(doc("pto", "carryover days"), 100, 10))
    assert {h.chunk.doc_id for h in index.search("salary days", k=5)} == {"bands", "pto"}
    assert {h.chunk.doc_id for h in index.search("salary days", k=5, user_groups={"all"})} == {"pto"}
    assert index.search("salary", k=5, user_groups={"nobody"}) == []


# --------------------------------------------------------------------------- pack / generate / validate
def _hits() -> list[Hit]:
    chunks = chunk_fixed(doc("hr-pto-policy", "Carry over up to 10 days."), 100, 10)
    return [Hit(chunks[0], 0.9)]


def test_evidence_is_labeled_and_prompt_has_contract():
    req = build_request("How many days?", _hits())
    user = req.messages[-1].text
    assert '<evidence id="hr-pto-policy#c0"' in user and "</evidence>" in user
    assert user.rstrip().endswith("Question: How many days?")
    assert "INSUFFICIENT_EVIDENCE" in req.messages[0].text and req.temperature == 0.0
    assert pack_evidence([]) == ""


def test_citation_validation_flags_ids_never_shown():
    cited, invalid = validate_citations("Ten [hr-pto-policy#c0]. Also [hr-faq#c3]. Again [hr-pto-policy#c0].", _hits())
    assert cited == ["hr-pto-policy#c0", "hr-faq#c3"]
    assert invalid == ["hr-faq#c3"]


def test_answer_end_to_end_records_citations_and_abstention():
    llm = FakeLLM(responses=["You may carry over 10 days [refund#c0].", "INSUFFICIENT_EVIDENCE"])
    rag = MinimalRAG(llm, FakeEmbeddings(vocabulary=VOCAB), k=1)
    rag.ingest([doc("refund", "refund policy carryover 10 days")])
    first = rag.answer("carryover days")
    assert first.cited_ids == ["refund#c0"] and not first.invalid_citations and not first.abstained
    assert "refund#c0" in llm.last_request.messages[-1].text
    assert rag.answer("vpn token").abstained


def test_default_build_runs_offline_and_cites_the_policy():
    result = build_default().answer("How far in advance must I request a two-week vacation?", user_groups={"all"})
    assert "14 calendar days" in result.text
    assert result.cited_ids and result.cited_ids[0].startswith("hr-pto-policy#")
    assert not result.invalid_citations


def test_fixture_is_hr_only():
    (bands,) = load_docs(FIXTURES_DIR)
    assert bands.acl_groups == ["hr"]


# --------------------------------------------------------------------------- failure catalogue
EXPECTED_FIX = {
    "wrong chunk boundary": True, "missing evidence": None, "distractor outranks evidence": None,
    "stale document version": None, "no abstention": True, "hallucinated citation": None,
    "permission leak": True,
}


@pytest.mark.parametrize("demo", fm.ALL_DEMOS, ids=lambda f: f.__name__)
def test_each_failure_mode_reproduces_and_is_detected(demo):
    case = demo()
    assert case.detected, f"{case.name}: {case.observed}"
    assert case.fixed is EXPECTED_FIX[case.name]


def test_lexical_grounding_check_catches_invented_figures_only():
    assert fm.unsupported_claims("It is 11 weeks [x#c0].", ["valid for 7 years"]) == ["11 weeks"]
    assert fm.unsupported_claims("Use **14 calendar** days.", ["at least 14 calendar days"]) == []


def test_catalogue_covers_every_stage_once():
    stages = [demo().stage for demo in fm.ALL_DEMOS]
    assert len(stages) == len(set(stages)) == 7
