# path: book/projects/ragkit/tests/test_retrieval_bm25.py
from __future__ import annotations

import math

import pytest
from retrieval_fixtures import EMPLOYEE, LOGISTICS_MANAGER, ONCALL, RETAIL_MANAGER, bench, make_chunk

from ragkit.retrieval.bm25 import BM25Index, BM25Tokenizer
from ragkit.retrieval.common import UnknownFilterError
from ragkit.retrieval.types import Principal, RetrievalQuery


def test_tokenizer_keeps_identifiers_whole_and_split():
    tok = BM25Tokenizer()
    assert tok.tokenize("Root cause of INC-2025-1142?") == ["root", "cause", "inc-2025-1142", "inc", "2025", "1142"]
    assert tok.tokenize("The laptops and policies") == ["laptop", "policy"]
    assert tok.tokenize("Error RET-002 on /v2/returns") == ["error", "ret-002", "ret", "002", "v2/returns", "v2", "return"]


def test_bm25_scores_match_the_formula():
    idx = BM25Index(k1=1.2, b=0.75)
    c1, c2, c3 = make_chunk("apple banana", doc_id="d1"), make_chunk("apple apple cherry", doc_id="d2"), make_chunk("durian", doc_id="d3")
    idx.add([c1, c2, c3])
    assert idx.avgdl == pytest.approx(2.0)
    idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))

    def expected(tf: int, dl: int) -> float:
        return idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * dl / 2.0))

    hits = idx.search("apple", EMPLOYEE, k=10)
    assert [h.chunk.id for h in hits] == [c2.id, c1.id]
    assert hits[0].score == pytest.approx(expected(2, 3))
    assert hits[1].score == pytest.approx(expected(1, 2))
    assert hits[0].rank == 1 and hits[0].stage == "bm25" and "bm25" in hits[0].signals
    assert idx.explain("apple durian", c2.id) == {"apple": pytest.approx(expected(2, 3), abs=1e-6)}


def test_b_zero_disables_length_normalization():
    idx = BM25Index(b=0.0)
    short, long_ = make_chunk("vpn", doc_id="s"), make_chunk("vpn " + "filler " * 50, doc_id="l")
    idx.add([short, long_])
    s, l_ = idx.search("vpn", EMPLOYEE)
    assert s.score == pytest.approx(l_.score)


def test_forbidden_chunks_are_never_scored_even_when_they_are_the_only_match():
    idx = BM25Index()
    secret = make_chunk("salary band senior engineer 104000", doc_id="hr-comp", acl=["hr"])
    public = make_chunk("office opening hours", doc_id="pub")
    idx.add([secret, public])
    assert idx.search("salary band", EMPLOYEE) == []
    assert secret.id not in idx.allowed_ids(EMPLOYEE)
    hr = Principal(user_id="h", tenant="shared", groups=["all", "hr"])
    assert idx.search("salary band", hr)[0].chunk.id == secret.id


def test_tenant_isolation():
    idx = BM25Index()
    retail = make_chunk("returns window thirty days", doc_id="r", tenant="retail")
    logistics = make_chunk("returns to depot window", doc_id="l", tenant="logistics")
    idx.add([retail, logistics])
    assert [h.chunk.doc_id for h in idx.search("returns window", RETAIL_MANAGER)] == ["r"]
    assert [h.chunk.doc_id for h in idx.search("returns window", LOGISTICS_MANAGER)] == ["l"]
    assert idx.search("returns window", EMPLOYEE) == []  # a "shared" user sees only shared docs


def test_metadata_filters_and_unknown_filter_fails_loudly():
    idx = BM25Index()
    old = make_chunk("pto carryover five days", doc_id="faq", tags=["hr", "faq"], updated_at="2025-06-10")
    new = make_chunk("pto carryover ten days", doc_id="policy", tags=["hr", "policy"], updated_at="2026-01-01")
    idx.add([old, new])
    assert [h.chunk.doc_id for h in idx.search("pto carryover", EMPLOYEE, filters={"tags_any": ["policy"]})] == ["policy"]
    assert [h.chunk.doc_id for h in idx.search("pto carryover", EMPLOYEE, filters={"updated_after": "2025-12-31"})] == ["policy"]
    assert [h.chunk.doc_id for h in idx.search("pto", EMPLOYEE, filters={"exclude_doc_ids": ["policy"]})] == ["faq"]
    with pytest.raises(UnknownFilterError):
        idx.search("pto", EMPLOYEE, filters={"tag_any": ["policy"]})  # typo must not widen results


def test_replace_and_delete_document_keep_statistics_consistent():
    idx = BM25Index()
    idx.add([make_chunk("alpha beta", doc_id="d1", position=0), make_chunk("gamma", doc_id="d2")])
    idx.replace_document("d1", [make_chunk("alpha beta delta epsilon", doc_id="d1", position=0, version="2")])
    assert len(idx) == 2 and idx.avgdl == pytest.approx((4 + 1) / 2)
    assert idx.delete_document("d1") == 1
    assert len(idx) == 1 and idx.search("alpha", EMPLOYEE) == [] and "alpha" not in idx._postings


def test_persistence_round_trip_and_tokenizer_fingerprint(tmp_path):
    idx = bench().bm25
    path = idx.save(tmp_path / "bm25.json")
    loaded = BM25Index.load(path)
    q = "How many failed login attempts lock an account?"
    assert [h.chunk.id for h in loaded.search(q, EMPLOYEE, 5)] == [h.chunk.id for h in idx.search(q, EMPLOYEE, 5)]
    with pytest.raises(ValueError, match="tokenizer"):
        BM25Index.load(path, tokenizer=BM25Tokenizer(fold_plurals=False))


def test_bm25_finds_exact_identifiers_that_dense_misses():
    b = bench()
    # Incident ID: BM25 puts the incident report first; dense prefers a document that merely mentions it.
    q = RetrievalQuery(text="INC-2025-1142", principal=RETAIL_MANAGER, k=5)
    lexical, dense = b.bm25.retrieve(q), b.dense.retrieve(q)
    assert lexical.doc_ids[0] == "inc-2025-11-pos-outage"
    assert dense.doc_ids[0] != "inc-2025-11-pos-outage"
    # Error code: dense returns k plausible-looking results with no relation to the code.
    q = RetrievalQuery(text="SH-201", principal=LOGISTICS_MANAGER, k=5)
    assert "SH-201" in b.bm25.retrieve(q).hits[0].chunk.text
    dense_hits = b.dense.retrieve(q).hits
    assert len(dense_hits) == 5 and not any("SH-201" in h.chunk.text for h in dense_hits)


def test_gold_forbidden_questions_never_return_restricted_docs_in_bm25():
    from retrieval_fixtures import gold

    for g in [g for g in gold() if g.forbidden]:
        result = bench().bm25.retrieve(RetrievalQuery(text=g.question, principal=g.principal, k=50))
        assert not set(g.required_doc_ids) & set(result.doc_ids), g.id
    # the same question under an authorized identity does find the runbook
    ok = bench().bm25.retrieve(RetrievalQuery(text="What is the response target for a SEV1 incident?", principal=ONCALL, k=3))
    assert "it-incident-response-runbook" in ok.doc_ids


def test_get_chunks_by_id_is_acl_checked_and_ordered():
    open_c = make_chunk("VPN error 412 means the certificate expired", doc_id="it-vpn")
    secret = make_chunk("salary bands for engineers", doc_id="hr-comp", acl=["hr"])
    ix = BM25Index()
    ix.add([open_c, secret])
    assert ix.get_chunks([secret.id, "missing:0", open_c.id], EMPLOYEE) == [open_c]  # forbidden == absent
    hr = Principal(user_id="h", tenant="shared", groups=["all", "hr"])
    assert [c.id for c in ix.get_chunks([secret.id, open_c.id], hr)] == [secret.id, open_c.id]
    assert ix.doc_ids() == {"it-vpn", "hr-comp"}
    ix.delete_document("hr-comp")
    assert ix.doc_ids() == {"it-vpn"}
