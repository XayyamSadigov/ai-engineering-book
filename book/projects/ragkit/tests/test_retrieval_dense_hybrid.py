# path: book/projects/ragkit/tests/test_retrieval_dense_hybrid.py
from __future__ import annotations

import pytest
from retrieval_fixtures import EMPLOYEE, RETAIL_MANAGER, bench, make_chunk
from semsearch.adapters import NumpyVectorStore

from aie_core.embeddings import FakeEmbeddings
from ragkit.retrieval.dense import DenseRetriever
from ragkit.retrieval.hybrid import HybridRetriever, min_max, reciprocal_rank_fusion, weighted_score_fusion
from ragkit.retrieval.types import Principal, RetrievalQuery, ScoredChunk

VOCAB = ["vpn", "error", "token", "laptop", "return", "window", "pto", "carryover", "days", "policy"]


def sc(chunk, score: float, rank: int, stage: str = "x") -> ScoredChunk:
    return ScoredChunk(chunk=chunk, score=score, stage=stage, rank=rank)


# ----------------------------------------------------------------------------- RRF and weighted fusion
@pytest.fixture
def lists():
    x, y, z, w = (make_chunk(t, doc_id=t) for t in ("x", "y", "z", "w"))
    a = [sc(x, 12.0, 1), sc(y, 7.0, 2), sc(z, 1.0, 3)]
    b = [sc(y, 0.91, 1), sc(w, 0.42, 2)]
    return {"bm25": a, "dense": b}, (x, y, z, w)


def test_rrf_math(lists):
    rankings, (x, y, z, w) = lists
    fused = reciprocal_rank_fusion(rankings, k=60)
    assert [h.chunk.id for h in fused] == [y.id, x.id, w.id, z.id]
    by_id = {h.chunk.id: h for h in fused}
    assert by_id[y.id].score == pytest.approx(1 / 62 + 1 / 61)
    assert by_id[x.id].score == pytest.approx(1 / 61)
    assert by_id[w.id].score == pytest.approx(1 / 62)
    assert by_id[z.id].score == pytest.approx(1 / 63)
    assert by_id[y.id].signals["bm25_rank"] == 2 and by_id[y.id].signals["dense_rank"] == 1
    assert by_id[y.id].signals["dense_score"] == pytest.approx(0.91)
    assert all(h.stage == "fusion" for h in fused) and [h.rank for h in fused] == [1, 2, 3, 4]


def test_rrf_weights_k_and_limit(lists):
    rankings, (x, y, z, w) = lists
    fused = reciprocal_rank_fusion(rankings, k=0, weights={"dense": 0.0}, limit=2)
    assert [h.chunk.id for h in fused] == [x.id, y.id]  # dense ignored; with k=0 rank 1 scores 1.0
    assert fused[0].score == pytest.approx(1.0)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(rankings, weights={"sparse": 1.0})


def test_rrf_counts_a_repeated_chunk_once_per_list():
    c = make_chunk("dup", doc_id="d")
    fused = reciprocal_rank_fusion({"multi": [sc(c, 1.0, 1), sc(c, 0.9, 2)]}, k=60)
    assert len(fused) == 1 and fused[0].score == pytest.approx(1 / 61)


def test_rrf_ties_break_by_best_rank_then_id():
    a, b = make_chunk("a", doc_id="a"), make_chunk("b", doc_id="b")
    fused = reciprocal_rank_fusion({"l1": [sc(a, 1, 1), sc(b, 1, 2)], "l2": [sc(b, 1, 1), sc(a, 1, 2)]})
    assert fused[0].score == fused[1].score
    assert [h.chunk.id for h in fused] == sorted([a.id, b.id])


def test_min_max_edges():
    assert min_max([]) == []
    assert min_max([3.0]) == [1.0]
    assert min_max([2.0, 2.0]) == [1.0, 1.0]
    assert min_max([10.0, 5.0, 0.0]) == [1.0, 0.5, 0.0]


def test_weighted_fusion(lists):
    rankings, (x, y, z, w) = lists
    fused = weighted_score_fusion(rankings)
    by_id = {h.chunk.id: h for h in fused}
    assert fused[0].chunk.id == y.id and by_id[y.id].score == pytest.approx(10 / 12 * 0 + (7 - 1) / 11 + 1.0)
    assert by_id[x.id].score == pytest.approx(1.0)
    assert by_id[y.id].signals["bm25_norm"] == pytest.approx(6 / 11, abs=1e-6)
    heavy_dense = weighted_score_fusion(rankings, weights={"bm25": 0.1, "dense": 1.0})
    assert heavy_dense[0].chunk.id == y.id and heavy_dense[1].chunk.id == x.id


# ----------------------------------------------------------------------------- dense retriever
def _dense(chunks, **kw) -> DenseRetriever:
    d = DenseRetriever(FakeEmbeddings(vocabulary=VOCAB), **kw)
    d.index(chunks)
    return d


def test_dense_namespace_and_dimension_check():
    d = DenseRetriever(FakeEmbeddings(vocabulary=VOCAB), index_version="v7")
    assert d.namespace == "northwind:fake-embedding:v7"
    with pytest.raises(ValueError, match="dimensions"):
        DenseRetriever(FakeEmbeddings(vocabulary=VOCAB), NumpyVectorStore(dimensions=3))


def test_dense_ranks_by_similarity_and_prefilters_acl_and_tenant():
    chunks = [
        make_chunk("vpn error token", doc_id="vpn"),
        make_chunk("laptop return window days", doc_id="laptop"),
        make_chunk("vpn error token reset secret", doc_id="secret", acl=["it-oncall"]),
        make_chunk("return window days", doc_id="retail", tenant="retail"),
    ]
    d = _dense(chunks)
    hits = d.search("vpn error", EMPLOYEE, k=10)
    assert hits[0].chunk.doc_id == "vpn" and "secret" not in {h.chunk.doc_id for h in hits}
    assert "retail" not in {h.chunk.doc_id for h in hits}
    assert "retail" in {h.chunk.doc_id for h in d.search("return window", RETAIL_MANAGER, k=10)}
    assert hits[0].signals["dense"] == pytest.approx(hits[0].score)


def test_dense_document_level_filters_and_min_score():
    chunks = [
        make_chunk("pto carryover days policy", doc_id="policy", tags=["policy"], updated_at="2026-01-01"),
        make_chunk("pto carryover days", doc_id="faq", tags=["faq"], updated_at="2025-06-10"),
        make_chunk("vpn token", doc_id="vpn"),
    ]
    d = _dense(chunks, min_score=0.1)
    assert [h.chunk.doc_id for h in d.search("pto carryover", EMPLOYEE, 10, {"tags_any": ["faq"]})] == ["faq"]
    assert [h.chunk.doc_id for h in d.search("pto carryover", EMPLOYEE, 10, {"updated_after": "2025-12-01"})] == ["policy"]
    assert d.search("pto", EMPLOYEE, 10, {"doc_ids": ["nope"]}) == []
    assert "vpn" not in {h.chunk.doc_id for h in d.search("pto carryover", EMPLOYEE, 10)}  # below min_score


def test_dense_drops_forbidden_results_from_a_broken_store():
    class LeakyStore(NumpyVectorStore):
        def search(self, namespace, query, k, flt=None, *, exact=False):
            return super().search(namespace, query, k, None)  # ignores the filter: a store bug

    chunks = [make_chunk("vpn error token", doc_id="public"), make_chunk("vpn error token", doc_id="secret", acl=["hr"])]
    d = DenseRetriever(FakeEmbeddings(vocabulary=VOCAB), LeakyStore(len(VOCAB)))
    d.index(chunks)
    result = d.retrieve(RetrievalQuery(text="vpn error", principal=EMPLOYEE, k=5))
    assert result.doc_ids == ["public"] and result.trace["acl_dropped"] == 1


def test_dense_reindex_replaces_old_version_and_persists(tmp_path):
    d = _dense([make_chunk("vpn error", doc_id="vpn", version="1"), make_chunk("laptop", doc_id="lap")])
    d.index([make_chunk("vpn token", doc_id="vpn", version="2", position=1)])
    hits = d.search("vpn", EMPLOYEE, 10)
    assert [h.chunk.version for h in hits if h.chunk.doc_id == "vpn"] == ["2"]
    d.save(tmp_path)
    loaded = DenseRetriever.load(tmp_path, FakeEmbeddings(vocabulary=VOCAB))
    assert [h.chunk.id for h in loaded.search("vpn token", EMPLOYEE, 3)] == [h.chunk.id for h in d.search("vpn token", EMPLOYEE, 3)]
    with pytest.raises(ValueError, match="re-embed"):
        DenseRetriever.load(tmp_path, FakeEmbeddings(vocabulary=VOCAB, model="other-model"))
    with pytest.raises(ValueError):
        d.index([make_chunk("a", doc_id="mixed", version="1"), make_chunk("b", doc_id="mixed", version="2", position=1)])


# ----------------------------------------------------------------------------- hybrid retriever
def test_hybrid_retriever_fuses_and_traces_each_retriever():
    b = bench()
    hybrid = HybridRetriever({"bm25": b.bm25, "dense": b.dense}, candidate_k=20)
    q = RetrievalQuery(text="What does error RET-002 mean in the Returns API?", principal=RETAIL_MANAGER, k=5)
    result = hybrid.retrieve(q)
    assert result.doc_ids[0] == "prod-retail-returns-api"
    assert set(result.trace["retrievers"]) == {"bm25", "dense"}
    assert result.trace["retrievers"]["bm25"]["k"] == 20 and len(result.hits) == 5
    top = result.hits[0]
    assert top.stage == "fusion" and "bm25_rank" in top.signals and "dense_rank" in top.signals
    sequential = HybridRetriever({"bm25": b.bm25, "dense": b.dense}, candidate_k=20, parallel=False).retrieve(q)
    assert sequential.chunk_ids == result.chunk_ids


def test_fusion_recovers_the_identifier_dense_misranks():
    b = bench()
    q = RetrievalQuery(text="INC-2025-1142", principal=RETAIL_MANAGER, k=5)
    dense_top = b.dense.retrieve(q).doc_ids[0]
    fused = HybridRetriever({"bm25": b.bm25, "dense": b.dense}, candidate_k=20).retrieve(q)
    assert dense_top != "inc-2025-11-pos-outage"
    assert fused.doc_ids[0] == "inc-2025-11-pos-outage"


def test_filters_without_a_catalog_fail_loudly():
    d = DenseRetriever(FakeEmbeddings(vocabulary=VOCAB))
    with pytest.raises(RuntimeError, match="catalog"):
        d.search("vpn", EMPLOYEE, 5, {"tags_any": ["policy"]})


def test_records_carry_flat_fields_for_sql_lexical_search():
    from ragkit.retrieval.common import extract_identifiers

    assert extract_identifiers("See INC-2025-1142 and RET-002, not well-known or 2025.") == ["INC-2025-1142", "RET-002"]
    c = make_chunk("Error RET-002 means expired.", doc_id="api", title="Returns API", section=["Returns API", "Errors"])
    d = _dense([c])
    rec = d.store._ns[d.namespace].records[c.id]
    assert rec.metadata["identifiers"] == "RET-002" and rec.metadata["breadcrumb"] == "Returns API > Errors"
    assert rec.metadata["title"] == "Returns API" and rec.metadata["chunk"]["id"] == c.id
