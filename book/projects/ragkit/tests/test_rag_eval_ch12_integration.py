# path: book/projects/ragkit/tests/test_rag_eval_ch12_integration.py
"""Stage isolation on REAL Chapter 12 traces: RetrievalPipeline (BM25 + dense over shared-data,
FakeEmbeddings in vocabulary mode, lexical reranker) and a flat BM25Index retriever.

Offline and deterministic. Forced-label tests first assert their preconditions with
`stage_candidates`, so if Chapter 12's rankings change the failure says which premise broke
rather than reporting a wrong label.
"""
from __future__ import annotations

from functools import lru_cache

import pytest
from retrieval_fixtures import bench

from ragkit.eval.rag_dataset import RagExpectation, RagOutput, expectation, load_gold_dataset, rag_input
from ragkit.eval.run_rag_eval import ExtractiveGenerator, evaluate_system
from ragkit.eval.stage_isolation import FailureStage as S
from ragkit.eval.stage_isolation import diagnose, diagnose_run, stage_lists
from ragkit.retrieval import (
    LexicalOverlapReranker,
    Principal,
    RetrievalPipeline,
    RetrievalQuery,
    RetrievalResult,
    stage_candidates,
)

EMPLOYEE = Principal(user_id="u", tenant="shared", groups=["all"])
VPN_Q = "How do I reset my VPN token?"
PER_DIEM_Q = "What per diem do I get on an international trip?"


def pipeline(**kw) -> RetrievalPipeline:
    b = bench()
    kw.setdefault("candidate_k", 5)
    kw.setdefault("rerank_k", 5)
    kw.setdefault("final_k", 3)
    return RetrievalPipeline({"bm25": b.bm25, "dense": b.dense}, reranker=LexicalOverlapReranker(), parallel=False, **kw)


@lru_cache(maxsize=None)
def corpus() -> dict:
    return {d.id: d for d in bench().docs}


def visible_doc(doc: str, p: Principal) -> bool:
    return doc in corpus() and corpus()[doc].visible_to(p.groups, p.tenant)


def docs_of(ids: list[str]) -> set[str]:
    return {i.rsplit(":", 1)[0] for i in ids}


def retrieve(text: str, principal: Principal = EMPLOYEE, **kw) -> RetrievalResult:
    return pipeline(**kw).retrieve(RetrievalQuery(text=text, principal=principal, k=3))


def label(result: RetrievalResult, required: str, *, packed: int | None = None, principal: Principal = EMPLOYEE):
    hits = [h.chunk for h in result.hits]
    packed_chunks = hits if packed is None else hits[:packed]
    cited = [c.id for c in packed_chunks if c.doc_id == required][:1] or [packed_chunks[0].id]
    out = RagOutput(answer="answer", cited_chunk_ids=cited, packed_chunks=packed_chunks, retrieval=result)
    return diagnose("C", RagExpectation(required_doc_ids=[required]), principal, out, answer_ok=True,
                    corpus=corpus(), doc_visible=visible_doc)


# ----------------------------------------------------------------------------- trace reading
def test_reader_sees_every_pipeline_stage_with_ids():
    r = retrieve(VPN_Q)
    lists = stage_lists(r)
    assert [(s.name, s.kind) for s in lists] == [
        ("bm25#q0", "candidate"), ("dense#q0", "candidate"), ("fusion", "fusion"), ("rerank", "rerank")]
    for s in lists:
        assert s.chunk_ids == stage_candidates(r, s.name)  # same ids Ch 12's own helper returns


# ----------------------------------------------------------------------------- forced labels
def test_ok_on_real_trace():
    r = retrieve(VPN_Q)
    assert "it-vpn-access-runbook" in docs_of(r.chunk_ids)
    assert label(r, "it-vpn-access-runbook").stage == S.OK


def test_not_retrieved_on_real_trace():
    r = retrieve(VPN_Q)
    first_stage = stage_candidates(r, "bm25#q0") + stage_candidates(r, "dense#q0")
    assert "hr-pto-policy" not in docs_of(first_stage)  # precondition
    d = label(r, "hr-pto-policy")
    assert d.stage == S.NOT_RETRIEVED
    assert d.paths[0].ranks == {"bm25#q0": None, "dense#q0": None, "fusion": None, "rerank": None}


def test_dropped_by_rerank_on_real_trace():
    r = retrieve(VPN_Q)
    doc = "it-password-reset-runbook"
    assert doc in docs_of(stage_candidates(r, "fusion")) and doc not in docs_of(stage_candidates(r, "rerank"))
    d = label(r, doc)
    assert d.stage == S.DROPPED_BY_RERANK
    assert d.paths[0].ranks["fusion"] is not None and d.paths[0].ranks["rerank"] is None


def test_dropped_by_fusion_on_real_trace():
    r = retrieve(PER_DIEM_Q)
    doc = "hr-faq"
    assert doc in docs_of(stage_candidates(r, "dense#q0")) and doc not in docs_of(stage_candidates(r, "fusion"))
    assert label(r, doc).stage == S.DROPPED_BY_FUSION


def test_truncated_in_packing_on_real_trace():
    r = retrieve(VPN_Q)
    assert r.doc_ids[0] != "it-faq" and "it-faq" in r.doc_ids
    assert label(r, "it-faq", packed=1).stage == S.TRUNCATED_IN_PACKING


def test_not_in_corpus_and_over_filtering_on_real_trace():
    r = retrieve(VPN_Q)
    assert label(r, "no-such-doc").stage == S.NOT_IN_CORPUS
    # the on-call runbook exists but an ordinary employee may not see it: a permission problem, not retrieval
    assert label(r, "it-incident-response-runbook").stage == S.PERMISSION


# ----------------------------------------------------------------------------- flat single retriever
def test_flat_bm25_index_trace():
    r = bench().bm25.retrieve(RetrievalQuery(text=VPN_Q, principal=EMPLOYEE, k=3))
    assert "stages" not in r.trace and r.trace["stage"] == "bm25"
    assert [(s.name, s.kind) for s in stage_lists(r)] == [("bm25", "candidate")]
    assert label(r, "hr-pto-policy").stage == S.NOT_RETRIEVED
    second = r.doc_ids[1]
    assert label(r, second, packed=1).stage == S.TRUNCATED_IN_PACKING
    assert label(r, r.doc_ids[0]).stage == S.OK


# ----------------------------------------------------------------------------- permissions
class IgnoresAcl:
    """A broken retriever that searches as a superuser: the pipeline's final check must catch it."""

    def __init__(self, inner) -> None:
        self.inner = inner

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        root = Principal(user_id="root", tenant=query.principal.tenant,
                         groups=["all", "it-oncall", "hr", "security", "managers", "retail", "logistics"])
        return self.inner.retrieve(query.model_copy(update={"principal": root}))


def test_final_check_violations_from_a_broken_retriever_are_a_permission_failure():
    b = bench()
    p = RetrievalPipeline({"bm25": IgnoresAcl(b.bm25)}, parallel=False, candidate_k=10, rerank_k=10, final_k=5)
    r = p.retrieve(RetrievalQuery(text="What is the response target for a SEV1 incident?", principal=EMPLOYEE, k=5))
    assert r.trace["acl_violations"], "precondition: the broken retriever surfaced restricted chunks"
    assert all(h.chunk.doc_id != "it-incident-response-runbook" for h in r.hits)  # Ch 12 removed them
    out = RagOutput(abstained=True, packed_chunks=[h.chunk for h in r.hits], retrieval=r)
    exp = RagExpectation(forbidden_doc_ids=["it-incident-response-runbook"], expect_abstain=True)
    d = diagnose("RQ-020", exp, EMPLOYEE, out, corpus=corpus(), doc_visible=visible_doc)
    assert d.stage == S.PERMISSION  # the user saw nothing, but the pre-filter failed: release blocker


# ----------------------------------------------------------------------------- whole gold set
def oracle(result: RetrievalResult, exp: RagExpectation, packed_ids: list[str]) -> S:
    """Independent re-derivation of the retrieval-stage label from Ch 12's own helper."""
    order = [S.NOT_RETRIEVED, S.DROPPED_BY_FUSION, S.DROPPED_BY_RERANK, S.TRUNCATED_IN_PACKING]
    names = [s["name"] for s in result.trace["stages"] if s["kind"] == "retrieve"]
    first = docs_of([cid for n in names for cid in stage_candidates(result, n)])
    fused = docs_of(stage_candidates(result, "fusion"))
    final = docs_of(result.chunk_ids)
    losses = []
    for doc in exp.required_doc_ids:
        if doc not in first:
            losses.append(S.NOT_RETRIEVED)
        elif doc not in fused:
            losses.append(S.DROPPED_BY_FUSION)
        elif doc not in final:
            losses.append(S.DROPPED_BY_RERANK)
        elif doc not in docs_of(packed_ids):
            losses.append(S.TRUNCATED_IN_PACKING)
    return min(losses, key=order.index) if losses else S.OK


@pytest.mark.parametrize("cfg", [dict(candidate_k=30, rerank_k=15, final_k=5), dict(candidate_k=3, rerank_k=3, final_k=1)])
def test_diagnose_run_agrees_with_oracle_on_the_gold_set(cfg):
    ds = load_gold_dataset()
    p = pipeline(**cfg)
    gen = ExtractiveGenerator(pack_k=2)

    def system(question: str, principal: Principal) -> RagOutput:
        return gen.answer(question, p.retrieve(RetrievalQuery(text=question, principal=principal, k=10)))

    outcome = evaluate_system(system, ds, name=f"ch12-{cfg['final_k']}", concurrency=1)
    diagnoses = {d.case_id: d for d in diagnose_run(outcome.run, ds, corpus=corpus(), doc_visible=visible_doc)}
    retrieval_labels = {S.NOT_RETRIEVED, S.DROPPED_BY_FUSION, S.DROPPED_BY_RERANK, S.TRUNCATED_IN_PACKING}
    checked = 0
    for r in outcome.run.results:
        case = ds.get(r.case_id)
        exp = expectation(case)
        out = RagOutput.coerce(r.output)
        assert out.retrieval is not None and out.retrieval.trace["acl_violations"] == []
        assert r.scores["no_permission_leak"] == 1.0  # real ACL pre-filter: zero leaks on all 40 cases
        if not exp.answerable:
            continue
        expected = oracle(out.retrieval, exp, out.packed_chunk_ids)
        got = diagnoses[r.case_id].stage
        if expected in retrieval_labels:
            assert got == expected, (r.case_id, got, expected)
        else:
            assert got not in retrieval_labels, (r.case_id, got)
        checked += 1
    assert checked == 37
    if cfg["final_k"] == 1:
        labels = {d.stage for d in diagnoses.values()}
        assert S.DROPPED_BY_RERANK in labels  # a tight funnel produces real rerank losses
    assert rag_input(ds.get("RQ-020")).principal.groups == ["all"]
