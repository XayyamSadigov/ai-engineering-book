# path: book/projects/ragkit/tests/test_rag_eval_stage_isolation.py
from __future__ import annotations

from rag_eval_fixtures import EMPLOYEE, chunk, output, result

from ragkit.eval.rag_dataset import RagExpectation, RagOutput
from ragkit.eval.stage_isolation import FailureStage as S
from ragkit.eval.stage_isolation import diagnose, stage_lists

PTO, FAQ, NOISE, NOISE2 = chunk("pto"), chunk("faq"), chunk("noise"), chunk("noise2")
REQ = RagExpectation(required_doc_ids=["pto"], acceptable_doc_ids=["faq"])
CORPUS = {"pto": 1, "faq": 1, "noise": 1, "noise2": 1, "runbook": 1}


def stages(candidate, fusion=None, rerank=None):
    out = [{"name": "bm25", "kind": "candidate", "chunk_ids": [c.id for c in candidate]}]
    if fusion is not None:
        out.append({"name": "fusion", "kind": "fusion", "chunk_ids": [c.id for c in fusion]})
    if rerank is not None:
        out.append({"name": "rerank", "kind": "rerank", "chunk_ids": [c.id for c in rerank]})
    return out


def dx(out: RagOutput, exp: RagExpectation = REQ, **kw):
    return diagnose("C", exp, EMPLOYEE, out, corpus=CORPUS, **kw)


def test_ok_when_everything_lines_up():
    d = dx(output([PTO, FAQ], stages=stages([PTO, FAQ])), answer_ok=True)
    assert d.stage == S.OK and d.paths[0].ranks == {"bm25": 1} and d.paths[0].cited


def test_not_in_corpus():
    exp = RagExpectation(required_doc_ids=["missing"])
    assert dx(output([NOISE], stages=stages([NOISE])), exp).stage == S.NOT_IN_CORPUS


def test_permission_when_required_doc_not_visible_to_principal():
    exp = RagExpectation(required_doc_ids=["runbook"])
    d = dx(output([NOISE], stages=stages([NOISE])), exp, doc_visible=lambda doc, p: doc != "runbook")
    assert d.stage == S.PERMISSION


def test_permission_leak_overrides_everything():
    secret = chunk("runbook", groups=["it-oncall"])
    d = dx(output([PTO, secret], stages=stages([PTO, secret])), answer_ok=True)
    assert d.stage == S.PERMISSION and "runbook" in d.detail


def test_not_retrieved():
    assert dx(output([NOISE, NOISE2], stages=stages([NOISE, NOISE2]))).stage == S.NOT_RETRIEVED


def test_not_retrieved_without_any_trace():
    assert dx(output([NOISE])).stage == S.NOT_RETRIEVED


def test_dropped_by_fusion():
    out = output([NOISE], stages=stages([NOISE, PTO], fusion=[NOISE]))
    assert dx(out).stage == S.DROPPED_BY_FUSION


def test_dropped_by_rerank():
    out = output([NOISE, NOISE2], stages=stages([NOISE, PTO], fusion=[NOISE, PTO, NOISE2], rerank=[NOISE, NOISE2]))
    d = dx(out)
    assert d.stage == S.DROPPED_BY_RERANK
    assert d.paths[0].ranks == {"bm25": 2, "fusion": 2, "rerank": None}


def test_truncated_in_packing():
    out = output([NOISE, PTO], packed=1, stages=stages([NOISE, PTO]))
    assert dx(out).stage == S.TRUNCATED_IN_PACKING


def test_generation_ignored_evidence_when_judged_wrong_or_false_abstain():
    assert dx(output([PTO], stages=stages([PTO])), answer_ok=False).stage == S.GENERATION_IGNORED_EVIDENCE
    assert dx(output([PTO], abstained=True, stages=stages([PTO]))).stage == S.GENERATION_IGNORED_EVIDENCE


def test_citation_error_when_required_doc_not_cited_or_id_invalid():
    out = output([PTO, FAQ], cited=["faq:c0"], stages=stages([PTO, FAQ]))
    assert dx(out, answer_ok=True).stage == S.CITATION_ERROR
    out = output([PTO, NOISE], packed=1, cited=["pto:c0", "noise:c0"], stages=stages([PTO, NOISE]))
    assert dx(out, answer_ok=True).stage == S.CITATION_ERROR


def test_abstention_expected():
    exp = RagExpectation(forbidden_doc_ids=["runbook"], expect_abstain=True)
    assert dx(output([NOISE], abstained=True), exp).stage == S.OK
    assert dx(output([NOISE]), exp).stage == S.ABSTENTION_MISSED


def test_earliest_loss_wins_with_two_required_docs():
    exp = RagExpectation(required_doc_ids=["pto", "faq"])
    out = output([NOISE, PTO], packed=1, stages=stages([NOISE, PTO]))  # faq never retrieved; pto truncated
    d = dx(out, exp)
    assert d.stage == S.NOT_RETRIEVED
    assert {p.doc_id: p.lost_at for p in d.paths} == {"pto": S.TRUNCATED_IN_PACKING, "faq": S.NOT_RETRIEVED}


def test_flat_trace_layout_is_accepted():
    r = result([NOISE])
    r.trace = {"bm25_ids": ["pto:c0", "noise:c0"], "rerank": ["noise:c0"], "latency_ms": 3.0}
    lists = stage_lists(r)
    assert [(s.name, s.kind) for s in lists] == [("bm25", "candidate"), ("rerank", "rerank")]
    out = RagOutput(answer="x", cited_chunk_ids=["noise:c0"], packed_chunks=[NOISE], retrieval=r)
    assert dx(out).stage == S.DROPPED_BY_RERANK


def test_chapter12_pipeline_trace_layout():
    r = result([NOISE])
    r.trace = {"stages": [
        {"name": "transform", "kind": "transform", "queries": ["q"]},
        {"name": "bm25#q0", "kind": "retrieve", "candidate_ids": ["pto:c0", "noise:c0"]},
        {"name": "dense#q0", "kind": "retrieve", "candidate_ids": ["noise:c0"]},
        {"name": "fusion", "kind": "fusion", "candidate_ids": ["noise:c0", "pto:c0"]},
        {"name": "rerank", "kind": "rerank", "candidate_ids": ["noise:c0"]},
    ]}
    assert [s.kind for s in stage_lists(r)] == ["candidate", "candidate", "fusion", "rerank"]
    out = RagOutput(answer="x", cited_chunk_ids=["noise:c0"], packed_chunks=[NOISE], retrieval=r)
    d = dx(out)
    assert d.stage == S.DROPPED_BY_RERANK and d.paths[0].ranks["bm25#q0"] == 1


def test_partially_truncated_block_blames_packing_not_generation():
    out = output([PTO], abstained=True, stages=stages([PTO]))
    out.metadata["truncated_chunk_ids"] = ["pto:c0"]
    assert dx(out).stage == S.TRUNCATED_IN_PACKING
    out.metadata["truncated_chunk_ids"] = []
    assert dx(out).stage == S.GENERATION_IGNORED_EVIDENCE


def test_chapter12_single_and_hybrid_flat_traces():
    r = result([NOISE])
    r.trace = {"stage": "bm25", "k": 5, "candidate_ids": ["noise:c0", "pto:c0"], "latency_ms": 1.0}
    assert [(s.name, s.kind) for s in stage_lists(r)] == [("bm25", "candidate")]
    r.trace = {"retrievers": {"bm25": {"candidate_ids": ["pto:c0"]}, "dense": {"candidate_ids": ["noise:c0"]}},
               "fused_ids": ["noise:c0"]}
    assert [(s.name, s.kind) for s in stage_lists(r)] == [("bm25", "candidate"), ("dense", "candidate"), ("fusion", "fusion")]
    out = RagOutput(answer="x", cited_chunk_ids=["noise:c0"], packed_chunks=[NOISE], retrieval=r)
    assert dx(out).stage == S.DROPPED_BY_FUSION


def test_final_check_acl_violations_in_trace_count_as_permission_failure():
    r = result([PTO])
    r.trace = {"stages": [{"name": "bm25#q0", "kind": "retrieve", "candidate_ids": ["pto:c0", "runbook:c0"]}],
               "acl_violations": ["runbook:c0"]}
    out = RagOutput(answer="x", cited_chunk_ids=["pto:c0"], packed_chunks=[PTO], retrieval=r)
    d = dx(out, answer_ok=True)
    assert d.stage == S.PERMISSION and "runbook" in d.detail


def test_diversify_stage_counts_as_part_of_the_precision_stage():
    from ragkit.eval.stage_isolation import stage_lists
    from ragkit.retrieval.types import Principal, RetrievalQuery, RetrievalResult

    q = RetrievalQuery(text="q", principal=Principal(user_id="u", tenant="shared", groups=["all"]))
    trace = {"stages": [{"name": "bm25#q0", "kind": "retrieve", "candidate_ids": ["a:1", "b:1"]},
                        {"name": "fusion", "kind": "fusion", "candidate_ids": ["a:1", "b:1"]},
                        {"name": "rerank", "kind": "rerank", "candidate_ids": ["a:1", "b:1"]},
                        {"name": "diversify", "kind": "diversify", "candidate_ids": ["a:1"]}]}
    kinds = {s.name: s.kind for s in stage_lists(RetrievalResult(query=q, hits=[], trace=trace))}
    assert kinds["diversify"] == "rerank" and kinds["bm25#q0"] == "candidate"
