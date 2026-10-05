# path: book/projects/ragkit/tests/test_retrieval_pipeline.py
from __future__ import annotations

import pytest
from retrieval_fixtures import EMPLOYEE, HISTORY_PTO, RETAIL_MANAGER, bench, gold, make_chunk

from aie_core.llm.providers import FakeLLM
from aie_core.observability import InMemoryTracer
from ragkit.retrieval.common import UnknownFilterError
from ragkit.retrieval.pipeline import RetrievalError, RetrievalPipeline, stage_candidates
from ragkit.retrieval.query import HyDEGenerator, MultiQueryExpander, QueryRewriter
from ragkit.retrieval.rerank import LexicalOverlapReranker
from ragkit.retrieval.settings import RetrievalSettings
from ragkit.retrieval.types import RetrievalQuery, RetrievalResult, ScoredChunk


def pipeline(**kw) -> RetrievalPipeline:
    b = bench()
    kw.setdefault("candidate_k", 30)
    kw.setdefault("rerank_k", 15)
    return RetrievalPipeline({"bm25": b.bm25, "dense": b.dense}, **kw)


def test_trace_records_every_stage_with_ids_k_and_latency():
    p = pipeline(reranker=LexicalOverlapReranker(), final_k=5)
    q = RetrievalQuery(text="What does error RET-002 mean in the Returns API?", principal=RETAIL_MANAGER, k=10)
    result = p.retrieve(q)
    names = [s["name"] for s in result.trace["stages"]]
    assert names == ["transform", "bm25#q0", "dense#q0", "fusion", "rerank"]
    assert result.trace["k"] == {"candidate_k": 30, "rerank_k": 15, "final_k": 5}
    for stage in result.trace["stages"][1:3]:
        assert stage["kind"] == "retrieve" and stage["k"] == 30 and 0 < len(stage["candidate_ids"]) <= 30
        assert stage["latency_ms"] >= 0 and stage["query"] == q.text
    assert len(stage_candidates(result, "fusion")) == 15
    assert stage_candidates(result, "rerank") == result.chunk_ids and len(result.hits) == 5
    assert set(result.trace["latency_ms"]) == {"transform", "retrieve", "fusion", "rerank", "total"}
    assert result.trace["latency_ms"]["total"] >= result.trace["latency_ms"]["retrieve"]
    assert result.trace["degraded"] == [] and result.trace["acl_violations"] == []
    assert result.doc_ids[0] == "prod-retail-returns-api" and result.hits[0].stage == "rerank"
    # the reranker's input is a subset of the fused list, which is a subset of the union of retriever lists
    union = set(stage_candidates(result, "bm25#q0")) | set(stage_candidates(result, "dense#q0"))
    assert set(stage_candidates(result, "fusion")) <= union
    assert set(result.chunk_ids) <= set(stage_candidates(result, "fusion"))


def test_without_reranker_returns_fused_order_and_query_k():
    result = pipeline().retrieve(RetrievalQuery(text="VPN error 412", principal=EMPLOYEE, k=4))
    assert len(result.hits) == 4 and all(h.stage == "fusion" for h in result.hits)
    assert result.chunk_ids == stage_candidates(result, "fusion")[:4]


def test_hyde_passages_go_to_dense_only_and_original_is_kept():
    passage = "Unused PTO days may be carried over up to a limit and must be used by a deadline."
    p = pipeline(transformer=HyDEGenerator(FakeLLM(responses=[{"passage": passage}])))
    result = p.retrieve(RetrievalQuery(text="pto rollover?", principal=EMPLOYEE, k=5))
    by_name = {s["name"]: s for s in result.trace["stages"]}
    assert "dense#hyde0" in by_name and "bm25#hyde0" not in by_name
    assert by_name["dense#hyde0"]["query"] == passage and by_name["bm25#q0"]["query"] == "pto rollover?"
    assert result.trace["plan"]["original_text"] == "pto rollover?"
    assert result.query.text == "pto rollover?"


def test_multi_query_fans_out_to_every_retriever():
    llm = FakeLLM(responses=[{"queries": ["PTO carry-over limit", "unused vacation days next year"]}])
    result = pipeline(transformer=MultiQueryExpander(llm, n=2)).retrieve(
        RetrievalQuery(text="How many PTO days carry over?", principal=EMPLOYEE, k=5))
    names = {s["name"] for s in result.trace["stages"] if s["kind"] == "retrieve"}
    assert names == {f"{r}#q{i}" for r in ("bm25", "dense") for i in range(3)}
    assert "hr-pto-policy" in result.doc_ids


def test_reranker_judges_the_rewritten_standalone_question():
    seen = {}

    class Spy:
        def rerank(self, query, candidates, k):
            seen["text"], seen["original"] = query.text, query.original_text
            return candidates[:k]

    llm = FakeLLM(responses=[{"query": "PTO carryover use-by deadline"}])
    pipeline(transformer=QueryRewriter(llm), reranker=Spy()).retrieve(
        RetrievalQuery(text="and by when?", principal=EMPLOYEE, k=3), history=HISTORY_PTO)
    assert seen == {"text": "PTO carryover use-by deadline", "original": "and by when?"}


@pytest.mark.parametrize("expand", [False, True])
def test_gold_forbidden_questions_never_leak_through_the_full_pipeline(expand):
    forbidden = [g for g in gold() if g.forbidden]
    assert len(forbidden) == 3
    for g in forbidden:
        transformer = None
        if expand:  # an expansion that names the restricted runbooks must not get around the ACL
            transformer = MultiQueryExpander(FakeLLM(responses=[{"queries": [
                "incident response runbook SEV1 response target", "database failover runbook RTO",
                "access control policy leaver access disabled hours"]}]), n=3)
        p = pipeline(transformer=transformer, reranker=LexicalOverlapReranker())
        result = p.retrieve(RetrievalQuery(text=g.question, principal=g.principal, k=10))
        every_candidate = {cid for s in result.trace["stages"] for cid in s.get("candidate_ids", [])}
        restricted = {c.id for c in bench().chunks if c.doc_id in g.required_doc_ids}
        assert not every_candidate & restricted, g.id  # not in any stage, not only the final list
        assert result.trace["acl_violations"] == []


def test_final_acl_check_drops_a_leaking_retriever():
    secret = make_chunk("salary bands for engineers", doc_id="hr-comp", acl=["hr"])

    class Leaky:
        def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
            return RetrievalResult(query=query, hits=[ScoredChunk(chunk=secret, score=9, stage="leaky", rank=1)])

    result = RetrievalPipeline({"leaky": Leaky(), "bm25": bench().bm25}).retrieve(
        RetrievalQuery(text="salary bands", principal=EMPLOYEE, k=5))
    assert "hr-comp" not in result.doc_ids and result.trace["acl_violations"] == [secret.id]


def test_failing_retriever_degrades_and_all_failing_raises():
    class Broken:
        def retrieve(self, query):
            raise ConnectionError("vector store unreachable")

    result = RetrievalPipeline({"bm25": bench().bm25, "dense": Broken()}).retrieve(
        RetrievalQuery(text="VPN error 412", principal=EMPLOYEE, k=3))
    assert result.trace["degraded"] == ["retrieve:dense#q0"] and "it-vpn-access-runbook" in result.doc_ids
    errored = [s for s in result.trace["stages"] if "error" in s]
    assert errored[0]["error"].startswith("ConnectionError")
    with pytest.raises(RetrievalError):
        RetrievalPipeline({"a": Broken(), "b": Broken()}).retrieve(RetrievalQuery(text="x", principal=EMPLOYEE))


def test_failing_reranker_keeps_fused_order():
    class Boom:
        def rerank(self, query, candidates, k):
            raise RuntimeError("model OOM")

    result = pipeline(reranker=Boom()).retrieve(RetrievalQuery(text="VPN error 412", principal=EMPLOYEE, k=3))
    assert result.trace["degraded"] == ["rerank:RuntimeError"]
    assert result.chunk_ids == stage_candidates(result, "fusion")[:3]


def test_unknown_filter_raises_before_any_retrieval():
    with pytest.raises(UnknownFilterError):
        pipeline().retrieve(RetrievalQuery(text="x", principal=EMPLOYEE, filters={"tenant": "retail"}))


def test_metadata_filters_reach_every_retriever():
    result = pipeline().retrieve(RetrievalQuery(text="carry over PTO days", principal=EMPLOYEE, k=5,
                                                filters={"tags_any": ["faq"]}))
    assert result.doc_ids[0] == "hr-faq"
    assert all("faq" in h.chunk.metadata["tags"] for h in result.hits)  # nothing outside the filter


def test_tracer_gets_one_span_per_stage_and_job():
    tracer = InMemoryTracer()
    pipeline(tracer=tracer, reranker=LexicalOverlapReranker()).retrieve(
        RetrievalQuery(text="VPN error 412", principal=EMPLOYEE, k=3))
    assert len(tracer.find("retrieval.retrieve")) == 2
    for name in ("retrieval.pipeline", "retrieval.transform", "retrieval.fusion", "retrieval.rerank"):
        assert len(tracer.find(name)) == 1
    assert tracer.find("retrieval.retrieve")[0].attributes["candidates"] > 0


def test_weighted_fusion_option_and_settings(monkeypatch):
    p = pipeline(fusion="weighted", weights={"bm25": 0.7, "dense": 0.3})
    result = p.retrieve(RetrievalQuery(text="NorthGate VPN error 412", principal=EMPLOYEE, k=3))
    assert "weighted" in result.hits[0].signals and "it-vpn-access-runbook" in result.doc_ids
    with pytest.raises(ValueError):
        pipeline(weights={"sparse": 1.0})
    monkeypatch.setenv("RAGKIT_RETRIEVAL_CANDIDATE_K", "7")
    monkeypatch.setenv("RAGKIT_RETRIEVAL_FINAL_K", "2")
    p = RetrievalPipeline.from_settings({"bm25": bench().bm25}, RetrievalSettings())
    result = p.retrieve(RetrievalQuery(text="VPN error 412", principal=EMPLOYEE))
    assert result.trace["k"]["candidate_k"] == 7 and len(result.hits) == 2


class _Slow:
    """A retriever that answers long after any sensible deadline (a hung vector store)."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        import time

        time.sleep(self.seconds)
        return RetrievalResult(query=query, hits=[])


def test_retrieve_deadline_skips_a_hung_retriever_and_keeps_the_others():
    import time

    p = RetrievalPipeline({"bm25": bench().bm25, "dense": _Slow(1.0)}, retrieve_timeout_s=0.2)
    started = time.perf_counter()
    result = p.retrieve(RetrievalQuery(text="VPN error 412", principal=EMPLOYEE, k=3))
    assert time.perf_counter() - started < 0.8  # did not wait for the hung retriever
    assert result.trace["degraded"] == ["retrieve:dense#q0"] and "it-vpn-access-runbook" in result.doc_ids
    timed_out = [s for s in result.trace["stages"] if s.get("timed_out")]
    assert [s["name"] for s in timed_out] == ["dense#q0"] and timed_out[0]["error"].startswith("TimeoutError")
    with pytest.raises(RetrievalError):  # every retriever past the deadline: nothing to rank
        RetrievalPipeline({"a": _Slow(1.0)}, retrieve_timeout_s=0.05, parallel=False).retrieve(
            RetrievalQuery(text="x", principal=EMPLOYEE))


def test_rerank_deadline_keeps_fused_order():
    import time

    class SlowReranker:
        def rerank(self, query, candidates, k):
            time.sleep(1.0)
            return list(candidates)[:k]

    result = pipeline(reranker=SlowReranker(), rerank_timeout_s=0.1).retrieve(
        RetrievalQuery(text="VPN error 412", principal=EMPLOYEE, k=3))
    assert result.trace["degraded"] == ["rerank:TimeoutError"]
    assert result.chunk_ids == stage_candidates(result, "fusion")[:3]


def test_shared_executor_is_used_and_left_open():
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as pool:
        p = pipeline(executor=pool, retrieve_timeout_s=5.0)
        for _ in range(2):  # the pool survives the first request
            result = p.retrieve(RetrievalQuery(text="VPN error 412", principal=EMPLOYEE, k=3))
            assert result.trace["degraded"] == [] and result.hits
        assert pool.submit(lambda: 1).result() == 1
    with pytest.raises(ValueError):
        pipeline(rerank_timeout_s=0)


def test_timeouts_come_from_settings(monkeypatch):
    monkeypatch.setenv("RAGKIT_RETRIEVAL_RETRIEVE_TIMEOUT_S", "0.25")
    monkeypatch.setenv("RAGKIT_RETRIEVAL_RERANK_TIMEOUT_S", "0.12")
    p = RetrievalPipeline.from_settings({"bm25": bench().bm25}, RetrievalSettings())
    assert p.retrieve_timeout_s == 0.25 and p.rerank_timeout_s == 0.12


# ----------------------------------------------------------------------------- diversity (MMR)
def _copies():
    """Three near-copies of the carryover rule and one chunk with the other facet (the deadline)."""
    return [
        make_chunk("Employees may carry over up to 10 unused PTO days into the next year.", doc_id="pto-a"),
        make_chunk("Employees may carry over up to 10 unused PTO days into the next calendar year.", doc_id="pto-b"),
        make_chunk("Up to 10 unused PTO days may be carried over into the next year by employees.", doc_id="pto-c"),
        make_chunk("Carried-over PTO days must be used by 31 March or they expire.", doc_id="pto-deadline"),
    ]


def _ranked(chunks, scores):
    return [ScoredChunk(chunk=c, score=s, stage="rerank", rank=i + 1) for i, (c, s) in enumerate(zip(chunks, scores))]


def test_mmr_replaces_a_near_copy_with_the_missing_facet():
    from ragkit.retrieval import MMRDiversifier

    q = RetrievalQuery(text="How many PTO days carry over and by when must I use them?", principal=EMPLOYEE)
    pool = _ranked(_copies(), [0.95, 0.94, 0.93, 0.80])
    plain = [h.chunk.doc_id for h in pool[:2]]
    picked = MMRDiversifier(lambda_=0.5).rerank(q, pool, 2)
    assert plain == ["pto-a", "pto-b"]  # relevance alone spends both slots on the same fact
    assert [h.chunk.doc_id for h in picked] == ["pto-a", "pto-deadline"]
    assert picked[1].signals["prior_rank"] == 4.0 and picked[1].signals["mmr_redundancy"] < 0.5
    assert all(h.stage == "diversify" for h in picked) and [h.rank for h in picked] == [1, 2]


def test_mmr_lambda_one_is_relevance_order_and_floor_keeps_junk_out():
    from ragkit.retrieval import MMRDiversifier

    q = RetrievalQuery(text="pto carryover", principal=EMPLOYEE)
    junk = make_chunk("The cafeteria opens at 8.", doc_id="junk")
    pool = _ranked(_copies()[:3] + [junk], [0.9, 0.89, 0.88, 0.1])
    assert [h.chunk.doc_id for h in MMRDiversifier(lambda_=1.0).rerank(q, pool, 3)] == ["pto-a", "pto-b", "pto-c"]
    # without a floor, the unrelated chunk is "maximally diverse" and wins a slot
    assert "junk" in [h.chunk.doc_id for h in MMRDiversifier(lambda_=0.3).rerank(q, pool, 2)]
    floored = MMRDiversifier(lambda_=0.3, min_relevance=0.5).rerank(q, pool, 2)
    assert "junk" not in [h.chunk.doc_id for h in floored]
    with pytest.raises(ValueError):
        MMRDiversifier(lambda_=1.5)


def test_mmr_uses_embeddings_when_given():
    from aie_core.embeddings import FakeEmbeddings
    from ragkit.retrieval import MMRDiversifier

    emb = FakeEmbeddings(vocabulary=["carry", "over", "unused", "days", "expire", "march", "used"])
    d = MMRDiversifier(lambda_=0.5, embeddings=emb)
    picked = d.rerank(RetrievalQuery(text="pto", principal=EMPLOYEE), _ranked(_copies(), [0.95, 0.94, 0.93, 0.8]), 2)
    assert d.similarity == "cosine" and picked[1].chunk.doc_id == "pto-deadline"


def test_pipeline_diversify_stage_is_optional_traced_and_degrades():
    from ragkit.retrieval import MMRDiversifier

    q = RetrievalQuery(text="How many PTO days can I carry over and when do they expire?", principal=EMPLOYEE, k=4)
    base = pipeline(reranker=LexicalOverlapReranker()).retrieve(q)
    assert "diversify" not in [s["name"] for s in base.trace["stages"]]  # off by default, trace unchanged
    result = pipeline(reranker=LexicalOverlapReranker(), diversifier=MMRDiversifier(lambda_=0.5),
                      diversify_pool_k=10).retrieve(q)
    by = {s["name"]: s for s in result.trace["stages"]}
    assert by["rerank"]["k"] == 10 and len(by["rerank"]["candidate_ids"]) == 10
    assert by["diversify"]["k"] == 4 and by["diversify"]["candidate_ids"] == result.chunk_ids
    assert set(result.chunk_ids) <= set(by["rerank"]["candidate_ids"]) and len(result.hits) == 4
    assert result.trace["k"]["diversify_pool_k"] == 10 and "diversify" in result.trace["latency_ms"]

    class Broken:
        def rerank(self, query, candidates, k):
            raise RuntimeError("boom")

    degraded = pipeline(reranker=LexicalOverlapReranker(), diversifier=Broken()).retrieve(q)
    assert degraded.trace["degraded"] == ["diversify:RuntimeError"] and len(degraded.hits) == 4


def test_diversity_from_settings(monkeypatch):
    monkeypatch.setenv("RAGKIT_RETRIEVAL_DIVERSITY", "mmr")
    monkeypatch.setenv("RAGKIT_RETRIEVAL_MMR_LAMBDA", "0.6")
    p = RetrievalPipeline.from_settings({"bm25": bench().bm25}, RetrievalSettings())
    assert p.diversifier is not None and p.diversifier.lambda_ == 0.6


def test_mmr_select_in_embedding_space_matches_chapter_8_formula():
    from aie_core.embeddings import FakeEmbeddings
    from ragkit.retrieval import mmr_select

    emb = FakeEmbeddings(vocabulary=["carry", "over", "unused", "days", "expire", "march", "used", "cafeteria"])
    pool = _ranked(_copies() + [make_chunk("The cafeteria opens at 8.", doc_id="junk")], [0.9, 0.9, 0.9, 0.8, 0.1])
    qv = emb.embed_query("how many unused days carry over and when do they expire")
    picked = mmr_select(qv, pool, 2, lambda_=0.5, embeddings=emb)
    assert [h.chunk.doc_id for h in picked][1] == "pto-deadline" and picked[0].stage == "diversify"
    # vectors supplied by the caller give the same result without an embedding call
    vecs = emb.embed([h.chunk.text for h in pool])
    assert [h.chunk.id for h in mmr_select(qv, pool, 2, lambda_=0.5, vectors=vecs)] == [h.chunk.id for h in picked]
    # lambda_=1 is pure cosine relevance; the floor keeps the unrelated chunk out even at low lambda
    assert "junk" not in [h.chunk.doc_id for h in mmr_select(qv, pool, 4, lambda_=0.2, min_relevance=0.1, vectors=vecs)]
    with pytest.raises(ValueError):
        mmr_select(qv, pool, 2)  # neither vectors nor embeddings


def test_parallel_retriever_spans_keep_their_parent_across_the_thread_pool():
    from concurrent.futures import ThreadPoolExecutor

    tracer = InMemoryTracer()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for kw in ({}, {"executor": pool}, {"retrieve_timeout_s": 5.0}):
            tracer = InMemoryTracer()
            with tracer.span("rag.request") as root:
                pipeline(tracer=tracer, **kw).retrieve(RetrievalQuery(text="VPN error 412", principal=EMPLOYEE, k=3))
            outer = tracer.find("retrieval.pipeline")[0]
            jobs = tracer.find("retrieval.retrieve")
            assert len(jobs) == 2
            assert {s.trace_id for s in [outer, *jobs]} == {root.trace_id}  # one trace, not three
            assert all(s.parent_span_id == outer.span_id for s in jobs), kw
