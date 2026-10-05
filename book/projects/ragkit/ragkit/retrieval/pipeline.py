# path: book/projects/ragkit/ragkit/retrieval/pipeline.py
"""The retrieval funnel: transform -> first-stage retrievers in parallel -> fusion -> rerank -> top-k.

    query ──transform──> plan.queries (+ HyDE passages for dense retrievers)
          ──retrieve───> one ranked list per (retriever, search text), candidate_k each
          ──fuse───────> one list (RRF or weighted), cut to rerank_k
          ──rerank─────> final_k hits (or a diversify pool), judged against plan.primary
          ──diversify──> optional MMR: final_k relevant hits that are not near-copies of each other
          ──ACL check──> drop anything the principal may not read (should never fire)

Every stage records its candidate ids, k, and latency in `RetrievalResult.trace`, so Chapter 14
can say which stage lost the evidence, and Chapter 31's tracer can show where the time went.
Failures degrade instead of failing the request: a retriever that raises is skipped (and named
in `trace["degraded"]`), a reranker that raises leaves the fused order. Only a caller error
(unknown filter) or the failure of every retriever raises.

Deadlines are optional and off by default. `retrieve_timeout_s` bounds the whole first-stage
phase: a job still running when it expires is recorded as timed out and treated like a failed
retriever. `rerank_timeout_s` bounds the reranker and degrades to the fused order. A deadline
stops the pipeline from waiting; it does not stop the work. The straggling thread keeps running
until its own client timeout fires, so set timeouts in the store and model clients as well.
Pass a shared `executor` in a service so a request does not create its own thread pool.
Work submitted to a pool runs in a copy of the caller's context (`contextvars.copy_context`), so
the span that is open when `retrieve` runs stays the parent of every per-retriever span instead
of each worker thread starting a new trace.
"""
from __future__ import annotations

import contextvars
from concurrent.futures import Executor, Future, ThreadPoolExecutor, wait
from contextlib import nullcontext
from typing import Any, Literal, Mapping, Sequence

from aie_core.llm.types import Message
from aie_core.observability import Tracer

from .common import Stopwatch, validate_filters
from .hybrid import reciprocal_rank_fusion, weighted_score_fusion
from .query import IdentityTransformer, QueryPlan, QueryTransformer
from .settings import RetrievalSettings
from .types import RetrievalQuery, RetrievalResult, Retriever, Reranker, ScoredChunk, visible


class RetrievalError(RuntimeError):
    """Every first-stage retriever failed; there is nothing to rank."""


class RetrievalPipeline:
    def __init__(
        self,
        retrievers: Mapping[str, Retriever],
        *,
        transformer: QueryTransformer | None = None,
        reranker: Reranker | None = None,
        fusion: Literal["rrf", "weighted"] = "rrf",
        rrf_k: int = 60,
        weights: Mapping[str, float] | None = None,
        candidate_k: int = 50,
        rerank_k: int = 20,
        final_k: int | None = None,
        hyde_retrievers: Sequence[str] = ("dense",),
        parallel: bool = True,
        max_workers: int = 8,
        tracer: Tracer | None = None,
        retrieve_timeout_s: float | None = None,
        rerank_timeout_s: float | None = None,
        executor: Executor | None = None,
        diversifier: Reranker | None = None,
        diversify_pool_k: int | None = None,
    ) -> None:
        if not retrievers:
            raise ValueError("RetrievalPipeline needs at least one retriever")
        if candidate_k <= 0 or rerank_k <= 0 or (final_k is not None and final_k <= 0):
            raise ValueError("candidate_k, rerank_k and final_k must be positive")
        if any(t is not None and t <= 0 for t in (retrieve_timeout_s, rerank_timeout_s)):
            raise ValueError("timeouts must be positive seconds or None")
        if diversify_pool_k is not None and diversify_pool_k <= 0:
            raise ValueError("diversify_pool_k must be positive")
        unknown = set(weights or {}) - set(retrievers)
        if unknown:
            raise ValueError(f"weights for unknown retrievers {sorted(unknown)}")
        self.retrievers = dict(retrievers)
        self.transformer = transformer or IdentityTransformer()
        self.reranker = reranker
        self.fusion = fusion
        self.rrf_k = rrf_k
        self.weights = dict(weights or {})
        self.candidate_k = candidate_k
        self.rerank_k = rerank_k
        self.final_k = final_k
        self.hyde_retrievers = set(hyde_retrievers)
        self.parallel = parallel
        self.max_workers = max_workers
        self.tracer = tracer
        self.retrieve_timeout_s = retrieve_timeout_s
        self.rerank_timeout_s = rerank_timeout_s
        self.executor = executor  # owned by the caller; never shut down here
        self.diversifier = diversifier  # e.g. MMRDiversifier; None keeps the reranker's order
        self.diversify_pool_k = diversify_pool_k

    @classmethod
    def from_settings(cls, retrievers: Mapping[str, Retriever], settings: RetrievalSettings | None = None,
                      **kwargs: Any) -> "RetrievalPipeline":
        s = settings or RetrievalSettings()
        defaults: dict[str, Any] = dict(fusion=s.fusion, rrf_k=s.rrf_k, candidate_k=s.candidate_k,
                                        rerank_k=s.rerank_k, final_k=s.final_k, parallel=s.parallel,
                                        retrieve_timeout_s=s.retrieve_timeout_s,
                                        rerank_timeout_s=s.rerank_timeout_s,
                                        diversify_pool_k=s.diversify_pool_k)
        if s.diversity == "mmr":
            from .diversity import MMRDiversifier  # Jaccard similarity; pass diversifier= for embeddings

            defaults["diversifier"] = MMRDiversifier(lambda_=s.mmr_lambda)
        defaults.update(kwargs)
        return cls(retrievers, **defaults)

    # ------------------------------------------------------------------ helpers
    def _span(self, name: str, **attrs: Any) -> Any:
        return self.tracer.span(name, **attrs) if self.tracer else nullcontext()

    def _jobs(self, plan: QueryPlan) -> list[tuple[str, str, str]]:
        """(list name, retriever name, search text). HyDE passages go to dense retrievers only."""
        jobs: list[tuple[str, str, str]] = []
        for name in self.retrievers:
            for i, text in enumerate(plan.queries):
                jobs.append((f"{name}#q{i}", name, text))
            if name in self.hyde_retrievers:
                for i, passage in enumerate(plan.hyde_passages):
                    jobs.append((f"{name}#hyde{i}", name, passage))
        return jobs

    def _run_job(self, job: tuple[str, str, str], query: RetrievalQuery, plan: QueryPlan) -> dict[str, Any]:
        list_name, retriever_name, text = job
        sub = plan.search_queries(query, text).model_copy(update={"k": self.candidate_k})
        record: dict[str, Any] = {"name": list_name, "kind": "retrieve", "retriever": retriever_name,
                                  "query": text, "k": self.candidate_k}
        with Stopwatch() as sw, self._span("retrieval.retrieve", list=list_name, k=self.candidate_k) as span:
            try:
                result = self.retrievers[retriever_name].retrieve(sub)
                record["hits"] = result.hits
                record["candidate_ids"] = [h.chunk.id for h in result.hits]
                record["acl_dropped"] = int(result.trace.get("acl_dropped", 0))
            except Exception as exc:  # degrade: one broken retriever must not fail the request
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["hits"] = []
                record["candidate_ids"] = []
            if span is not None:
                span.set_attribute("candidates", len(record["candidate_ids"]))
        record["latency_ms"] = sw.ms
        return record

    def _run_jobs(self, jobs: list[tuple[str, str, str]], query: RetrievalQuery,
                  plan: QueryPlan) -> list[dict[str, Any]]:
        """Run every first-stage job; under a deadline, unfinished jobs become timed-out records."""
        concurrent = self.parallel and len(jobs) > 1
        if not concurrent and self.retrieve_timeout_s is None:
            return [self._run_job(j, query, plan) for j in jobs]
        own = None if self.executor else ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(jobs)) if concurrent else 1)
        pool = self.executor or own
        assert pool is not None
        try:
            # one copied context per job: the open span (aie_core's contextvar, or Chapter 31's tracer)
            # crosses into the worker thread, so retriever spans stay children of retrieval.pipeline
            futures = [pool.submit(contextvars.copy_context().run, self._run_job, j, query, plan) for j in jobs]
            done, _ = wait(futures, timeout=self.retrieve_timeout_s)
            return [f.result() if f in done else self._timed_out(j, f) for j, f in zip(jobs, futures)]
        finally:
            if own is not None:
                own.shutdown(wait=False, cancel_futures=True)

    def _timed_out(self, job: tuple[str, str, str], future: Future[dict[str, Any]]) -> dict[str, Any]:
        future.cancel()  # succeeds only if the job never started
        list_name, retriever_name, text = job
        return {"name": list_name, "kind": "retrieve", "retriever": retriever_name, "query": text,
                "k": self.candidate_k, "hits": [], "candidate_ids": [], "timed_out": True,
                "error": f"TimeoutError: no result within {self.retrieve_timeout_s:g} s",
                "latency_ms": round(1000 * (self.retrieve_timeout_s or 0.0), 3)}

    def _rerank(self, query: RetrievalQuery, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        assert self.reranker is not None
        if self.rerank_timeout_s is None:
            return self.reranker.rerank(query, candidates, k)
        own = None if self.executor else ThreadPoolExecutor(max_workers=1)
        pool = self.executor or own
        assert pool is not None
        try:  # concurrent.futures.TimeoutError is the builtin TimeoutError on Python 3.11+
            ctx = contextvars.copy_context()
            return pool.submit(ctx.run, self.reranker.rerank, query, candidates, k).result(timeout=self.rerank_timeout_s)
        finally:
            if own is not None:
                own.shutdown(wait=False, cancel_futures=True)

    def _pool_k(self, final_k: int) -> int:
        """How many hits the reranker returns: final_k, or a larger pool for the diversifier to choose from."""
        if self.diversifier is None:
            return final_k
        return max(final_k, self.diversify_pool_k or min(2 * final_k, self.rerank_k))

    def _fuse(self, rankings: Mapping[str, list[ScoredChunk]], limit: int) -> list[ScoredChunk]:
        per_list = {ln: self.weights.get(ln.split("#", 1)[0], 1.0) for ln in rankings}
        if self.fusion == "rrf":
            return reciprocal_rank_fusion(rankings, k=self.rrf_k, weights=per_list, limit=limit)
        return weighted_score_fusion(rankings, weights=per_list, limit=limit)

    # ------------------------------------------------------------------ main entry
    def retrieve(self, query: RetrievalQuery, history: Sequence[Message] | None = None) -> RetrievalResult:
        validate_filters(query.filters)  # a caller bug, so it raises instead of degrading
        final_k = self.final_k or query.k
        stages: list[dict[str, Any]] = []
        latency: dict[str, float] = {}
        degraded: list[str] = []
        with Stopwatch() as total, self._span("retrieval.pipeline", final_k=final_k):
            # 1. transform
            with Stopwatch() as sw, self._span("retrieval.transform"):
                plan = self.transformer.transform(query, history)
            latency["transform"] = sw.ms
            if plan.fallback:
                degraded.append(f"transform:{plan.strategy}")
            stages.append({"name": "transform", "kind": "transform", "strategy": plan.strategy,
                           "queries": plan.queries, "hyde_passages": len(plan.hyde_passages),
                           "fallback": plan.fallback, "latency_ms": sw.ms})

            # 2. first-stage retrieval, all (retriever, text) jobs in parallel, under an optional deadline
            jobs = self._jobs(plan)
            with Stopwatch() as sw:
                records = self._run_jobs(jobs, query, plan)
            latency["retrieve"] = sw.ms  # wall clock of the parallel phase, not the sum
            ok = [r for r in records if "error" not in r]
            degraded.extend(f"retrieve:{r['name']}" for r in records if "error" in r)
            if not ok:
                raise RetrievalError("all retrievers failed: " + "; ".join(r["error"] for r in records))
            for r in records:
                stages.append({k: v for k, v in r.items() if k != "hits"})

            # 3. fusion
            with Stopwatch() as sw, self._span("retrieval.fusion", lists=len(ok)):
                fused = self._fuse({r["name"]: r["hits"] for r in ok}, limit=self.rerank_k)
            latency["fusion"] = sw.ms
            stages.append({"name": "fusion", "kind": "fusion", "method": self.fusion, "k": self.rerank_k,
                           "candidate_ids": [h.chunk.id for h in fused], "latency_ms": sw.ms})

            # 4. rerank, judged against the self-contained question
            rerank_query = query.model_copy(update={"text": plan.primary, "original_text": plan.original_text})
            pool_k = self._pool_k(final_k)
            hits = fused[:pool_k]
            if self.reranker is not None and fused:
                with Stopwatch() as sw, self._span("retrieval.rerank", candidates=len(fused)):
                    try:
                        hits = self._rerank(rerank_query, fused, pool_k)
                    except Exception as exc:  # degrade to fused order, including on timeout
                        degraded.append(f"rerank:{type(exc).__name__}")
                        hits = fused[:pool_k]
                if any(h.signals.get("rerank_failed") or h.signals.get("cross_encoder_fallback") for h in hits):
                    degraded.append("rerank:fallback")
                latency["rerank"] = sw.ms
                stages.append({"name": "rerank", "kind": "rerank", "reranker": type(self.reranker).__name__,
                               "k": pool_k, "candidate_ids": [h.chunk.id for h in hits], "latency_ms": sw.ms})

            # 4b. diversify (optional): pick final_k from the pool, trading relevance against redundancy
            if self.diversifier is not None and hits:
                with Stopwatch() as sw, self._span("retrieval.diversify", candidates=len(hits)):
                    try:
                        hits = self.diversifier.rerank(rerank_query, hits, final_k)
                    except Exception as exc:  # degrade to the relevance order
                        degraded.append(f"diversify:{type(exc).__name__}")
                        hits = hits[:final_k]
                latency["diversify"] = sw.ms
                stages.append({"name": "diversify", "kind": "diversify",
                               "diversifier": type(self.diversifier).__name__, "pool_k": pool_k, "k": final_k,
                               "candidate_ids": [h.chunk.id for h in hits], "latency_ms": sw.ms})
            hits = hits[:final_k]

            # 5. final authorization check: belt and braces; any violation is a security event
            violations = [h.chunk.id for h in hits if not visible(h.chunk, query.principal)]
            hits = [h for h in hits if visible(h.chunk, query.principal)]
        latency["total"] = total.ms
        trace = {
            "plan": plan.model_dump(),
            "stages": stages,
            "k": {"candidate_k": self.candidate_k, "rerank_k": self.rerank_k, "final_k": final_k,
                  **({"diversify_pool_k": pool_k} if self.diversifier is not None else {})},
            "latency_ms": latency,
            "degraded": degraded,
            "acl_violations": violations,
            "acl_dropped": sum(r.get("acl_dropped", 0) for r in records),
        }
        return RetrievalResult(query=query, hits=hits, trace=trace)


def stage_candidates(result: RetrievalResult, stage: str) -> list[str]:
    """Candidate ids recorded for a stage name ("fusion", "rerank", "bm25#q0", ...)."""
    for s in result.trace.get("stages", []):
        if s["name"] == stage:
            return list(s.get("candidate_ids", []))
    raise KeyError(f"no stage {stage!r} in trace")


__all__ = ["RetrievalPipeline", "RetrievalError", "stage_candidates"]
