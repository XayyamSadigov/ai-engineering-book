# path: book/projects/ragkit/ragkit/retrieval/hybrid.py
"""Fusing ranked lists from several retrievers.

BM25 scores are unbounded sums of idf-weighted term matches; cosine similarities sit in
[-1, 1] and cluster in a model-specific band. Adding them raw lets one scale dominate.
Two fixes:

- Reciprocal Rank Fusion (RRF) ignores scores and uses ranks: fused(d) = sum_i w_i / (k + rank_i(d)).
  Robust, parameter-light, no calibration. k (default 60) flattens the head: with k=60 rank 1
  is worth 1/61 and rank 10 is worth 1/70, so agreement across lists beats one strong vote.
- Weighted score fusion normalizes each list's scores to [0, 1] with min-max and takes a weighted
  sum. It keeps score gaps (a runaway BM25 match stays a runaway) but its weights must be tuned on
  judged queries, and min-max is sensitive to the outliers in each list.

Both keep every list's rank and score in `signals`, so a fused hit can be explained afterwards.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Mapping, Sequence

from .common import Stopwatch
from .types import RetrievalQuery, RetrievalResult, Retriever, ScoredChunk

Ranking = Sequence[ScoredChunk]


def _check_weights(rankings: Mapping[str, Ranking], weights: Mapping[str, float] | None) -> dict[str, float]:
    weights = dict(weights or {})
    unknown = set(weights) - set(rankings)
    if unknown:
        raise ValueError(f"weights given for unknown lists {sorted(unknown)}")
    if any(w < 0 for w in weights.values()):
        raise ValueError("fusion weights must be non-negative")
    return {name: weights.get(name, 1.0) for name in rankings}


def reciprocal_rank_fusion(
    rankings: Mapping[str, Ranking],
    *,
    k: int = 60,
    weights: Mapping[str, float] | None = None,
    limit: int | None = None,
) -> list[ScoredChunk]:
    """Fuse ranked lists by sum of w / (k + rank). Ranks are 1-based positions in each list."""
    if k < 0:
        raise ValueError("RRF k must be >= 0")
    w = _check_weights(rankings, weights)
    fused: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    signals: dict[str, dict[str, float]] = {}
    chunks: dict[str, ScoredChunk] = {}
    for name, ranking in rankings.items():
        seen: set[str] = set()
        for position, hit in enumerate(ranking, start=1):
            cid = hit.chunk.id
            if cid in seen:  # a list that repeats a chunk counts it once, at its best rank
                continue
            seen.add(cid)
            chunks.setdefault(cid, hit)
            fused[cid] = fused.get(cid, 0.0) + w[name] / (k + position)
            best_rank[cid] = min(best_rank.get(cid, position), position)
            sig = signals.setdefault(cid, {})
            sig.update({k_: v for k_, v in hit.signals.items() if k_ not in sig})
            sig[f"{name}_rank"] = float(position)
            sig[f"{name}_score"] = round(hit.score, 6)
    order = sorted(fused, key=lambda cid: (-fused[cid], best_rank[cid], cid))
    if limit is not None:
        order = order[:limit]
    return [
        ScoredChunk(chunk=chunks[cid].chunk, score=round(fused[cid], 8), stage="fusion", rank=i,
                    signals={**signals[cid], "rrf": round(fused[cid], 8)})
        for i, cid in enumerate(order, start=1)
    ]


def min_max(scores: Sequence[float]) -> list[float]:
    """Scale to [0, 1]. A list whose scores are all equal maps to 1.0: it expresses no preference
    among its own items, but each item was still retrieved."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def weighted_score_fusion(
    rankings: Mapping[str, Ranking],
    *,
    weights: Mapping[str, float] | None = None,
    limit: int | None = None,
) -> list[ScoredChunk]:
    """Weighted sum of per-list min-max normalized scores; a chunk missing from a list gets 0 there."""
    w = _check_weights(rankings, weights)
    fused: dict[str, float] = {}
    signals: dict[str, dict[str, float]] = {}
    chunks: dict[str, ScoredChunk] = {}
    for name, ranking in rankings.items():
        dedup: dict[str, ScoredChunk] = {}
        for hit in ranking:
            if hit.chunk.id not in dedup or hit.score > dedup[hit.chunk.id].score:
                dedup[hit.chunk.id] = hit
        hits = list(dedup.values())
        for hit, norm in zip(hits, min_max([h.score for h in hits])):
            cid = hit.chunk.id
            chunks.setdefault(cid, hit)
            fused[cid] = fused.get(cid, 0.0) + w[name] * norm
            sig = signals.setdefault(cid, {})
            sig.update({k_: v for k_, v in hit.signals.items() if k_ not in sig})
            sig[f"{name}_score"] = round(hit.score, 6)
            sig[f"{name}_norm"] = round(norm, 6)
    order = sorted(fused, key=lambda cid: (-fused[cid], cid))
    if limit is not None:
        order = order[:limit]
    return [
        ScoredChunk(chunk=chunks[cid].chunk, score=round(fused[cid], 8), stage="fusion", rank=i,
                    signals={**signals[cid], "weighted": round(fused[cid], 8)})
        for i, cid in enumerate(order, start=1)
    ]


class HybridRetriever:
    """Run several retrievers on the same query (in parallel) and fuse their rankings.

    Each retriever is asked for `candidate_k` hits; the fused list is cut to `query.k`. For the
    full funnel with query transformation and reranking, use `RetrievalPipeline`.
    """

    name = "hybrid"

    def __init__(
        self,
        retrievers: Mapping[str, Retriever],
        *,
        fusion: Literal["rrf", "weighted"] = "rrf",
        rrf_k: int = 60,
        weights: Mapping[str, float] | None = None,
        candidate_k: int = 50,
        parallel: bool = True,
    ) -> None:
        if not retrievers:
            raise ValueError("HybridRetriever needs at least one retriever")
        self.retrievers = dict(retrievers)
        self.fusion = fusion
        self.rrf_k = rrf_k
        self.weights = dict(weights or {})
        self.candidate_k = candidate_k
        self.parallel = parallel

    def fuse(self, rankings: Mapping[str, Ranking], limit: int | None = None) -> list[ScoredChunk]:
        if self.fusion == "rrf":
            return reciprocal_rank_fusion(rankings, k=self.rrf_k, weights=self.weights or None, limit=limit)
        return weighted_score_fusion(rankings, weights=self.weights or None, limit=limit)

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        sub = query.model_copy(update={"k": self.candidate_k})
        with Stopwatch() as total:
            if self.parallel and len(self.retrievers) > 1:
                with ThreadPoolExecutor(max_workers=len(self.retrievers)) as pool:
                    futures = {name: pool.submit(r.retrieve, sub) for name, r in self.retrievers.items()}
                    results = {name: f.result() for name, f in futures.items()}
            else:
                results = {name: r.retrieve(sub) for name, r in self.retrievers.items()}
            with Stopwatch() as fuse_sw:
                hits = self.fuse({name: res.hits for name, res in results.items()}, limit=query.k)
        trace = {
            "stage": self.name,
            "fusion": self.fusion,
            "candidate_k": self.candidate_k,
            "retrievers": {name: res.trace for name, res in results.items()},
            "fused_ids": [h.chunk.id for h in hits],
            "fusion_latency_ms": fuse_sw.ms,
            "latency_ms": total.ms,
        }
        return RetrievalResult(query=query, hits=hits, trace=trace)


__all__ = ["reciprocal_rank_fusion", "weighted_score_fusion", "min_max", "HybridRetriever"]
