# path: book/projects/p2-semantic-search/semsearch/bench.py
"""Two experiments the chapter's numbers come from. Both run offline in seconds.

1. ann_sweep: IVF recall@k and work per query versus nprobe, measured against exact search.
2. filter_experiment: why post-filtering an ANN candidate list fails for selective filters,
   and why pre-filtering (or an iterative scan) does not.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .domain.ivf import IVFIndex, _unit_rows, exact_top_k
from .domain.metrics import overlap_at_k


def clustered_vectors(n: int, dims: int, clusters: int, spread: float = 0.35, seed: int = 0) -> np.ndarray:
    """Synthetic embeddings with topic structure: real embeddings are clustered, not uniform."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(clusters, dims))
    labels = rng.integers(clusters, size=n)
    return _unit_rows(centers[labels] + spread * rng.normal(size=(n, dims)))


@dataclass
class SweepRow:
    nprobe: int
    recall: float
    scanned_fraction: float
    ms_per_query: float


def ann_sweep(
    n: int = 20000, dims: int = 64, n_lists: int = 128, k: int = 10, queries: int = 200,
    nprobes: tuple[int, ...] = (1, 2, 4, 8, 16, 32), seed: int = 0,
) -> tuple[list[SweepRow], float]:
    data = clustered_vectors(n, dims, clusters=64, seed=seed)
    qs = clustered_vectors(queries, dims, clusters=64, seed=seed + 1)
    index = IVFIndex(n_lists, seed=seed).fit(data)
    truth = [exact_top_k(data, q, k) for q in qs]
    t0 = time.perf_counter()
    for q in qs:
        exact_top_k(data, q, k)
    exact_ms = (time.perf_counter() - t0) * 1000 / queries
    rows: list[SweepRow] = []
    for nprobe in nprobes:
        recalls, scanned = [], 0
        t0 = time.perf_counter()
        for q, gold in zip(qs, truth):
            ids, work = index.search(q, k, nprobe)
            scanned += work
            recalls.append(overlap_at_k([str(i) for i in ids], [str(i) for i in gold], k))
        ms = (time.perf_counter() - t0) * 1000 / queries
        rows.append(SweepRow(nprobe, float(np.mean(recalls)), scanned / (queries * n), ms))
    return rows, exact_ms


@dataclass
class FilterRow:
    selectivity: float
    post_filter_returned: float  # mean results out of k when filtering an ANN candidate list
    pre_filter_returned: float   # mean results out of k when filtering before scoring


def filter_experiment(
    n: int = 20000, dims: int = 64, k: int = 10, candidates: int = 40, queries: int = 100,
    selectivities: tuple[float, ...] = (0.5, 0.1, 0.02, 0.005), seed: int = 0,
) -> list[FilterRow]:
    """`candidates` plays the role of ef_search: the ANN stage returns that many neighbors,
    then the filter runs. Exact top-`candidates` is used as an optimistic stand-in for ANN."""
    rng = np.random.default_rng(seed)
    data = clustered_vectors(n, dims, clusters=64, seed=seed)
    qs = clustered_vectors(queries, dims, clusters=64, seed=seed + 1)
    rows: list[FilterRow] = []
    for sel in selectivities:
        allowed = rng.random(n) < sel  # e.g. "tenant = logistics AND group = it-oncall"
        allowed_idx = np.flatnonzero(allowed)
        post, pre = [], []
        for q in qs:
            cand = exact_top_k(data, q, candidates)
            post.append(min(k, int(allowed[cand].sum())))
            pre.append(min(k, allowed_idx.size))
        rows.append(FilterRow(sel, float(np.mean(post)), float(np.mean(pre))))
    return rows


def format_sweep(rows: list[SweepRow], exact_ms: float) -> str:
    lines = [f"exact search: {exact_ms:.3f} ms/query", "nprobe  recall@10  scanned  ms/query"]
    lines += [f"{r.nprobe:>6}  {r.recall:>9.3f}  {r.scanned_fraction:>7.1%}  {r.ms_per_query:>8.3f}" for r in rows]
    return "\n".join(lines)


def format_filter(rows: list[FilterRow], k: int = 10) -> str:
    lines = [f"selectivity  post-filter hits/{k}  pre-filter hits/{k}"]
    lines += [f"{r.selectivity:>11.1%}  {r.post_filter_returned:>18.2f}  {r.pre_filter_returned:>17.2f}" for r in rows]
    return "\n".join(lines)


__all__ = ["ann_sweep", "filter_experiment", "clustered_vectors", "format_sweep", "format_filter"]
