# path: book/projects/examples/ch37/long_context_cost.py
"""Long context versus RAG versus the hybrid (retrieve documents, send them whole): cost math.

All prices and window sizes are illustrative parameters, not facts about any provider. The point
is the shape of the curves: long context costs scale with corpus size per query, RAG costs scale
with k * chunk size, caching discounts a stable prefix, and permission scopes fragment the cache.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

MINUTES_PER_MONTH = 30 * 24 * 60

Strategy = Literal["long_context", "rag", "hybrid"]


class Workload(BaseModel):
    queries_per_month: int = 50_000
    corpus_tokens: int = 22_600  # the shared Northwind corpus
    system_tokens: int = 300
    question_tokens: int = 40
    answer_tokens: int = 250
    rag_context_tokens: int = 600  # k=4 chunks of about 150 tokens
    hybrid_docs: int = 2  # documents sent whole after document-level retrieval
    avg_doc_tokens: int = 950
    permission_scopes: int = 1  # distinct corpora users may see; each needs its own cached prefix
    context_window: int = 128_000  # illustrative
    headroom: float = 0.5  # use at most this fraction of the window for documents


class Prices(BaseModel):
    """Illustrative USD per million tokens."""

    input: float = 3.0
    cached_input: float = 0.3
    cache_write: float = 3.75
    output: float = 15.0
    prefill_tokens_per_s: float = 5_000.0  # illustrative, for a rough time-to-first-token term


class Estimate(BaseModel):
    strategy: Strategy
    cached: bool
    input_tokens_per_query: int
    monthly_usd: float
    prefill_s: float  # uncached prefill time per query, a lower bound on time to first token
    fits_window: bool


def estimate(strategy: Strategy, w: Workload, p: Prices, cache: bool = False, cache_ttl_minutes: int = 5) -> Estimate:
    fixed = w.system_tokens + w.question_tokens
    docs = {"long_context": w.corpus_tokens, "rag": w.rag_context_tokens, "hybrid": w.hybrid_docs * w.avg_doc_tokens}[strategy]
    per_query_in = fixed + docs
    out_cost = w.queries_per_month * w.answer_tokens * p.output / 1e6

    if cache and strategy == "long_context":
        prefix = w.system_tokens + w.corpus_tokens  # identical across requests within one scope
        # with steady traffic, each scope's prefix is rewritten once per TTL window
        writes = min(w.queries_per_month, w.permission_scopes * MINUTES_PER_MONTH // cache_ttl_minutes)
        hits = w.queries_per_month - writes
        in_cost = (
            writes * prefix * p.cache_write
            + hits * prefix * p.cached_input
            + w.queries_per_month * w.question_tokens * p.input
        ) / 1e6
    else:
        in_cost = w.queries_per_month * per_query_in * p.input / 1e6

    return Estimate(
        strategy=strategy,
        cached=cache,
        input_tokens_per_query=per_query_in,
        monthly_usd=round(in_cost + out_cost, 2),
        prefill_s=round(per_query_in / p.prefill_tokens_per_s, 3),
        fits_window=docs + fixed + w.answer_tokens <= w.context_window * w.headroom if strategy == "long_context" else True,
    )


def compare(w: Workload, p: Prices | None = None) -> list[Estimate]:
    p = p or Prices()
    return [
        estimate("long_context", w, p),
        estimate("long_context", w, p, cache=True),
        estimate("hybrid", w, p),
        estimate("rag", w, p),
    ]


def main() -> None:
    for label, w in (
        ("shared corpus, one scope", Workload()),
        ("shared corpus, 12 permission scopes, low traffic", Workload(permission_scopes=12, queries_per_month=20_000)),
        ("knowledge base of 45M tokens", Workload(corpus_tokens=45_000_000)),
    ):
        print(f"\n{label}")
        for e in compare(w):
            name = e.strategy + (" +cache" if e.cached else "")
            print(f"  {name:20} in/query={e.input_tokens_per_query:>10,}  monthly=${e.monthly_usd:>12,.2f}  prefill~{e.prefill_s:>8.2f}s  fits={e.fits_window}")


if __name__ == "__main__":
    main()
