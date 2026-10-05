# path: book/projects/examples/ch37/maxsim.py
"""Late interaction (ColBERT-style MaxSim) versus single-vector retrieval, in NumPy.

A real late-interaction model produces one contextual vector per token from a trained encoder.
Here token vectors come from a tiny deterministic "concept" table (synonyms share a concept
vector plus a little token-specific noise), which is enough to show the mechanism:

    single vector:  score = cos(mean(Q), mean(D))         one comparison, details averaged away
    late interaction: score = sum_i max_j  q_i . d_j      every query token finds its best match
"""
from __future__ import annotations

import hashlib

import numpy as np

from corpus import terms

DIM = 48
NOISE = 0.25

# surface token -> concept; tokens not listed are their own concept
CONCEPTS: dict[str, str] = {
    "laptop": "device", "notebook": "device", "computer": "device", "device": "device", "machine": "device",
    "return": "return", "returns": "return", "returned": "return", "returning": "return", "hand": "return",
    "window": "deadline", "deadline": "deadline", "within": "deadline", "period": "deadline",
    "old": "old", "previous": "old", "replaced": "old",
    "refund": "refund", "refunds": "refund", "exchange": "refund", "exchanges": "refund",
    "store": "store", "stores": "store", "retail": "store", "customer": "store", "customers": "store",
    "product": "product", "products": "product", "item": "product", "items": "product", "purchase": "product",
    **{w: "data" for w in "data files file backup back onedrive migration migrate migrated copy copies folders documents sync synced".split()},
}


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def _seeded(key: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    return np.random.default_rng(seed).standard_normal(DIM)


def token_vectors(text: str) -> tuple[list[str], np.ndarray]:
    """One unit vector per content token: concept direction plus token-specific noise."""
    toks = terms(text)
    if not toks:
        return [], np.zeros((0, DIM))
    rows = [_unit(_unit(_seeded("concept:" + CONCEPTS.get(t, t))) + NOISE * _unit(_seeded("token:" + t))) for t in toks]
    return toks, np.vstack(rows)


def pooled_vector(text: str) -> np.ndarray:
    """The single-vector baseline: mean-pool the token vectors, then normalize."""
    _, m = token_vectors(text)
    return _unit(m.mean(axis=0))


def single_vector_score(query: str, doc: str) -> float:
    return float(pooled_vector(query) @ pooled_vector(doc))


def maxsim(q: np.ndarray, d: np.ndarray) -> float:
    """sum over query tokens of the best cosine against any document token."""
    if q.size == 0 or d.size == 0:
        return 0.0
    return float((q @ d.T).max(axis=1).sum())


def maxsim_score(query: str, doc: str) -> float:
    return maxsim(token_vectors(query)[1], token_vectors(doc)[1])


def explain(query: str, doc: str) -> list[tuple[str, str, float]]:
    """For each query token: the document token it aligned with and the similarity. This is the
    interpretability bonus of late interaction: you can see which words matched."""
    qt, q = token_vectors(query)
    dt, d = token_vectors(doc)
    sims = q @ d.T
    return [(qt[i], dt[int(sims[i].argmax())], round(float(sims[i].max()), 3)) for i in range(len(qt))]


def index_bytes(n_passages: int, tokens_per_passage: int, dim: int, bytes_per_value: int, late_interaction: bool) -> int:
    """Storage for the vectors alone (no graph or posting overhead)."""
    vectors = n_passages * (tokens_per_passage if late_interaction else 1)
    return vectors * dim * bytes_per_value


# Two passages that compete for "return window for my old laptop". The retail one is about
# returns in general and repeats the topic; the runbook one answers the question in its last
# sentence but spends most of its words on data migration, which dominates its pooled vector.
RUNBOOK = (
    "Data migration before the swap: back up all files and folders to OneDrive, confirm the OneDrive sync "
    "finished, copy local data not covered by sync, and verify the migrated files open on the new device. "
    "The migration tool copies documents and browser data; files outside synced folders are not migrated. "
    "Hand the previous notebook to the service desk within ten business days."
)
RETAIL = (
    "Retail returns: customers may return products within a 30 day return window. Returns of items "
    "purchased in stores are refunded to the original payment; exchanges of products are processed at "
    "any store; returned items must include the receipt."
)
QUERY = "return window for my old laptop"


def main() -> None:
    for name, doc in (("runbook", RUNBOOK), ("retail", RETAIL)):
        print(f"{name:8} single-vector={single_vector_score(QUERY, doc):.3f}  maxsim={maxsim_score(QUERY, doc):.3f}")
        for qt, dt, s in explain(QUERY, doc):
            print(f"         {qt:>8} -> {dt:<10} {s:.2f}")
    single = index_bytes(1_000_000, 200, 768, 4, late_interaction=False)
    late = index_bytes(1_000_000, 200, 128, 2, late_interaction=True)
    print(f"\nillustrative storage for 1M passages: single-vector {single / 1e9:.1f} GB, late interaction {late / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
