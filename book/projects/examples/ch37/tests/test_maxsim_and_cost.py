# path: book/projects/examples/ch37/tests/test_maxsim_and_cost.py
from __future__ import annotations

import numpy as np
import pytest

from long_context_cost import Prices, Workload, compare, estimate
from maxsim import QUERY, RETAIL, RUNBOOK, explain, index_bytes, maxsim, maxsim_score, single_vector_score, token_vectors


def test_maxsim_matches_the_definition():
    q = np.array([[1.0, 0.0], [0.0, 1.0]])
    d = np.array([[0.6, 0.8], [1.0, 0.0], [0.0, -1.0]])
    # row 1 best = 1.0 (second doc token), row 2 best = 0.8 (first doc token)
    assert maxsim(q, d) == pytest.approx(1.8)
    assert maxsim(q, np.zeros((0, 2))) == 0.0


def test_token_vectors_are_unit_and_synonyms_align():
    toks, m = token_vectors("laptop notebook refund")
    assert toks == ["laptop", "notebook", "refund"]
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0)
    assert m[0] @ m[1] > 0.85 > m[0] @ m[2]


def test_pooling_dilutes_while_late_interaction_aligns_every_query_token():
    assert single_vector_score(QUERY, RETAIL) > single_vector_score(QUERY, RUNBOOK)  # the failure
    assert maxsim_score(QUERY, RUNBOOK) > maxsim_score(QUERY, RETAIL)  # the fix
    alignment = {q: d for q, d, _ in explain(QUERY, RUNBOOK)}
    assert alignment["old"] == "previous" and alignment["window"] == "within"


def test_late_interaction_index_is_much_larger():
    single = index_bytes(1_000, 200, 768, 4, late_interaction=False)
    late = index_bytes(1_000, 200, 128, 2, late_interaction=True)
    assert late / single == pytest.approx(200 * 128 * 2 / (768 * 4))


def test_long_context_costs_about_25x_rag_on_the_shared_corpus():
    w, p = Workload(), Prices()
    lc, rag = estimate("long_context", w, p), estimate("rag", w, p)
    assert 20 < lc.input_tokens_per_query / rag.input_tokens_per_query < 30
    assert lc.monthly_usd > rag.monthly_usd


def test_caching_helps_one_scope_but_can_hurt_many_low_traffic_scopes():
    p = Prices()
    one = Workload()
    assert estimate("long_context", one, p, cache=True).monthly_usd < estimate("long_context", one, p).monthly_usd
    many = Workload(permission_scopes=12, queries_per_month=20_000)
    assert estimate("long_context", many, p, cache=True).monthly_usd > estimate("long_context", many, p).monthly_usd


def test_a_large_knowledge_base_does_not_fit_and_hybrid_stays_bounded():
    results = {(e.strategy, e.cached): e for e in compare(Workload(corpus_tokens=45_000_000))}
    assert not results[("long_context", False)].fits_window
    assert results[("hybrid", False)].input_tokens_per_query < 5_000
