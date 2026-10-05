# path: book/projects/examples/ch08/tests/test_ch08_vector_math.py
from __future__ import annotations

import numpy as np
import pytest

from embedlab.vector_math import (
    centroid,
    euclidean_from_cosine,
    is_normalized,
    l2_normalize_rows,
    mean_center,
    pairwise,
    rank,
    similarity,
    similarity_profile,
    truncate,
)


def test_workshop_example_cosine_ignores_magnitude():
    q, d1, d2 = [1, 2], [2, 4], [2, -1]
    assert similarity(q, d1) == pytest.approx(1.0)
    assert similarity(q, d2) == pytest.approx(0.0)
    assert similarity(q, d1, "dot") == pytest.approx(10.0)


def test_metrics_agree_on_unit_vectors_and_disagree_otherwise():
    rng = np.random.default_rng(7)
    docs = rng.normal(size=(50, 16))
    query = rng.normal(size=16)
    unit_docs = l2_normalize_rows(docs)
    by_metric = {m: [i for i, _ in rank(query, unit_docs, m, k=10)] for m in ("cosine", "dot", "euclidean")}
    assert by_metric["cosine"] == by_metric["dot"] == by_metric["euclidean"]

    # Recipe: a long vector wins under dot product even though its direction is worse.
    raw = np.array([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [6.0, 6.0, 2.0]])
    q = [0.8, 0.2, 0.0]
    assert rank(q, raw, "cosine", k=1)[0][0] == 0
    assert rank(q, raw, "dot", k=1)[0][0] == 2


def test_euclidean_cosine_identity_for_unit_vectors():
    rng = np.random.default_rng(1)
    a, b = l2_normalize_rows(rng.normal(size=(2, 32)))
    cos = float(a @ b)
    assert np.linalg.norm(a - b) == pytest.approx(euclidean_from_cosine(cos))


def test_normalize_keeps_zero_rows_finite():
    m = l2_normalize_rows([[3.0, 4.0], [0.0, 0.0]])
    assert m[0].tolist() == pytest.approx([0.6, 0.8])
    assert m[1].tolist() == [0.0, 0.0]
    assert is_normalized(m)


def test_truncate_renormalizes_and_validates():
    m = l2_normalize_rows(np.ones((3, 8)))
    cut = truncate(m, 4)
    assert cut.shape == (3, 4)
    assert np.allclose(np.linalg.norm(cut, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(truncate(m, 4, renormalize=False), axis=1), np.sqrt(0.5))
    with pytest.raises(ValueError):
        truncate(m, 9)


def test_pairwise_rejects_dimension_mismatch():
    with pytest.raises(ValueError):
        pairwise([[1, 0]], [[1, 0, 0]])


def test_centroid_is_unit_and_not_dominated_by_long_vectors():
    c = centroid([[1.0, 0.0], [0.0, 100.0]])
    assert c.tolist() == pytest.approx([np.sqrt(0.5), np.sqrt(0.5)])


def test_mean_centering_spreads_an_anisotropic_space():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(40, 24)) * 0.3 + 2.0  # every vector shares a large common component
    before = similarity_profile(base)["mean"]
    centered, mu = mean_center(base)
    after = similarity_profile(centered)["mean"]
    assert before > 0.8
    assert abs(after) < 0.1
    # the same mean must be applied to queries
    q_centered, _ = mean_center(base[:1], mu)
    assert np.allclose(q_centered, centered[:1])
