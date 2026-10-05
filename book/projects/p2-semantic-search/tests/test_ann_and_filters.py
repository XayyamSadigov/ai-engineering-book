# path: book/projects/p2-semantic-search/tests/test_ann_and_filters.py
from __future__ import annotations

import numpy as np
import pytest

from semsearch.bench import ann_sweep, clustered_vectors, filter_experiment
from semsearch.domain.ivf import IVFIndex, exact_top_k


def test_ivf_full_probe_equals_exact():
    data = clustered_vectors(2000, 32, clusters=16, seed=1)
    index = IVFIndex(16, seed=1).fit(data)
    q = data[7]
    ids, scanned = index.search(q, 10, nprobe=16)
    assert scanned == 2000
    assert list(ids) == list(exact_top_k(data, q, 10))


def test_ivf_recall_grows_with_nprobe():
    rows, _ = ann_sweep(n=4000, dims=32, n_lists=32, queries=50, nprobes=(1, 4, 32))
    recalls = [r.recall for r in rows]
    assert recalls[0] < recalls[1] <= recalls[2]
    assert recalls[2] == pytest.approx(1.0)
    assert rows[0].scanned_fraction < rows[2].scanned_fraction


def test_ivf_needs_training_data():
    with pytest.raises(ValueError):
        IVFIndex(10).fit(np.zeros((5, 4)))
    with pytest.raises(RuntimeError):
        IVFIndex(2).search(np.ones(4), 1, 1)


def test_post_filter_underfills_selective_filters():
    rows = {r.selectivity: r for r in filter_experiment(n=5000, queries=30, selectivities=(0.5, 0.01))}
    assert rows[0.5].post_filter_returned == pytest.approx(10.0, abs=0.5)
    assert rows[0.01].post_filter_returned < 2.0
    assert rows[0.01].pre_filter_returned == 10.0
