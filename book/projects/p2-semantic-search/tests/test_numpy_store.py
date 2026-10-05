# path: book/projects/p2-semantic-search/tests/test_numpy_store.py
from __future__ import annotations

import pytest

from semsearch.adapters.numpy_store import NumpyVectorStore
from semsearch.domain.models import SearchFilter

from .conftest import NS, make_record


def test_persistence_round_trip(tmp_path, vocab_embedder):
    e = vocab_embedder
    store = NumpyVectorStore(e.dimensions, persist_dir=tmp_path)
    store.upsert([
        make_record(e, "a", "refund policy", tags=("finance",)),
        make_record(e, "b", "vpn error", acl=("it-oncall",), tenant="logistics"),
    ])
    store.save()
    assert (tmp_path / "test__fake__v1.npz").exists()
    assert (tmp_path / "test__fake__v1.json").exists()

    loaded = NumpyVectorStore.load(tmp_path)
    assert loaded.dimensions == e.dimensions
    assert loaded.count(NS) == 2
    q = e.embed_query("vpn error")
    a = [(h.id, round(h.score, 5)) for h in store.search(NS, q, 2)]
    b = [(h.id, round(h.score, 5)) for h in loaded.search(NS, q, 2)]
    assert a == b
    hit = loaded.search(NS, q, 1, SearchFilter(acl_groups=("it-oncall",)))[0]
    assert hit.tenant == "logistics"


def test_load_rejects_dimension_change(tmp_path, vocab_embedder):
    store = NumpyVectorStore(vocab_embedder.dimensions, persist_dir=tmp_path)
    store.upsert([make_record(vocab_embedder, "a", "refund")])
    store.save()
    with pytest.raises(ValueError):
        NumpyVectorStore.load(tmp_path, dimensions=vocab_embedder.dimensions + 1)


def test_record_requires_acl(vocab_embedder):
    with pytest.raises(ValueError):
        make_record(vocab_embedder, "a", "refund", acl=())


def test_record_rejects_bad_namespace(vocab_embedder):
    with pytest.raises(ValueError):
        make_record(vocab_embedder, "a", "refund", namespace="Bad Namespace!")


def test_writes_after_search_are_visible(vocab_embedder):
    e = vocab_embedder
    store = NumpyVectorStore(e.dimensions)
    store.upsert([make_record(e, "a", "refund")])
    assert len(store.search(NS, e.embed_query("refund"), 5)) == 1
    store.upsert([make_record(e, "b", "refund policy")])  # marks the matrix dirty
    assert len(store.search(NS, e.embed_query("refund"), 5)) == 2
    assert isinstance(store.search(NS, e.embed_query("refund"), 1)[0].score, float)
