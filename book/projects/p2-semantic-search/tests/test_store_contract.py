# path: book/projects/p2-semantic-search/tests/test_store_contract.py
"""One behavioral contract, run against every adapter. NumPy always; pgvector when
DATABASE_URL is set and the integration marker is selected (pytest -m integration)."""
from __future__ import annotations

import pytest

from semsearch.adapters.base import DimensionMismatchError
from semsearch.domain.models import SearchFilter

from .conftest import NS, make_record


@pytest.fixture
def store(store_factory, vocab_embedder):
    s = store_factory(vocab_embedder.dimensions)
    e = vocab_embedder
    s.upsert([
        make_record(e, "refund-policy", "refund policy deadline", tags=("finance",)),
        make_record(e, "vpn-runbook", "vpn error laptop", tags=("it",)),
        make_record(e, "pto-policy", "pto carryover policy deadline", tags=("hr",)),
        make_record(e, "sev1-runbook", "incident sev1 failover", acl=("it-oncall",), tags=("it",)),
        make_record(e, "retail-returns", "returns api refund error", tenant="retail", tags=("api",)),
        make_record(e, "logistics-tracking", "tracking api webhook error", tenant="logistics", tags=("api",)),
    ])
    return s


def ids(hits):
    return [h.doc_id for h in hits]


def test_nearest_first(store, vocab_embedder):
    hits = store.search(NS, vocab_embedder.embed_query("refund deadline"), k=3)
    assert hits[0].doc_id == "refund-policy"
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_tenant_filter_excludes_other_tenants(store, vocab_embedder):
    q = vocab_embedder.embed_query("api error")
    hits = store.search(NS, q, k=10, flt=SearchFilter(tenants=("retail", "shared")))
    assert "retail-returns" in ids(hits)
    assert "logistics-tracking" not in ids(hits)


def test_acl_filter_hides_restricted_documents(store, vocab_embedder):
    q = vocab_embedder.embed_query("incident sev1")
    visible = ids(store.search(NS, q, k=10, flt=SearchFilter(acl_groups=("all",))))
    assert "sev1-runbook" not in visible
    oncall = ids(store.search(NS, q, k=10, flt=SearchFilter(acl_groups=("all", "it-oncall"))))
    assert oncall[0] == "sev1-runbook"


def test_selective_filter_still_returns_matches(store, vocab_embedder):
    # The only allowed record is far from the query; a post-filtered ANN list would miss it.
    q = vocab_embedder.embed_query("refund policy deadline")
    hits = store.search(NS, q, k=3, flt=SearchFilter(tags_any=("api",), tenants=("logistics",)))
    assert ids(hits) == ["logistics-tracking"]


def test_namespaces_are_isolated(store, vocab_embedder):
    e = vocab_embedder
    store.upsert([make_record(e, "other-doc", "refund policy deadline", namespace="other:fake:v1")])
    assert "other-doc" not in ids(store.search(NS, e.embed_query("refund"), k=10))
    assert ids(store.search("other:fake:v1", e.embed_query("refund"), k=10)) == ["other-doc"]


def test_replace_document_removes_old_version(store, vocab_embedder):
    e = vocab_embedder
    new = [
        make_record(e, "pto-policy", "pto carryover deadline", version="2", ordinal=0),
        make_record(e, "pto-policy", "pto policy", version="2", ordinal=1),
    ]
    store.replace_document(NS, "pto-policy", "2", new)
    versions = store.doc_versions(NS)
    assert versions["pto-policy"].doc_version == "2"
    assert versions["pto-policy"].chunks == 2
    hits = store.search(NS, e.embed_query("pto carryover"), k=10, flt=SearchFilter(doc_ids=("pto-policy",)))
    assert {h.doc_version for h in hits} == {"2"}


def test_replace_document_rejects_foreign_records(store, vocab_embedder):
    rec = make_record(vocab_embedder, "vpn-runbook", "vpn", version="9")
    with pytest.raises(ValueError):
        store.replace_document(NS, "pto-policy", "9", [rec])


def test_delete_document(store, vocab_embedder):
    assert store.delete_document(NS, "vpn-runbook") == 1
    assert "vpn-runbook" not in store.doc_versions(NS)
    assert store.delete_document(NS, "vpn-runbook") == 0


def test_upsert_is_idempotent(store, vocab_embedder):
    before = store.count(NS)
    store.upsert([make_record(vocab_embedder, "vpn-runbook", "vpn error laptop", tags=("it",))])
    assert store.count(NS) == before


def test_dimension_mismatch_is_rejected(store, vocab_embedder):
    bad = make_record(vocab_embedder, "x", "vpn").model_copy(update={"vector": [1.0, 0.0]})
    with pytest.raises(DimensionMismatchError):
        store.upsert([bad])


def test_empty_namespace_and_zero_k(store, vocab_embedder):
    q = vocab_embedder.embed_query("vpn")
    assert store.search("missing:ns:v1", q, k=5) == []
    assert store.search(NS, q, k=0) == []


def test_exact_and_default_agree_on_small_data(store, vocab_embedder):
    q = vocab_embedder.embed_query("api error webhook")
    assert ids(store.search(NS, q, k=3)) == ids(store.search(NS, q, k=3, exact=True))
