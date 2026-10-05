# path: book/projects/p2-semantic-search/tests/test_api.py
from __future__ import annotations

import pytest
from aie_core.observability import InMemoryTracer
from fastapi.testclient import TestClient

from semsearch.adapters.numpy_store import NumpyVectorStore
from semsearch.api.app import create_app
from semsearch.service import SearchService

from .conftest import NS, make_record


@pytest.fixture
def tracer():
    return InMemoryTracer()


@pytest.fixture
def client(vocab_embedder, tracer):
    e = vocab_embedder
    store = NumpyVectorStore(e.dimensions)
    store.upsert([
        make_record(e, "refund-policy", "refund policy deadline", tags=("finance",)),
        make_record(e, "sev1-runbook", "incident sev1 failover", acl=("it-oncall",), tags=("it",)),
        make_record(e, "retail-returns", "returns api refund error", tenant="retail", tags=("api",)),
        make_record(e, "logistics-tracking", "tracking api webhook error", tenant="logistics", tags=("api",)),
    ])
    svc = SearchService(store, e, NS, default_k=5, max_k=10, tracer=tracer)
    return TestClient(create_app(svc))


def headers(tenant="shared", groups="all"):
    return {"X-User": "u1", "X-Tenant": tenant, "X-Groups": groups}


def doc_ids(resp):
    return [h["doc_id"] for h in resp.json()["hits"]]


def test_missing_identity_is_rejected(client):
    assert client.post("/search", json={"query": "refund"}).status_code == 401
    r = client.post("/search", json={"query": "refund"}, headers={"X-User": "u", "X-Tenant": "retail", "X-Groups": " , "})
    assert r.status_code == 401


def test_tenant_and_acl_come_from_identity(client):
    r = client.post("/search", json={"query": "api error refund"}, headers=headers("retail", "all"))
    assert r.status_code == 200
    got = doc_ids(r)
    assert "retail-returns" in got
    assert "logistics-tracking" not in got
    assert "sev1-runbook" not in got


def test_oncall_group_sees_runbook(client):
    r = client.post("/search", json={"query": "incident sev1"}, headers=headers("shared", "all,it-oncall"))
    assert doc_ids(r)[0] == "sev1-runbook"


def test_tags_narrow_results(client):
    r = client.post("/search", json={"query": "refund", "tags": ["api"]}, headers=headers("retail"))
    assert doc_ids(r) == ["retail-returns"]


def test_k_is_capped_and_validated(client):
    assert client.post("/search", json={"query": "refund", "k": 0}, headers=headers()).status_code == 422
    assert client.post("/search", json={"query": ""}, headers=headers()).status_code == 422
    r = client.post("/search", json={"query": "refund", "k": 200}, headers=headers())
    assert r.status_code == 200 and len(r.json()["hits"]) <= 10


def test_search_span_records_underfill(client, tracer):
    client.post("/search", json={"query": "refund", "k": 5}, headers=headers())
    span = tracer.find("search")[-1]
    assert span.attributes["underfilled"] is True  # only one doc visible to shared/all
    assert span.attributes["doc_ids"] == ["refund-policy"]


def test_healthz(client, vocab_embedder):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["chunks"] == 4
    empty = SearchService(NumpyVectorStore(vocab_embedder.dimensions), vocab_embedder, NS)
    assert TestClient(create_app(empty)).get("/healthz").status_code == 503
