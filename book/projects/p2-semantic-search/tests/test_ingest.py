# path: book/projects/p2-semantic-search/tests/test_ingest.py
from __future__ import annotations

from aie_core.embeddings import FakeEmbeddings
from aie_core.observability import InMemoryTracer

from semsearch.adapters.numpy_store import NumpyVectorStore
from semsearch.ingest.loader import load_corpus
from semsearch.ingest.pipeline import ingest

NS = "knowledge:fake-embedding:v1"


def write(d, name, doc_id, body, version="1", acl='["all"]', tenant="shared"):
    (d / name).write_text(
        f"---\nid: {doc_id}\ntitle: {doc_id}\nversion: \"{version}\"\ntenant: {tenant}\n"
        f"acl_groups: {acl}\ntags: [t]\n---\n\n# {doc_id}\n\n{body}\n",
        encoding="utf-8",
    )


def test_incremental_ingest_and_prune(tmp_path):
    src = tmp_path / "docs"
    src.mkdir()
    write(src, "a.md", "doc-a", "refund policy deadline")
    write(src, "b.md", "doc-b", "vpn error 412")
    emb = FakeEmbeddings(dimensions=128)
    store = NumpyVectorStore(emb.dimensions, persist_dir=tmp_path / "idx")
    tracer = InMemoryTracer()

    r1 = ingest(load_corpus(src), store, emb, NS, tracer=tracer)
    assert sorted(r1.added) == ["doc-a", "doc-b"]
    assert tracer.find("ingest")[0].attributes["added"] == 2
    assert (tmp_path / "idx").exists()  # persisted after ingest

    calls_before = len(emb.calls)
    r2 = ingest(load_corpus(src), store, emb, NS)
    assert sorted(r2.unchanged) == ["doc-a", "doc-b"]
    assert len(emb.calls) == calls_before  # nothing re-embedded

    write(src, "a.md", "doc-a", "refund policy deadline is 30 days")  # content edit, same version
    (src / "b.md").unlink()
    r3 = ingest(load_corpus(src), store, emb, NS, prune=True)
    assert r3.updated == ["doc-a"]
    assert r3.deleted == ["doc-b"]
    assert set(store.doc_versions(NS)) == {"doc-a"}
    hits = store.search(NS, emb.embed_query("refund"), 10)
    assert any("30 days" in h.text for h in hits)
    assert {h.doc_version for h in hits} == {store.doc_versions(NS)["doc-a"].doc_version}


def test_without_prune_deleted_sources_stay_searchable(tmp_path):
    src = tmp_path / "docs"
    src.mkdir()
    write(src, "a.md", "doc-a", "refund")
    write(src, "b.md", "doc-b", "revoked content")
    emb = FakeEmbeddings(dimensions=64)
    store = NumpyVectorStore(emb.dimensions)
    ingest(load_corpus(src), store, emb, NS)
    (src / "b.md").unlink()
    ingest(load_corpus(src), store, emb, NS)  # no prune: the classic stale-index bug
    assert "doc-b" in store.doc_versions(NS)


def test_records_carry_acl_tenant_and_model(tmp_path):
    src = tmp_path / "docs"
    src.mkdir()
    write(src, "a.md", "doc-a", "incident sev1", acl='["it-oncall"]', tenant="logistics")
    emb = FakeEmbeddings(dimensions=64)
    store = NumpyVectorStore(emb.dimensions)
    ingest(load_corpus(src), store, emb, NS)
    hit = store.search(NS, emb.embed_query("incident sev1"), 1)[0]
    assert hit.tenant == "logistics"
    assert hit.metadata["declared_version"] == "1"
    assert hit.doc_version.startswith("1+")
