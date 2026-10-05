# path: book/projects/p3-rag-assistant/tests/test_p3_deletion.py
"""Deletion propagates to every store and cache, and nothing brings a deleted document back."""
from __future__ import annotations

from conftest import PTO_Q, inner_embeddings

from rag_assistant.domain.models import IngestPayload, now
from rag_assistant.retrieval.index import _bm25_doc_ids

UPLOAD = """---
id: hr-sabbatical-policy
title: Sabbatical Policy
version: "1.0"
updated_at: 2026-03-01
tenant: shared
acl_groups: ["all"]
tags: [hr, sabbatical, leave, policy]
---

# Sabbatical Policy

## Eligibility

Employees with seven years of continuous service may take an unpaid sabbatical of up to twelve weeks.
Requests go to People Operations at least three months before the planned start date.
"""
SABBATICAL_Q = "How long can an unpaid sabbatical last and who is eligible?"


def _traces(c, doc_id: str) -> dict[str, bool]:  # type: ignore[no-untyped-def]
    """Every place a document can live. All must be False after a purge."""
    return {
        "bm25": any(doc_id in _bm25_doc_ids(ix.bm25) for ix in c.index_set.existing()),
        "vectors": any(doc_id in ix.dense_doc_ids() for ix in c.index_set.existing()),
        "embedding_cache": c.embeddings.has_doc(doc_id),
        "blob": c.blobs.has_doc(doc_id),
        "request_caches": c.caches.entries_for(doc_id) > 0,
    }


def test_delete_leaves_no_trace_in_any_store_or_cache(container, employee):
    container.ingestion.submit_upload(UPLOAD.encode(), "sabbatical.md")
    container.drain()
    first = container.answers.ask(SABBATICAL_Q, employee)
    assert first.response.answer.action == "answer"
    assert "hr-sabbatical-policy" in first.retrieval.doc_ids
    assert container.answers.ask(SABBATICAL_Q, employee).response.cache == "hit"
    assert all(_traces(container, "hr-sabbatical-policy").values())

    container.ingestion.delete("hr-sabbatical-policy")
    # logically deleted at once: caches purged and retrieval hides it before the purge job runs
    assert not _traces(container, "hr-sabbatical-policy")["request_caches"]
    hidden = container.answers.ask(SABBATICAL_Q, employee)
    assert "hr-sabbatical-policy" not in hidden.retrieval.doc_ids

    container.drain()
    assert _traces(container, "hr-sabbatical-policy") == {k: False for k in _traces(container, "hr-sabbatical-policy")}
    rec = container.registry.get("hr-sabbatical-policy")
    assert rec.status == "deleted" and rec.chunk_ids == [] and rec.purged_at is not None
    after = container.answers.ask(SABBATICAL_Q, employee)
    assert "hr-sabbatical-policy" not in after.retrieval.doc_ids
    assert "hr-sabbatical-policy" not in {c.doc_id for c in after.response.answer.citations}
    assert "twelve weeks" not in after.response.answer.text


def test_stale_job_cannot_resurrect_a_deleted_document(container):
    rec = container.registry.get("hr-faq")
    stale_seq = container.registry.next_seq()  # a job planned before the delete, delivered after it
    container.ingestion.delete("hr-faq")
    container.drain()
    job = container.queue.enqueue("ingest.upsert", IngestPayload(
        op="upsert", doc_id="hr-faq", connector="folder", uri=rec.uri, seq=stale_seq,
        submitted_at=now()).model_dump(mode="json"))
    assert container.handlers.upsert(job)["change"] == "skipped_tombstone"
    assert container.registry.get("hr-faq").status == "deleted"
    assert not any("hr-faq" in _bm25_doc_ids(ix.bm25) for ix in container.index_set.existing())


def test_folder_sync_respects_the_tombstone(container):
    container.ingestion.delete("hr-faq")
    container.drain()
    out = container.ingestion.sync("folder")  # the file is still in the folder
    assert "hr-faq" in out["suppressed"]
    container.drain()
    assert container.registry.get("hr-faq").status == "deleted"


def test_source_deletion_propagates_on_sync(container, docs_dir, employee):
    (docs_dir / "travel-policy.md").unlink()
    out = container.ingestion.sync("folder")
    assert out["deleted"] == ["hr-travel-policy"]
    container.drain()
    assert not _traces(container, "hr-travel-policy")["vectors"]


def test_old_snapshot_is_reconciled_against_the_registry(make_container, tmp_path):
    c = make_container(snapshot_dir=tmp_path / "snap")
    old = {p.name: p.read_bytes() for p in (tmp_path / "snap" / "v1").iterdir()}
    c.ingestion.delete("hr-faq")
    c.drain()
    for name, data in old.items():  # restore a backup taken before the delete
        (tmp_path / "snap" / "v1" / name).write_bytes(data)
    c.registry.set_state("snapshot_gen", "999")  # make the API side reload it
    c.index_set.refresh()  # refresh() reconciles after loading
    assert not any("hr-faq" in _bm25_doc_ids(ix.bm25) for ix in c.index_set.existing())


def test_reupload_after_delete_is_a_new_ingestion(container, employee):
    container.ingestion.submit_upload(UPLOAD.encode(), "sabbatical.md")
    container.drain()
    container.ingestion.delete("hr-sabbatical-policy")
    container.drain()
    before = inner_embeddings(container).texts_embedded
    out = container.ingestion.submit_upload(UPLOAD.encode(), "sabbatical.md")
    assert not out["deduplicated"]  # the tombstone sequence is part of the idempotency key
    container.drain()
    assert container.registry.get("hr-sabbatical-policy").status == "active"
    assert inner_embeddings(container).texts_embedded > before  # the forgotten vectors are recomputed
    assert "hr-sabbatical-policy" in container.answers.ask(SABBATICAL_Q, employee).retrieval.doc_ids


def test_delete_also_removes_cached_answers_that_cited_it(container, employee):
    assert container.answers.ask(PTO_Q, employee).response.answer.action in ("answer", "answer_with_caveat")
    assert container.caches.entries_for("hr-pto-policy") >= 1
    container.ingestion.delete("hr-pto-policy")
    assert container.caches.entries_for("hr-pto-policy") == 0
    assert container.answers.ask(PTO_Q, employee).response.cache == "miss"
