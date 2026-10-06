# path: book/projects/p3-rag-assistant/tests/test_p3_ingestion.py
"""Ingestion: idempotency, incremental updates, ACL changes, authority metadata, blue/green."""
from __future__ import annotations

import pytest
from conftest import PTO_Q, edit_doc, inner_embeddings
from ragkit.retrieval import Principal

from rag_assistant.ingestion.service import ReindexIncomplete


def test_full_sync_indexes_every_document_once(container):
    status = container.status()
    assert status.documents_active == 24
    assert status.dead_letters == 0
    assert status.freshness_slo_met
    assert sum(status.chunks_indexed.values()) == sum(len(r.chunk_ids) for r in container.registry.all("active"))


def test_resubmitting_unchanged_sources_is_free(container):
    before = inner_embeddings(container).texts_embedded
    gens = container.registry.generations()
    out = container.ingestion.sync("folder")
    assert all(o for o in out["submitted"])
    assert container.queue.depth() == 0  # every upsert was deduplicated at enqueue (same idempotency key)
    assert container.drain() == 0
    assert inner_embeddings(container).texts_embedded == before
    assert container.registry.generations() == gens  # nothing changed, so no cache was invalidated
    assert out["submitted"]


def test_handler_is_idempotent_on_redelivery(container):
    """At-least-once delivery: running the same job twice converges to the same state."""
    rec = container.registry.get("hr-pto-policy")
    job = container.queue.enqueue("ingest.upsert", {
        "op": "upsert", "doc_id": rec.doc_id, "connector": "folder", "uri": rec.uri,
        "content_hash": rec.content_hash, "seq": container.registry.next_seq(), "submitted_at": 0.0,
    })
    first = container.handlers.upsert(job)
    second = container.handlers.upsert(job)
    assert first["change"] == "unchanged" and second["change"] == "unchanged"
    assert container.registry.get("hr-pto-policy").chunk_ids == rec.chunk_ids


def test_update_reembeds_only_the_changed_chunks(container, docs_dir):
    rec = container.registry.get("hr-pto-policy")
    before = inner_embeddings(container).texts_embedded
    edit_doc(docs_dir / "pto-policy.md", "Questions go to People Operations through the Beacon portal",
             "Questions go to the People Operations desk through the Beacon portal")
    container.ingestion.sync("folder")
    assert container.drain() == 1
    after = container.registry.get("hr-pto-policy")
    added = set(after.chunk_ids) - set(rec.chunk_ids)
    removed = set(rec.chunk_ids) - set(after.chunk_ids)
    assert len(added) == 1 and len(removed) == 1  # one section changed, one chunk replaced
    assert inner_embeddings(container).texts_embedded - before == 1  # unchanged chunks hit the embedding cache
    assert after.content_hash != rec.content_hash


def test_acl_change_reaches_indexes_without_reembedding(container, docs_dir, employee):
    assert "hr-pto-policy" in container.answers.ask(PTO_Q, employee).retrieval.doc_ids
    before = inner_embeddings(container).texts_embedded
    edit_doc(docs_dir / "pto-policy.md", 'acl_groups: ["all"]', 'acl_groups: ["hr"]')
    container.ingestion.sync("folder")
    container.drain()
    out = container.answers.ask(PTO_Q, employee)
    assert "hr-pto-policy" not in out.retrieval.doc_ids  # the cached result from before was not reused
    assert out.response.cache == "miss"
    assert inner_embeddings(container).texts_embedded == before  # same text, same vectors
    hr = Principal(user_id="hr-1", tenant="retail", groups=["all", "hr"])
    assert "hr-pto-policy" in container.answers.ask(PTO_Q, hr).retrieval.doc_ids


def test_invalid_document_without_acl_is_rejected_before_queueing(container):
    with pytest.raises(Exception, match="no tenant or acl_groups"):
        container.ingestion.submit_upload(b"---\nid: x-no-acl\ntitle: X\n---\n\n# X\n\nSome text here.", "x.md")


def test_authority_metadata_fixes_faq_over_policy(container, employee, make_container):
    rec = container.registry.get("hr-pto-policy")
    assert rec.authority == 3 and rec.effective_date == "2026-01-01" and rec.supersedes == ["hr-faq"]
    ranked = container.answers.ask(PTO_Q, employee).retrieval.doc_ids
    assert ranked[0] == "hr-pto-policy"
    # without the authority layer the FAQ, the closer textual match, ranks first (Chapters 10-14)
    plain = make_container(authority_weight=0.0, authority_rules_path=_empty_rules(container))
    assert plain.answers.ask(PTO_Q, employee).retrieval.doc_ids[0] == "hr-faq"


def _empty_rules(container) -> str:  # type: ignore[no-untyped-def]
    path = container.settings.docs_dir.parent / "no_rules.json"
    path.write_text('{"levels": [], "default_level": 1, "supersessions": []}', encoding="utf-8")
    return str(path)


def test_blue_green_reindex_dual_writes_and_promotes(container, docs_dir, employee):
    assert container.ingestion.start_reindex("v2") == 24
    with pytest.raises(ReindexIncomplete):
        container.ingestion.promote()
    # a change during the rebuild must land in both versions
    edit_doc(docs_dir / "pto-policy.md", "Questions go to People Operations", "Questions go to People Ops")
    container.ingestion.sync("folder")
    container.drain()
    assert set(container.registry.get("hr-pto-policy").index_versions) == {"v1", "v2"}
    assert container.answers.ask(PTO_Q, employee).response.index_version == "v1"
    assert container.ingestion.promote() == "v2"
    out = container.answers.ask(PTO_Q, employee)
    assert out.response.index_version == "v2" and out.retrieval.doc_ids[0] == "hr-pto-policy"
    assert container.ingestion.rollback() == "v1"
    assert container.answers.ask(PTO_Q, employee).response.index_version == "v1"


def test_reverting_a_document_is_not_deduplicated(container, docs_dir, employee):
    """A -> B -> A: the second A is a new change, not a duplicate of the first (succeeded) A job."""
    a = container.registry.get("hr-pto-policy")
    edit_doc(docs_dir / "pto-policy.md", "Questions go to People Operations", "Questions go to People Ops")
    container.ingestion.sync("folder")
    assert container.drain() == 1
    assert container.registry.get("hr-pto-policy").fingerprint != a.fingerprint
    edit_doc(docs_dir / "pto-policy.md", "Questions go to People Ops", "Questions go to People Operations")
    out = container.ingestion.submit_uri("folder", a.uri)
    assert not out["deduplicated"]
    assert container.drain() == 1
    back = container.registry.get("hr-pto-policy")
    assert back.fingerprint == a.fingerprint and back.chunk_ids == a.chunk_ids
    served = container.index_set.get_chunks(a.chunk_ids, employee)
    assert len(served) == len(a.chunk_ids)
    assert any("Questions go to People Operations" in c.text for c in served)
