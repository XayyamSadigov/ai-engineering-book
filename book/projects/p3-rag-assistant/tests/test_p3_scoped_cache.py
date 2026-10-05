# path: book/projects/p3-rag-assistant/tests/test_p3_scoped_cache.py
"""ScopedCache uses guardrails' public removal API, so metadata never outlives an entry."""
from __future__ import annotations

from guardrails import GuardContext

from rag_assistant.caching.caches import ScopedCache

RETAIL = GuardContext(tenant="retail", user_id="u1", groups=frozenset({"all"}))
LOGI = GuardContext(tenant="logistics", user_id="u2", groups=frozenset({"all"}))


def make(now: list[float]) -> ScopedCache[str]:
    return ScopedCache("answers", ttl_s=60, clock=lambda: now[0])


def test_invalidate_tenant_removes_metadata_too():
    now = [0.0]
    c = make(now)
    c.put(RETAIL, "a", "q", doc_ids=["hr-pto-policy"])
    c.put(LOGI, "b", "q", doc_ids=["logi-routes"])
    assert c.invalidate_tenant("retail") == 1
    assert len(c) == 1 and c.entries_for("hr-pto-policy") == 0
    assert c.get(RETAIL, "q") is None and c.misses == 1 and c.hits == 0  # a miss, not a phantom hit


def test_discard_ttl_and_clear():
    now = [0.0]
    c = make(now)
    c.put(RETAIL, "a", "q1", doc_ids=["d1"])
    c.put(RETAIL, "short", "q2", doc_ids=["d2"], ttl_s=5)
    assert c.discard(RETAIL, "q1") is True and c.get(RETAIL, "q1") is None
    now[0] = 6.0
    assert c.get(RETAIL, "q2") is None and len(c) == 0  # per-entry TTL expired and was dropped
    c.put(RETAIL, "a", "q1", doc_ids=["d1"])
    assert c.clear() == 1 and len(c) == 0 and c.keys() == []


def test_index_set_chunk_lookup_respects_acl_and_tombstones(container, employee):  # type: ignore[no-untyped-def]
    from ragkit.retrieval import Principal

    ix = container.index_set
    security = Principal(user_id="sec-1", tenant="shared", groups=["all", "security"])
    hits = [h for i in ix.existing(ix.active_version)
            for h in i.bm25.search("leavers access disabled within 24 hours", security, k=10)]
    policy = [h.chunk.id for h in hits if h.chunk.doc_id == "sec-access-control-policy"]
    faq = [h.chunk.id for h in hits if h.chunk.doc_id == "hr-faq"]
    assert policy and faq, "fixture premise: both documents are indexed and match"
    asked = [policy[0], faq[0], "no-such-doc:0"]
    assert [c.id for c in ix.get_chunks(asked, security)] == [policy[0], faq[0]]  # order as asked
    assert [c.id for c in ix.get_chunks(asked, employee)] == [faq[0]]  # restricted == absent
    container.ingestion.delete("hr-faq")  # logical delete: hidden before any purge runs
    assert ix.get_chunks(asked, employee) == []
