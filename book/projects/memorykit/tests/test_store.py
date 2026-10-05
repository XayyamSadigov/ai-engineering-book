# path: book/projects/memorykit/tests/test_store.py
"""Store contract tests. Every test runs against both InMemoryStore and SQLiteStore."""
from __future__ import annotations

import json
from datetime import timedelta

import pytest

from memorykit import MemoryKind, MemoryRecord, MemoryStatus, Owner, Source, SQLiteStore, VersionConflict


def rec(owner: Owner, content: str, clock, **kw) -> MemoryRecord:
    return MemoryRecord(owner=owner, kind=kw.pop("kind", MemoryKind.SEMANTIC), content=content,
                        source=kw.pop("source", Source.USER_STATED), created_at=clock(), updated_at=clock(), **kw)


def test_tenant_and_user_isolation(store, clock, ana, ben, ana_logistics):
    r = store.put(rec(ana, "Ana prefers Spanish", clock))
    assert store.get(ana, r.id) is not None
    assert store.get(ana_logistics, r.id) is None   # same user id, other tenant
    assert store.get(ben, r.id) is None             # same tenant, other user
    assert store.query(ana_logistics, now=clock()) == []
    assert store.query(ben, now=clock()) == []
    assert [x.id for x in store.query(ana, now=clock())] == [r.id]


def test_shared_records_only_when_requested(store, clock, ana):
    shared = store.put(rec(ana.shared(), "Retail POS restarts happen at 03:00", clock, source=Source.SYSTEM_OF_RECORD))
    assert store.query(ana, now=clock()) == []
    assert [x.id for x in store.query(ana, include_shared=True, now=clock())] == [shared.id]
    assert store.query(Owner(tenant="logistics", user="ana"), include_shared=True, now=clock()) == []


def test_record_id_cannot_move_across_tenants(store, clock, ana, ana_logistics):
    r = store.put(rec(ana, "Ana works night shift", clock))
    hijack = r.model_copy(update={"owner": ana_logistics})
    with pytest.raises(PermissionError):
        store.put(hijack)


def test_expired_records_are_invisible_before_purge(store, clock, ana):
    r = store.put(rec(ana, "Temporary laptop replacement loaner", clock, expires_at=clock() + timedelta(days=7)))
    assert store.query(ana, now=clock()) != []
    clock.advance(days=8)
    assert store.query(ana, now=clock()) == []                        # hidden at once
    assert store.query(ana, include_expired=True, now=clock())[0].id == r.id
    assert store.purge_expired(clock()) == 1                          # removed by the job
    assert store.query(ana, include_expired=True, now=clock()) == []


def test_hard_delete_leaves_content_free_tombstone(store, clock, ana):
    r = store.put(rec(ana, "Ana's office is in Lisbon", clock, key="office", value="Lisbon", kind=MemoryKind.PROFILE))
    stones = store.delete(ana, r.id, reason="user request", now=clock())
    assert len(stones) == 1 and stones[0].record_id == r.id
    assert "Lisbon" not in json.dumps(stones[0].model_dump(mode="json"))
    assert store.get(ana, r.id) is None
    # The same fact proposed again is recognized as previously deleted.
    again = rec(ana, "office: lisbon", clock, key="office", value="lisbon", kind=MemoryKind.PROFILE)
    assert store.is_suppressed(again)
    assert not store.is_suppressed(again.model_copy(update={"value": "Berlin"}))


def test_delete_propagates_to_derived_records(store, clock, ana):
    base = store.put(rec(ana, "Ana's manager is Ben", clock))
    child = store.put(rec(ana, "Summary: Ana reports to Ben", clock, source=Source.MODEL_INFERRED, provenance=[f"mem:{base.id}"]))
    shared_child = store.put(rec(ana.shared(), "Team note citing Ana", clock, source=Source.MODEL_INFERRED, provenance=[f"mem:{child.id}"]))
    unrelated = store.put(rec(ana, "Ana prefers Spanish", clock))
    stones = store.delete(ana, base.id, reason="user request", now=clock())
    assert {s.record_id for s in stones} == {base.id, child.id, shared_child.id}
    assert store.get(ana, child.id) is None and store.get(ana.shared(), shared_child.id) is None
    assert store.get(ana, unrelated.id) is not None
    assert all("derived from" in s.reason for s in stones if s.record_id != base.id)


def test_delete_by_wrong_owner_is_a_noop(store, clock, ana, ben):
    r = store.put(rec(ana, "Ana prefers Spanish", clock))
    assert store.delete(ben, r.id, reason="x", now=clock()) == []
    assert store.get(ana, r.id) is not None


def test_delete_owner_erases_one_user_only(store, clock, ana, ben):
    r = store.put(rec(ana, "Ana prefers Spanish", clock))
    store.delete(ana, r.id, reason="x", now=clock())
    store.put(rec(ana, "Ana works night shift", clock))
    keep = store.put(rec(ben, "Ben uses a docking station", clock))
    assert store.delete_owner(ana) == 1
    assert store.query(ana, statuses=tuple(MemoryStatus), include_expired=True, now=clock()) == []
    assert store.tombstones(ana) == []
    assert store.get(ben, keep.id) is not None


def test_export_has_records_and_deletions_but_no_vectors(store, clock, ana):
    store.put(rec(ana, "Ana prefers Spanish", clock, embedding=[0.1, 0.2], embedding_model="m"))
    gone = store.put(rec(ana, "Ana's office is in Lisbon", clock))
    store.delete(ana, gone.id, reason="user request", now=clock())
    data = store.export(ana)
    assert [r["content"] for r in data["records"]] == ["Ana prefers Spanish"]
    assert "embedding" not in data["records"][0]
    assert data["deleted"][0]["record_id"] == gone.id
    json.dumps(data)  # serializable as delivered to the user


def test_optimistic_concurrency(store, clock, ana):
    r = store.put(rec(ana, "Ana prefers Spanish", clock))
    store.put(r.model_copy(update={"version": 2, "salience": 0.9}), expected_version=1)
    with pytest.raises(VersionConflict):
        store.put(r.model_copy(update={"version": 2, "salience": 0.1}), expected_version=1)


def test_sqlite_persists_across_connections(tmp_path, clock, ana):
    path = str(tmp_path / "memory.db")
    r = SQLiteStore(path).put(rec(ana, "Ana prefers Spanish", clock, value={"lang": "es"}, provenance=["turn:s1#2"]))
    loaded = SQLiteStore(path).get(ana, r.id)
    assert loaded is not None and loaded.value == {"lang": "es"} and loaded.provenance == ["turn:s1#2"]
    assert loaded.created_at == r.created_at
