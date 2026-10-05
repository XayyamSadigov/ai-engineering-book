# path: book/projects/guardrails/tests/test_tenancy.py
"""Tenant isolation tests: the CI tripwires Chapter 26 asked for (threats R3 and R4)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from guardrails import (GuardContext, TenantIsolationError, TenantScopedCache, assert_tenant_scope, in_scope,
                        scoped_cache_key, tenant_guarded)

RETAIL = GuardContext(tenant="retail", user_id="r1", groups=frozenset({"all"}))
RETAIL_HR = GuardContext(tenant="retail", user_id="r2", groups=frozenset({"all", "hr"}))
LOGISTICS = GuardContext(tenant="logistics", user_id="l1", groups=frozenset({"all"}))

RECORDS = [
    {"id": "r-doc", "tenant": "retail", "acl_groups": ["all"]},
    {"id": "s-doc", "tenant": "shared", "acl_groups": ["all"]},
    {"id": "l-doc", "tenant": "logistics", "acl_groups": ["all"]},
    {"id": "hr-doc", "tenant": "retail", "acl_groups": ["hr"]},
    {"id": "untagged"},
]


def test_in_scope_matrix():
    assert [r["id"] for r in RECORDS if in_scope(RETAIL, r)] == ["r-doc", "s-doc"]
    assert [r["id"] for r in RECORDS if in_scope(RETAIL_HR, r)] == ["r-doc", "s-doc", "hr-doc"]
    assert [r["id"] for r in RECORDS if in_scope(LOGISTICS, r)] == ["s-doc", "l-doc"]


def test_assert_raises_with_offending_ids():
    with pytest.raises(TenantIsolationError) as exc:
        assert_tenant_scope(RETAIL, RECORDS)
    assert set(exc.value.record_ids) == {"l-doc", "hr-doc", "untagged"}


def test_metadata_and_attribute_records():
    class Chunk:
        def __init__(self):
            self.chunk_id = "c1"
            self.metadata = {"tenant": "retail", "acl_groups": ["all"]}
    assert assert_tenant_scope(RETAIL, [Chunk()])


def test_decorated_retriever_with_missing_filter_fails_loudly():
    @tenant_guarded
    def buggy_retrieve(ctx, query):          # forgot the tenant filter
        return RECORDS[:3]

    @tenant_guarded
    def good_retrieve(ctx, query):
        return [r for r in RECORDS if in_scope(ctx, r)]

    with pytest.raises(TenantIsolationError):
        buggy_retrieve(RETAIL, "q")
    assert [r["id"] for r in good_retrieve(LOGISTICS, "q")] == ["s-doc", "l-doc"]


def test_cache_key_includes_authorization_context():
    q = "what is the bonus policy?"
    keys = {scoped_cache_key(c, "answers", q) for c in (RETAIL, RETAIL_HR, LOGISTICS)}
    assert len(keys) == 3
    assert scoped_cache_key(RETAIL, "answers", q) == scoped_cache_key(
        GuardContext(tenant="retail", user_id="other", groups=frozenset({"all"})), "answers", q)
    assert scoped_cache_key(RETAIL, "answers", q, per_user=True) != scoped_cache_key(
        GuardContext(tenant="retail", user_id="other", groups=frozenset({"all"})), "answers", q, per_user=True)


def test_scoped_cache_never_serves_across_acl():
    cache: TenantScopedCache[str] = TenantScopedCache("answers")
    cache.set(RETAIL_HR, "answer built from HR documents", "bonus?")
    assert cache.get(RETAIL_HR, "bonus?") == "answer built from HR documents"
    assert cache.get(RETAIL, "bonus?") is None
    assert cache.get(LOGISTICS, "bonus?") is None


def test_scoped_cache_rechecks_on_read():
    cache: TenantScopedCache[str] = TenantScopedCache("answers")
    cache.set(RETAIL, "v", "q")
    key = cache.key(RETAIL, "q")
    cache._store[key].tenant = "logistics"     # simulate a corrupted or hand-built entry
    with pytest.raises(TenantIsolationError):
        cache.get(RETAIL, "q")


def test_invalidate_tenant_for_deletion():
    cache: TenantScopedCache[str] = TenantScopedCache("answers")
    cache.set(RETAIL, "a", "q1")
    cache.set(LOGISTICS, "b", "q1")
    assert cache.invalidate_tenant("retail") == 1 and len(cache) == 1


def test_discard_and_clear_are_public_and_scoped():
    cache: TenantScopedCache[str] = TenantScopedCache("answers")
    cache.set(RETAIL, "a", "q1")
    cache.set(RETAIL_HR, "hr", "q1")
    assert cache.discard(RETAIL, "q1") is True and cache.discard(RETAIL, "q1") is False
    assert cache.get(RETAIL_HR, "q1") == "hr"  # another scope's entry for the same question survives
    assert cache.keys() == [cache.key(RETAIL_HR, "q1")]
    assert cache.discard_key(cache.key(RETAIL_HR, "q1")) is True and len(cache) == 0
    cache.set(RETAIL, "a", "q1")
    cache.set(LOGISTICS, "b", "q2")
    assert cache.clear() == 2 and len(cache) == 0


def test_removal_paths_go_through_discard_key():
    class Tracking(TenantScopedCache[str]):
        def __init__(self) -> None:
            super().__init__("answers")
            self.removed: list[str] = []

        def discard_key(self, key: str) -> bool:
            self.removed.append(key)
            return super().discard_key(key)

    cache = Tracking()
    cache.set(RETAIL, "a", "q1")
    cache.set(LOGISTICS, "b", "q1")
    cache.invalidate_tenant("retail")
    cache.clear()
    assert cache.removed == [cache.key(RETAIL, "q1"), cache.key(LOGISTICS, "q1")]


def test_shared_corpus_cross_tenant_retrieval():
    shared = Path(__file__).resolve().parents[2] / "shared-data"
    if not (shared / "shared_data.py").exists():
        pytest.skip("shared-data not present")
    sys.path.insert(0, str(shared))
    from shared_data import load_docs

    docs = [d.model_dump() for d in load_docs()]
    visible_retail = [d for d in docs if in_scope(RETAIL, d)]
    assert visible_retail, "retail users see something"
    assert all(d["tenant"] in {"retail", "shared"} for d in visible_retail)
    with pytest.raises(TenantIsolationError):
        assert_tenant_scope(RETAIL, docs)        # the unfiltered corpus must never pass
