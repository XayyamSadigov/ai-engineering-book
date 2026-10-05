# path: book/projects/p2-semantic-search/semsearch/domain/filters.py
"""Filter semantics in one place so every adapter and every test agrees on them."""
from __future__ import annotations

from .models import SearchFilter, VectorRecord


def matches(record: VectorRecord, f: SearchFilter | None) -> bool:
    if f is None:
        return True
    if f.tenants is not None and record.tenant not in f.tenants:
        return False
    if f.acl_groups is not None and not set(record.acl_groups) & set(f.acl_groups):
        return False
    if f.tags_any is not None and not set(record.tags) & set(f.tags_any):
        return False
    if f.doc_ids is not None and record.doc_id not in f.doc_ids:
        return False
    return True


def visible_tenants(user_tenant: str, shared_tenant: str = "shared") -> tuple[str, ...]:
    """A tenant user sees their own tenant plus the cross-tenant 'shared' corpus."""
    if user_tenant == shared_tenant:
        return (shared_tenant,)
    return (user_tenant, shared_tenant)


__all__ = ["matches", "visible_tenants"]
