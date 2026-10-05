# path: book/projects/guardrails/guardrails/tenancy.py
"""Tenant and ACL isolation helpers for retrieval results, caches and tool results.

The primary control is *filtering inside retrieval* (Chapter 15): forbidden records never get
scored. These helpers are the second line: assertions that run on whatever a retriever, cache
or tool returned and raise loudly if a record from another tenant, or outside the caller's
groups, got through. A raised `TenantIsolationError` is a bug report, not a user error.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Generic, Iterable, Sequence, TypeVar

from .pipeline import GuardContext

T = TypeVar("T")
SHARED_TENANTS: frozenset[str] = frozenset({"shared"})


class TenantIsolationError(PermissionError):
    def __init__(self, message: str, record_ids: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.record_ids = list(record_ids)


def _get(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        if key in record:
            return record[key]
        meta = record.get("metadata")
        return meta.get(key, default) if isinstance(meta, dict) else default
    if hasattr(record, key):
        return getattr(record, key)
    meta = getattr(record, "metadata", None)
    return meta.get(key, default) if isinstance(meta, dict) else default


def _rid(record: Any) -> str:
    return str(_get(record, "id") or _get(record, "doc_id") or _get(record, "chunk_id") or "?")


def in_scope(ctx: GuardContext, record: Any, tenant_key: str = "tenant", groups_key: str = "acl_groups",
             shared: frozenset[str] = SHARED_TENANTS) -> bool:
    tenant = _get(record, tenant_key)
    if tenant is None:
        return False                                  # untagged data is out of scope: deny by default
    if tenant != ctx.tenant and tenant not in shared:
        return False
    groups = _get(record, groups_key)
    if groups is None:
        return False
    return bool(set(groups) & set(ctx.groups))


def assert_tenant_scope(ctx: GuardContext, records: Iterable[T], tenant_key: str = "tenant",
                        groups_key: str = "acl_groups", shared: frozenset[str] = SHARED_TENANTS) -> list[T]:
    """Return `records` unchanged if every one is in scope; raise otherwise. Never filters silently:
    a silent filter hides the retrieval bug that let the record through."""
    records = list(records)
    bad = [_rid(r) for r in records if not in_scope(ctx, r, tenant_key, groups_key, shared)]
    if bad:
        raise TenantIsolationError(
            f"{len(bad)} record(s) outside scope tenant={ctx.tenant} groups={sorted(ctx.groups)}", bad)
    return records


def tenant_guarded(fn: Callable[..., Iterable[T]]) -> Callable[..., list[T]]:
    """Decorator for retrieval functions with signature `fn(ctx, *args, **kwargs)`."""

    def wrapper(ctx: GuardContext, *args: Any, **kwargs: Any) -> list[T]:
        return assert_tenant_scope(ctx, fn(ctx, *args, **kwargs))

    wrapper.__name__ = getattr(fn, "__name__", "tenant_guarded")
    wrapper.__doc__ = fn.__doc__
    return wrapper


def scoped_cache_key(ctx: GuardContext, namespace: str, *parts: Any, per_user: bool = False,
                     policy_version: str = "v1") -> str:
    """Cache key that includes the full authorization context. Two callers who could see
    different documents can never share an entry. `per_user=True` for anything personalized."""
    payload = {
        "ns": namespace,
        "tenant": ctx.tenant,
        "groups": sorted(ctx.groups),
        "user": ctx.user_id if per_user else None,
        "policy": policy_version,
        "parts": parts,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    return f"{namespace}:{ctx.tenant}:{digest[:32]}"


@dataclass
class _Entry(Generic[T]):
    tenant: str
    groups: frozenset[str]
    value: T


class TenantScopedCache(Generic[T]):
    """A dict-backed cache whose keys come from `scoped_cache_key` and whose entries remember the
    authorization context that produced them. `get` re-checks the context on read, so even a key
    collision or a caller who builds keys by hand cannot read across tenants."""

    def __init__(self, namespace: str, per_user: bool = False, policy_version: str = "v1") -> None:
        self.namespace = namespace
        self.per_user = per_user
        self.policy_version = policy_version
        self._store: dict[str, _Entry[T]] = {}

    def key(self, ctx: GuardContext, *parts: Any) -> str:
        return scoped_cache_key(ctx, self.namespace, *parts, per_user=self.per_user,
                                policy_version=self.policy_version)

    def get(self, ctx: GuardContext, *parts: Any) -> T | None:
        entry = self._store.get(self.key(ctx, *parts))
        if entry is None:
            return None
        if entry.tenant != ctx.tenant or entry.groups != frozenset(ctx.groups):
            raise TenantIsolationError("cache entry authorization context does not match caller")
        return entry.value

    def set(self, ctx: GuardContext, value: T, *parts: Any) -> None:
        self._store[self.key(ctx, *parts)] = _Entry(ctx.tenant, frozenset(ctx.groups), value)

    def discard(self, ctx: GuardContext, *parts: Any) -> bool:
        """Remove the entry this caller would read for these parts. True if something was removed."""
        return self.discard_key(self.key(ctx, *parts))

    def discard_key(self, key: str) -> bool:
        """Remove one entry by its full key (as returned by `key`). Every removal path goes through
        here, so a subclass that keeps per-key metadata (TTL, tags) overrides this one method."""
        return self._store.pop(key, None) is not None

    def keys(self) -> list[str]:
        return list(self._store)

    def clear(self) -> int:
        """Drop everything (for example after a permission-model change). Returns the count removed."""
        doomed = list(self._store)
        for k in doomed:
            self.discard_key(k)
        return len(doomed)

    def invalidate_tenant(self, tenant: str) -> int:
        """Deletion must reach caches too: drop every entry produced for a tenant."""
        doomed = [k for k, e in self._store.items() if e.tenant == tenant]
        for k in doomed:
            self.discard_key(k)
        return len(doomed)

    def __len__(self) -> int:
        return len(self._store)


__all__ = [
    "SHARED_TENANTS", "TenantIsolationError", "in_scope", "assert_tenant_scope", "tenant_guarded",
    "scoped_cache_key", "TenantScopedCache",
]
