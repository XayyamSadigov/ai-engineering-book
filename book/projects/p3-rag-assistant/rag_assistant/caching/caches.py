# path: book/projects/p3-rag-assistant/rag_assistant/caching/caches.py
"""Three caches, three key disciplines.

1. Embeddings (`ForgettableEmbeddings`): key = embedding-space fingerprint + exact text
   (aie_core.CachedEmbeddings). Safe to share across tenants because a vector depends only on
   the text. What it adds is deletion: vectors are derived data, so the cache remembers which
   document each chunk text came from and can forget them.
2. Retrieval results and 3. answers (`ScopedCache`): key = guardrails.scoped_cache_key over
   the caller's tenant and groups, plus the normalized query, index version, and the
   generation counters of every partition the caller can read. Entries are tagged with the
   documents they contain, so a delete purges them physically, and stamped with the
   generations, so a replica that sees a newer generation can sweep what became unreachable.

Freshness has two mechanisms on purpose: generations make any committed change invisible to
new lookups immediately; TTLs bound how long a correct-but-aging answer can live.
"""
from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from collections.abc import Iterable, Iterator, MutableMapping
from typing import Any, Callable, Generic, TypeVar

from aie_core.embeddings import CachedEmbeddings, EmbeddingClient
from guardrails import GuardContext, TenantScopedCache
from ragkit.retrieval import Principal

T = TypeVar("T")
_TAG_PREFIX = "doc-tags:"


def guard_context(principal: Principal, request_id: str = "") -> GuardContext:
    """The authorization context, built from the verified principal and nothing else."""
    return GuardContext(tenant=principal.tenant, user_id=principal.user_id,
                        groups=frozenset(principal.groups), request_id=request_id)


# ============================================================================ embeddings
class ForgettableEmbeddings(CachedEmbeddings):
    """CachedEmbeddings plus a per-document index of cache keys, kept in the same store."""

    def __init__(self, inner: EmbeddingClient, store: MutableMapping[str, bytes] | None = None, **kwargs: Any) -> None:
        super().__init__(inner, store, **kwargs)
        self._lock = threading.Lock()

    def _tags(self, doc_id: str) -> set[str]:
        raw = self.store.get(_TAG_PREFIX + doc_id)
        return set(json.loads(raw)) if raw else set()

    def track(self, doc_id: str, texts: Iterable[str]) -> None:
        """Record that these texts (already embedded) belong to doc_id; forget the doc's old texts."""
        keep = {self._key(t) for t in texts}
        with self._lock:
            for key in self._tags(doc_id) - keep:
                self.store.pop(key, None)
            self.store[_TAG_PREFIX + doc_id] = json.dumps(sorted(keep)).encode("utf-8")

    def forget_doc(self, doc_id: str) -> int:
        with self._lock:
            keys = self._tags(doc_id)
            for key in keys:
                self.store.pop(key, None)
            self.store.pop(_TAG_PREFIX + doc_id, None)
        return len(keys)

    def has_doc(self, doc_id: str) -> bool:
        return bool(self._tags(doc_id))


# ============================================================================ scoped caches
class _Meta:
    __slots__ = ("expires_at", "doc_ids", "stamp")

    def __init__(self, expires_at: float, doc_ids: frozenset[str], stamp: Any) -> None:
        self.expires_at = expires_at
        self.doc_ids = doc_ids
        self.stamp = stamp


class ScopedCache(TenantScopedCache[T], Generic[T]):
    """guardrails.TenantScopedCache (authorization-checked reads) with TTL, document tags,
    generation stamps and a size bound. In-process: one per API replica."""

    def __init__(self, namespace: str, *, ttl_s: float, max_entries: int = 10_000, policy_version: str = "v1",
                 clock: Callable[[], float] = time.monotonic) -> None:
        super().__init__(namespace, per_user=False, policy_version=policy_version)
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self._clock = clock
        self._meta: OrderedDict[str, _Meta] = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def get(self, ctx: GuardContext, *parts: Any) -> T | None:  # type: ignore[override]
        key = self.key(ctx, *parts)
        with self._lock:
            meta = self._meta.get(key)
            if meta is None or meta.expires_at <= self._clock():
                if meta is not None:
                    self._drop(key)
                self.misses += 1
                return None
            value = super().get(ctx, *parts)  # raises TenantIsolationError on a context mismatch
            self._meta.move_to_end(key)
            self.hits += 1
            return value

    def put(self, ctx: GuardContext, value: T, *parts: Any, doc_ids: Iterable[str], stamp: Any = None,
            ttl_s: float | None = None) -> str:
        key = self.key(ctx, *parts)
        with self._lock:
            super().set(ctx, value, *parts)
            self._meta[key] = _Meta(self._clock() + (self.ttl_s if ttl_s is None else ttl_s),
                                    frozenset(doc_ids), stamp)
            self._meta.move_to_end(key)
            while len(self._meta) > self.max_entries:
                oldest = next(iter(self._meta))
                self._drop(oldest)
        return key

    def discard_key(self, key: str) -> bool:
        """The one removal path (guardrails routes discard, clear and invalidate_tenant through it),
        so the TTL/tag metadata can never outlive its entry."""
        with self._lock:
            had_meta = self._meta.pop(key, None) is not None
            return super().discard_key(key) or had_meta

    def _drop(self, key: str) -> None:
        self.discard_key(key)

    # ------------------------------------------------------------------ invalidation
    def invalidate_docs(self, doc_ids: Iterable[str]) -> int:
        """Physically remove every entry that contains any of these documents."""
        doomed_docs = set(doc_ids)
        with self._lock:
            doomed = [k for k, m in self._meta.items() if m.doc_ids & doomed_docs]
            for k in doomed:
                self._drop(k)
        return len(doomed)

    def sweep(self, is_stale: Callable[[Any], bool]) -> int:
        """Drop entries whose stamp is stale (e.g. older generations) or whose TTL passed."""
        now = self._clock()
        with self._lock:
            doomed = [k for k, m in self._meta.items() if m.expires_at <= now or is_stale(m.stamp)]
            for k in doomed:
                self._drop(k)
        return len(doomed)

    def clear(self) -> int:
        with self._lock:
            removed = super().clear()
            self._meta.clear()
            return removed

    def entries_for(self, doc_id: str) -> int:
        with self._lock:
            return sum(1 for m in self._meta.values() if doc_id in m.doc_ids)

    def __len__(self) -> int:
        return len(self._meta)


class CacheSet:
    """The request-path caches, invalidated together."""

    def __init__(self, *, retrieval_ttl_s: float, answer_ttl_s: float, max_entries: int, policy_version: str) -> None:
        self.retrieval: ScopedCache[Any] = ScopedCache("retrieval", ttl_s=retrieval_ttl_s, max_entries=max_entries,
                                                       policy_version=policy_version)
        self.answers: ScopedCache[Any] = ScopedCache("answer", ttl_s=answer_ttl_s, max_entries=max_entries,
                                                     policy_version=policy_version)

    def invalidate_docs(self, doc_ids: Iterable[str]) -> int:
        ids = list(doc_ids)
        return self.retrieval.invalidate_docs(ids) + self.answers.invalidate_docs(ids)

    def sweep(self, is_stale: Callable[[Any], bool]) -> int:
        return self.retrieval.sweep(is_stale) + self.answers.sweep(is_stale)

    def entries_for(self, doc_id: str) -> int:
        return self.retrieval.entries_for(doc_id) + self.answers.entries_for(doc_id)


# ============================================================================ redis-backed byte map
class RedisBytesMap(MutableMapping[str, bytes]):
    """A MutableMapping over Redis, so CachedEmbeddings and ContextualEnricher caches can be
    shared by every worker and API replica. Keys are prefixed; iteration uses SCAN."""

    def __init__(self, client: Any, prefix: str = "rag:emb:", ttl_s: int | None = None) -> None:
        self.r = client
        self.prefix = prefix
        self.ttl_s = ttl_s

    def __getitem__(self, key: str) -> bytes:
        value = self.r.get(self.prefix + key)
        if value is None:
            raise KeyError(key)
        return value if isinstance(value, bytes) else str(value).encode("utf-8")

    def __setitem__(self, key: str, value: bytes | str) -> None:
        self.r.set(self.prefix + key, value, ex=self.ttl_s)

    def __delitem__(self, key: str) -> None:
        if not self.r.delete(self.prefix + key):
            raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        for raw in self.r.scan_iter(match=self.prefix + "*"):
            k = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            yield k[len(self.prefix):]

    def __len__(self) -> int:
        return sum(1 for _ in self)


class StrMapView(MutableMapping[str, str]):
    """Adapts a bytes map to the str->str MutableMapping ragkit's ContextualEnricher expects."""

    def __init__(self, inner: MutableMapping[str, bytes]) -> None:
        self.inner = inner

    def __getitem__(self, key: str) -> str:
        return self.inner[key].decode("utf-8")

    def __setitem__(self, key: str, value: str) -> None:
        self.inner[key] = value.encode("utf-8")

    def __delitem__(self, key: str) -> None:
        del self.inner[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.inner)

    def __len__(self) -> int:
        return len(self.inner)


__all__ = ["CacheSet", "ForgettableEmbeddings", "RedisBytesMap", "ScopedCache", "StrMapView", "guard_context"]
