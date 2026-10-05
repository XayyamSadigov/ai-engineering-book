# path: book/capstone/northwind-assist/northwind_assist/rag/caches.py
"""Scoped caches (Chapter 30): retrieval ids and whole grounded answers.

Every key carries the tenant, a hash of the caller's group set (the ACL scope), and the version
of what produced the value: the index version for retrieval, plus prompt version and model for
answers. The tenant also prefixes the key, so a tenant can be purged in one call. Retrieval caches
*ids*, not text: hits are re-hydrated from the live chunk table and re-checked against the ACL,
so a document deleted or re-permissioned after caching cannot be served from the cache.
"""
from __future__ import annotations

from typing import Any

from caching import RetrievalCache, Scope, TTLCache, make_key, normalize_text  # type: ignore[import-not-found]


class CacheLayer:
    def __init__(self, *, retrieval_ttl_s: float = 900.0, answer_ttl_s: float = 3600.0, enabled: bool = True) -> None:
        self.retrieval = RetrievalCache(TTLCache(max_size=20_000), ttl_s=retrieval_ttl_s)
        self.answers: TTLCache[dict[str, Any]] = TTLCache(max_size=5_000)
        self.answer_ttl_s = answer_ttl_s
        self.enabled = enabled

    @staticmethod
    def scope(tenant: str, groups: frozenset[str] | set[str]) -> Scope:
        return Scope.of(tenant, groups)

    @staticmethod
    def answer_key(question: str, scope: Scope, *, prompt_version: str, index_version: str, model: str,
                   plan_level: int) -> str:
        return make_key(f"ans:{scope.tenant}", {
            "question": normalize_text(question), "tenant": scope.tenant, "acl_scope": scope.acl_scope,
            "prompt_version": prompt_version, "index_version": index_version, "model": model,
            "plan_level": plan_level,
        })

    def get_answer(self, key: str) -> dict[str, Any] | None:
        return self.answers.get(key) if self.enabled else None

    def put_answer(self, key: str, value: dict[str, Any]) -> None:
        if self.enabled:
            self.answers.set(key, value, self.answer_ttl_s)

    def purge_tenant(self, tenant: str) -> int:
        return self.retrieval.purge_tenant(tenant) + self.answers.delete_prefix(f"ans:{tenant}:")

    def clear(self) -> None:
        self.retrieval.store.delete_prefix("")
        self.answers.delete_prefix("")

    def stats(self) -> dict[str, float]:
        return {"retrieval_hit_rate": round(self.retrieval.store.hit_rate, 3),
                "answer_hit_rate": round(self.answers.hit_rate, 3),
                "retrieval_entries": len(self.retrieval.store), "answer_entries": len(self.answers)}


__all__ = ["CacheLayer", "Scope"]
