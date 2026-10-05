# path: book/projects/examples/ch05/context/filters.py
"""Filter hooks: permission first, relevance second.

A filter receives an item and the request scope and returns None to keep the item or a
short reason string to drop it. Permission filters run on every item, pinned or not: a
pinned item the caller may not see is a bug upstream, and the builder refuses it loudly.
Relevance filters skip pinned items, because pinning is the caller saying "relevant".
"""
from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, Field

from .items import ContextItem, Trust

# Kinds produced inside the session itself (the user's own turns, state we derived from them).
# They carry no document ACL; the session boundary is their permission check.
SESSION_KINDS = frozenset({"turn", "query", "fact", "summary"})


class RequestScope(BaseModel):
    """Who is asking. Comes from the authenticated session, never from the model."""

    user_id: str
    tenant: str
    groups: list[str] = Field(default_factory=list)


Filter = Callable[[ContextItem, RequestScope], str | None]


def acl_filter(item: ContextItem, scope: RequestScope) -> str | None:
    """Tenant and group check against metadata the indexer attached (Chapter 15 owns ACL design).

    Items with no ACL metadata pass only if we wrote them (trusted) or they belong to the
    session. An untrusted document or tool result without ACL metadata fails closed.
    """
    tenant = item.metadata.get("tenant")
    groups = item.metadata.get("acl_groups")
    if tenant is None and groups is None:
        if item.trust is Trust.TRUSTED or item.kind in SESSION_KINDS:
            return None
        return "no_acl_metadata"
    if tenant not in (None, "shared", scope.tenant):
        return f"tenant:{tenant}"
    if groups is not None and "all" not in groups and not set(groups) & set(scope.groups):
        return "group"
    return None


def min_score_filter(threshold: float, kinds: tuple[str, ...] = ("evidence", "memory")) -> Filter:
    """Drop retrieved items whose retrieval score is below a calibrated threshold."""

    def _filter(item: ContextItem, scope: RequestScope) -> str | None:
        if item.kind not in kinds:
            return None
        score = item.metadata.get("score")
        if score is None:
            return None
        return None if score >= threshold else f"score<{threshold}"

    return _filter


__all__ = ["RequestScope", "Filter", "acl_filter", "min_score_filter", "SESSION_KINDS"]
