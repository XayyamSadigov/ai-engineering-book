# path: book/capstone/northwind-assist/northwind_assist/domain/context.py
"""RequestContext: identity, tenant and budget, built once by auth and passed everywhere.

Chapter 28's rule is that nothing below the auth dependency re-derives identity from headers or
bodies. Every package the capstone composes has its own principal type (ragkit Principal,
toolkit ToolContext, guardrails GuardContext, memorykit Owner, Chapter 30 Scope, Chapter 5
RequestScope). They are all projections of this one object, so they cannot drift apart.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from guardrails import GuardContext, PIIVault
from memorykit import Owner
from ragkit.retrieval import Principal
from reliability import Deadline
from toolkit import ToolContext

ROLE_SCOPES: dict[str, frozenset[str]] = {
    "employee": frozenset({"directory:read", "status:read"}),
    "agent": frozenset({"directory:read", "status:read", "tickets:read", "tickets:write",
                        "replies:draft", "replies:send"}),
    "lead": frozenset({"directory:read", "status:read", "tickets:read", "tickets:write",
                       "replies:draft", "replies:send", "replies:approve"}),
    "admin": frozenset({"cost:read", "admin:reindex", "status:read"}),
}


def scopes_for(roles: frozenset[str]) -> frozenset[str]:
    out: set[str] = set()
    for role in roles:
        out |= ROLE_SCOPES.get(role, frozenset())
    return frozenset(out)


@dataclass(frozen=True)
class RequestContext:
    user_id: str
    tenant: str
    groups: frozenset[str]
    roles: frozenset[str]
    request_id: str = field(default_factory=lambda: f"req_{uuid.uuid4().hex[:16]}")
    session_id: str | None = None
    idempotency_key: str | None = None
    deadline: Deadline | None = None
    token_id: str | None = None

    @property
    def scopes(self) -> frozenset[str]:
        return scopes_for(self.roles)

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes

    def with_request(self, *, session_id: str | None, idempotency_key: str | None,
                     deadline: Deadline | None) -> RequestContext:
        """A fresh request id per request: the token may be reused, the request never is."""
        return RequestContext(self.user_id, self.tenant, self.groups, self.roles, f"req_{uuid.uuid4().hex[:16]}",
                              session_id, idempotency_key, deadline, self.token_id)

    # ------------------------------------------------------------------ projections
    def principal(self) -> Principal:
        return Principal(user_id=self.user_id, tenant=self.tenant, groups=sorted(self.groups))

    def tool_context(self) -> ToolContext:
        return ToolContext(user_id=self.user_id, tenant=self.tenant, groups=self.groups, scopes=self.scopes,
                           session_id=self.session_id or self.request_id, request_id=self.request_id)

    def guard_context(self) -> GuardContext:
        ctx = GuardContext(tenant=self.tenant, user_id=self.user_id, groups=self.groups, request_id=self.request_id)
        ctx.vault = PIIVault(tenant=self.tenant, scope_id=self.session_id or self.request_id)
        return ctx

    def owner(self) -> Owner:
        return Owner(tenant=self.tenant, user=self.user_id)


__all__ = ["RequestContext", "ROLE_SCOPES", "scopes_for"]
