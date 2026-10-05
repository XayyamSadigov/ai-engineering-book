# path: book/projects/p1-extraction-api/extraction_api/api/auth.py
"""Caller authentication and tenant scoping.

The review queue holds full document text, which includes personal and financial data, so
"who may list it" is a security requirement, not a nicety. Each API key maps to a principal,
a role, and optionally a tenant. A tenant-bound key can only submit documents for its tenant
and only sees review items of its tenant; another tenant's item is reported as 404, not 403,
so ids cannot be probed. With no keys configured, authentication is off: that mode exists for
local development and tests and is logged loudly at startup.
"""
from __future__ import annotations

import hmac
import logging
from dataclasses import dataclass
from typing import Literal

from fastapi import HTTPException, Request

from ..config import ApiKey

log = logging.getLogger(__name__)

Role = Literal["submitter", "reviewer", "admin"]
_ALLOWED: dict[str, set[str]] = {"submit": {"submitter", "admin"}, "review": {"reviewer", "admin"}}


@dataclass(frozen=True)
class Principal:
    name: str
    role: Role
    tenant: str | None   # None means not tenant-bound

    def can_see(self, tenant: str | None) -> bool:
        return self.tenant is None or self.tenant == tenant


ANONYMOUS = Principal(name="anonymous", role="admin", tenant=None)


class Authenticator:
    def __init__(self, keys: list[ApiKey]) -> None:
        self._keys = keys
        if not keys:
            log.warning("EXTRACT_API_KEYS is empty: authentication is DISABLED (development mode)")

    @property
    def enabled(self) -> bool:
        return bool(self._keys)

    def authenticate(self, request: Request, action: Literal["submit", "review"]) -> Principal:
        if not self._keys:
            return ANONYMOUS
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(status_code=401, detail="missing bearer token",
                                headers={"WWW-Authenticate": "Bearer"})
        match: ApiKey | None = None
        for k in self._keys:  # compare against every key in constant time; never short-circuit on a prefix
            if hmac.compare_digest(k.key.get_secret_value().encode(), token.encode()):
                match = k
        if match is None:
            raise HTTPException(status_code=401, detail="invalid token", headers={"WWW-Authenticate": "Bearer"})
        if match.role not in _ALLOWED[action]:
            raise HTTPException(status_code=403, detail=f"role {match.role} may not {action}")
        return Principal(name=match.principal, role=match.role, tenant=match.tenant)


__all__ = ["Principal", "Authenticator", "ANONYMOUS"]
