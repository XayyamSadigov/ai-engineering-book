# path: book/capstone/northwind-assist/northwind_assist/security/auth.py
"""JWT validation: signature, expiry, issuer, audience, required claims, tenant allow-list.

Three key sources share one code path. `hs256` uses a shared secret and exists for local
development and tests. `rs256` verifies with a PEM public key. `jwks` fetches signing keys from
the identity provider's JWKS endpoint by `kid`, which is what makes key rotation an IdP-side
operation (runbook: rotate keys). The algorithm list is pinned per mode, so a token cannot pick
its own algorithm (the classic `alg: none` and HS/RS confusion attacks).
"""
from __future__ import annotations

import time
import uuid
from typing import Any

import jwt
from pydantic import BaseModel, Field, ValidationError, field_validator

from ..config import Settings
from ..domain.context import RequestContext

KNOWN_ROLES = frozenset({"employee", "agent", "lead", "admin"})


class AuthError(Exception):
    """401: the credential is missing, malformed, expired or not ours."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ForbiddenTenant(AuthError):
    """403: a valid token for a tenant this deployment does not serve."""


class Claims(BaseModel):
    sub: str = Field(min_length=1, max_length=128)
    tenant: str = Field(min_length=1, max_length=64)
    groups: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)
    exp: int
    iat: int | None = None
    jti: str | None = None

    @field_validator("groups", "roles")
    @classmethod
    def _strings(cls, v: list[str]) -> list[str]:
        if any(not isinstance(x, str) or not x or len(x) > 64 for x in v):
            raise ValueError("groups and roles must be short non-empty strings")
        return v


class TokenValidator:
    def __init__(self, settings: Settings, *, jwks_client: Any | None = None) -> None:
        self.s = settings
        if settings.auth_mode == "hs256":
            self.algorithms = ["HS256"]
        else:
            self.algorithms = ["RS256"]
        self._jwks = jwks_client
        if settings.auth_mode == "jwks" and self._jwks is None:
            self._jwks = jwt.PyJWKClient(str(settings.jwks_url), cache_keys=True, lifespan=300)

    def _key(self, token: str) -> Any:
        if self.s.auth_mode == "hs256":
            return self.s.jwt_secret.get_secret_value()
        if self.s.auth_mode == "rs256":
            return self.s.jwt_public_key
        try:
            return self._jwks.get_signing_key_from_jwt(token).key  # type: ignore[union-attr]
        except jwt.PyJWKClientError as exc:
            raise AuthError("unknown_key", f"no signing key for token: {exc}") from exc

    def validate(self, token: str) -> RequestContext:
        try:
            raw = jwt.decode(
                token, self._key(token), algorithms=self.algorithms, audience=self.s.jwt_audience,
                issuer=self.s.jwt_issuer, leeway=self.s.jwt_leeway_s,
                options={"require": ["exp", "sub", "iss", "aud"]},
            )
        except jwt.ExpiredSignatureError as exc:
            raise AuthError("token_expired", "token has expired") from exc
        except jwt.InvalidTokenError as exc:
            raise AuthError("invalid_token", f"invalid token: {type(exc).__name__}") from exc
        try:
            claims = Claims.model_validate(raw)
        except ValidationError as exc:
            raise AuthError("invalid_claims", f"token claims rejected: {exc.errors()[0]['msg']}") from exc
        if claims.tenant not in self.s.allowed_tenants:
            raise ForbiddenTenant("tenant_not_served", f"tenant {claims.tenant!r} is not served here")
        roles = frozenset(r for r in claims.roles if r in KNOWN_ROLES) or frozenset({"employee"})
        groups = frozenset(claims.groups) | {"all"}   # every employee is in `all` (shared-data ACL rule)
        return RequestContext(user_id=claims.sub, tenant=claims.tenant, groups=groups, roles=roles,
                              token_id=claims.jti)


def issue_dev_token(settings: Settings, *, sub: str, tenant: str, groups: list[str], roles: list[str],
                    ttl_s: int = 3600, now: float | None = None) -> str:
    """HS256 token for local development and tests. Refused in any other mode."""
    if settings.auth_mode != "hs256":
        raise AuthError("dev_tokens_disabled", "dev tokens exist only in hs256 mode")
    issued = int(now if now is not None else time.time())
    payload = {"sub": sub, "tenant": tenant, "groups": groups, "roles": roles, "iss": settings.jwt_issuer,
               "aud": settings.jwt_audience, "iat": issued, "exp": issued + ttl_s, "jti": uuid.uuid4().hex}
    return jwt.encode(payload, settings.jwt_secret.get_secret_value(), algorithm="HS256")


# Synthetic personas for the login stub. They mirror Project 4's staff accounts.
DEV_PERSONAS: dict[str, dict[str, Any]] = {
    "ana": {"tenant": "retail", "groups": ["support"], "roles": ["agent"], "title": "Retail service desk agent"},
    "sam": {"tenant": "retail", "groups": ["support", "support-leads", "managers"], "roles": ["lead"],
            "title": "Retail service desk lead (approver)"},
    "lee": {"tenant": "logistics", "groups": ["it-oncall"], "roles": ["employee"], "title": "Logistics on-call engineer"},
    "kai": {"tenant": "logistics", "groups": ["support", "contractor"], "roles": ["agent"],
            "title": "Logistics contractor (may not send)"},
    "ops": {"tenant": "retail", "groups": [], "roles": ["admin"], "title": "Platform operator (cost reports)"},
}

__all__ = ["AuthError", "ForbiddenTenant", "Claims", "TokenValidator", "issue_dev_token", "DEV_PERSONAS", "KNOWN_ROLES"]
