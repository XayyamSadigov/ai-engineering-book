# path: book/projects/p3-rag-assistant/rag_assistant/api/auth.py
"""Principal from a verified bearer token. A STUB: Chapter 39 replaces it with real JWT/OIDC.

What the stub gets right, and the real implementation must keep:
- the principal (user, tenant, groups) comes only from a token the server verified, never from
  request fields, query text, or anything a document or model said;
- verification is a constant-time HMAC comparison plus an expiry check;
- a missing or invalid token is a 401, not an anonymous principal with group "all".

What it lacks: key rotation, asymmetric signatures (an identity provider signs, services only
verify), audience/issuer checks, and revocation. Do not deploy it outside a closed network.

Token format: base64url(JSON claims) "." base64url(HMAC-SHA256(secret, claims part)).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

from fastapi import Header, HTTPException, Request
from ragkit.retrieval import Principal


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def issue_token(secret: str, *, user_id: str, tenant: str, groups: list[str], ttl_s: int = 3600) -> str:
    """For tests and local development only; in production an identity provider issues tokens."""
    claims = {"sub": user_id, "tenant": tenant, "groups": sorted(set(groups) | {"all"}),
              "exp": int(time.time()) + ttl_s}
    body = _b64(json.dumps(claims, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    sig = _b64(hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest())
    return f"{body}.{sig}"


def verify_token(secret: str, token: str) -> Principal:
    try:
        body, sig = token.split(".", 1)
        expected = _b64(hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            raise ValueError("bad signature")
        claims = json.loads(_unb64(body))
        if int(claims["exp"]) < time.time():
            raise ValueError("expired")
        return Principal(user_id=str(claims["sub"]), tenant=str(claims["tenant"]),
                         groups=[str(g) for g in claims["groups"]])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=401, detail="invalid or expired token") from exc


def current_principal(request: Request, authorization: str | None = Header(default=None)) -> Principal:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token", headers={"WWW-Authenticate": "Bearer"})
    secret = request.app.state.container.settings.auth_secret
    return verify_token(secret, authorization.split(" ", 1)[1].strip())


__all__ = ["current_principal", "issue_token", "verify_token"]
