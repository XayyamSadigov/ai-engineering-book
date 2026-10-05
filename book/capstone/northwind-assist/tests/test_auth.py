# path: book/capstone/northwind-assist/tests/test_auth.py
"""JWT validation: signature, expiry, tenant allow-list, algorithm pinning, RS256 and JWKS."""
from __future__ import annotations

import time

import jwt
import pytest
from conftest import auth, token_for
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from northwind_assist.config import Settings
from northwind_assist.security.auth import AuthError, ForbiddenTenant, TokenValidator, issue_dev_token


def test_valid_token_builds_context(client, container):
    r = client.get("/v1/me", headers=auth(container, "ana"))
    assert r.status_code == 200
    body = r.json()
    assert body["tenant"] == "retail" and body["user_id"] == "ana"
    assert "all" in body["groups"] and "replies:send" in body["scopes"]
    assert "replies:approve" not in body["scopes"]


def test_missing_and_garbage_tokens_are_401(client):
    assert client.get("/v1/me").status_code == 401
    r = client.get("/v1/me", headers={"Authorization": "Bearer not-a-jwt"})
    assert r.status_code == 401 and r.json()["detail"] == "invalid_token"


def test_expired_token_is_401(client, container):
    old = issue_dev_token(container.settings, sub="ana", tenant="retail", groups=[], roles=["agent"],
                          ttl_s=60, now=time.time() - 3600)
    r = client.get("/v1/me", headers={"Authorization": f"Bearer {old}"})
    assert r.status_code == 401 and r.json()["detail"] == "token_expired"


def test_wrong_tenant_is_403(client, container):
    t = token_for(container, "ana", tenant="acme-corp")
    r = client.get("/v1/me", headers={"Authorization": f"Bearer {t}"})
    assert r.status_code == 403 and r.json()["detail"] == "tenant_not_served"


def test_wrong_signature_audience_and_alg_none_are_rejected(container):
    v = container.validator
    s = container.settings
    payload = {"sub": "ana", "tenant": "retail", "iss": s.jwt_issuer, "aud": s.jwt_audience, "exp": int(time.time()) + 60}
    with pytest.raises(AuthError):
        v.validate(jwt.encode(payload, "another-secret-of-sufficient-length-123456", algorithm="HS256"))
    with pytest.raises(AuthError):
        v.validate(jwt.encode({**payload, "aud": "other-app"}, s.jwt_secret.get_secret_value(), algorithm="HS256"))
    with pytest.raises(AuthError):
        v.validate(jwt.encode(payload, None, algorithm="none"))


def test_body_cannot_override_token_tenant(client, container):
    """The request body has no tenant field; an injected one is ignored, the token wins."""
    r = client.post("/v1/chat?stream=false", headers=auth(container, "ana"),
                    json={"message": "What was the root cause of the Trackline API latency incident?",
                          "tenant": "logistics"})
    assert r.status_code == 200
    cited = [c["doc_id"] for c in r.json()["citations"]]
    assert "inc-2026-02-tracking-latency" not in cited


def _rsa_pair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    return key, pem.decode()


def test_rs256_mode_verifies_with_public_key_and_refuses_hs256():
    key, pub = _rsa_pair()
    s = Settings(environment="test", auth_mode="rs256", jwt_public_key=pub)
    v = TokenValidator(s)
    claims = {"sub": "sam", "tenant": "retail", "groups": ["support-leads"], "roles": ["lead"], "iss": s.jwt_issuer,
              "aud": s.jwt_audience, "exp": int(time.time()) + 60}
    ctx = v.validate(jwt.encode(claims, key, algorithm="RS256"))
    assert ctx.has_scope("replies:approve")
    with pytest.raises(AuthError):   # HS/RS confusion: the public key used as an HMAC secret
        v.validate(jwt.encode(claims, "x" * 40, algorithm="HS256"))


def test_jwks_mode_picks_key_by_kid():
    key, _ = _rsa_pair()

    class StubJWKS:
        def get_signing_key_from_jwt(self, token):
            assert jwt.get_unverified_header(token)["kid"] == "k-2026-10"
            return jwt.PyJWK.from_dict({**jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key(), as_dict=True),
                                        "kid": "k-2026-10", "alg": "RS256"})

    s = Settings(environment="test", auth_mode="jwks", jwks_url="https://sso.northwind.example/jwks.json")
    v = TokenValidator(s, jwks_client=StubJWKS())
    claims = {"sub": "lee", "tenant": "logistics", "iss": s.jwt_issuer, "aud": s.jwt_audience,
              "exp": int(time.time()) + 60}
    ctx = v.validate(jwt.encode(claims, key, algorithm="RS256", headers={"kid": "k-2026-10"}))
    assert ctx.tenant == "logistics" and ctx.roles == frozenset({"employee"})


def test_forbidden_tenant_is_distinct_from_invalid():
    s = Settings(environment="test", allowed_tenants=["retail"])
    t = issue_dev_token(s, sub="lee", tenant="logistics", groups=[], roles=[])
    with pytest.raises(ForbiddenTenant):
        TokenValidator(s).validate(t)


def test_prod_settings_refuse_dev_auth():
    with pytest.raises(ValueError):
        Settings(environment="prod")
    with pytest.raises(ValueError):
        Settings(environment="prod", auth_mode="jwks", jwks_url="https://x", dev_login=False, rag_enforce_acl=False)
