# path: book/projects/guardrails/tests/test_secrets.py
from __future__ import annotations

import pytest

from guardrails import Action, SecretsCheck, Stage, Subject, detect_secrets, redact_secrets, shannon_entropy

FAKE_GH = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"


@pytest.mark.parametrize("text,kind", [
    ("key AKIAABCDEFGHIJKLMNOP in config", "aws_access_key_id"),
    (f"token {FAKE_GH}", "github_token"),
    ("OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwx1234", "sk_api_key"),
    ("Authorization: Bearer abcdefghijklmnop1234567890", "bearer_header"),
    ("postgres://app:hunter2pass@db.internal:5432/x", "url_credentials"),
    ("password = 'Winter2026!xyz'", "assigned_secret"),
    ("-----BEGIN RSA PRIVATE KEY-----\nMIIabc\n-----END RSA PRIVATE KEY-----", "private_key"),
    ("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyMSJ9.c2lnbmF0dXJlLXZhbHVl", "jwt"),
    ("blob q7Zx9LmP2vR8tY4wK1nB6cJ3hG5fD0sA", "high_entropy"),
])
def test_detects_known_formats(text, kind):
    assert kind in [m.kind for m in detect_secrets(text)]


@pytest.mark.parametrize("text", [
    "commit 3f9a1c0b7d2e4f5a6b8c9d0e1f2a3b4c5d6e7f80 fixed it",       # hex SHA
    "request id 123e4567-e89b-12d3-a456-426614174000",               # UUID
    "see retail-returns-api-v2-handler for details",                  # readable identifier
    "MAX_RETRY_ATTEMPTS_FOR_PAYMENT_GATEWAY is 3",                    # constant
    "Temporary passwords expire after 24 hours.",                     # prose about passwords
    "https://intranet.northwind.example/it/vpn/troubleshooting-guide",
])
def test_false_positive_cases(text):
    assert detect_secrets(text) == []


def test_entropy_ordering():
    assert shannon_entropy("aaaaaaaa") == 0.0
    assert shannon_entropy("q7Zx9LmP2vR8tY4wK1nB") > shannon_entropy("the quick brown fox")


def test_redaction_labels_kind():
    text, matches = redact_secrets("use AKIAABCDEFGHIJKLMNOP now")
    assert text == "use [REDACTED:aws_access_key_id] now" and len(matches) == 1


def test_check_actions(ctx):
    s = Subject(Stage.OUTPUT, text="AKIAABCDEFGHIJKLMNOP")
    assert SecretsCheck(on_detect=Action.BLOCK).evaluate(s, ctx).action is Action.BLOCK
    v = SecretsCheck().evaluate(s, ctx)
    assert v.action is Action.REDACT and "AKIA" not in (v.redacted or "")
    assert "AKIA" not in v.reason
