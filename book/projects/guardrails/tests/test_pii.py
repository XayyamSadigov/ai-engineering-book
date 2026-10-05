# path: book/projects/guardrails/tests/test_pii.py
from __future__ import annotations

import pytest

from guardrails import (Action, GuardContext, GuardrailPipeline, PIIRedactionCheck, PIIVault, detect_pii,
                        iban_valid, luhn_valid, redact_pii)


def kinds(text):
    return [m.kind for m in detect_pii(text)]


def test_luhn():
    assert luhn_valid("4111 1111 1111 1111")
    assert not luhn_valid("4111 1111 1111 1112")
    assert not luhn_valid("1234")


def test_iban_checksum_and_length():
    assert iban_valid("DE89 3704 0044 0532 0130 00")
    assert iban_valid("GB82WEST12345698765432")
    assert not iban_valid("DE89 3704 0044 0532 0130 01")   # checksum
    assert not iban_valid("DE89 3704 0044 0532 0130")      # length for DE


@pytest.mark.parametrize("text,expected", [
    ("mail jane.doe@northwind.example now", ["email"]),
    ("call +1-555-0142 or (012) 555 0199 88", ["phone", "phone"]),
    ("card 4111-1111-1111-1111 charged", ["card"]),
    ("IBAN DE89 3704 0044 0532 0130 00 please", ["iban"]),
    ("from 10.20.30.40 and 2001:db8::1", ["ip", "ip"]),
])
def test_detects_each_kind(text, expected):
    assert kinds(text) == expected


@pytest.mark.parametrize("text", [
    "card 4111 1111 1111 1112 is not valid",            # fails Luhn
    "on 2026-01-09 we shipped",                          # date
    "transaction TX-2025-0293-118-0007",                 # order id with dashes
    "return RET-20260215-004412 refunded",               # reference id
    "police report PR-2026-00912",
    "version 2.4.1 and host 999.10.10.10",               # not an IP
    "IBAN-like DE00 1234 5678 9012 3456 78",             # bad checksum
])
def test_false_positive_cases(text):
    assert kinds(text) == []


def test_card_wins_overlap_with_phone():
    assert kinds("4111 1111 1111 1111") == ["card"]


def test_mask_mode():
    res = redact_pii("card 4111 1111 1111 1111, mail a.b@northwind.example")
    assert "[CARD ****1111]" in res.text and "[EMAIL @northwind.example]" in res.text
    assert res.counts == {"card": 1, "email": 1}


def test_tokenize_round_trip_with_authorized_rehydration(vault_ctx):
    original = "Refund card 4111 1111 1111 1111 for jane@northwind.example, again jane@northwind.example"
    res = redact_pii(original, mode="tokenize", vault=vault_ctx.vault)
    assert "4111" not in res.text and "jane@" not in res.text
    from guardrails.pii import TOKEN_RE
    tokens = {m.group(0) for m in TOKEN_RE.finditer(res.text)}
    assert len(tokens) == 2, "same value maps to the same token"
    model_answer = res.text.replace("Refund", "I have refunded")
    restored = vault_ctx.vault.rehydrate(model_answer, vault_ctx, policy=lambda c, k: True)
    assert restored == original.replace("Refund", "I have refunded")


def test_rehydration_policy_per_kind(vault_ctx):
    res = redact_pii("jane@northwind.example card 4111 1111 1111 1111", mode="tokenize", vault=vault_ctx.vault)
    out = vault_ctx.vault.rehydrate(res.text, vault_ctx, policy=lambda c, k: k == "email")
    assert "jane@northwind.example" in out and "4111" not in out and "<PII:card:" in out


def test_rehydration_refuses_other_tenant(vault_ctx):
    res = redact_pii("jane@northwind.example", mode="tokenize", vault=vault_ctx.vault)
    other = GuardContext(tenant="logistics", user_id="x")
    with pytest.raises(PermissionError):
        vault_ctx.vault.rehydrate(res.text, other, policy=lambda c, k: True)


def test_forged_and_foreign_tokens_stay_tokens(vault_ctx):
    other_vault = PIIVault(tenant="retail", scope_id="req-2")
    foreign = other_vault.token_for("email", "boss@northwind.example")
    forged = "<PII:email:0123456789>"
    out = vault_ctx.vault.rehydrate(f"{foreign} {forged}", vault_ctx, policy=lambda c, k: True)
    assert out == f"{foreign} {forged}"


def test_vault_clear_makes_tokens_unrecoverable(vault_ctx):
    res = redact_pii("jane@northwind.example", mode="tokenize", vault=vault_ctx.vault)
    vault_ctx.vault.clear()
    assert vault_ctx.vault.rehydrate(res.text, vault_ctx, lambda c, k: True) == res.text


def test_tokenize_without_vault_is_an_error():
    with pytest.raises(ValueError):
        redact_pii("a@northwind.example", mode="tokenize")


def test_check_uses_vault_when_present_and_masks_otherwise(ctx, vault_ctx):
    pipe = GuardrailPipeline([PIIRedactionCheck(mode="tokenize")])
    tok = pipe.check_input("my card 4111 1111 1111 1111", vault_ctx)
    assert "<PII:card:" in tok.text and tok.action is Action.REDACT
    plain_ctx = GuardContext(tenant="retail", user_id="u")
    masked = pipe.check_input("my card 4111 1111 1111 1111", plain_ctx)
    assert "[CARD ****1111]" in masked.text


def test_check_block_mode_and_findings_have_no_values(ctx):
    v = PIIRedactionCheck(on_detect=Action.BLOCK).evaluate(
        __import__("guardrails").Subject(__import__("guardrails").Stage.OUTPUT, text="a@northwind.example"), ctx)
    assert v.action is Action.BLOCK and all(f.value == "" for f in v.findings)
