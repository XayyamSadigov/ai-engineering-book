# path: book/projects/guardrails/guardrails/pii.py
"""PII detection and redaction: regexes for candidates, checksums to cut false positives,
masking or reversible tokenization through a per-request vault.

Detectors (illustrative coverage; production systems add names, addresses and national IDs,
usually with an NER model or a dedicated service, see Chapter 27's text):

  email   RFC-lite pattern
  phone   international or common national formats, 9 to 15 digits
  card    13 to 19 digits, spaces or dashes allowed, must pass the Luhn checksum
  iban    country code + check digits + BBAN, must pass ISO 7064 mod-97
  ip      IPv4 with octet validation; IPv6 via the standard library parser
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import re
import secrets as _secrets
from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal

from .pipeline import Action, BaseCheck, FailMode, Finding, GuardContext, Stage, Subject, Verdict

PIIKind = Literal["email", "phone", "card", "iban", "ip"]
ALL_KINDS: tuple[str, ...] = ("email", "card", "iban", "ip", "phone")  # resolution priority order

EMAIL_RE = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?![\w-])")
CARD_RE = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]){11,30}\b")
IPV4_RE = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
IPV6_RE = re.compile(r"(?<![:\w])(?:[0-9A-Fa-f]{1,4}:){2,7}[0-9A-Fa-f]{0,4}(?![:\w])|::1\b")
PHONE_RE = re.compile(r"(?<![\w+/-])(?:\+\d{1,3}[ .-]?)?(?:\(\d{1,4}\)[ .-]?)?\d{2,4}(?:[ .-]?\d{2,4}){2,4}(?![\w/]|-\w)")
TOKEN_RE = re.compile(r"<PII:(email|phone|card|iban|ip):([0-9a-f]{10})>")


@dataclass(frozen=True)
class PIIMatch:
    kind: str
    start: int
    end: int
    value: str


# --------------------------------------------------------------------------- checksums
def luhn_valid(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


_IBAN_LENGTHS = {"AZ": 28, "DE": 22, "GB": 22, "FR": 27, "NL": 18, "ES": 24, "IT": 27, "CH": 21, "TR": 26,
                 "BE": 16, "AT": 20, "PL": 28, "SE": 24, "NO": 15, "DK": 18, "IE": 22, "PT": 25}


def iban_valid(candidate: str) -> bool:
    s = candidate.replace(" ", "").upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", s):
        return False
    expected = _IBAN_LENGTHS.get(s[:2])
    if expected is not None and len(s) != expected:
        return False
    rearranged = s[4:] + s[:4]
    numeric = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(numeric) % 97 == 1


def ipv4_valid(candidate: str) -> bool:
    parts = candidate.split(".")
    return len(parts) == 4 and all(p.isdigit() and int(p) <= 255 and (p == "0" or not p.startswith("0"))
                                   for p in parts)


def _ipv6_valid(candidate: str) -> bool:
    try:
        ipaddress.IPv6Address(candidate)
        return True
    except ValueError:
        return False


def _phone_valid(candidate: str) -> bool:
    digits = re.sub(r"\D", "", candidate)
    min_digits = 8 if candidate.strip().startswith("+") else 9
    if not min_digits <= len(digits) <= 15:
        return False
    if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", candidate.strip()):
        return False          # IPv4-shaped (valid or not) is never a phone number
    # Dates and version-like strings are the classic false positives.
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", candidate.strip()):
        return False
    has_separator_or_plus = candidate.strip().startswith("+") or bool(re.search(r"[ .()-]", candidate))
    return has_separator_or_plus or len(digits) >= 10


# --------------------------------------------------------------------------- detection
def detect_pii(text: str, kinds: Iterable[str] = ALL_KINDS) -> list[PIIMatch]:
    """Return non-overlapping matches. Earlier kinds in `ALL_KINDS` win overlaps, so a card
    number is not also reported as a phone number."""
    wanted = [k for k in ALL_KINDS if k in set(kinds)]
    candidates: list[PIIMatch] = []
    for kind in wanted:
        if kind == "email":
            candidates += [PIIMatch("email", m.start(), m.end(), m.group(0)) for m in EMAIL_RE.finditer(text)]
        elif kind == "card":
            candidates += [PIIMatch("card", m.start(), m.end(), m.group(0)) for m in CARD_RE.finditer(text)
                           if luhn_valid(m.group(0))]
        elif kind == "iban":
            candidates += [PIIMatch("iban", m.start(), m.end(), m.group(0)) for m in IBAN_RE.finditer(text)
                           if iban_valid(m.group(0))]
        elif kind == "ip":
            candidates += [PIIMatch("ip", m.start(), m.end(), m.group(0)) for m in IPV4_RE.finditer(text)
                           if ipv4_valid(m.group(0))]
            candidates += [PIIMatch("ip", m.start(), m.end(), m.group(0)) for m in IPV6_RE.finditer(text)
                           if _ipv6_valid(m.group(0))]
        elif kind == "phone":
            for m in PHONE_RE.finditer(text):
                value = m.group(0).strip()
                if _phone_valid(value):
                    start = m.start() + (len(m.group(0)) - len(m.group(0).lstrip()))
                    candidates.append(PIIMatch("phone", start, start + len(value), value))
    chosen: list[PIIMatch] = []
    taken: list[tuple[int, int]] = []
    priority = {k: i for i, k in enumerate(ALL_KINDS)}
    for c in sorted(candidates, key=lambda c: (priority[c.kind], c.start)):
        if any(c.start < e and s < c.end for s, e in taken):
            continue
        chosen.append(c)
        taken.append((c.start, c.end))
    return sorted(chosen, key=lambda c: c.start)


# --------------------------------------------------------------------------- vault
RehydratePolicy = Callable[[GuardContext, str], bool]


class PIIVault:
    """Reversible tokenization scoped to one tenant and one request (or conversation).

    Tokens are HMAC-derived, so the same value maps to the same token within a vault (the model
    can still reason "the same customer emailed twice") and to a different token in any other
    vault. Values never leave the vault except through `rehydrate`, which requires a matching
    tenant and a policy decision per PII kind. Unknown tokens, including ones a model invents or
    copies from another tenant's conversation, stay as tokens.
    """

    def __init__(self, tenant: str, scope_id: str, key: bytes | None = None) -> None:
        self.tenant = tenant
        self.scope_id = scope_id
        self._key = key or _secrets.token_bytes(32)
        self._by_token: dict[str, tuple[str, str]] = {}

    def token_for(self, kind: str, value: str) -> str:
        digest = hmac.new(self._key, f"{kind}\x00{value}".encode(), hashlib.sha256).hexdigest()[:10]
        token = f"<PII:{kind}:{digest}>"
        self._by_token[token] = (kind, value)
        return token

    def __len__(self) -> int:
        return len(self._by_token)

    def __contains__(self, token: str) -> bool:
        return token in self._by_token

    def rehydrate(self, text: str, ctx: GuardContext, policy: RehydratePolicy) -> str:
        if ctx.tenant != self.tenant:
            raise PermissionError("vault belongs to a different tenant")

        def sub(m: re.Match[str]) -> str:
            token = m.group(0)
            entry = self._by_token.get(token)
            if entry is None:
                return token
            kind, value = entry
            return value if policy(ctx, kind) else token

        return TOKEN_RE.sub(sub, text)

    def clear(self) -> None:
        """Call at the end of the retention window; afterwards tokens are unrecoverable."""
        self._by_token.clear()


# --------------------------------------------------------------------------- redaction
@dataclass(frozen=True)
class RedactionResult:
    text: str
    matches: tuple[PIIMatch, ...] = field(default_factory=tuple)

    @property
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for m in self.matches:
            out[m.kind] = out.get(m.kind, 0) + 1
        return out


def mask_value(kind: str, value: str) -> str:
    if kind == "card":
        digits = re.sub(r"\D", "", value)
        return f"[CARD ****{digits[-4:]}]"
    if kind == "email":
        domain = value.rsplit("@", 1)[-1]
        return f"[EMAIL @{domain}]"
    return f"[{kind.upper()}]"


def redact_pii(text: str, mode: Literal["mask", "tokenize"] = "mask", vault: PIIVault | None = None,
               kinds: Iterable[str] = ALL_KINDS) -> RedactionResult:
    if mode == "tokenize" and vault is None:
        raise ValueError("tokenize mode needs a PIIVault")
    matches = detect_pii(text, kinds)
    out: list[str] = []
    pos = 0
    for m in matches:
        out.append(text[pos:m.start])
        out.append(vault.token_for(m.kind, m.value) if mode == "tokenize" and vault is not None
                   else mask_value(m.kind, m.value))
        pos = m.end
    out.append(text[pos:])
    return RedactionResult("".join(out), tuple(matches))


class PIIRedactionCheck(BaseCheck):
    """Redacts PII at any text stage. With `mode="tokenize"` and a vault in the context, values
    are replaced by reversible tokens; otherwise they are masked irreversibly.

    Fail-closed by default: a broken detector should not silently pass raw PII to a provider.
    Set `on_detect=Action.BLOCK` for surfaces that must never carry PII at all.
    """

    name = "pii"
    fail_mode = FailMode.CLOSED

    def __init__(self, mode: Literal["mask", "tokenize"] = "mask", kinds: Iterable[str] = ALL_KINDS,
                 on_detect: Action = Action.REDACT,
                 stages: frozenset[Stage] = frozenset({Stage.INPUT, Stage.CONTEXT, Stage.OUTPUT})) -> None:
        self.mode = mode
        self.kinds = tuple(kinds)
        self.on_detect = on_detect
        self.stages = stages

    def evaluate(self, subject: Subject, ctx: GuardContext) -> Verdict:
        vault = ctx.vault if self.mode == "tokenize" else None
        mode = "tokenize" if vault is not None else "mask"
        res = redact_pii(subject.text, mode=mode, vault=vault, kinds=self.kinds)
        if not res.matches:
            return Verdict.allow()
        findings = [Finding(m.kind, m.start, m.end) for m in res.matches]   # positions only, no values
        reason = "pii " + ", ".join(f"{k}={v}" for k, v in sorted(res.counts.items()))
        if self.on_detect is Action.BLOCK:
            return Verdict.block(reason, findings=findings, counts=res.counts)
        if self.on_detect is Action.FLAG:
            return Verdict.flag(reason, 1.0, findings, counts=res.counts)
        return Verdict.redact(res.text, reason, 1.0, findings, counts=res.counts, mode=mode)


__all__ = [
    "PIIMatch", "ALL_KINDS", "luhn_valid", "iban_valid", "ipv4_valid", "detect_pii", "PIIVault",
    "RedactionResult", "redact_pii", "mask_value", "PIIRedactionCheck", "TOKEN_RE",
]
