# path: book/projects/examples/ch07/fakes.py
"""Two FakeLLM instances that behave like a small and a large model on ticket classification.

The "small" model is a keyword scorer: fast, cheap, right on tickets that use the obvious
words, unsure (and sometimes wrong) on tickets that mix topics. Its confidence comes from
the score margin, so it is informative but not calibrated. The "large" model knows the gold
labels and errs on a fixed, small subset. Both are deterministic, so tests and the demo
reproduce exactly. Latencies are illustrative constants reported in Completion.latency_ms.
"""
from __future__ import annotations

import hashlib
import re

from aie_core import CompletionRequest, FakeLLM

KEYWORDS: dict[str, tuple[str, ...]] = {
    "account_access": ("access", "permission", "role", "group", "approve", "console", "account", "shared drive"),
    "benefits_leave": ("parental", "benefit", "insurance", "leave", "pension", "maternity", "paternity", "salary", "bank"),
    "expenses_travel": ("expense", "receipt", "travel", "hotel", "flight", "reimburse", "per diem", "mileage", "card statement"),
    "hardware": ("laptop", "monitor", "keyboard", "screen", "battery", "dock", "headset", "broken"),
    "password_mfa": ("password", "mfa", "authenticator", "locked", "log in", "login", "2fa", "reset"),
    "pos_payments": ("register", "till", "card", "payment", "pos", "declin", "receipt printer"),
    "returns": ("return", "refund", "rma", "exchange", "restock", "final sale", "final-sale"),
    "security_report": ("phishing", "suspicious", "malware", "security", "leak", "virus", "scam", "stolen", "personal data", "conflict of interest", "unknown login"),
    "shipment_tracking": ("tracking", "parcel", "shipment", "delivery", "eta", "carrier", "consignment", "late", "route", "stops", "webhook"),
    "time_off": ("pto", "vacation", "holiday", "time off", "day off", "sick", "work from", "remotely"),
    "vpn_network": ("vpn", "wifi", "wi-fi", "network", "connection", "disconnect", "tunnel"),
    "warehouse_scanner": ("scanner", "handheld", "barcode", "scan", "picking", "warehouse"),
}

SUBJECT_RE = re.compile(r"Subject: (.*)")


def _user_text(req: CompletionRequest) -> str:
    return "\n".join(m.text for m in req.messages if m.role.value == "user")


def keyword_scores(text: str) -> dict[str, int]:
    low = text.lower()
    return {cat: sum(low.count(k) for k in words) for cat, words in KEYWORDS.items()}


def small_answer(text: str) -> dict[str, object]:
    scores = keyword_scores(text)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    (best, s1), (_, s2) = ranked[0], ranked[1]
    if s1 == 0:
        return {"category": "account_access", "confidence": 0.2}   # a guess, flagged as one
    margin = (s1 - s2) / s1
    confidence = round(min(0.97, 0.45 + 0.5 * margin + 0.04 * min(s1, 3)), 2)
    return {"category": best, "confidence": confidence}


def make_small_model(*, latency_ms: float = 250.0) -> FakeLLM:
    return FakeLLM(handler=lambda req: small_answer(_user_text(req)),
                   provider="local", model="small-instruct-2026-03", latency_ms=latency_ms)


def _stable_bucket(key: str, buckets: int) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % buckets


def make_large_model(gold_by_subject: dict[str, str], *, error_every: int = 20,
                     latency_ms: float = 1800.0, model: str = "general-2026-02",
                     provider: str = "cloud-a", salt: str = "") -> FakeLLM:
    """gold_by_subject maps the ticket subject line to its label. About 1 in error_every
    subjects is answered wrongly (deterministically) so the large model is not an oracle."""
    others = sorted(set(gold_by_subject.values()))

    def handler(req: CompletionRequest) -> dict[str, object]:
        m = SUBJECT_RE.search(_user_text(req))
        subject = m.group(1).strip() if m else ""
        gold = gold_by_subject.get(subject)
        if gold is None:
            return small_answer(_user_text(req))   # unseen ticket: behave sensibly
        if _stable_bucket(salt + subject, error_every) == 0:
            wrong = others[(others.index(gold) + 1) % len(others)]
            return {"category": wrong, "confidence": 0.81}
        return {"category": gold, "confidence": 0.93}

    return FakeLLM(handler=handler, provider=provider, model=model, latency_ms=latency_ms)


def gold_from_cases(cases: list) -> dict[str, str]:
    out: dict[str, str] = {}
    for c in cases:
        m = SUBJECT_RE.search(_user_text(c.request))
        if m:
            out[m.group(1).strip()] = c.expected
    return out


__all__ = ["KEYWORDS", "keyword_scores", "small_answer", "make_small_model", "make_large_model",
           "gold_from_cases"]
