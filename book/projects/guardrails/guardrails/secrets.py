# path: book/projects/guardrails/guardrails/secrets.py
"""Secret detection: known credential formats plus a high-entropy fallback.

Run it in three places: on user input (people paste keys into chat), on model output (a key
that reached the context can be echoed back), and on everything bound for logs and traces
(see `telemetry.RedactingTracer`). Detection is the backstop. The primary control is that
secrets never enter model context in the first place: tools hold credentials server side.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .pipeline import Action, BaseCheck, FailMode, Finding, GuardContext, Stage, Subject, Verdict

# Known formats. The vendor-specific shapes are examples of public, documented prefixes; add the
# formats your own organization issues (internal service tokens, signed URLs) to this table.
SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "private_key": re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----[\s\S]*?(?:-----END (?:[A-Z]+ )?PRIVATE KEY-----|\Z)"),
    "aws_access_key_id": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "slack_token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    "sk_api_key": re.compile(r"\bsk-(?:[A-Za-z0-9_-]{2,20}-)?[A-Za-z0-9_-]{20,}\b"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    "bearer_header": re.compile(r"(?i)\bauthorization\s*:\s*bearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    "url_credentials": re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s:/@]+:[^\s@/]+@[^\s/]+"),
    "assigned_secret": re.compile(
        r"(?i)\b(?:api[_-]?key|secret|token|passw(?:or)?d|pwd|client[_-]?secret)\b\s*[:=]\s*['\"]?([^\s'\"]{8,})"),
}

_CANDIDATE = re.compile(r"[A-Za-z0-9+/_-]{24,}={0,2}")


@dataclass(frozen=True)
class SecretMatch:
    kind: str
    start: int
    end: int
    value: str


def shannon_entropy(s: str) -> float:
    """Bits per character. Random base64 approaches 6, English prose is near 4, hex caps at 4."""
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def _looks_random(token: str, min_entropy: float) -> bool:
    core = token.strip("=")
    if re.fullmatch(r"[0-9a-fA-F-]+", core):
        return False          # hex (commit SHAs, UUIDs, hashes): too many benign uses; see trade-offs
    if not (re.search(r"[a-z]", core) and re.search(r"[A-Z]", core) and re.search(r"\d", core)):
        return False          # identifiers like ALL_CAPS_CONSTANT or snake_case_names
    if re.fullmatch(r"[A-Za-z]+(?:[_-][A-Za-z0-9]+)+", core) and shannon_entropy(core) < 4.5:
        return False          # long readable identifiers: retail-returns-api-v2-handler
    return shannon_entropy(core) >= min_entropy


def detect_secrets(text: str, high_entropy: bool = True, min_entropy: float = 4.2) -> list[SecretMatch]:
    known: list[SecretMatch] = []
    for kind, rx in SECRET_PATTERNS.items():
        for m in rx.finditer(text):
            known.append(SecretMatch(kind, m.start(), m.end(), m.group(0)))
    # Among known formats keep the earliest-starting, longest match where they overlap.
    known.sort(key=lambda s: (s.start, -(s.end - s.start)))
    out: list[SecretMatch] = []
    for s in known:
        if out and s.start < out[-1].end:
            continue
        out.append(s)
    # The entropy fallback only reports spans no named format already explains.
    if high_entropy:
        for m in _CANDIDATE.finditer(text):
            if any(m.start() < s.end and s.start < m.end() for s in out):
                continue
            if _looks_random(m.group(0), min_entropy):
                out.append(SecretMatch("high_entropy", m.start(), m.end(), m.group(0)))
    return sorted(out, key=lambda s: s.start)


def redact_secrets(text: str, high_entropy: bool = True) -> tuple[str, list[SecretMatch]]:
    matches = detect_secrets(text, high_entropy=high_entropy)
    pieces: list[str] = []
    pos = 0
    for m in matches:
        pieces.append(text[pos:m.start])
        pieces.append(f"[REDACTED:{m.kind}]")
        pos = m.end
    pieces.append(text[pos:])
    return "".join(pieces), matches


class SecretsCheck(BaseCheck):
    """Detects credentials. Typical wiring:

    * INPUT:  `on_detect=REDACT`, the user still gets help but the key never reaches the provider.
    * OUTPUT: `on_detect=BLOCK`, a credential in an answer means something upstream is broken.
    * CONTEXT: `on_detect=REDACT`, documents and tool results with embedded keys get scrubbed.
    """

    name = "secrets"
    fail_mode = FailMode.CLOSED

    def __init__(self, on_detect: Action = Action.REDACT, high_entropy: bool = True,
                 stages: frozenset[Stage] = frozenset({Stage.INPUT, Stage.CONTEXT, Stage.OUTPUT}),
                 name: str = "secrets") -> None:
        self.on_detect = on_detect
        self.high_entropy = high_entropy
        self.stages = stages
        self.name = name

    def evaluate(self, subject: Subject, ctx: GuardContext) -> Verdict:
        redacted, matches = redact_secrets(subject.text, high_entropy=self.high_entropy)
        if not matches:
            return Verdict.allow()
        kinds = sorted({m.kind for m in matches})
        findings = [Finding(m.kind, m.start, m.end) for m in matches]
        reason = "possible secret(s): " + ", ".join(kinds)
        if self.on_detect is Action.BLOCK:
            return Verdict.block(reason, findings=findings)
        if self.on_detect is Action.FLAG:
            return Verdict.flag(reason, 1.0, findings)
        return Verdict.redact(redacted, reason, 1.0, findings)


__all__ = ["SECRET_PATTERNS", "SecretMatch", "shannon_entropy", "detect_secrets", "redact_secrets", "SecretsCheck"]
