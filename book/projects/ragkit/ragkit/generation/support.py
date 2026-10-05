# path: book/projects/ragkit/ragkit/generation/support.py
"""Deterministic text helpers shared by the packer, validator and streamer.

Everything here is cheap and explainable: citation-marker parsing, content-word extraction,
and a lexical support score. Lexical support is a smoke alarm, not a faithfulness judge. It
catches invented numbers and claims that share almost no vocabulary with their evidence; it
cannot catch a paraphrase that reverses meaning. Use an LLM judge (validator hook) or human
review for that, and measure both against labels (Chapter 14).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# "[E1]", "[E1, E3]", "[E2][E4]". Evidence ids are assigned by the packer, never by the model.
MARKER_RE = re.compile(r"\[\s*(E\d+(?:\s*,\s*E\d+)*)\s*\]")
EID_RE = re.compile(r"E\d+")

# ISO dates stay one token: splitting "2026-01-15" would make "15" look supported.
_TOKEN_RE = re.compile(r"\d{4}-\d{2}-\d{2}|[a-z]+|\d+(?:[.,]\d+)*")
_NUMBER_RE = re.compile(r"^\d+(?:[.,]\d+)*$|^\d{4}-\d{2}-\d{2}$")

STOPWORDS = frozenset(
    """a about above after again all also am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has have
    having he her here hers him his how i if in into is it its itself just may me might more most must my
    no nor not now of off on once only or other our ours out over own same shall she should so some such
    than that the their them then there these they this those through to too under until up upon us very
    was we were what when where which while who whom why will with would you your yours per via e g""".split()
)

NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
    "fifteen": "15", "twenty": "20", "thirty": "30",
}


def markers(text: str) -> list[str]:
    """Evidence ids cited inline, in first-seen order, without duplicates."""
    seen: dict[str, None] = {}
    for group in MARKER_RE.findall(text):
        for eid in EID_RE.findall(group):
            seen.setdefault(eid, None)
    return list(seen)


def strip_markers(text: str) -> str:
    out = MARKER_RE.sub("", text)
    out = re.sub(r"\s+([.,;:!?])", r"\1", out)
    return re.sub(r"[ \t]{2,}", " ", out).strip()


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def content_tokens(text: str) -> list[str]:
    """Lower-cased content words (light stemming) and normalized numbers."""
    out: list[str] = []
    for raw in _TOKEN_RE.findall(strip_markers(text).lower()):
        tok = NUMBER_WORDS.get(raw, raw)
        if _NUMBER_RE.match(tok):
            out.append(tok if "-" in tok else tok.replace(",", ""))
        elif tok not in STOPWORDS and len(tok) > 1:
            out.append(_stem(tok))
    return out


def numbers(tokens: list[str]) -> set[str]:
    return {t for t in tokens if _NUMBER_RE.match(t)}


def normalize_ws(text: str) -> str:
    """Collapse whitespace and Markdown emphasis so quotes match across formatting."""
    text = re.sub(r"[*_`]+", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()


@dataclass(frozen=True)
class Support:
    coverage: float  # share of the claim's content tokens found in the evidence
    missing_numbers: frozenset[str]  # numbers in the claim that the evidence never states
    missing_tokens: tuple[str, ...]

    def supported(self, min_coverage: float) -> bool:
        return not self.missing_numbers and self.coverage >= min_coverage


def lexical_support(claim: str, evidence_text: str) -> Support:
    """How much of `claim` is lexically present in `evidence_text`.

    Numbers are strict: one number that the evidence does not contain makes the claim
    unsupported regardless of coverage, because invented figures are the most common and
    most damaging grounded-answer failure.
    """
    claim_toks = content_tokens(claim)
    if not claim_toks:
        return Support(1.0, frozenset(), ())
    ev = set(content_tokens(evidence_text))
    missing = tuple(t for t in claim_toks if t not in ev)
    coverage = 1.0 - len(missing) / len(claim_toks)
    return Support(coverage, frozenset(numbers(claim_toks) - ev), missing)


def jaccard(a: str, b: str) -> float:
    sa, sb = set(content_tokens(a)), set(content_tokens(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


_SENTENCE_END_RE = re.compile(r"(?<=[.!?])((?:\s*\[\s*E\d+(?:\s*,\s*E\d+)*\s*\])*)\s+(?=[^\s\[])")


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, keeping trailing citation markers with their sentence."""
    parts: list[str] = []
    start = 0
    for m in _SENTENCE_END_RE.finditer(text):
        end = m.start() + len(m.group(1))
        parts.append(text[start:end].strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return [p for p in parts if p]


__all__ = [
    "MARKER_RE",
    "STOPWORDS",
    "Support",
    "content_tokens",
    "jaccard",
    "lexical_support",
    "markers",
    "normalize_ws",
    "numbers",
    "split_sentences",
    "strip_markers",
]
