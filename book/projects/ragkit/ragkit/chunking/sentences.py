# path: book/projects/ragkit/ragkit/chunking/sentences.py
"""A rule-based sentence splitter that returns character spans.

Rules: a sentence ends at '.', '!' or '?' (plus closing quotes or brackets) followed by
whitespace, unless the period follows a known abbreviation or a single initial, or the next
word starts lowercase. Every newline is also a boundary: in parsed documents, single newlines
separate list items, table rows, and code lines, none of which should be glued together.
Statistical splitters do better on messy prose; this one is deterministic and has no
dependencies, which is what an ingestion library needs by default.
"""
from __future__ import annotations

import re

from ..tokenizers import Span, Tokenizer

ABBREVIATIONS = frozenset(
    {"e.g", "i.e", "etc", "vs", "dr", "mr", "mrs", "ms", "no", "fig", "approx", "inc", "ltd", "jr", "sr", "st",
     "u.s", "a.m", "p.m", "dept", "est", "min", "max", "cf", "al", "sec", "ch", "vol", "rev"}
)
_CANDIDATE = re.compile(r"(?P<p>[.!?]+[\"')\]]*)\s+|(?P<nl>[ \t]*\n\s*)")
_LAST_WORD = re.compile(r"(\S+)$")


def split_sentences(text: str) -> list[Span]:
    spans: list[Span] = []
    start = 0

    def add(s: int, e: int) -> None:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            spans.append((s, e))

    for m in _CANDIDATE.finditer(text):
        if m.group("p") is not None:
            punct = m.group("p")
            if punct[0] == ".":
                w = _LAST_WORD.search(text, start, m.start())
                word = w.group(1).lower().lstrip("(\"'[") if w else ""
                nxt = text[m.end() : m.end() + 1]
                if word in ABBREVIATIONS or (len(word) == 1 and word.isalpha()) or (nxt.islower() and "\n" not in m.group(0)):
                    continue
            add(start, m.start() + len(punct))
        else:
            add(start, m.start())
        start = m.end()
    add(start, len(text))
    return spans


def token_windows(text: str, start: int, end: int, max_tokens: int, tokenizer: Tokenizer) -> list[Span]:
    """Cut [start, end) into contiguous pieces of at most max_tokens tokens, at token starts."""
    spans = tokenizer.spans(text[start:end])
    if len(spans) <= max_tokens:
        return [(start, end)]
    cuts = [start + spans[k][0] for k in range(max_tokens, len(spans), max_tokens)]
    bounds = [start, *cuts, end]
    return [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]


def pack_spans(text: str, spans: list[Span], max_tokens: int, tokenizer: Tokenizer) -> list[Span]:
    """Greedily merge consecutive spans into groups of at most max_tokens (single long spans are cut)."""
    units: list[Span] = []
    for s, e in spans:
        units.extend(token_windows(text, s, e, max_tokens, tokenizer))
    out: list[Span] = []
    cur_start: int | None = None
    cur_end = 0
    cur_tokens = 0
    for s, e in units:
        t = tokenizer.count(text[s:e])
        if cur_start is not None and cur_tokens + t > max_tokens:
            out.append((cur_start, cur_end))
            cur_start, cur_tokens = None, 0
        if cur_start is None:
            cur_start = s
        cur_end = e
        cur_tokens += t
    if cur_start is not None:
        out.append((cur_start, cur_end))
    return out


__all__ = ["ABBREVIATIONS", "pack_spans", "split_sentences", "token_windows"]
