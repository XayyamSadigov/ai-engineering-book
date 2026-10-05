# path: book/projects/examples/ch25/taskevals/text_support.py
"""Small lexical helpers shared by the summarization and RAG evaluators.

These are deliberately simple, deterministic proxies. They catch invented numbers, invented
identifiers, and claims with no lexical support at all, which is a large and cheap share of
real faithfulness failures. They cannot recognize a paraphrase that changes meaning, so they
run first and an LLM judge (evalkit.LLMJudge) handles what they cannot decide.
"""
from __future__ import annotations

import re

STOPWORDS = frozenset(
    "a an the and or but if of to in on at by for with from as is are was were be been being it its this that "
    "these those there here what which who whom when where why how do does did can could should would will may "
    "might must not no yes your you we our they their he she his her i me my so than then also any all each per "
    "into about over under up down out only just more most very".split()
)

_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?")
_IDENT = re.compile(r"\b[A-Z][A-Z0-9]+(?:-[A-Z0-9]+)+\b")  # ticket ids, PO numbers, error codes
_CITATION = re.compile(r"\[[^\]]+\]")


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS and len(w) > 2}


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", _CITATION.sub("", text))
    return [p.strip() for p in parts if len(p.strip()) > 3]


def numbers_in(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUMBER.findall(text)}


def identifiers_in(text: str) -> set[str]:
    return set(_IDENT.findall(text))


def unsupported_atoms(claim: str, source: str) -> list[str]:
    """Numbers and identifiers in `claim` that never appear in `source`."""
    src_numbers = numbers_in(source)
    src_ids = identifiers_in(source)
    return sorted((numbers_in(claim) - src_numbers) | (identifiers_in(claim) - src_ids))


def overlap(claim: str, passage: str) -> float:
    """Share of the claim's content words found in the passage."""
    cw = content_words(claim)
    return len(cw & content_words(passage)) / len(cw) if cw else 1.0


__all__ = ["STOPWORDS", "content_words", "split_sentences", "numbers_in", "identifiers_in",
           "unsupported_atoms", "overlap"]
