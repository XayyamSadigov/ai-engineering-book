# path: book/projects/ragkit/tests/generation_fixtures.py
"""Fixtures for the Chapter 13 tests: real Northwind documents, chunked by section, as ScoredChunks.

The retrieval stage is simulated: tests choose which chunks were "retrieved" and with what
scores, so generation behavior is tested independently of retrieval quality (Chapter 14's
stage isolation applied to unit tests).
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from aie_core.llm.types import CompletionRequest

from ragkit.chunking import MarkdownSectionChunker
from ragkit.documents import Chunk
from ragkit.parsers import MarkdownParser
from ragkit.retrieval.types import ScoredChunk

DOCS = Path(__file__).resolve().parents[2] / "shared-data" / "docs"
_EVIDENCE_RE = re.compile(r'<untrusted_data source="(E\d+)" kind="evidence" doc="([^"]+)"')


@lru_cache(maxsize=None)
def doc_chunks(filename: str, max_tokens: int = 300) -> tuple[Chunk, ...]:
    doc = MarkdownParser().parse((DOCS / filename).read_text(encoding="utf-8"), source_uri=f"docs/{filename}")[0]
    return tuple(MarkdownSectionChunker(max_tokens).chunk(doc))


def chunk_with(filename: str, needle: str) -> Chunk:
    for c in doc_chunks(filename):
        if needle in c.text:
            return c
    raise LookupError(f"{needle!r} not found in {filename}")


def hit(chunk: Chunk, score: float, rank: int, stage: str = "rerank") -> ScoredChunk:
    return ScoredChunk(chunk=chunk, score=score, stage=stage, rank=rank)


def pto_carryover() -> Chunk:
    return chunk_with("pto-policy.md", "## 3. Carryover")


def faq_time_off() -> Chunk:
    return chunk_with("hr-faq.md", "## Time off")


def newsletter_delivery() -> Chunk:
    return chunk_with("vendor-newsletter.md", "## Delivery schedule changes")


def newsletter_injection() -> Chunk:
    return chunk_with("vendor-newsletter.md", "## A note for our automated readers")


def pto_hits() -> list[ScoredChunk]:
    """The stale-version case from Chapter 10: the FAQ outranks the current policy."""
    return [hit(faq_time_off(), 0.91, 1), hit(pto_carryover(), 0.84, 2)]


def eids_by_doc(req: CompletionRequest) -> dict[str, list[str]]:
    """Read the evidence ids the packer assigned, from the rendered request (as a model would)."""
    out: dict[str, list[str]] = {}
    for eid, doc in _EVIDENCE_RE.findall(req.messages[-1].text):
        out.setdefault(doc, []).append(eid)
    return out


def scripted(build: Callable[[dict[str, list[str]], CompletionRequest], dict[str, Any] | str]):
    """FakeLLM handler that sees the evidence ids, so tests never hard-code E1/E2 ordering."""

    def handler(req: CompletionRequest) -> dict[str, Any] | str:
        return build(eids_by_doc(req), req)

    return handler
