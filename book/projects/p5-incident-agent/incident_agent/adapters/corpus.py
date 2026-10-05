# path: book/projects/p5-incident-agent/incident_agent/adapters/corpus.py
"""Runbooks and past incident reports from shared-data, chunked with ragkit and ranked with BM25.

ACL filtering uses the trusted principal (groups and tenant) and happens before ranking
results are returned, so a document the caller cannot read never reaches the model, not
even as a title. Results are grouped to the best chunk per document: the citation unit is
the document id, which is what the report, the DoD, and the reviewer all reason about.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rank_bm25 import BM25Okapi

from ragkit import Chunk, MarkdownSectionChunker, load_documents

Kind = Literal["runbook", "incident"]
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {"the", "and", "for", "with", "that", "this", "from", "are", "was", "not", "but", "into", "our", "all"}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if len(t) > 1 and t not in _STOP]


@dataclass(frozen=True)
class Hit:
    doc_id: str
    title: str
    section: str
    text: str
    score: float


class _Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.embedding_text()) for c in chunks]) if chunks else None

    def search(self, query: str, principal: dict[str, Any], k: int) -> list[Hit]:
        terms = tokenize(query)
        if self.bm25 is None or not terms:
            return []
        groups, tenant = set(principal.get("groups", [])) | {"all"}, principal.get("tenant")
        scores = self.bm25.get_scores(terms)
        best: dict[str, Hit] = {}
        for chunk, score in sorted(zip(self.chunks, scores), key=lambda p: -p[1]):
            if score <= 0:
                break
            if not set(chunk.acl_groups) & groups or chunk.tenant not in ("shared", tenant):
                continue                                     # ACL before anything leaves this function
            if chunk.doc_id not in best:
                best[chunk.doc_id] = Hit(chunk.doc_id, str(chunk.metadata.get("title", "")),
                                         " > ".join(chunk.section_path[1:]) or "overview",
                                         " ".join(chunk.text.split()), float(score))
            if len(best) >= k:
                break
        return list(best.values())


class KnowledgeBase:
    def __init__(self, docs_dir: Path, *, chunk_tokens: int = 250) -> None:
        report = load_documents(docs_dir, root=docs_dir.parent)
        chunker = MarkdownSectionChunker(max_tokens=chunk_tokens)
        runbooks = [d for d in report.documents if "runbook" in d.metadata.get("tags", [])]
        incidents = [d for d in report.documents if d.id.startswith("inc-")]
        self.titles = {d.id: d.title for d in report.documents}
        self.runbook_ids = frozenset(d.id for d in runbooks)          # the catalog the DoD checks against
        self._indexes: dict[Kind, _Index] = {
            "runbook": _Index(chunker.chunk_many(runbooks)),
            "incident": _Index(chunker.chunk_many(incidents)),
        }

    def search(self, kind: Kind, query: str, principal: dict[str, Any], k: int = 3) -> list[Hit]:
        return self._indexes[kind].search(query, principal, k)


__all__ = ["KnowledgeBase", "Hit", "tokenize"]
