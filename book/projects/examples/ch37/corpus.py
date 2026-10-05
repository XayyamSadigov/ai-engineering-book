# path: book/projects/examples/ch37/corpus.py
"""Shared fixtures for the Chapter 37 examples.

- The Northwind corpus as ragkit Documents and section-aware Chunks (Chapter 11).
- `Principal` and an ACL check with the book-wide rule (tenant matches or is shared, groups overlap).
- `LexicalIndex`: a deliberately small TF-IDF scorer. It is a stand-in for the real retrievers
  of Chapter 12 (BM25, dense, hybrid, rerank); the patterns in this chapter sit on top of any of them.
- `FullTextIndex`: SQLite FTS5 with metadata and ACL filtering in SQL, the "vectorless" baseline.
"""
from __future__ import annotations

import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

from ragkit import Chunk, Document, MarkdownSectionChunker, load_documents

SHARED_DATA = Path(__file__).resolve().parents[2] / "shared-data"

STOPWORDS = frozenset(
    """a an and are as at be by can could did do does for from had has have how i if in into is it its
    me my not of on or our should so than that the their then there these this to was we were what when
    where which who why will with would you your""".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")


def terms(text: str) -> list[str]:
    """Lowercased alphanumeric tokens without stopwords. "INC-2025-1142" -> inc, 2025, 1142."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


@dataclass(frozen=True)
class Principal:
    """Who is asking. Every retrieval path in this chapter takes one and filters by it."""

    user: str
    groups: frozenset[str]
    tenant: str

    def can_read(self, tenant: str | None, acl_groups: Iterable[str]) -> bool:
        if tenant not in ("shared", self.tenant):
            return False
        return bool(self.groups & set(acl_groups))

    def can_read_chunk(self, chunk: Chunk) -> bool:
        return self.can_read(chunk.tenant, chunk.acl_groups)

    def can_read_doc(self, doc: Document) -> bool:
        return self.can_read(doc.tenant, doc.acl_groups)


EMPLOYEE = Principal("amira", frozenset({"all"}), "retail")
ONCALL = Principal("jonas", frozenset({"all", "it-oncall"}), "retail")


@lru_cache(maxsize=1)
def load_corpus() -> tuple[Document, ...]:
    report = load_documents(SHARED_DATA / "docs", root=SHARED_DATA)
    return tuple(sorted(report.documents, key=lambda d: d.id))


def section_chunks(docs: Sequence[Document] | None = None, max_tokens: int = 300) -> list[Chunk]:
    chunker = MarkdownSectionChunker(max_tokens)
    out: list[Chunk] = []
    for d in docs if docs is not None else load_corpus():
        out.extend(chunker.chunk(d))
    return out


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class LexicalIndex:
    """TF-IDF with sublinear term frequency and length damping. Small, explainable, offline."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        self._tf = [Counter(terms(c.embedding_text())) for c in self.chunks]
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self.chunks)
        self._idf = {t: math.log((n + 1) / (f + 1)) + 1.0 for t, f in df.items()}
        self._len = [math.sqrt(max(1, sum(tf.values()))) for tf in self._tf]
        self.by_id = {c.id: c for c in self.chunks}

    def search(
        self, query: str, principal: Principal, k: int = 4, exclude: Iterable[str] = ()
    ) -> list[Hit]:
        q = set(terms(query))
        skip = set(exclude)
        scored: list[Hit] = []
        for i, chunk in enumerate(self.chunks):
            if chunk.id in skip or not principal.can_read_chunk(chunk):  # ACL before scoring
                continue
            tf = self._tf[i]
            s = sum(self._idf[t] * (1.0 + math.log(tf[t])) for t in q if t in tf)
            if s > 0:
                scored.append(Hit(chunk, s / self._len[i] ** 0.5))
        scored.sort(key=lambda h: (-h.score, h.chunk.id))
        return scored[:k]


class FullTextIndex:
    """SQLite FTS5: phrase, prefix and boolean queries ranked by bm25(), filters in SQL."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.executescript(
            """
            CREATE VIRTUAL TABLE chunk_fts USING fts5(chunk_id UNINDEXED, title, section, body);
            CREATE TABLE chunk_meta (chunk_id TEXT PRIMARY KEY, doc_id TEXT, tenant TEXT, updated_at TEXT);
            CREATE TABLE chunk_acl (chunk_id TEXT, grp TEXT);
            """
        )
        for c in chunks:
            self.conn.execute(
                "INSERT INTO chunk_fts VALUES (?, ?, ?, ?)",
                (c.id, str(c.metadata.get("title", "")), " / ".join(c.section_path), c.text),
            )
            self.conn.execute(
                "INSERT INTO chunk_meta VALUES (?, ?, ?, ?)",
                (c.id, c.doc_id, c.tenant, str(c.metadata.get("updated_at", ""))),
            )
            self.conn.executemany("INSERT INTO chunk_acl VALUES (?, ?)", [(c.id, g) for g in c.acl_groups])

    def search(
        self,
        match: str,
        principal: Principal,
        k: int = 5,
        updated_after: str | None = None,
    ) -> list[tuple[str, float]]:
        groups = sorted(principal.groups)
        sql = f"""
            SELECT f.chunk_id, bm25(chunk_fts) AS rank
            FROM chunk_fts f JOIN chunk_meta m ON m.chunk_id = f.chunk_id
            WHERE chunk_fts MATCH ?
              AND m.tenant IN ('shared', ?)
              AND f.chunk_id IN (SELECT chunk_id FROM chunk_acl WHERE grp IN ({",".join("?" * len(groups))}))
              AND (? IS NULL OR m.updated_at > ?)
            ORDER BY rank LIMIT ?
        """
        rows = self.conn.execute(sql, (match, principal.tenant, *groups, updated_after, updated_after, k))
        return [(cid, float(rank)) for cid, rank in rows]


__all__ = [
    "EMPLOYEE",
    "ONCALL",
    "FullTextIndex",
    "Hit",
    "LexicalIndex",
    "Principal",
    "SHARED_DATA",
    "load_corpus",
    "section_chunks",
    "terms",
]
