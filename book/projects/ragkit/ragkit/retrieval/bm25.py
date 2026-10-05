# path: book/projects/ragkit/ragkit/retrieval/bm25.py
"""BM25 lexical retrieval, implemented from scratch.

Three decisions matter more than the formula:
- Tokenization. Identifiers such as "INC-2025-1142" or "RET-002" are indexed both whole and as
  parts, so an exact-ID query matches strongly and a partial one ("1142") still matches.
- Authorization before scoring. The candidate set is restricted to chunks the principal may
  read *before* any score is computed, so a forbidden chunk never occupies a top-k slot and
  never appears in a debug log of scores.
- Persistence with a fingerprint. The saved index records the tokenizer configuration; loading
  it with a different tokenizer is an error, because query and index terms would not line up.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from ..documents import Chunk
from .common import Stopwatch, indexed_text, matches_filters, validate_filters
from .types import Principal, RetrievalQuery, RetrievalResult, ScoredChunk, visible

FORMAT_VERSION = 1

STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have how i if in into is it
    its me my of on or our so than that the their them then there these they this to up us was we were what
    when where which who why will with would you your""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_./][a-z0-9]+)*")
_PART_RE = re.compile(r"[-_./]")


class BM25Tokenizer:
    """Lowercase word tokens; compound identifiers emitted whole and split; optional plural folding."""

    def __init__(self, *, stopwords: Iterable[str] = STOPWORDS, fold_plurals: bool = True, min_len: int = 1) -> None:
        self.stopwords = frozenset(stopwords)
        self.fold_plurals = fold_plurals
        self.min_len = min_len

    @property
    def fingerprint(self) -> str:
        return f"bm25tok-v1:plurals={self.fold_plurals}:min={self.min_len}:stop={len(self.stopwords)}"

    def _norm(self, word: str) -> str:
        if self.fold_plurals and len(word) > 3 and word.isalpha():
            if word.endswith("ies") and len(word) > 4:
                return word[:-3] + "y"
            if word.endswith("s") and not word.endswith("ss"):
                return word[:-1]
        return word

    def tokenize(self, text: str) -> list[str]:
        out: list[str] = []
        for match in _TOKEN_RE.finditer(text.lower()):
            tok = match.group(0)
            if _PART_RE.search(tok):
                out.append(tok)  # the whole identifier: "inc-2025-1142"
                parts = [p for p in _PART_RE.split(tok) if p]
                out.extend(self._norm(p) for p in parts if p not in self.stopwords and len(p) >= self.min_len)
            elif tok not in self.stopwords and len(tok) >= self.min_len:
                out.append(self._norm(tok))
        return out


class BM25Index:
    """An in-memory inverted index with Okapi BM25 scoring and ACL-aware candidate selection.

    score(q, d) = sum over query terms t of  idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
    idf(t)      = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))      (always positive)

    Corpus statistics (N, n_t, avgdl) are global over the index, not per principal.
    """

    name = "bm25"

    def __init__(self, *, k1: float = 1.2, b: float = 0.75, tokenizer: BM25Tokenizer | None = None) -> None:
        if k1 < 0 or not 0.0 <= b <= 1.0:
            raise ValueError("BM25 needs k1 >= 0 and 0 <= b <= 1")
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer or BM25Tokenizer()
        self._chunks: dict[str, Chunk] = {}
        self._tf: dict[str, Counter[str]] = {}  # forward index: chunk id -> term counts
        self._length: dict[str, int] = {}
        self._postings: dict[str, dict[str, int]] = {}  # inverted index: term -> {chunk id: tf}
        self._by_doc: dict[str, set[str]] = {}
        self._total_length = 0

    # ------------------------------------------------------------------ writes
    def add(self, chunks: Iterable[Chunk]) -> int:
        """Insert or replace chunks by id. Returns the number written."""
        n = 0
        for chunk in chunks:
            if chunk.id in self._chunks:
                self._remove_one(chunk.id)
            terms = Counter(self.tokenizer.tokenize(indexed_text(chunk)))
            self._chunks[chunk.id] = chunk
            self._tf[chunk.id] = terms
            length = sum(terms.values())
            self._length[chunk.id] = length
            self._total_length += length
            self._by_doc.setdefault(chunk.doc_id, set()).add(chunk.id)
            for term, tf in terms.items():
                self._postings.setdefault(term, {})[chunk.id] = tf
            n += 1
        return n

    def replace_document(self, doc_id: str, chunks: Iterable[Chunk]) -> int:
        """Make `chunks` the only indexed chunks of `doc_id` (re-ingestion of a new version)."""
        self.delete_document(doc_id)
        return self.add(chunks)

    def delete_document(self, doc_id: str) -> int:
        ids = list(self._by_doc.pop(doc_id, set()))
        for cid in ids:
            self._remove_one(cid, drop_doc_entry=False)
        return len(ids)

    def _remove_one(self, chunk_id: str, *, drop_doc_entry: bool = True) -> None:
        chunk = self._chunks.pop(chunk_id)
        for term in self._tf.pop(chunk_id):
            posting = self._postings.get(term)
            if posting is not None:
                posting.pop(chunk_id, None)
                if not posting:
                    del self._postings[term]
        self._total_length -= self._length.pop(chunk_id)
        if drop_doc_entry:
            ids = self._by_doc.get(chunk.doc_id)
            if ids is not None:
                ids.discard(chunk_id)
                if not ids:
                    del self._by_doc[chunk.doc_id]

    # ------------------------------------------------------------------ lookups
    def get_chunks(self, chunk_ids: Iterable[str], principal: Principal) -> list[Chunk]:
        """Chunks by id, in the order asked, for re-display or citation resolution.

        A lookup by id bypasses search, so it must not bypass authorization: the principal is
        required and chunks it may not read are omitted, exactly like unknown ids. Callers that
        need to tell "forbidden" from "absent" are asking the wrong question of a user-facing path.
        """
        out: list[Chunk] = []
        for cid in chunk_ids:
            chunk = self._chunks.get(cid)
            if chunk is not None and visible(chunk, principal):
                out.append(chunk)
        return out

    def doc_ids(self) -> set[str]:
        """Documents with at least one indexed chunk (for reconciliation and audits)."""
        return set(self._by_doc)

    # ------------------------------------------------------------------ stats
    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def avgdl(self) -> float:
        return self._total_length / len(self._chunks) if self._chunks else 0.0

    def idf(self, term: str) -> float:
        n_t = len(self._postings.get(term, {}))
        n = len(self._chunks)
        return math.log(1.0 + (n - n_t + 0.5) / (n_t + 0.5))

    def _term_score(self, term: str, chunk_id: str, tf: int) -> float:
        dl = self._length[chunk_id]
        avgdl = self.avgdl or 1.0
        denom = tf + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
        return self.idf(term) * tf * (self.k1 + 1.0) / denom

    # ------------------------------------------------------------------ reads
    def allowed_ids(self, principal: Principal, filters: dict | None = None) -> set[str]:
        """Chunk ids this principal may see under these filters. Computed before any scoring."""
        filters = filters or {}
        validate_filters(filters)
        return {cid for cid, c in self._chunks.items() if visible(c, principal) and matches_filters(c, filters)}

    def search(self, text: str, principal: Principal, k: int = 10, filters: dict | None = None) -> list[ScoredChunk]:
        if k <= 0 or not self._chunks:
            return []
        allowed = self.allowed_ids(principal, filters)
        if not allowed:
            return []
        query_terms = Counter(self.tokenizer.tokenize(text))
        scores: dict[str, float] = {}
        matched: Counter[str] = Counter()
        for term, qtf in query_terms.items():
            for cid, tf in self._postings.get(term, {}).items():
                if cid not in allowed:  # authorization first: forbidden chunks are never scored
                    continue
                scores[cid] = scores.get(cid, 0.0) + qtf * self._term_score(term, cid, tf)
                matched[cid] += 1
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [
            ScoredChunk(
                chunk=self._chunks[cid],
                score=score,
                stage=self.name,
                rank=i,
                signals={"bm25": round(score, 6), "bm25_matched_terms": float(matched[cid])},
            )
            for i, (cid, score) in enumerate(ranked, start=1)
        ]

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        with Stopwatch() as sw:
            hits = self.search(query.text, query.principal, query.k, query.filters)
        trace = {"stage": self.name, "k": query.k, "candidate_ids": [h.chunk.id for h in hits], "latency_ms": sw.ms}
        return RetrievalResult(query=query, hits=hits, trace=trace)

    def explain(self, text: str, chunk_id: str) -> dict[str, float]:
        """Per-term contribution to one chunk's score. Useful when a ranking looks wrong."""
        tf = self._tf[chunk_id]
        out: dict[str, float] = {}
        for term, qtf in Counter(self.tokenizer.tokenize(text)).items():
            if tf.get(term):
                out[term] = round(qtf * self._term_score(term, chunk_id, tf[term]), 6)
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> Path:
        """Write chunks and their term vectors as JSON (write-then-rename). Postings are rebuilt on load."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": FORMAT_VERSION,
            "k1": self.k1,
            "b": self.b,
            "tokenizer": self.tokenizer.fingerprint,
            "chunks": [
                {"chunk": self._chunks[cid].model_dump(mode="json"), "tf": dict(self._tf[cid])}
                for cid in sorted(self._chunks)
            ],
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path, *, tokenizer: BM25Tokenizer | None = None) -> "BM25Index":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != FORMAT_VERSION:
            raise ValueError(f"unsupported BM25 index format {data.get('format')!r}")
        index = cls(k1=float(data["k1"]), b=float(data["b"]), tokenizer=tokenizer)
        if data["tokenizer"] != index.tokenizer.fingerprint:
            raise ValueError(
                f"index was built with tokenizer {data['tokenizer']!r}, loader uses {index.tokenizer.fingerprint!r}"
            )
        for row in data["chunks"]:
            chunk = Chunk.model_validate(row["chunk"])
            terms = Counter({t: int(n) for t, n in row["tf"].items()})
            index._chunks[chunk.id] = chunk
            index._tf[chunk.id] = terms
            index._length[chunk.id] = sum(terms.values())
            index._total_length += index._length[chunk.id]
            index._by_doc.setdefault(chunk.doc_id, set()).add(chunk.id)
            for term, tf in terms.items():
                index._postings.setdefault(term, {})[chunk.id] = tf
        return index


__all__ = ["BM25Index", "BM25Tokenizer", "STOPWORDS"]
