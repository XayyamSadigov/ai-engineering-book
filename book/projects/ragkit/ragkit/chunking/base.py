# path: book/projects/ragkit/ragkit/chunking/base.py
"""Chunker protocol, shared finalization, and deterministic chunk identity.

Every strategy only decides *where to cut*. It returns `Piece`s (character spans, optionally
with replacement text). `BaseChunker.finalize` turns pieces into `Chunk`s the same way for all
strategies: token counts, section paths and pages from the document's blocks, inherited
permissions, and ids.

Chunk id = doc_id + hash(chunker fingerprint, section path, content hash, occurrence, parent id).
Position and document version are deliberately *not* in the id: inserting a paragraph near the
top of a document, or bumping its version without touching a section, leaves the ids of
unchanged chunks unchanged, so an incremental indexer re-embeds only what changed (Chapter 15).
The chunker fingerprint *is* in the id, so two chunking configurations can be indexed side by
side for comparison without colliding.
"""
from __future__ import annotations

import bisect
import json
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, ClassVar, Iterable, Protocol, runtime_checkable

from ..documents import Block, Chunk, ChunkKind, ChunkRole, Document, sha256_text, short_hash
from ..tokenizers import Tokenizer, default_tokenizer


@dataclass
class Piece:
    start: int  # character span in Document.text
    end: int
    text: str | None = None  # None: use Document.text[start:end]
    kind: ChunkKind | None = None  # None: infer from the blocks the span covers
    section_path: list[str] | None = None  # None: section of the block at `start`
    meta: dict[str, Any] = field(default_factory=dict)
    role: ChunkRole = "leaf"
    parent_id: str | None = None


@runtime_checkable
class Chunker(Protocol):
    name: str

    def fingerprint(self) -> str: ...

    def chunk(self, doc: Document) -> list[Chunk]: ...


def make_chunk_id(
    doc_id: str,
    chunker_fingerprint: str,
    section_path: list[str],
    content_hash: str,
    occurrence: int,
    parent_id: str | None = None,
) -> str:
    digest = short_hash(chunker_fingerprint, " / ".join(section_path), content_hash, occurrence, parent_id or "")
    return f"{doc_id}:{digest}"


_DOC_FIELDS_IN_CHUNK = ("title", "source_uri", "updated_at")


class BaseChunker(ABC):
    name: ClassVar[str] = "base"
    algorithm_version: ClassVar[int] = 1

    def __init__(self, tokenizer: Tokenizer | None = None) -> None:
        self.tokenizer = tokenizer or default_tokenizer()

    # -- strategy hooks
    @abstractmethod
    def config(self) -> dict[str, Any]:
        """Every parameter that changes the output. Feeds the fingerprint."""

    @abstractmethod
    def split(self, doc: Document) -> list[Piece]:
        """Decide where to cut."""

    # -- shared behavior
    def fingerprint(self) -> str:
        cfg = json.dumps({**self.config(), "tokenizer": self.tokenizer.name}, sort_keys=True, default=str)
        return f"{self.name}/{self.algorithm_version}:{short_hash(cfg, length=8)}"

    def count(self, text: str) -> int:
        return self.tokenizer.count(text)

    def chunk(self, doc: Document) -> list[Chunk]:
        return self.finalize(doc, self.split(doc))

    def chunk_many(self, docs: Iterable[Document]) -> list[Chunk]:
        out: list[Chunk] = []
        for d in docs:
            out.extend(self.chunk(d))
        return out

    def finalize(self, doc: Document, pieces: list[Piece]) -> list[Chunk]:
        fp = self.fingerprint()
        index = _BlockIndex(doc.blocks)
        seen: Counter[tuple[Any, ...]] = Counter()
        chunks: list[Chunk] = []
        base_meta = {"source_type": doc.source_type.value, **doc.metadata}
        base_meta.update({k: getattr(doc, k) for k in _DOC_FIELDS_IN_CHUNK})
        for position, p in enumerate(pieces):
            text = p.text if p.text is not None else doc.text[p.start : p.end]
            if not text.strip():
                continue
            covered = index.covering(p.start, p.end)
            section = p.section_path if p.section_path is not None else index.section_at(p.start)
            content_hash = sha256_text(text)
            key = (tuple(section), content_hash, p.parent_id)
            occurrence = seen[key]
            seen[key] += 1
            pages = [b.page for b in covered if b.page is not None]
            meta = {**base_meta, **_aggregate_block_meta(covered), **p.meta}
            chunks.append(
                Chunk(
                    id=make_chunk_id(doc.id, fp, section, content_hash, occurrence, p.parent_id),
                    doc_id=doc.id,
                    version=doc.version,
                    text=text,
                    content_hash=content_hash,
                    tenant=doc.tenant,
                    acl_groups=list(doc.acl_groups),
                    metadata=meta,
                    section_path=list(section),
                    parent_id=p.parent_id,
                    position=len(chunks),
                    char_start=p.start,
                    char_end=p.end,
                    token_count=self.count(text),
                    kind=p.kind or _infer_kind(covered),
                    role=p.role,
                    page_start=min(pages) if pages else None,
                    page_end=max(pages) if pages else None,
                    chunker=fp,
                )
            )
        return chunks


class _BlockIndex:
    def __init__(self, blocks: list[Block]) -> None:
        self.blocks = blocks
        self.starts = [b.char_start for b in blocks]

    def section_at(self, pos: int) -> list[str]:
        if not self.blocks:
            return []
        i = max(0, bisect.bisect_right(self.starts, pos) - 1)
        return self.blocks[i].section_path

    def covering(self, start: int, end: int) -> list[Block]:
        if not self.blocks:
            return []
        i = max(0, bisect.bisect_right(self.starts, start) - 1)
        out = []
        while i < len(self.blocks) and self.blocks[i].char_start < max(end, start + 1):
            if self.blocks[i].char_end > start:
                out.append(self.blocks[i])
            i += 1
        return out


def _infer_kind(blocks: list[Block]) -> ChunkKind:
    kinds = {b.kind for b in blocks if b.kind != "heading"}
    if not kinds:
        return "text"
    if kinds == {"table"}:
        return "table"
    if kinds == {"code"}:
        return "code"
    if kinds & {"table", "code"}:
        return "mixed"
    return "text"


def _aggregate_block_meta(blocks: list[Block]) -> dict[str, Any]:
    """Lift per-block facts that matter for retrieval filters: speakers, timestamps, OCR."""
    out: dict[str, Any] = {}
    speakers = sorted({str(b.meta["speaker"]) for b in blocks if b.meta.get("speaker")})
    if speakers:
        out["speakers"] = speakers
    stamps = [b.meta["timestamp"] for b in blocks if b.meta.get("timestamp")]
    if stamps:
        out["start_time"] = stamps[0]
    confidences = [b.meta["ocr_confidence"] for b in blocks if b.meta.get("ocr")]
    if confidences:
        out["ocr"] = True
        out["ocr_confidence_min"] = min(c for c in confidences if c is not None) if any(
            c is not None for c in confidences
        ) else None
    return out


__all__ = ["BaseChunker", "Chunker", "Piece", "make_chunk_id"]
