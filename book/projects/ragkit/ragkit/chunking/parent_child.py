# path: book/projects/ragkit/ragkit/chunking/parent_child.py
"""Parent-child chunking: search small, read big.

Parents are produced by one chunker (sections by default) and children by running a second
chunker inside each parent. Index the children; when a child is retrieved, hand the generator
its parent. Small children give precise embeddings; parents restore the definitions and
qualifiers that a child alone would lose.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from ..documents import Chunk, Document
from ..tokenizers import Tokenizer
from .base import BaseChunker, Piece
from .section import MarkdownSectionChunker
from .sentence import SentenceChunker


class ParentChildChunker(BaseChunker):
    name = "parent_child"

    def __init__(
        self,
        parent: BaseChunker | None = None,
        child: BaseChunker | None = None,
        *,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        super().__init__(tokenizer)
        self.parent = parent or MarkdownSectionChunker(800, tokenizer=self.tokenizer)
        self.child = child or SentenceChunker(128, overlap_sentences=0, tokenizer=self.tokenizer)

    def config(self) -> dict[str, Any]:
        return {"parent": self.parent.fingerprint(), "child": self.child.fingerprint()}

    def split(self, doc: Document) -> list[Piece]:
        """Leaf view (children without parent ids). Use `chunk` to get linked parents and children."""
        out: list[Piece] = []
        for p in self.parent.split(doc):
            out.extend(self._children_of(doc, p))
        return out

    def _children_of(self, doc: Document, parent: Piece) -> list[Piece]:
        parent_text = parent.text if parent.text is not None else doc.text[parent.start : parent.end]
        contiguous = parent.text is None or parent.text == doc.text[parent.start : parent.end]
        sub = doc.model_copy(update={"text": parent_text, "blocks": [], "pages": []})
        out = []
        for c in self.child.split(sub):
            child_text = parent_text[c.start : c.end]
            if contiguous:
                start, end, text = parent.start + c.start, parent.start + c.end, None
            else:  # parent text was synthesized (e.g. repeated table header); keep the parent span
                start, end, text = parent.start, parent.end, child_text
            out.append(
                Piece(start, end, text=text, kind=parent.kind, section_path=parent.section_path, role="child",
                      meta={**c.meta})
            )
        return out

    def chunk(self, doc: Document) -> list[Chunk]:
        parent_pieces = self.parent.split(doc)
        for p in parent_pieces:
            p.role = "parent"
        parents = self.finalize(doc, parent_pieces)
        # finalize skips blank pieces, so re-pair surviving parents with their pieces by span
        by_span = {(c.char_start, c.char_end, c.text): c for c in parents}
        out: list[Chunk] = []
        child_pieces: list[Piece] = []
        groups: list[tuple[Chunk, list[Piece]]] = []
        for piece in parent_pieces:
            text = piece.text if piece.text is not None else doc.text[piece.start : piece.end]
            parent_chunk = by_span.get((piece.start, piece.end, text))
            if parent_chunk is None:
                continue
            kids = self._children_of(doc, piece)
            for k in kids:
                k.parent_id = parent_chunk.id
            groups.append((parent_chunk, kids))
            child_pieces.extend(kids)
        children = self.finalize(doc, child_pieces)
        by_parent: dict[str, list[Chunk]] = {}
        for c in children:
            by_parent.setdefault(c.parent_id or "", []).append(c)
        for parent_chunk, _ in groups:
            parent_chunk.metadata["child_count"] = len(by_parent.get(parent_chunk.id, []))
            out.append(parent_chunk)
            out.extend(by_parent.get(parent_chunk.id, []))
        return out


def split_roles(chunks: Iterable[Chunk]) -> tuple[list[Chunk], list[Chunk]]:
    """(parents, children). Leaf chunks from other strategies count as children of nobody."""
    parents, children = [], []
    for c in chunks:
        (parents if c.role == "parent" else children).append(c)
    return parents, children


def expand_to_parents(hits: Iterable[Chunk], parents: Mapping[str, Chunk]) -> list[Chunk]:
    """Map ranked child hits to their parents, keeping the best rank and dropping repeats."""
    out: list[Chunk] = []
    seen: set[str] = set()
    for hit in hits:
        target = parents.get(hit.parent_id or "", hit)
        if target.id not in seen:
            seen.add(target.id)
            out.append(target)
    return out


__all__ = ["ParentChildChunker", "expand_to_parents", "split_roles"]
