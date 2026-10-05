# path: book/projects/ragkit/ragkit/chunking/recursive.py
"""Recursive splitting: try the coarsest separator first, recurse only into pieces that are too big."""
from __future__ import annotations

from typing import Any, Sequence

from ..documents import Document
from ..tokenizers import Span, Tokenizer
from .base import BaseChunker, Piece
from .sentences import token_windows

DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


class RecursiveChunker(BaseChunker):
    """Split on paragraphs, then lines, then sentences, then words, then tokens; merge small neighbors.

    Overlap is measured in tokens but applied in whole atoms (the pieces produced by splitting),
    so with paragraph-sized atoms the effective overlap can be zero. That is usually what you want:
    repeating half a paragraph rarely helps retrieval.
    """

    name = "recursive"

    def __init__(
        self,
        chunk_size: int = 256,
        overlap: int = 0,
        *,
        separators: Sequence[str] = DEFAULT_SEPARATORS,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        super().__init__(tokenizer)
        if chunk_size <= 0 or not 0 <= overlap < chunk_size:
            raise ValueError("need chunk_size > 0 and 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = tuple(separators)

    def config(self) -> dict[str, Any]:
        return {"chunk_size": self.chunk_size, "overlap": self.overlap, "separators": list(self.separators)}

    # -- splitting
    def _split_span(self, text: str, s: int, e: int, seps: tuple[str, ...]) -> list[Span]:
        if self.count(text[s:e]) <= self.chunk_size:
            return [(s, e)]
        while seps and seps[0] and text.find(seps[0], s, e) == -1:
            seps = seps[1:]
        if not seps or seps[0] == "":
            return token_windows(text, s, e, self.chunk_size, self.tokenizer)
        sep, rest = seps[0], seps[1:]
        parts: list[Span] = []
        idx = s
        pos = text.find(sep, idx, e)
        while pos != -1:
            parts.append((idx, pos + len(sep)))  # separator stays with the left part: spans stay contiguous
            idx = pos + len(sep)
            pos = text.find(sep, idx, e)
        if idx < e:
            parts.append((idx, e))
        out: list[Span] = []
        for ps, pe in parts:
            if self.count(text[ps:pe]) > self.chunk_size:
                out.extend(self._split_span(text, ps, pe, rest))
            else:
                out.append((ps, pe))
        return out

    def _merge(self, text: str, atoms: list[Span]) -> list[Span]:
        out: list[Span] = []
        cur: list[tuple[int, int, int]] = []  # (start, end, tokens)
        cur_tokens = 0
        for s, e in atoms:
            t = self.count(text[s:e])
            if t == 0:
                if cur:
                    cur[-1] = (cur[-1][0], e, cur[-1][2])
                continue
            if cur and cur_tokens + t > self.chunk_size:
                out.append((cur[0][0], cur[-1][1]))
                tail: list[tuple[int, int, int]] = []
                tail_tokens = 0
                for atom in reversed(cur):
                    if tail_tokens + atom[2] > self.overlap or tail_tokens + atom[2] + t > self.chunk_size:
                        break
                    tail.insert(0, atom)
                    tail_tokens += atom[2]
                cur, cur_tokens = tail, tail_tokens
            cur.append((s, e, t))
            cur_tokens += t
        if cur:
            out.append((cur[0][0], cur[-1][1]))
        return out

    def split(self, doc: Document) -> list[Piece]:
        text = doc.text
        if not text.strip():
            return []
        atoms = self._split_span(text, 0, len(text), self.separators)
        pieces = []
        for s, e in self._merge(text, atoms):
            while s < e and text[s].isspace():
                s += 1
            while e > s and text[e - 1].isspace():
                e -= 1
            if e > s:
                pieces.append(Piece(s, e))
        return pieces


__all__ = ["DEFAULT_SEPARATORS", "RecursiveChunker"]
