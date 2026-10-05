# path: book/projects/ragkit/ragkit/chunking/fixed.py
"""Fixed-size token windows with overlap: the baseline every other strategy must beat."""
from __future__ import annotations

from typing import Any

from ..documents import Document
from ..tokenizers import Tokenizer
from .base import BaseChunker, Piece


class FixedTokenChunker(BaseChunker):
    """Windows of `chunk_size` tokens; consecutive windows share exactly `overlap` tokens.

    Ignores structure entirely: windows cut through sentences, tables, and code. Its virtues are
    predictability (every chunk fits the embedding model's input limit) and speed.
    """

    name = "fixed"

    def __init__(self, chunk_size: int = 256, overlap: int = 32, *, tokenizer: Tokenizer | None = None) -> None:
        super().__init__(tokenizer)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must be in [0, chunk_size)")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def config(self) -> dict[str, Any]:
        return {"chunk_size": self.chunk_size, "overlap": self.overlap}

    def split(self, doc: Document) -> list[Piece]:
        spans = self.tokenizer.spans(doc.text)
        if not spans:
            return []
        step = self.chunk_size - self.overlap
        pieces: list[Piece] = []
        i = 0
        while True:
            window = spans[i : i + self.chunk_size]
            pieces.append(Piece(window[0][0], window[-1][1], meta={"token_offset": i}))
            if i + self.chunk_size >= len(spans):
                break
            i += step
        return pieces


__all__ = ["FixedTokenChunker"]
