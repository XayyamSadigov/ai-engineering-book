# path: book/projects/ragkit/ragkit/chunking/sentence.py
"""Sentence packing that prefers paragraph boundaries."""
from __future__ import annotations

from typing import Any

from ..documents import Document
from ..tokenizers import Span, Tokenizer
from .base import BaseChunker, Piece
from .sentences import split_sentences, token_windows


class SentenceChunker(BaseChunker):
    """Never cuts inside a sentence (unless one sentence alone exceeds `max_tokens`).

    Paragraphs that fit are kept whole as packing units, so chunk boundaries fall between
    paragraphs whenever possible. Overlap is a number of trailing sentences repeated at the start
    of the next chunk; it is dropped when it would push the next chunk over budget.
    """

    name = "sentence"

    def __init__(self, max_tokens: int = 200, overlap_sentences: int = 1, *, tokenizer: Tokenizer | None = None) -> None:
        super().__init__(tokenizer)
        if max_tokens <= 0 or overlap_sentences < 0:
            raise ValueError("need max_tokens > 0 and overlap_sentences >= 0")
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences

    def config(self) -> dict[str, Any]:
        return {"max_tokens": self.max_tokens, "overlap_sentences": self.overlap_sentences}

    def units(self, text: str) -> list[list[tuple[Span, int]]]:
        """Packing units: whole paragraphs when they fit, otherwise single sentences."""
        sentences: list[tuple[Span, int]] = []
        starts_paragraph: list[bool] = []
        prev_end = 0
        for s, e in split_sentences(text):
            for ws, we in token_windows(text, s, e, self.max_tokens, self.tokenizer):
                sentences.append(((ws, we), self.count(text[ws:we])))
                starts_paragraph.append(prev_end == 0 or "\n\n" in text[prev_end:ws])
                prev_end = we
        paragraphs: list[list[tuple[Span, int]]] = []
        for sent, starts in zip(sentences, starts_paragraph):
            if starts or not paragraphs:
                paragraphs.append([])
            paragraphs[-1].append(sent)
        units: list[list[tuple[Span, int]]] = []
        for para in paragraphs:
            if sum(t for _, t in para) <= self.max_tokens:
                units.append(para)
            else:
                units.extend([s] for s in para)
        return units

    def split(self, doc: Document) -> list[Piece]:
        text = doc.text
        pieces: list[Piece] = []
        cur: list[tuple[Span, int]] = []
        cur_tokens = 0

        def emit() -> None:
            pieces.append(Piece(cur[0][0][0], cur[-1][0][1], meta={"sentences": len(cur)}))

        for unit in self.units(text):
            unit_tokens = sum(t for _, t in unit)
            if cur and cur_tokens + unit_tokens > self.max_tokens:
                emit()
                tail = cur[-self.overlap_sentences :] if self.overlap_sentences else []
                if sum(t for _, t in tail) + unit_tokens > self.max_tokens:
                    tail = []
                cur = list(tail)
                cur_tokens = sum(t for _, t in cur)
            cur.extend(unit)
            cur_tokens += unit_tokens
        if cur:
            emit()
        return pieces


__all__ = ["SentenceChunker"]
