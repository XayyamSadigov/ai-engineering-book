# path: book/projects/ragkit/ragkit/chunking/semantic.py
"""Semantic chunking: cut where consecutive sentences stop being about the same thing.

Each sentence (optionally with `window` neighbors on each side, to smooth noise) is embedded.
The cosine distance between consecutive embeddings is a topic-shift signal. Cuts go where the
distance exceeds a threshold: either an absolute value or a percentile of the document's own
distances (adaptive, so every document gets roughly the same share of cuts). Size limits are
then enforced: groups over `max_tokens` are packed by sentences, groups under `min_tokens` are
merged into a neighbor.

Cost: one embedding per sentence at ingestion time, typically several times the embedding
volume of the final chunks. The boundaries depend on the embedding model, so the model id is
part of the fingerprint; switching models re-chunks (and re-ids) the corpus.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from aie_core.embeddings import EmbeddingClient

from ..documents import Document
from ..tokenizers import Span, Tokenizer
from .base import BaseChunker, Piece
from .sentences import pack_spans, split_sentences


class SemanticChunker(BaseChunker):
    name = "semantic"

    def __init__(
        self,
        embeddings: EmbeddingClient,
        *,
        breakpoint_percentile: float = 90.0,
        threshold: float | None = None,
        window: int = 1,
        max_tokens: int = 400,
        min_tokens: int = 30,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        super().__init__(tokenizer)
        if not 0 < breakpoint_percentile < 100:
            raise ValueError("breakpoint_percentile must be in (0, 100)")
        if min_tokens >= max_tokens:
            raise ValueError("min_tokens must be below max_tokens")
        self.embeddings = embeddings
        self.breakpoint_percentile = breakpoint_percentile
        self.threshold = threshold
        self.window = window
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.last_distances: list[float] = []  # exposed for debugging and plots

    def config(self) -> dict[str, Any]:
        return {
            "embedding_model": getattr(self.embeddings, "model", "unknown"),
            "dimensions": getattr(self.embeddings, "dimensions", 0),
            "breakpoint_percentile": self.breakpoint_percentile,
            "threshold": self.threshold,
            "window": self.window,
            "max_tokens": self.max_tokens,
            "min_tokens": self.min_tokens,
        }

    def distances(self, text: str, sentences: list[Span]) -> list[float]:
        windows = []
        for i in range(len(sentences)):
            lo, hi = max(0, i - self.window), min(len(sentences), i + self.window + 1)
            windows.append(text[sentences[lo][0] : sentences[hi - 1][1]])
        vectors = np.asarray(self.embeddings.embed(windows), dtype=np.float64)
        norms = np.linalg.norm(vectors, axis=1)
        norms[norms == 0] = 1.0
        unit = vectors / norms[:, None]
        sims = np.sum(unit[:-1] * unit[1:], axis=1)
        return [float(1.0 - s) for s in sims]

    def split(self, doc: Document) -> list[Piece]:
        text = doc.text
        sentences = split_sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return [Piece(s, e) for s, e in pack_spans(text, sentences, self.max_tokens, self.tokenizer)]
        dist = self.distances(text, sentences)
        self.last_distances = dist
        cut_at = self.threshold if self.threshold is not None else float(np.percentile(dist, self.breakpoint_percentile))
        groups: list[list[Span]] = [[sentences[0]]]
        for i, d in enumerate(dist):
            if d > cut_at:
                groups.append([])
            groups[-1].append(sentences[i + 1])

        # enforce the upper bound
        sized: list[list[Span]] = []
        for g in groups:
            g_tokens = self.count(text[g[0][0] : g[-1][1]])
            if g_tokens <= self.max_tokens:
                sized.append(g)
            else:
                sized.extend([[span] for span in pack_spans(text, g, self.max_tokens, self.tokenizer)])
        # enforce the lower bound by merging small groups into the previous one when it fits
        merged: list[list[Span]] = []
        for g in sized:
            g_tokens = self.count(text[g[0][0] : g[-1][1]])
            if merged and g_tokens < self.min_tokens:
                prev = merged[-1]
                if self.count(text[prev[0][0] : g[-1][1]]) <= self.max_tokens:
                    prev.extend(g)
                    continue
            merged.append(list(g))
        return [Piece(g[0][0], g[-1][1], meta={"cut_threshold": round(cut_at, 4)}) for g in merged]


__all__ = ["SemanticChunker"]
