# path: book/projects/examples/ch08/embedlab/usecases/routing.py
"""Intent routing by similarity to example utterances, with an explicit fallback.

The router answers "which handler?" in one embedding call. When it is unsure (best score too
low, or two routes too close), it returns no route and the caller falls back, typically to an
LLM classifier (Chapter 7) or a clarifying question."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from pydantic import BaseModel, Field

from ..pipeline import EmbeddingPipeline
from ..vector_math import l2_normalize_rows


class Route(BaseModel):
    name: str
    description: str = ""
    utterances: list[str] = Field(min_length=1)


@dataclass(frozen=True)
class RouteDecision:
    route: str | None
    score: float
    margin: float
    reason: Literal["matched", "below_threshold", "ambiguous"]


class IntentRouter:
    def __init__(self, pipeline: EmbeddingPipeline, routes: Sequence[Route], threshold: float = 0.3, min_margin: float = 0.05) -> None:
        self.pipeline = pipeline
        self.routes = list(routes)
        self.threshold = threshold
        self.min_margin = min_margin
        texts = [u for r in self.routes for u in r.utterances]
        self.owner = np.asarray([n for n, r in enumerate(self.routes) for _ in r.utterances])
        # Route utterances play the passage role; incoming messages are queries.
        self.matrix = l2_normalize_rows(pipeline.embed_passages(texts))

    def scores(self, text: str) -> np.ndarray:
        """Best exemplar similarity per route (max, not mean: one close example is enough)."""
        sims = self.matrix @ l2_normalize_rows(self.pipeline.embed_query(text))[0]
        per_route = np.full(len(self.routes), -1.0)
        np.maximum.at(per_route, self.owner, sims)
        return per_route

    def route(self, text: str) -> RouteDecision:
        s = self.scores(text)
        order = np.argsort(-s, kind="stable")
        best = float(s[order[0]])
        margin = best - float(s[order[1]]) if len(order) > 1 else best
        if best < self.threshold:
            return RouteDecision(None, best, margin, "below_threshold")
        if margin < self.min_margin:
            return RouteDecision(None, best, margin, "ambiguous")
        return RouteDecision(self.routes[order[0]].name, best, margin, "matched")

    def calibrate(self, labeled: Sequence[tuple[str, str | None]], thresholds: Sequence[float]) -> list[dict[str, float]]:
        """For each candidate threshold: accuracy (None is the right answer for out-of-scope),
        wrong-route rate (the expensive error), and fallback rate (the cheap one)."""
        all_scores = [self.scores(text) for text, _ in labeled]
        rows = []
        for t in thresholds:
            correct = wrong = fallback = 0
            for s, (_, gold) in zip(all_scores, labeled):
                order = np.argsort(-s, kind="stable")
                best = float(s[order[0]])
                margin = best - float(s[order[1]]) if len(order) > 1 else best
                pred = self.routes[order[0]].name if best >= t and margin >= self.min_margin else None
                if pred is None:
                    fallback += 1
                if pred == gold:
                    correct += 1
                elif pred is not None:
                    wrong += 1
            n = len(labeled)
            rows.append({"threshold": t, "accuracy": correct / n, "wrong_route": wrong / n, "fallback": fallback / n})
        return rows


__all__ = ["IntentRouter", "Route", "RouteDecision"]
