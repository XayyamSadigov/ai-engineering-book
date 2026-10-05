# path: book/projects/examples/ch04/prompts/examples.py
"""Dynamic few-shot selection: pick examples that resemble the input, cover several labels,
and fit a token budget. Static examples in the prompt file are the default; reach for this
only when an evaluation shows that input-specific examples beat a fixed set."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from aie_core.embeddings import EmbeddingClient, cosine_similarity
from aie_core.llm.tokens import count_tokens


class FewShotExample(BaseModel):
    input: str
    output: str
    label: str  # the decision this example demonstrates, used for label diversity


def select_examples(
    pool: Sequence[FewShotExample],
    query: str,
    embedder: EmbeddingClient,
    *,
    k: int = 3,
    max_per_label: int = 1,
    token_budget: int = 600,
    exclude_inputs: set[str] | None = None,
) -> list[FewShotExample]:
    """Greedy: rank by similarity, skip labels already shown `max_per_label` times, stop at
    `k` examples or when the next one would exceed `token_budget`.

    `exclude_inputs` keeps evaluation inputs out of the prompt, the leak that makes a
    regression suite measure memorization instead of behavior.
    """
    exclude = exclude_inputs or set()
    candidates = [ex for ex in pool if ex.input not in exclude]
    if not candidates or k <= 0:
        return []
    query_vec = embedder.embed_query(query)
    vectors = embedder.embed([ex.input for ex in candidates])
    ranked = sorted(
        zip(candidates, vectors),
        key=lambda pair: cosine_similarity(query_vec, pair[1]),
        reverse=True,
    )
    chosen: list[FewShotExample] = []
    per_label: dict[str, int] = {}
    spent = 0
    for ex, _vec in ranked:
        if per_label.get(ex.label, 0) >= max_per_label:
            continue
        cost = count_tokens(ex.input) + count_tokens(ex.output)
        if spent + cost > token_budget:
            continue
        chosen.append(ex)
        per_label[ex.label] = per_label.get(ex.label, 0) + 1
        spent += cost
        if len(chosen) == k:
            break
    return chosen


def as_variables(examples: Sequence[FewShotExample]) -> list[dict[str, Any]]:
    """Shape for a template loop: `{% for ex in examples %}...{% endfor %}`."""
    return [{"input": ex.input, "output": ex.output} for ex in examples]


__all__ = ["FewShotExample", "select_examples", "as_variables"]
