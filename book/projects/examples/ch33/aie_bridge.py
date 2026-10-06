# path: book/projects/examples/ch33/aie_bridge.py
"""Where the Chapter 33 pipeline plugs into ``aie_core``.

The dataset builder and the evaluation protocol are deliberately free of model clients: one
takes an embedding *function*, the other takes *predictions*. This module supplies both from the
shared library, so a real run uses the same clients, gateway, pricing, and tracing as the rest of
the book:

- ``embed_fn_from_client`` turns any ``aie_core`` ``EmbeddingClient`` (hosted, ``FakeEmbeddings``,
  or ``CachedEmbeddings`` around either) into the ``EmbedFn`` that ``dedupe_near`` expects, and
  reports the embedding space it used so the data card can record it.
- ``run_classifier`` runs a classifier (a prompted baseline or the fine-tuned model id) over the
  frozen holdout through any ``LLMClient``, normally a ``ModelGateway``, and returns the
  ``SystemRun`` that ``evaluate`` and ``cascade`` consume. Cost comes from the gateway's pricing
  table, latency from the completion, and a label outside the taxonomy becomes an invalid
  prediction with zero confidence, so a cascade escalates it instead of routing it.
"""
from __future__ import annotations

import hashlib
from typing import Callable

import numpy as np

from aie_core import CompletionRequest, LLMClient, LLMError, Message
from aie_core.embeddings import EmbeddingClient
from aie_core.llm.types import Completion

from dataset_builder import EmbedFn
from eval_protocol import HoldoutExample, Prediction, SystemRun

INVALID_LABEL = "__invalid__"

ConfidenceFn = Callable[[Completion, str], float]


def embed_fn_from_client(client: EmbeddingClient) -> EmbedFn:
    """Adapt an ``aie_core`` embedding client to ``dedupe_near``'s ``EmbedFn`` (texts -> matrix).

    Wrap a hosted client in ``CachedEmbeddings`` so a rebuild does not re-embed unchanged rows;
    the cache keys on the embedding space, so changing model or instruction re-embeds instead of
    mixing spaces.
    """

    def embed(texts: list[str]) -> np.ndarray:
        vectors = client.embed(list(texts))
        return np.asarray(vectors, dtype=np.float32)

    return embed


def embedding_space(client: EmbeddingClient) -> dict[str, str]:
    """What the data card should record about the near-dedup embedder."""
    fingerprint = getattr(client, "space_fingerprint", None)
    return {
        "model": client.model,
        "provider": str(getattr(client, "provider", type(client).__name__)),
        "space_fingerprint": fingerprint or "uncached",
    }


def prompt_sha256(system_prompt: str) -> str:
    """Same fingerprint the data card stores as ``system_prompt_sha256``."""
    return hashlib.sha256(system_prompt.encode()).hexdigest()[:16]


def _no_confidence(completion: Completion, label: str) -> float:
    """Chat APIs return text, not class probabilities. Without a real confidence source every
    valid label gets 1.0, which makes a cascade escalate only invalid outputs."""
    return 1.0


def logprob_confidence(completion: Completion, label: str) -> float:
    """Probability of the emitted label from token log-probabilities, when the server returned them.

    Request them with ``OpenAICompatibleClient(extra_body={"logprobs": True})`` on an engine that
    supports it. The label's probability is the product of its tokens' probabilities, that is
    ``exp(sum of logprobs)``. Returns 1.0 when the response carries no log-probabilities, so the
    caller can tell from the ECE check that confidence is missing rather than silently wrong.
    """
    import math

    choices = (completion.raw or {}).get("choices") or []
    content = ((choices[0].get("logprobs") or {}).get("content") or []) if choices else []
    if not content:
        return 1.0
    total = sum(float(tok.get("logprob", 0.0)) for tok in content)
    return float(min(1.0, math.exp(total)))


def run_classifier(
    llm: LLMClient,
    holdout: list[HoldoutExample],
    *,
    name: str,
    model: str,
    system_prompt: str,
    labels: list[str],
    user_template: str = "{text}",
    expected_prompt_sha256: str | None = None,
    max_tokens: int = 16,
    confidence_fn: ConfidenceFn = _no_confidence,
    expected_provider: str | None = None,
) -> SystemRun:
    """Classify every holdout row with ``model`` and return predictions for the protocol.

    ``system_prompt`` and ``user_template`` must be the ones the training file was rendered with;
    pass the data card's ``system_prompt_sha256`` as ``expected_prompt_sha256`` and a mismatch
    raises before any call is made. ``confidence_fn(completion, label)`` supplies the probability
    of the predicted label, for example from token log-probabilities that a self-hosted engine
    returns in ``completion.raw``; whatever the source, the ECE check in the ship rule decides
    whether the cascade may trust it.
    """
    if expected_prompt_sha256 is not None and prompt_sha256(system_prompt) != expected_prompt_sha256:
        raise ValueError(
            "system prompt differs from the one the model was trained with; "
            "re-evaluate the model under the new prompt instead of serving it silently"
        )
    allowed = set(labels)
    predictions: list[Prediction] = []
    for row in holdout:
        req = CompletionRequest(
            messages=[Message.system(system_prompt), Message.user(user_template.format(text=row.text))],
            model=model,
            temperature=0.0,
            max_tokens=max_tokens,
            metadata={"eval.system": name, "eval.example_id": row.id},
        )
        try:
            completion = llm.complete(req)
        except LLMError:
            # A failed call is a wrong answer for the protocol, not a crash of the evaluation.
            predictions.append(Prediction(example_id=row.id, label=INVALID_LABEL, confidence=0.0))
            continue
        label = completion.text.strip()
        # With a fallback-capable gateway, pass expected_provider: an answer served by the fallback
        # provider says nothing about the model under test, so it counts as invalid.
        served_by_expected = expected_provider is None or completion.provider == expected_provider
        valid = label in allowed and served_by_expected
        predictions.append(
            Prediction(
                example_id=row.id,
                label=label if valid else INVALID_LABEL,
                confidence=confidence_fn(completion, label) if valid else 0.0,
                latency_ms=completion.latency_ms,
                # a cache hit is billed 0 but still costs what the model charges: compare full prices
                cost_usd=float((completion.raw or {}).get("cost_usd", 0.0))
                + float((completion.raw or {}).get("avoided_cost_usd", 0.0)),
            )
        )
    return SystemRun(name=name, predictions=predictions)


__all__ = ["INVALID_LABEL", "embed_fn_from_client", "embedding_space", "logprob_confidence", "prompt_sha256", "run_classifier"]
