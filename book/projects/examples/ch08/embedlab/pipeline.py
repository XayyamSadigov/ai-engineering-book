# path: book/projects/examples/ch08/embedlab/pipeline.py
"""EmbeddingPipeline: text preparation, query/passage prefixes, token-aware batching,
a cache keyed by the full embedding space, cost metering, and tracing.

    texts -> prepare -> prefix -> cache lookup -> batch misses -> client.embed -> cache write
"""
from __future__ import annotations

import random
import time
import unicodedata
from collections.abc import Callable, Iterator, MutableMapping
from dataclasses import dataclass

import numpy as np

from aie_core.embeddings import CachedEmbeddings, EmbeddingClient
from aie_core.llm.errors import LLMError
from aie_core.llm.gateway import RetryPolicy
from aie_core.llm.tokens import count_tokens
from aie_core.observability import NoopTracer, Tracer

from .space import EmbeddingSpace

TEXT_PREP_VERSION = "v1"


def prepare_text(text: str, max_chars: int = 8000) -> str:
    """Deterministic normalization. Any change here changes vectors, so bump TEXT_PREP_VERSION."""
    t = unicodedata.normalize("NFC", text)
    t = " ".join(t.split())
    return t[:max_chars]


class NamespacedStore(MutableMapping[str, bytes]):
    """Prefixes every cache key with this chapter's EmbeddingSpace fingerprint.

    aie_core's CachedEmbeddings already salts its keys with a fingerprint of what it can see
    (inner client, model, dimensions, instruction, text-prep version). It cannot see the
    normalization flag or post-processing recorded in EmbeddingSpace, and a client whose
    dimensions are learned lazily contributes "unknown". Namespacing by the full space makes a
    stale hit impossible after any field of the space changes, and lets one shared store
    (dict, Redis) be listed and purged per space."""

    def __init__(self, inner: MutableMapping[str, bytes], namespace: str) -> None:
        self.inner = inner
        self.prefix = f"emb:{namespace}:"

    def __getitem__(self, key: str) -> bytes:
        return self.inner[self.prefix + key]

    def __setitem__(self, key: str, value: bytes) -> None:
        self.inner[self.prefix + key] = value

    def __delitem__(self, key: str) -> None:
        del self.inner[self.prefix + key]

    def __iter__(self) -> Iterator[str]:
        return (k[len(self.prefix):] for k in self.inner if k.startswith(self.prefix))

    def __len__(self) -> int:
        return sum(1 for _ in self)


@dataclass
class EmbeddingStats:
    requests: int = 0
    texts_sent: int = 0
    tokens_sent: int = 0
    cost_usd: float = 0.0
    retries: int = 0


class BatchingEmbeddings:
    """An EmbeddingClient that splits work into batches bounded by count and by tokens,
    retries transient provider errors per batch, and meters what is actually sent.

    Embedding calls have no side effects, so retrying a timed-out batch is safe. The policy is
    aie_core's RetryPolicy (ModelGateway covers chat completions, not embeddings)."""

    def __init__(
        self,
        inner: EmbeddingClient,
        *,
        max_batch_texts: int = 64,
        max_batch_tokens: int = 8000,
        price_per_million_tokens: float = 0.0,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.inner = inner
        self.model = inner.model
        self.max_batch_texts = max_batch_texts
        self.max_batch_tokens = max_batch_tokens
        self.price = price_per_million_tokens
        self.retry = retry or RetryPolicy()
        self._sleep = sleep
        self._rng = random.Random()
        self.stats = EmbeddingStats()

    @property
    def dimensions(self) -> int:
        return self.inner.dimensions

    def batches(self, texts: list[str]) -> Iterator[list[int]]:
        batch: list[int] = []
        tokens = 0
        for i, t in enumerate(texts):
            n = count_tokens(t)
            if batch and (len(batch) >= self.max_batch_texts or tokens + n > self.max_batch_tokens):
                yield batch
                batch, tokens = [], 0
            batch.append(i)
            tokens += n
        if batch:
            yield batch

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = [[] for _ in texts]
        for idx in self.batches(texts):
            chunk = [texts[i] for i in idx]
            vectors = self._embed_with_retry(chunk)
            n_tokens = sum(count_tokens(t) for t in chunk)
            self.stats.requests += 1
            self.stats.texts_sent += len(chunk)
            self.stats.tokens_sent += n_tokens
            self.stats.cost_usd += n_tokens / 1_000_000 * self.price
            for i, v in zip(idx, vectors):
                out[i] = v
        return out

    def _embed_with_retry(self, chunk: list[str]) -> list[list[float]]:
        attempt = 1
        while True:
            try:
                return self.inner.embed(chunk)
            except LLMError as err:
                if not self.retry.should_retry(err, attempt):
                    raise  # non-retryable (bad input) or attempts exhausted: caller degrades
                self.stats.retries += 1
                self._sleep(self.retry.delay_for(attempt, err.retry_after_s, self._rng))
                attempt += 1

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class EmbeddingPipeline:
    def __init__(
        self,
        client: EmbeddingClient,
        *,
        store: MutableMapping[str, bytes] | None = None,
        query_prefix: str = "",
        passage_prefix: str = "",
        max_batch_texts: int = 64,
        max_batch_tokens: int = 8000,
        price_per_million_tokens: float = 0.0,  # illustrative; set from your provider's price sheet
        tracer: Tracer | None = None,
        retry: RetryPolicy | None = None,
    ) -> None:
        self.client = client
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.tracer = tracer or NoopTracer()
        self.batcher = BatchingEmbeddings(
            client,
            max_batch_texts=max_batch_texts,
            max_batch_tokens=max_batch_tokens,
            price_per_million_tokens=price_per_million_tokens,
            retry=retry,
        )
        self.store: MutableMapping[str, bytes] = store if store is not None else {}
        self._cached: CachedEmbeddings | None = None
        self._cached_for: str | None = None

    @property
    def space(self) -> EmbeddingSpace:
        dims = self.client.dimensions
        if not dims:  # OpenAI-compatible clients learn dimensions from the first response
            self.client.embed(["dimension probe"])
            dims = self.client.dimensions
        return EmbeddingSpace(
            model=self.client.model,
            dimensions=dims,
            text_prep=TEXT_PREP_VERSION,
            query_prefix=self.query_prefix,
            passage_prefix=self.passage_prefix,
        )

    @property
    def stats(self) -> EmbeddingStats:
        return self.batcher.stats

    def _cache(self) -> CachedEmbeddings:
        fp = self.space.fingerprint
        if self._cached is None or self._cached_for != fp:
            self._cached = CachedEmbeddings(self.batcher, NamespacedStore(self.store, fp))
            self._cached_for = fp
        return self._cached

    def _embed(self, texts: list[str], prefix: str, kind: str) -> np.ndarray:
        prepared = [prefix + prepare_text(t) for t in texts]
        unique = list(dict.fromkeys(prepared))  # identical inputs are embedded once per call
        cache = self._cache()
        with self.tracer.span(
            "embed", kind=kind, model=self.client.model, space=self._cached_for, texts=len(texts), unique=len(unique)
        ) as span:
            hits_before, misses_before, retries_before = cache.hits, cache.misses, self.stats.retries
            vectors = cache.embed(unique) if unique else []
            span.set_attribute("cache_hits", cache.hits - hits_before)
            span.set_attribute("cache_misses", cache.misses - misses_before)
            span.set_attribute("retries", self.stats.retries - retries_before)
            span.set_attribute("zero_vectors", sum(1 for v in vectors if not any(v)))
            span.set_attribute("tokens_sent_total", self.stats.tokens_sent)
        by_text = dict(zip(unique, vectors))
        dims = self.client.dimensions
        if not prepared:
            return np.zeros((0, dims))
        return np.asarray([by_text[p] for p in prepared], dtype=np.float64)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, self.passage_prefix, "passage")

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, self.query_prefix, "query")

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_queries([text])[0]


__all__ = [
    "BatchingEmbeddings",
    "EmbeddingPipeline",
    "EmbeddingStats",
    "NamespacedStore",
    "TEXT_PREP_VERSION",
    "prepare_text",
]
