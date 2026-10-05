# path: book/projects/aie_core/aie_core/embeddings.py
"""Embedding clients and the small amount of vector math every chapter needs.

Chapter 8 owns the theory. Here: a protocol, an OpenAI-compatible HTTP client, two fakes
(hashing for determinism, bag-of-words for controllable similarity), a cache wrapper, and
cosine/normalize/top_k on NumPy.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import MutableMapping
from typing import Any, Protocol, Sequence, runtime_checkable

import httpx
import numpy as np

from .llm.errors import MalformedResponseError
from .llm.providers._http import (
    make_sync_client,
    map_transport_error,
    parse_json_body,
    raise_for_status,
    timeout_for,
)

Vector = list[float]


@runtime_checkable
class EmbeddingClient(Protocol):
    model: str
    dimensions: int

    def embed(self, texts: list[str]) -> list[Vector]: ...

    def embed_query(self, text: str) -> Vector: ...


# ------------------------------------------------------------------------------- math
def normalize(v: Sequence[float]) -> Vector:
    arr = np.asarray(v, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr.tolist()
    return (arr / norm).tolist()


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom == 0.0:
        return 0.0
    return float(np.dot(x, y) / denom)


def top_k(query: Sequence[float], matrix: Sequence[Sequence[float]] | np.ndarray, k: int) -> list[tuple[int, float]]:
    """Brute-force cosine search. Returns (row_index, similarity) sorted descending."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] == 0:
        return []
    q = np.asarray(query, dtype=np.float64)
    q_norm = np.linalg.norm(q)
    row_norms = np.linalg.norm(m, axis=1)
    denom = row_norms * q_norm
    with np.errstate(divide="ignore", invalid="ignore"):
        sims = np.where(denom > 0, (m @ q) / denom, 0.0)
    k = max(0, min(k, m.shape[0]))
    if k == 0:
        return []
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx], kind="stable")]
    return [(int(i), float(sims[i])) for i in idx]


# ------------------------------------------------------------------------------- HTTP client
class OpenAICompatibleEmbeddings:
    provider: str = "openai"

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        dimensions: int | None = None,
        timeout_s: float = 60.0,
        batch_size: int = 128,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.dimensions = dimensions or 0  # learned from the first response when not given
        self.timeout_s = timeout_s
        self.batch_size = batch_size
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = make_sync_client(base_url, headers, timeout_s, transport)

    def embed(self, texts: list[str]) -> list[Vector]:
        out: list[Vector] = []
        for start in range(0, len(texts), self.batch_size):
            out.extend(self._embed_batch(texts[start : start + self.batch_size]))
        return out

    def _embed_batch(self, texts: list[str]) -> list[Vector]:
        if not texts:
            return []
        payload: dict[str, Any] = {"model": self.model, "input": texts}
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        try:
            resp = self._client.post("/embeddings", json=payload, timeout=timeout_for(None, self.timeout_s))
        except httpx.HTTPError as exc:
            raise map_transport_error(exc, self.provider) from exc
        raise_for_status(resp, self.provider)
        data = parse_json_body(resp, self.provider)
        items = data.get("data")
        if not isinstance(items, list) or len(items) != len(texts):
            got = len(items) if isinstance(items, list) else "non-list"
            raise MalformedResponseError(
                f"embeddings response size mismatch: sent {len(texts)} inputs, got {got} vectors",
                provider=self.provider, raw=data,
            )
        indices = sorted(int(item.get("index", -1)) for item in items)
        if indices != list(range(len(texts))):
            raise MalformedResponseError(
                f"embeddings response indices are not exactly 0..{len(texts) - 1}: {indices}", provider=self.provider, raw=data
            )
        items = sorted(items, key=lambda d: int(d["index"]))
        vectors = [[float(x) for x in item["embedding"]] for item in items]
        if vectors and not self.dimensions:
            self.dimensions = len(vectors[0])
        return vectors

    def embed_query(self, text: str) -> Vector:
        return self.embed([text])[0]


# ------------------------------------------------------------------------------- fakes
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class FakeEmbeddings:
    """Deterministic, offline.

    Hashing mode (default): each token is hashed to a dimension with a sign, so equal
    strings get equal vectors but similar meanings do *not* get similar vectors.
    Vocabulary mode: `FakeEmbeddings(vocabulary=[...])` embeds bag-of-words counts over
    the given vocabulary, so texts sharing words are close. Use it when a test needs
    controllable similarity.
    """

    provider: str = "fake"

    def __init__(self, dimensions: int = 64, *, vocabulary: Sequence[str] | None = None, model: str = "fake-embedding") -> None:
        self.model = model
        self.vocabulary = [v.lower() for v in vocabulary] if vocabulary else None
        self._index = {w: i for i, w in enumerate(self.vocabulary)} if self.vocabulary else None
        self.dimensions = len(self.vocabulary) if self.vocabulary else dimensions
        self.calls: list[list[str]] = []

    def _embed_one(self, text: str) -> Vector:
        vec = np.zeros(self.dimensions, dtype=np.float64)
        tokens = _tokenize(text)
        if self._index is not None:
            for tok in tokens:
                i = self._index.get(tok)
                if i is not None:
                    vec[i] += 1.0
        else:
            for tok in tokens:
                digest = hashlib.sha256(tok.encode("utf-8")).digest()
                slot = int.from_bytes(digest[:4], "big") % self.dimensions
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[slot] += sign
        return normalize(vec)

    def embed(self, texts: list[str]) -> list[Vector]:
        self.calls.append(list(texts))
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> Vector:
        return self.embed([text])[0]


# ------------------------------------------------------------------------------- cache
class CachedEmbeddings:
    """Caches vectors by (embedding space, text).

    Two vectors are comparable only if they come from the same *space*: same provider, same
    model, same output dimensionality, same instruction or prefix prepended to the text, and
    the same text preparation. All of that goes into `space_fingerprint`, which is part of
    every key, so changing any of it invalidates the cache instead of mixing spaces. The
    fingerprint is frozen at construction; a client whose `dimensions` is learned lazily
    (0 at construction) contributes "unknown" rather than changing the key mid-life.
    `store` is any mutable mapping str->bytes (dict, Redis-like).
    """

    provider: str = "cached"

    def __init__(
        self,
        inner: EmbeddingClient,
        store: MutableMapping[str, bytes] | None = None,
        *,
        instruction: str | None = None,
        text_prep_version: str = "v1",
    ) -> None:
        self.inner = inner
        self.model = inner.model
        self.instruction = instruction
        self.text_prep_version = text_prep_version
        self.store: MutableMapping[str, bytes] = store if store is not None else {}
        self.hits = 0
        self.misses = 0
        self._fingerprint = self._compute_fingerprint()

    @property
    def dimensions(self) -> int:
        return self.inner.dimensions

    @property
    def innermost(self) -> EmbeddingClient:
        """The real provider client under any number of wrappers (each exposing `.inner`)."""
        client: Any = self.inner
        seen = 0
        while hasattr(client, "inner") and seen < 32:
            client = client.inner
            seen += 1
        return client

    def _compute_fingerprint(self) -> str:
        real = self.innermost
        dims = getattr(real, "dimensions", 0) or None
        material = {
            "provider": getattr(real, "provider", type(real).__name__),
            "model": real.model,
            "dimensions": dims if dims else "unknown",
            "instruction": self.instruction,
            "text_prep_version": self.text_prep_version,
        }
        blob = json.dumps(material, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    @property
    def space_fingerprint(self) -> str:
        """Stable identifier of the embedding space these cached vectors belong to."""
        return self._fingerprint

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self._fingerprint}\x00{text}".encode("utf-8")).hexdigest()

    def embed(self, texts: list[str]) -> list[Vector]:
        results: list[Vector | None] = [None] * len(texts)
        missing: list[int] = []
        for i, t in enumerate(texts):
            raw = self.store.get(self._key(t))
            if raw is not None:
                results[i] = json.loads(raw)
                self.hits += 1
            else:
                missing.append(i)
                self.misses += 1
        if missing:
            prepared = [self._prepare(texts[i]) for i in missing]
            fresh = self.inner.embed(prepared)
            if len(fresh) != len(missing):
                raise MalformedResponseError(
                    f"embedding client returned {len(fresh)} vectors for {len(missing)} inputs",
                    provider=getattr(self.inner, "provider", None),
                )
            for i, vec in zip(missing, fresh):
                results[i] = vec
                self.store[self._key(texts[i])] = json.dumps(vec).encode("utf-8")
        out = [r for r in results if r is not None]
        if len(out) != len(texts):  # defensive: a vector was lost between cache and provider
            raise MalformedResponseError(f"expected {len(texts)} vectors, assembled {len(out)}", provider="cached")
        return out

    def _prepare(self, text: str) -> str:
        return f"{self.instruction}{text}" if self.instruction else text

    def embed_query(self, text: str) -> Vector:
        return self.embed([text])[0]


__all__ = [
    "EmbeddingClient",
    "OpenAICompatibleEmbeddings",
    "FakeEmbeddings",
    "CachedEmbeddings",
    "cosine_similarity",
    "normalize",
    "top_k",
]
