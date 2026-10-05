# path: book/projects/examples/ch30/caching.py
"""Cache layers for a RAG assistant, each keyed by everything that changes the correct answer.

Layers, cheapest-to-get-right first:
- `EmbeddingCache`: text -> vector. Key: embedding model, normalization version, normalized text.
- `RetrievalCache`: query -> evidence ids. Key: normalized query, tenant, ACL scope, index version,
  retriever config.
- `ScopedResponseCache`: request -> completion. Key: `aie_core.cache_key` (model, messages, tools,
  schema, sampling) plus tenant, ACL scope, prompt version and index version.
- `SemanticCache`: similar question -> answer. Eligibility: same tenant, same ACL scope, same
  versions, unexpired; then cosine similarity above a threshold.
Plus `lint_cache_key`, which flags missing scope components (data leaks) and per-request
components (zero hit rate) in a key definition.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import unicodedata
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Iterable, Mapping, TypeVar

from aie_core.embeddings import EmbeddingClient, cosine_similarity
from aie_core.llm.client import LLMClient
from aie_core.llm.gateway import InMemoryResponseCache, ResponseCache, cache_key
from aie_core.llm.types import Completion, CompletionRequest

T = TypeVar("T")


# ----------------------------------------------------------------------------- keys
def make_key(namespace: str, components: Mapping[str, Any]) -> str:
    """Stable hash of named components. Sorted, so dict order never changes a key."""
    blob = json.dumps({"ns": namespace, **components}, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return f"{namespace}:{hashlib.sha256(blob.encode('utf-8')).hexdigest()}"


NORMALIZATION_VERSION = "nfkc-lower-ws-v1"


def normalize_text(text: str) -> str:
    """Unicode NFKC, lowercase, collapse whitespace. Changing this function means bumping the version."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().lower()


@dataclass(frozen=True)
class Scope:
    """Who is asking. The ACL scope is a hash of the sorted groups so keys do not leak group names."""

    tenant: str
    groups: frozenset[str]

    @classmethod
    def of(cls, tenant: str, groups: Iterable[str]) -> "Scope":
        return cls(tenant, frozenset(groups))

    @property
    def acl_scope(self) -> str:
        return hashlib.sha256("\x00".join(sorted(self.groups)).encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------------------- store
class TTLCache(Generic[T]):
    """Thread-safe LRU with per-entry TTL and hit/miss counters. Swap for Redis behind the same methods."""

    def __init__(self, max_size: int = 10_000, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.max_size = max_size
        self._clock = clock
        self._data: OrderedDict[str, tuple[T, float | None]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> T | None:
        with self._lock:
            item = self._data.get(key)
            if item is None or (item[1] is not None and self._clock() >= item[1]):
                self._data.pop(key, None)
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return item[0]

    def set(self, key: str, value: T, ttl_s: float | None = None) -> None:
        with self._lock:
            self._data[key] = (value, None if ttl_s is None else self._clock() + ttl_s)
            self._data.move_to_end(key)
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)

    def delete_prefix(self, prefix: str) -> int:
        with self._lock:
            doomed = [k for k in self._data if k.startswith(prefix)]
            for k in doomed:
                del self._data[k]
            return len(doomed)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def __len__(self) -> int:
        return len(self._data)


# ----------------------------------------------------------------------------- embedding
class EmbeddingCache:
    """An `EmbeddingClient` that caches by (embedding-space fingerprint, normalized text).

    A vector depends on the model version, the output dimensions, any instruction prefix, and how
    the text was prepared; change any of them and an old cached vector is silently wrong. Like
    `aie_core.CachedEmbeddings` and Chapter 8's `NamespacedStore`, the key carries a fingerprint of
    all of them (the "space"). Unlike `CachedEmbeddings`, the text is normalized first, which raises
    the hit rate on queries. Misses are embedded in one batched call.
    """

    key_components = frozenset({"text", "space"})

    def __init__(
        self,
        inner: EmbeddingClient,
        store: TTLCache[list[float]] | None = None,
        *,
        model_version: str = "",
        instruction_prefix: str = "",
    ) -> None:
        self.inner = inner
        self.model = inner.model
        self.model_version = model_version  # pin the provider's dated version when it exposes one
        self.instruction_prefix = instruction_prefix  # e.g. "query: " for models trained with prefixes
        self.store: TTLCache[list[float]] = store if store is not None else TTLCache()
        self.calls = 0

    @property
    def dimensions(self) -> int:
        return self.inner.dimensions

    @property
    def space(self) -> str:
        """Fingerprint of everything besides the text that determines the vector."""
        material = {
            "model": self.model,
            "model_version": self.model_version,
            "dimensions": self.dimensions,
            "instruction_prefix": self.instruction_prefix,
            "text_prep": NORMALIZATION_VERSION,
        }
        return hashlib.sha256(json.dumps(material, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def key(self, text: str) -> str:
        return make_key("emb", {"space": self.space, "text": normalize_text(text)})

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float] | None] = [None] * len(texts)
        missing: dict[str, list[int]] = {}
        for i, text in enumerate(texts):
            k = self.key(text)
            hit = self.store.get(k)
            if hit is not None:
                out[i] = hit
            else:
                missing.setdefault(k, []).append(i)  # duplicates inside one batch are embedded once
        if missing:
            firsts = [idx[0] for idx in missing.values()]
            self.calls += 1
            vectors = self.inner.embed([self.instruction_prefix + normalize_text(texts[i]) for i in firsts])
            for (k, idxs), vec in zip(missing.items(), vectors):
                self.store.set(k, vec)  # embeddings do not go stale; the version in the key handles change
                for i in idxs:
                    out[i] = vec
        return [v for v in out if v is not None]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


# ----------------------------------------------------------------------------- retrieval
class RetrievalCache:
    """Caches the ranked evidence ids for a query within one tenant, ACL scope and index version.

    Caching ids, not text, means a document deleted or re-permissioned after caching is re-checked
    when the ids are hydrated. Bumping `index_version` on every reindex retires old entries.
    """

    key_components = frozenset({"query", "tenant", "acl_scope", "index_version", "retriever_config"})

    def __init__(self, store: TTLCache[list[str]] | None = None, *, ttl_s: float = 900.0) -> None:
        self.store: TTLCache[list[str]] = store if store is not None else TTLCache()
        self.ttl_s = ttl_s

    def key(self, query: str, scope: Scope, *, index_version: str, retriever_config: str) -> str:
        return make_key(
            f"ret:{scope.tenant}",  # tenant in the prefix enables per-tenant purge
            {
                "query": normalize_text(query),
                "tenant": scope.tenant,
                "acl_scope": scope.acl_scope,
                "index_version": index_version,
                "retriever_config": retriever_config,
            },
        )

    def get_or_compute(
        self,
        query: str,
        scope: Scope,
        compute: Callable[[], list[str]],
        *,
        index_version: str,
        retriever_config: str = "default",
    ) -> tuple[list[str], bool]:
        k = self.key(query, scope, index_version=index_version, retriever_config=retriever_config)
        cached = self.store.get(k)
        if cached is not None:
            return cached, True
        ids = compute()
        self.store.set(k, ids, self.ttl_s)
        return ids, False

    def purge_tenant(self, tenant: str) -> int:
        return self.store.delete_prefix(f"ret:{tenant}:")


# ----------------------------------------------------------------------------- response
class ScopedResponseCache:
    """Exact-match response cache that adds the caller's scope and versions to `aie_core.cache_key`.

    Use it in front of a gateway built without its own cache. The gateway's key already covers
    model, messages, tools, schema and sampling; this adds what the gateway cannot see.
    """

    key_components = frozenset({"request", "model", "tenant", "acl_scope", "prompt_version", "index_version"})

    def __init__(self, client: LLMClient, store: ResponseCache | None = None, *, ttl_s: float = 3600.0, max_temperature: float = 0.0) -> None:
        self.client = client
        self.store: ResponseCache = store if store is not None else InMemoryResponseCache()
        self.ttl_s = ttl_s
        self.max_temperature = max_temperature
        self.hits = 0
        self.misses = 0

    def key(self, req: CompletionRequest, scope: Scope, *, prompt_version: str, index_version: str) -> str:
        return make_key(
            "resp",
            {
                "request": cache_key(req, req.model),
                "model": req.model,
                "tenant": scope.tenant,
                "acl_scope": scope.acl_scope,
                "prompt_version": prompt_version,
                "index_version": index_version,
            },
        )

    def complete(self, req: CompletionRequest, scope: Scope, *, prompt_version: str, index_version: str = "none") -> tuple[Completion, bool]:
        if req.temperature > self.max_temperature or req.metadata.get("cache") is False:
            return self.client.complete(req), False
        k = self.key(req, scope, prompt_version=prompt_version, index_version=index_version)
        hit = self.store.get(k)
        if hit is not None:
            self.hits += 1
            return hit, True
        self.misses += 1
        completion = self.client.complete(req)
        if completion.finish_reason in ("stop", "end_turn", "tool_calls", "tool_use"):
            self.store.set(k, completion, self.ttl_s)  # never cache truncated or filtered output
        return completion, False


# ----------------------------------------------------------------------------- semantic
@dataclass
class SemanticEntry:
    entry_id: str
    question: str
    vector: list[float]
    answer: str
    tenant: str
    acl_scope: str
    versions: dict[str, str]
    source_ids: frozenset[str]
    expires_at: float


@dataclass(frozen=True)
class SemanticHit:
    answer: str
    similarity: float
    entry_id: str
    question: str


@dataclass
class SemanticStats:
    hits: int = 0
    misses: int = 0
    near_misses: int = 0  # best similarity just under the threshold: candidates for threshold review
    ineligible: int = 0  # a similar entry existed but scope, version or TTL ruled it out
    refused_writes: int = 0
    near_miss_log: list[tuple[str, str, float]] = field(default_factory=list)


class SemanticCache:
    """Reuse an answer for a sufficiently similar question, inside strict eligibility rules.

    Eligibility is checked before similarity: an entry from another tenant, ACL scope, prompt
    or index version, or past its TTL is never a candidate, however similar. Entries record the
    source document ids they cite so a document change can invalidate exactly the answers built
    on it.
    """

    key_components = frozenset({"query_embedding", "tenant", "acl_scope", "ttl", "version"})

    def __init__(
        self,
        embedder: EmbeddingClient,
        *,
        threshold: float = 0.92,
        near_miss_margin: float = 0.05,
        ttl_s: float = 3600.0,
        max_entries: int = 50_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        self.embedder = embedder
        self.threshold = threshold
        self.near_miss_margin = near_miss_margin
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self._clock = clock
        self._entries: list[SemanticEntry] = []
        self._lock = threading.Lock()
        self.stats = SemanticStats()

    def _eligible(self, e: SemanticEntry, scope: Scope, versions: Mapping[str, str], now: float) -> bool:
        return e.tenant == scope.tenant and e.acl_scope == scope.acl_scope and e.versions == dict(versions) and e.expires_at > now

    def lookup(self, question: str, scope: Scope, versions: Mapping[str, str]) -> SemanticHit | None:
        vec = self.embedder.embed_query(normalize_text(question))
        now = self._clock()
        best: tuple[float, SemanticEntry] | None = None
        best_ineligible = 0.0
        with self._lock:
            for e in self._entries:
                sim = cosine_similarity(vec, e.vector)
                if not self._eligible(e, scope, versions, now):
                    best_ineligible = max(best_ineligible, sim)
                    continue
                if best is None or sim > best[0]:
                    best = (sim, e)
        if best is not None and best[0] >= self.threshold:
            self.stats.hits += 1
            return SemanticHit(best[1].answer, best[0], best[1].entry_id, best[1].question)
        self.stats.misses += 1
        if best_ineligible >= self.threshold:
            self.stats.ineligible += 1
        if best is not None and best[0] >= self.threshold - self.near_miss_margin:
            self.stats.near_misses += 1
            self.stats.near_miss_log.append((question, best[1].question, round(best[0], 4)))
        return None

    def store(
        self,
        question: str,
        answer: str,
        scope: Scope,
        versions: Mapping[str, str],
        *,
        source_ids: Iterable[str] = (),
        personalized: bool = False,
        grounded: bool = True,
    ) -> str | None:
        """Store an answer. Refuses personalized or ungrounded answers: they are wrong for the next asker."""
        if personalized or not grounded:
            self.stats.refused_writes += 1
            return None
        entry = SemanticEntry(
            entry_id=uuid.uuid4().hex[:12],
            question=question,
            vector=self.embedder.embed_query(normalize_text(question)),
            answer=answer,
            tenant=scope.tenant,
            acl_scope=scope.acl_scope,
            versions=dict(versions),
            source_ids=frozenset(source_ids),
            expires_at=self._clock() + self.ttl_s,
        )
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self.max_entries:
                self._entries.pop(0)
        return entry.entry_id

    def invalidate_version(self, name: str, current: str) -> int:
        """Drop entries built under any other value of version `name` (e.g. a new index version)."""
        with self._lock:
            before = len(self._entries)
            self._entries = [e for e in self._entries if e.versions.get(name) == current]
            return before - len(self._entries)

    def invalidate_sources(self, source_ids: Iterable[str]) -> int:
        """Dependency invalidation: drop every answer that cited a changed or deleted document."""
        changed = set(source_ids)
        with self._lock:
            before = len(self._entries)
            self._entries = [e for e in self._entries if not (e.source_ids & changed)]
            return before - len(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


# ----------------------------------------------------------------------------- linter
REQUIRED_COMPONENTS: dict[str, frozenset[str]] = {
    "embedding": frozenset({"text", "space"}),  # space = model, version, dimensions, prefix, text prep (Ch 8)
    "retrieval": frozenset({"query", "tenant", "acl_scope", "index_version"}),
    "response": frozenset({"request", "model", "tenant", "acl_scope", "prompt_version"}),
    "semantic": frozenset({"tenant", "acl_scope", "ttl", "version"}),
    "tool": frozenset({"tool", "arguments", "tenant", "tool_version"}),
}
SCOPE_COMPONENTS = frozenset({"tenant", "acl_scope"})
PER_REQUEST_COMPONENTS = frozenset({"request_id", "timestamp", "trace_id", "session_nonce", "now"})


@dataclass(frozen=True)
class LintFinding:
    layer: str
    severity: str  # "error" leaks or serves stale data; "warning" kills the hit rate
    component: str
    message: str


def lint_cache_key(layer: str, components: Iterable[str]) -> list[LintFinding]:
    """Check a cache key definition against the components its layer needs."""
    if layer not in REQUIRED_COMPONENTS:
        raise ValueError(f"unknown cache layer {layer!r}; known: {sorted(REQUIRED_COMPONENTS)}")
    have = set(components)
    findings: list[LintFinding] = []
    for missing in sorted(REQUIRED_COMPONENTS[layer] - have):
        if missing in SCOPE_COMPONENTS:
            msg = f"{layer} key lacks {missing}: two callers with different access can share an entry"
        elif missing == "space":
            msg = f"{layer} key lacks the embedding-space fingerprint: a model, dimension, prefix or text-prep change serves stale vectors"
        elif missing in ("index_version", "prompt_version", "version", "tool_version", "model"):
            msg = f"{layer} key lacks {missing}: entries survive a change that alters the correct value"
        elif missing == "ttl":
            msg = f"{layer} entries have no TTL: answers over changing data never expire"
        else:
            msg = f"{layer} key lacks {missing}: different inputs can collide"
        findings.append(LintFinding(layer, "error", missing, msg))
    for noisy in sorted(have & PER_REQUEST_COMPONENTS):
        findings.append(LintFinding(layer, "warning", noisy, f"{layer} key includes {noisy}: every request is a miss"))
    return findings


def lint_cache_classes(classes: Mapping[str, Any]) -> list[LintFinding]:
    """Lint every cache class's declared `key_components`. Wire this into CI."""
    out: list[LintFinding] = []
    for layer, cls in classes.items():
        out.extend(lint_cache_key(layer, getattr(cls, "key_components", frozenset())))
    return out


def semantic_key_components(cache: SemanticCache) -> frozenset[str]:
    """Semantic eligibility expressed as key components, so the linter covers it too."""
    comps = set(SemanticCache.key_components)
    if cache.ttl_s is None or cache.ttl_s <= 0:
        comps.discard("ttl")
    return frozenset(comps)


__all__ = [
    "make_key",
    "normalize_text",
    "NORMALIZATION_VERSION",
    "Scope",
    "TTLCache",
    "EmbeddingCache",
    "RetrievalCache",
    "ScopedResponseCache",
    "SemanticCache",
    "SemanticHit",
    "SemanticStats",
    "LintFinding",
    "lint_cache_key",
    "lint_cache_classes",
    "semantic_key_components",
    "REQUIRED_COMPONENTS",
]
