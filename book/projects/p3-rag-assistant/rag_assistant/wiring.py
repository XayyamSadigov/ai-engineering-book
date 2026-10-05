# path: book/projects/p3-rag-assistant/rag_assistant/wiring.py
"""Composition root: builds every component from settings, in one place.

Tests call `build_container(AssistantSettings(...), llm=FakeLLM(...), embeddings=FakeEmbeddings(...))`;
the API, worker and eval script call it with environment settings. Backends switch by setting:

    registry   memory | sql (SQLAlchemy URL; PostgreSQL in Compose)
    queue      memory | redis (reliability.RedisJobQueue)
    vectors    numpy  | pgvector (semsearch.PgVectorStore)
    cache      memory | redis (embedding cache and contextual-prefix cache)

One tracer is created per container and passed to every component, so a request produces
one connected tree of aie_core spans (request id on the root span).
"""
from __future__ import annotations

import time
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import Any

from aie_core.embeddings import EmbeddingClient, FakeEmbeddings
from aie_core.llm.client import LLMClient
from aie_core.llm.providers import FakeLLM
from aie_core.observability import Tracer, get_tracer
from aie_core.settings import Settings as CoreSettings
from aie_core.settings import make_embedding_client, make_llm_client
from guardrails import rag_pipeline
from ragkit.generation import (
    AbstentionPolicy,
    CitationValidator,
    EvidencePacker,
    GeneratorConfig,
    GroundedGenerator,
    GroundedQA,
    GroundedStreamer,
    PackerConfig,
)
from ragkit.retrieval import ContextualEnricher, DenseRetriever
from reliability import (
    AdmissionConfig,
    AdmissionController,
    CircuitBreakerClient,
    CircuitBreakerRegistry,
    InMemoryJobQueue,
    JobQueue,
    TenantQuota,
)

from .adapters.llm import DeadlineBoundLLM, corpus_vocabulary, extractive_handler
from .answering.service import AnswerService
from .caching.caches import CacheSet, ForgettableEmbeddings, RedisBytesMap, StrMapView
from .config import AssistantSettings
from .domain.authority import AuthorityRules
from .domain.models import IndexStatus, now
from .ingestion.handlers import IngestHandlers, IngestionDeps
from .ingestion.processing import DocumentProcessor
from .ingestion.registry import DocumentRegistry, InMemoryRegistry, SqlRegistry
from .ingestion.service import IngestionService
from .ingestion.sources import BlobConnector, FolderConnector
from .observability.metrics import Metrics, quantile
from .retrieval.index import IndexSet


@dataclass
class Container:
    settings: AssistantSettings
    tracer: Tracer
    registry: DocumentRegistry
    queue: JobQueue
    embeddings: ForgettableEmbeddings
    vector_store: Any
    index_set: IndexSet
    caches: CacheSet
    blobs: BlobConnector
    handlers: IngestHandlers
    ingestion: IngestionService
    answers: AnswerService
    breakers: CircuitBreakerRegistry
    metrics: Metrics
    llm: LLMClient
    raw_llm: LLMClient
    extra: dict[str, Any] = field(default_factory=dict)

    def drain(self, max_jobs: int | None = None) -> int:
        """Process queued ingestion jobs in this process until the queue is idle (tests, dev, CLI)."""
        from .ingestion.worker import build_worker

        worker = build_worker(self, poll_interval_s=0.0, sleep=lambda s: None)
        return worker.run(max_jobs=max_jobs, max_idle_polls=1)

    def startup(self) -> dict[str, list[str]]:
        """Load snapshots and remove anything the registry no longer lists as active."""
        self.index_set.refresh()
        return self.index_set.reconcile()

    def status(self) -> IndexStatus:
        active = self.registry.all("active")
        deleting = self.registry.all("deleting")
        lags = [r.freshness_lag_s for r in active if r.freshness_lag_s is not None]
        p95 = quantile(lags, 0.95)
        t = now()
        return IndexStatus(
            active_version=self.index_set.active_version,
            building_version=self.index_set.building_version,
            previous_version=self.index_set.previous_version,
            tenancy_mode=self.settings.tenancy_mode,
            documents_active=len(active),
            documents_deleting=len(deleting),
            chunks_indexed=self.index_set.stats(),
            generations=self.registry.generations(),
            queue_depth=self.queue.depth(),
            dead_letters=len(self.queue.dead_letters()),
            freshness_lag_p95_s=p95,
            freshness_lag_max_s=max(lags) if lags else None,
            freshness_slo_s=self.settings.freshness_slo_s,
            freshness_slo_met=p95 is None or p95 <= self.settings.freshness_slo_s,
            oldest_pending_purge_s=max((t - (r.deleted_at or t) for r in deleting), default=None),
            breakers={name: state.value for name, state in self.breakers.states().items()},
        )


def _registry(s: AssistantSettings, core: CoreSettings) -> DocumentRegistry:
    if s.registry_backend == "sql":
        url = s.registry_url or core.database_url
        if not url:
            raise ValueError("RAG_REGISTRY_BACKEND=sql needs RAG_REGISTRY_URL or DATABASE_URL")
        return SqlRegistry(url)
    return InMemoryRegistry()


def _redis(core: CoreSettings) -> Any:
    import redis

    if not core.redis_url:
        raise ValueError("a redis backend needs REDIS_URL")
    return redis.Redis.from_url(core.redis_url)


def build_container(
    settings: AssistantSettings | None = None,
    *,
    llm: LLMClient | None = None,
    embeddings: EmbeddingClient | None = None,
    tracer: Tracer | None = None,
    queue: JobQueue | None = None,
    registry: DocumentRegistry | None = None,
    vector_store: Any | None = None,
    cache_store: MutableMapping[str, bytes] | None = None,
    clock: Any = time.monotonic,
) -> Container:
    s = settings or AssistantSettings()
    core = CoreSettings()
    tracer = tracer or get_tracer(core)
    metrics = Metrics()
    registry = registry or _registry(s, core)
    if queue is None:
        queue = InMemoryJobQueue() if s.queue_backend == "memory" else _redis_queue(core)

    # --- embeddings: one space, cached by space fingerprint + text, forgettable per document
    if embeddings is None:
        if core.embedding_provider == "fake":  # hashing fakes cannot retrieve; vocabulary mode can
            embeddings = FakeEmbeddings(vocabulary=corpus_vocabulary(s.docs_dir), model=core.embedding_model)
        else:
            embeddings = make_embedding_client(core)
    if cache_store is None:
        cache_store = RedisBytesMap(_redis(core), prefix="rag:emb:") if s.cache_backend == "redis" else {}
    cached = ForgettableEmbeddings(embeddings, cache_store, text_prep_version=s.pipeline_version)

    # --- vectors (semsearch) and index versions
    if vector_store is None:
        if s.vector_backend == "pgvector":
            from semsearch.adapters import PgVectorStore

            if not core.database_url:
                raise ValueError("RAG_VECTOR_BACKEND=pgvector needs DATABASE_URL")
            vector_store = PgVectorStore(core.database_url, cached.dimensions,
                                         statement_timeout_ms=s.pg_statement_timeout_ms)
        else:
            from semsearch.adapters import NumpyVectorStore

            vector_store = NumpyVectorStore(cached.dimensions)

    def make_dense(version: str, partition: str) -> DenseRetriever:
        name = s.index_name if partition == "all" else f"{s.index_name}-{partition}"
        return DenseRetriever(cached, vector_store, index_name=name, index_version=version)

    index_set = IndexSet(registry, make_dense=make_dense, default_version=s.index_version,
                         tenancy_mode=s.tenancy_mode, snapshot_dir=s.snapshot_dir)

    # --- LLM stack: provider (gateway with retries for real providers) -> breaker -> deadline
    breakers = CircuitBreakerRegistry(min_calls=s.breaker_min_calls, open_s=s.breaker_open_s, clock=clock)
    if llm is None:
        llm = FakeLLM(handler=extractive_handler()) if core.llm_provider == "fake" else make_llm_client(core)
    llm_stack = DeadlineBoundLLM(CircuitBreakerClient(llm, breakers.get("llm")))

    # --- ingestion
    rules = AuthorityRules.load(s.authority_rules_path)
    context_cache: MutableMapping[str, str] | None = None
    enricher = None
    if s.contextual_enrichment:
        context_cache = StrMapView(RedisBytesMap(_redis(core), prefix="rag:ctx:")) if s.cache_backend == "redis" else {}
        enricher = ContextualEnricher(llm, cache=context_cache)
    processor = DocumentProcessor(s, rules, enricher)
    blobs = BlobConnector(s.blob_dir)
    caches = CacheSet(retrieval_ttl_s=s.retrieval_cache_ttl_s, answer_ttl_s=s.answer_cache_ttl_s,
                      max_entries=s.cache_max_entries, policy_version=s.guard_policy_version)
    deps = IngestionDeps(registry=registry, index_set=index_set,
                         connectors={"folder": FolderConnector(s.docs_dir), "blob": blobs},
                         processor=processor, embeddings=cached, caches=[caches], blobs=blobs,
                         tenancy_mode=s.tenancy_mode, metrics=metrics, tracer=tracer, context_cache=context_cache)
    handlers = IngestHandlers(deps)
    ingestion = IngestionService(deps, queue, max_inline_bytes=s.max_inline_bytes)

    # --- answering
    generator = GroundedGenerator(llm_stack, GeneratorConfig(max_tokens=s.generation_max_tokens), tracer=tracer)
    qa = GroundedQA(generator,
                    EvidencePacker(PackerConfig(token_budget=s.evidence_token_budget, max_blocks=s.max_evidence_blocks)),
                    CitationValidator(), AbstentionPolicy(), tracer=tracer)
    streamer = GroundedStreamer(llm_stack, generator=generator)
    admission = AdmissionController(AdmissionConfig(
        capacity=s.admission_capacity, max_queue=s.admission_capacity,
        default_quota=TenantQuota(requests_per_minute=s.tenant_requests_per_minute)))
    answers = AnswerService(s, registry=registry, index_set=index_set, qa=qa, streamer=streamer,
                            guards=rag_pipeline(tracer=tracer), caches=caches, breakers=breakers, metrics=metrics,
                            tracer=tracer, admission=admission,
                            model_id=f"{getattr(llm, 'provider', 'x')}:{getattr(llm, 'default_model', '') or core.llm_model}")
    return Container(settings=s, tracer=tracer, registry=registry, queue=queue, embeddings=cached,
                     vector_store=vector_store, index_set=index_set, caches=caches, blobs=blobs, handlers=handlers,
                     ingestion=ingestion, answers=answers, breakers=breakers, metrics=metrics, llm=llm_stack,
                     raw_llm=llm)


def _redis_queue(core: CoreSettings) -> JobQueue:
    from reliability import RedisJobQueue

    return RedisJobQueue(_redis(core), namespace="northwind:rag-ingest")


__all__ = ["Container", "build_container"]
