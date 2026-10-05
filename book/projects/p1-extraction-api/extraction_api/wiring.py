# path: book/projects/p1-extraction-api/extraction_api/wiring.py
"""Composition root: build the service from environment settings. The only module that
knows which concrete adapters exist."""
from __future__ import annotations

from aie_core import LLMClient, Settings, make_llm_client
from aie_core.observability import Tracer, get_tracer

from .adapters import InMemoryReviewQueue, ReplayLLM, SQLiteReviewQueue
from .application import DocumentClassifier, ExtractionService, ReviewQueue
from .config import AppSettings
from .domain import RoutingPolicy


def build_llm(core: Settings, app: AppSettings) -> LLMClient:
    if core.llm_provider == "fake":
        return ReplayLLM.from_shared_data(app.shared_data_dir)
    return make_llm_client(core)  # a ModelGateway with retries, limits, tracing


def build_queue(app: AppSettings) -> ReviewQueue:
    if app.review_backend == "sqlite":
        return SQLiteReviewQueue(app.review_db_path)
    return InMemoryReviewQueue()


def build_service(
    app: AppSettings | None = None,
    core: Settings | None = None,
    *,
    client: LLMClient | None = None,
    queue: ReviewQueue | None = None,
    tracer: Tracer | None = None,
) -> ExtractionService:
    app = app or AppSettings()
    core = core or Settings()
    return ExtractionService(
        client or build_llm(core, app),
        queue or build_queue(app),
        policy=RoutingPolicy(accept_threshold=app.accept_threshold, max_rule_repairs=app.max_rule_repairs),
        classifier=DocumentClassifier(threshold=app.classify_threshold, samples=app.classify_samples),
        tracer=tracer or get_tracer(core),
        max_schema_repairs=app.max_schema_repairs,
        require_po=app.require_po,
        dollar_means=app.dollar_means,
        max_document_chars=app.max_document_chars,
        batch_concurrency=app.batch_concurrency,
        document_deadline_s=app.document_deadline_s,
        call_timeout_s=app.call_timeout_s,
    )


__all__ = ["build_llm", "build_queue", "build_service"]
