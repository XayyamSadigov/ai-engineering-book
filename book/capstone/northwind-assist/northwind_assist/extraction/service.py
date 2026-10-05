# path: book/capstone/northwind-assist/northwind_assist/extraction/service.py
"""Structured extraction (Project 1, Chapter 6) as one route of the orchestrator.

Project 1's ExtractionService is used as is: classify, extract with a schema and a repair loop,
normalize, validate business rules, and route to `accept` or `human_review`. The result is either
schema-valid data or a review item; the capstone never returns unvalidated JSON. With
LLM_PROVIDER=fake the model is Project 1's ReplayLLM, which answers from labeled shared data.
"""
from __future__ import annotations

from typing import Any

from aie_core.llm.client import LLMClient
from aie_core.settings import Settings as CoreSettings
from extraction_api.adapters.replay_llm import ReplayLLM
from extraction_api.adapters.review_queue import InMemoryReviewQueue
from extraction_api.application.service import DocumentIn, ExtractionResult, ExtractionService

from .. import _paths


class ExtractionRoute:
    def __init__(self, client: LLMClient | None = None, *, tracer: Any = None) -> None:
        if client is None and CoreSettings().llm_provider == "fake":
            client = ReplayLLM.from_shared_data(_paths.SHARED_DATA)
        self.fixed_client = client
        self.queue = InMemoryReviewQueue()
        self.tracer = tracer

    def extract(self, text: str, *, tenant: str, request_id: str, client: LLMClient,
                doc_type: str | None = None) -> ExtractionResult:
        service = ExtractionService(self.fixed_client or client, self.queue, tracer=self.tracer)
        return service.extract(DocumentIn(text=text, tenant=tenant, doc_type=doc_type), request_id=request_id)

    def review_items(self, tenant: str) -> list[dict[str, Any]]:
        return [i.model_dump(mode="json", exclude={"document_text"}) for i in self.queue.list(tenant=tenant)]


__all__ = ["ExtractionRoute"]
