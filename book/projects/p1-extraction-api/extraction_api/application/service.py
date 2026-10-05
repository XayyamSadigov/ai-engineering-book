# path: book/projects/p1-extraction-api/extraction_api/application/service.py
"""ExtractionService: classify -> extract -> normalize -> validate -> route.

The control flow is a fixed workflow (Chapter 17). The model fills values at two points,
classification and extraction; code decides everything else, including whether a second
model attempt is worth paying for.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import date
from typing import Any, Callable

from aie_core import CompletionRequest, LLMClient, LLMError, MalformedResponseError, Message, Usage
from aie_core.llm.errors import TimeoutError as LLMTimeoutError
from aie_core.llm.structured import complete_structured
from aie_core.observability import NoopTracer, Tracer
from pydantic import BaseModel, Field

from ..domain.common import DocumentType, Route, RuleViolation
from ..domain.routing import RouteDecision, RoutingPolicy, decide
from .classifier import DocumentClassifier
from .metering import MeteredClient
from .pipelines import SPECS, ProcessContext, Processed
from .ports import ReviewItem, ReviewQueue
from .prompts import PROMPT_VERSION, render_document, render_repair


class DocumentIn(BaseModel):
    document_id: str | None = Field(default=None, max_length=128)
    text: str = Field(min_length=1)
    doc_type: DocumentType | None = Field(default=None, description="Skip classification when the caller knows the type.")
    tenant: str | None = Field(default=None, max_length=64)


class ExtractionResult(BaseModel):
    request_id: str
    document_id: str | None
    doc_type: DocumentType
    route: Route                                  # accept or human_review; repair is internal
    reasons: list[str] = Field(default_factory=list)
    data: dict[str, Any] | None = None
    violations: list[RuleViolation] = Field(default_factory=list)
    score: float = 0.0
    field_scores: dict[str, float] = Field(default_factory=dict)
    classification_confidence: float | None = None
    rule_repairs: int = 0
    llm_calls: int = 0
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0
    review_id: str | None = None
    prompt_version: str = PROMPT_VERSION


class BatchItem(BaseModel):
    index: int
    result: ExtractionResult | None = None
    error: str | None = None   # infrastructure failure: retry later, do not send to a human
    retryable: bool | None = None   # False: resubmitting the same item will fail the same way


class DocumentTooLarge(ValueError):
    pass


class ExtractionService:
    def __init__(
        self,
        client: LLMClient,
        review_queue: ReviewQueue,
        *,
        policy: RoutingPolicy | None = None,
        classifier: DocumentClassifier | None = None,
        tracer: Tracer | None = None,
        max_schema_repairs: int = 2,
        require_po: bool = True,
        dollar_means: str = "USD",
        max_document_chars: int = 50_000,
        batch_concurrency: int = 8,
        document_deadline_s: float = 120.0,
        call_timeout_s: float = 45.0,
        today: Callable[[], date] = date.today,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.client = client
        self.queue = review_queue
        self.policy = policy or RoutingPolicy()
        self.classifier = classifier or DocumentClassifier()
        self.tracer = tracer or NoopTracer()
        self.max_schema_repairs = max_schema_repairs
        self.require_po = require_po
        self.dollar_means = dollar_means
        self.max_document_chars = max_document_chars
        self.batch_concurrency = batch_concurrency
        self.document_deadline_s = document_deadline_s
        self.call_timeout_s = call_timeout_s
        self.today = today
        self.clock = clock

    # ------------------------------------------------------------------ single
    def extract(self, doc: DocumentIn, request_id: str | None = None) -> ExtractionResult:
        if len(doc.text) > self.max_document_chars:
            raise DocumentTooLarge(f"document has {len(doc.text)} characters; limit is {self.max_document_chars}")
        rid = request_id or uuid.uuid4().hex
        started = time.perf_counter()
        deadline = self.clock() + self.document_deadline_s
        metered = MeteredClient(self.client)
        with self.tracer.span("extract.document", request_id=rid, document_id=doc.document_id,
                              tenant=doc.tenant, prompt_version=PROMPT_VERSION) as span:
            result = self._run(doc, rid, metered, deadline)
            result.llm_calls = metered.calls
            result.usage = metered.usage
            result.latency_ms = round((time.perf_counter() - started) * 1000, 2)
            if result.route is Route.HUMAN_REVIEW:
                result.review_id = self._enqueue(doc, result).review_id
            span.set_attribute("doc_type", result.doc_type.value)
            span.set_attribute("route", result.route.value)
            span.set_attribute("reasons", result.reasons)
            span.set_attribute("score", result.score)
            span.set_attribute("rule_repairs", result.rule_repairs)
            span.set_attribute("llm_calls", metered.calls)
            span.set_attribute("input_tokens", metered.usage.input_tokens)
            span.set_attribute("output_tokens", metered.usage.output_tokens)
        return result

    def _call_timeout(self, deadline: float) -> float:
        """Per-call timeout capped by what is left of the document's deadline. An exhausted
        deadline is an infrastructure failure (retry the document later), not a review item."""
        remaining = deadline - self.clock()
        if remaining <= 0:
            raise LLMTimeoutError("document deadline exceeded", retryable=True)
        return min(self.call_timeout_s, remaining)

    def _run(self, doc: DocumentIn, rid: str, client: MeteredClient, deadline: float) -> ExtractionResult:
        # 1. classify, unless the caller already knows the type
        if doc.doc_type is not None and doc.doc_type is not DocumentType.OTHER:
            doc_type, class_conf = doc.doc_type, None
        else:
            with self.tracer.span("extract.classify", request_id=rid) as span:
                cls = self.classifier.classify(client, doc.text, timeout_s=self._call_timeout(deadline))
                span.set_attribute("doc_type", cls.doc_type.value)
                span.set_attribute("confidence", cls.confidence)
                span.set_attribute("abstained", cls.abstained)
            if cls.abstained or cls.doc_type not in SPECS:
                return ExtractionResult(
                    request_id=rid, document_id=doc.document_id, doc_type=cls.doc_type,
                    route=Route.HUMAN_REVIEW, reasons=["UNCLASSIFIED"], classification_confidence=cls.confidence,
                )
            doc_type, class_conf = cls.doc_type, cls.confidence

        spec = SPECS[doc_type]
        ctx = ProcessContext(today=self.today(), require_po=self.require_po, dollar_means=self.dollar_means)
        messages = [Message.system(spec.system_prompt), Message.user(render_document(doc.text))]
        repairs = 0
        processed: Processed | None = None

        # 2-5. extract, post-process, validate, route; loop only on the REPAIR route
        while True:
            task = f"extract_{doc_type.value}" if repairs == 0 else f"repair_{doc_type.value}"
            req = CompletionRequest(messages=messages, max_tokens=spec.max_tokens, metadata={"task": task},
                                    timeout_s=self._call_timeout(deadline))
            with self.tracer.span("extract.llm", request_id=rid, task=task) as span:
                try:
                    draft, _ = complete_structured(client, req, spec.draft_schema,
                                                   max_repair_attempts=self.max_schema_repairs)
                except MalformedResponseError as exc:
                    span.set_attribute("schema_failure", True)
                    return ExtractionResult(
                        request_id=rid, document_id=doc.document_id, doc_type=doc_type,
                        route=Route.HUMAN_REVIEW, reasons=["SCHEMA_FAILURE"],
                        data=processed.data if processed else None,
                        violations=[RuleViolation(code="SCHEMA_FAILURE", detail=str(exc)[:500])],
                        classification_confidence=class_conf, rule_repairs=repairs,
                    )
            with self.tracer.span("extract.validate", request_id=rid) as span:
                processed = spec.process(draft, doc.text, ctx)
                decision = decide(processed.violations, processed.score, repairs, self.policy)
                span.set_attribute("violations", [v.code for v in processed.violations])
                span.set_attribute("score", processed.score)
                span.set_attribute("route", decision.route.value)
            if decision.route is not Route.REPAIR:
                break
            if deadline - self.clock() < self.call_timeout_s / 3:
                # degraded mode: not enough time for a useful repair call; a person gets the
                # draft now instead of the whole document failing later
                decision = RouteDecision(route=Route.HUMAN_REVIEW,
                                         reasons=[*decision.reasons, "REPAIR_SKIPPED_DEADLINE"])
                break
            repairs += 1
            errors = [v for v in processed.violations if v.severity.value == "error"]
            messages = [
                *messages,
                Message.assistant(json.dumps(draft.model_dump(mode="json"), ensure_ascii=False)),
                Message.user(render_repair(errors)),
            ]

        return ExtractionResult(
            request_id=rid, document_id=doc.document_id, doc_type=doc_type, route=decision.route,
            reasons=decision.reasons, data=processed.data, violations=processed.violations,
            score=round(processed.score, 4), field_scores=processed.field_scores,
            classification_confidence=class_conf, rule_repairs=repairs,
        )

    def _enqueue(self, doc: DocumentIn, result: ExtractionResult) -> ReviewItem:
        item = ReviewItem(
            review_id=uuid.uuid4().hex, request_id=result.request_id, document_id=doc.document_id,
            tenant=doc.tenant, doc_type=result.doc_type, reasons=result.reasons, violations=result.violations,
            data=result.data, document_text=doc.text, prompt_version=result.prompt_version,
        )
        return self.queue.enqueue(item)

    # ------------------------------------------------------------------- batch
    async def extract_batch(
        self, docs: list[DocumentIn], request_id: str | None = None, concurrency: int | None = None
    ) -> list[BatchItem]:
        """Bounded fan-out. Order is preserved, and one document's failure never fails the
        batch: infrastructure errors come back as `error`, content problems as human_review."""
        rid = request_id or uuid.uuid4().hex
        sem = asyncio.Semaphore(concurrency or self.batch_concurrency)

        async def one(i: int, doc: DocumentIn) -> BatchItem:
            async with sem:
                try:
                    res = await asyncio.to_thread(self.extract, doc, f"{rid}-{i}")
                    return BatchItem(index=i, result=res)
                except (LLMError, DocumentTooLarge) as exc:
                    retryable = exc.retryable if isinstance(exc, LLMError) else False
                    return BatchItem(index=i, error=f"{type(exc).__name__}: {exc}"[:500], retryable=retryable)

        with self.tracer.span("extract.batch", request_id=rid, size=len(docs)) as span:
            items = await asyncio.gather(*(one(i, d) for i, d in enumerate(docs)))
            routes = [it.result.route.value if it.result else "error" for it in items]
            span.set_attribute("accepted", routes.count("accept"))
            span.set_attribute("human_review", routes.count("human_review"))
            span.set_attribute("errors", routes.count("error"))
        return list(items)


__all__ = ["DocumentIn", "ExtractionResult", "BatchItem", "DocumentTooLarge", "ExtractionService"]
