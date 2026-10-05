# path: book/projects/p1-extraction-api/extraction_api/api/app.py
"""FastAPI surface: extract one, extract a batch, list and resolve review items."""
from __future__ import annotations

import re
import time
import uuid
from typing import Literal

from aie_core import LLMError
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from ..application import (
    AlreadyResolved, DocumentIn, DocumentTooLarge, ExtractionResult, ExtractionService, ReviewItem,
    ReviewNotFound, ReviewResolution,
)
from ..config import AppSettings
from ..domain import DocumentType, Invoice, SupportTicket
from .auth import ANONYMOUS, Authenticator, Principal
from .schemas import BatchRequest, BatchResponse, BatchSummary, ResolveRequest

_REQUEST_ID_OK = re.compile(r"^[A-Za-z0-9._-]{1,64}$")  # never echo arbitrary header bytes into logs
_DOMAIN_MODEL = {DocumentType.INVOICE: Invoice, DocumentType.SUPPORT_TICKET: SupportTicket}


def create_app(service: ExtractionService | None = None, settings: AppSettings | None = None) -> FastAPI:
    settings = settings or AppSettings()
    if service is None:
        from ..wiring import build_service

        service = build_service(settings)
    app = FastAPI(title="Northwind Extraction API", version="0.1.0")
    app.state.service = service
    tracer = service.tracer
    auth = Authenticator(settings.api_keys)

    def submitter(request: Request) -> Principal:
        return auth.authenticate(request, "submit")

    def reviewer(request: Request) -> Principal:
        return auth.authenticate(request, "review")

    def scoped(doc: DocumentIn, who: Principal) -> DocumentIn:
        # a tenant-bound caller cannot submit for, or label a document as, another tenant
        if who.tenant is None:
            return doc
        if doc.tenant not in (None, who.tenant):
            raise HTTPException(status_code=403, detail="tenant does not match credentials")
        return doc.model_copy(update={"tenant": who.tenant})

    def visible_item(review_id: str, who: Principal) -> ReviewItem:
        try:
            item = service.queue.get(review_id)
        except ReviewNotFound:
            item = None
        if item is None or not who.can_see(item.tenant):
            raise HTTPException(status_code=404, detail="review item not found")  # 404, not 403: ids are not probeable
        return item

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        incoming = request.headers.get("x-request-id", "")
        rid = incoming if _REQUEST_ID_OK.match(incoming) else uuid.uuid4().hex
        request.state.request_id = rid
        length = request.headers.get("content-length")
        if length is not None and (not length.isdigit() or int(length) > settings.max_request_bytes):
            # refuse before parsing: a 2 GB JSON body must not reach pydantic (a proxy should cap it too)
            return JSONResponse(status_code=413, headers={"X-Request-ID": rid},
                                content={"detail": f"request body limit is {settings.max_request_bytes} bytes"})
        started = time.perf_counter()
        with tracer.span("http.request", request_id=rid, method=request.method, path=request.url.path) as span:
            response = await call_next(request)
            span.set_attribute("status_code", response.status_code)
            span.set_attribute("duration_ms", round((time.perf_counter() - started) * 1000, 2))
        response.headers["X-Request-ID"] = rid
        return response

    @app.exception_handler(LLMError)
    async def llm_unavailable(request: Request, exc: LLMError) -> JSONResponse:
        rid = getattr(request.state, "request_id", None)
        if not exc.retryable:
            # e.g. the provider rejected our request or schema: retrying the same call cannot
            # succeed, so do not tell the caller to retry (that turns a bug into a retry storm)
            return JSONResponse(status_code=502, content={
                "detail": "model provider rejected the request; not retryable", "error": type(exc).__name__,
                "retryable": False, "request_id": rid,
            })
        headers = {"Retry-After": str(int(exc.retry_after_s))} if exc.retry_after_s else {}
        return JSONResponse(status_code=503, headers=headers, content={
            "detail": "model provider unavailable, retry later", "error": type(exc).__name__,
            "retryable": True, "request_id": rid,
        })

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        return {"status": "ok", "review_counts": service.queue.counts()}

    @app.post("/extract", response_model=ExtractionResult)
    def extract(doc: DocumentIn, request: Request, who: Principal = Depends(submitter)) -> ExtractionResult:
        # a sync endpoint: FastAPI runs it on its thread pool, so a slow model call
        # does not block the event loop
        try:
            return service.extract(scoped(doc, who), request_id=request.state.request_id)
        except DocumentTooLarge as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc

    @app.post("/extract/batch", response_model=BatchResponse)
    async def extract_batch(
        body: BatchRequest, request: Request, who: Principal = Depends(submitter)
    ) -> BatchResponse:
        if len(body.documents) > settings.max_batch_size:
            raise HTTPException(status_code=413, detail=f"batch limit is {settings.max_batch_size} documents")
        rid = request.state.request_id
        items = await service.extract_batch([scoped(d, who) for d in body.documents], request_id=rid)
        results = [i.result for i in items if i.result is not None]
        summary = BatchSummary(
            total=len(items),
            accepted=sum(r.route.value == "accept" for r in results),
            human_review=sum(r.route.value == "human_review" for r in results),
            errors=sum(i.error is not None for i in items),
            llm_calls=sum(r.llm_calls for r in results),
            input_tokens=sum(r.usage.input_tokens for r in results),
            output_tokens=sum(r.usage.output_tokens for r in results),
        )
        return BatchResponse(request_id=rid, summary=summary, items=items)

    @app.get("/review", response_model=list[ReviewItem])
    def list_review(
        status: Literal["pending", "approved", "corrected", "rejected", "all"] = "pending",
        limit: int = Query(default=50, ge=1, le=500),
        who: Principal = Depends(reviewer),
    ) -> list[ReviewItem]:
        return service.queue.list(None if status == "all" else status, limit, tenant=who.tenant)

    @app.get("/review/{review_id}", response_model=ReviewItem)
    def get_review(review_id: str, who: Principal = Depends(reviewer)) -> ReviewItem:
        return visible_item(review_id, who)

    @app.post("/review/{review_id}/resolve", response_model=ReviewItem)
    def resolve_review(review_id: str, body: ResolveRequest, who: Principal = Depends(reviewer)) -> ReviewItem:
        item = visible_item(review_id, who)
        corrected = None
        if body.decision == "correct":
            model = _DOMAIN_MODEL.get(item.doc_type)
            if body.corrected_data is None or model is None:
                raise HTTPException(status_code=422, detail="correct requires corrected_data for a known doc_type")
            try:
                # a human correction must satisfy the same domain schema as the machine's output
                corrected = model.model_validate(body.corrected_data).model_dump(mode="json")
            except ValidationError as exc:
                raise HTTPException(status_code=422, detail=exc.errors(include_url=False)) from exc
        # with authentication on, the recorded reviewer is the authenticated principal, not a
        # name the client typed: the audit trail must not be self-asserted
        name = body.reviewer if who is ANONYMOUS else who.name
        resolution = ReviewResolution(decision=body.decision, reviewer=name,
                                      corrected_data=corrected, note=body.note)
        try:
            with tracer.span("review.resolve", review_id=review_id, decision=body.decision, reviewer=name):
                return service.queue.resolve(review_id, resolution)
        except AlreadyResolved:
            raise HTTPException(status_code=409, detail="review item already resolved") from None

    return app


def app_factory() -> FastAPI:
    """Entry point for `uvicorn extraction_api.api.app:app_factory --factory`."""
    return create_app()


__all__ = ["create_app", "app_factory"]
