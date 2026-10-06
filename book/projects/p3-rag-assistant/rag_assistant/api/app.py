# path: book/projects/p3-rag-assistant/rag_assistant/api/app.py
"""HTTP API.

    POST   /v1/ask                 answer a question (JSON), or stream it (SSE) with "stream": true
                                   or "Accept: text/event-stream"
    POST   /v1/documents           admin: upload a document (Markdown with front matter); 202 + job id
    DELETE /v1/documents/{doc_id}  admin: delete now (tombstone), purge asynchronously; 202
    GET    /v1/index/status        versions, generations, queue depth, freshness lag vs SLO
    GET    /healthz                liveness and dependency breakers
    GET    /metrics                Prometheus text (stage latencies, cache hits, degraded modes)

Run: uvicorn rag_assistant.api.app:app  (the module-level `app` builds a container from env).
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Iterator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field
from ragkit.retrieval import Principal
from reliability import AdmissionRejected

from ..config import DEV_AUTH_SECRET
from ..domain.models import SHARED_TENANT
from ..ingestion.processing import InvalidDocument
from ..wiring import Container, build_container
from .auth import current_principal


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    stream: bool = False


class UploadRequest(BaseModel):
    filename: str = Field(description="used to pick the parser, e.g. policy.md")
    content: str = Field(description="the document, including YAML front matter with id, tenant, acl_groups")


def _request_id(request: Request) -> str:
    rid = request.headers.get("x-request-id", "")
    return rid if rid and len(rid) <= 64 and rid.replace("-", "").isalnum() else uuid.uuid4().hex[:16]


def _require_admin(container: Container, principal: Principal, tenant: str) -> None:
    if container.settings.admin_group not in principal.groups:
        raise HTTPException(status_code=403, detail="admin group required")
    if tenant not in (principal.tenant, SHARED_TENANT):  # an admin manages own tenant + shared content
        raise HTTPException(status_code=403, detail="document belongs to another tenant")


def _sse(events: Iterator[tuple[str, dict[str, Any]]]) -> Iterator[bytes]:
    for event, data in events:
        yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n".encode("utf-8")


def create_app(container: Container | None = None) -> FastAPI:
    app = FastAPI(title="Northwind Assist RAG", version="0.1.0")
    app.state.container = container or build_container()
    settings = app.state.container.settings
    if settings.auth_secret == DEV_AUTH_SECRET and not settings.allow_dev_auth_secret:
        # fail closed: the default secret is in the source tree, so it authenticates nobody
        raise RuntimeError("RAG_AUTH_SECRET is the published dev default; set a real secret "
                           "(or RAG_ALLOW_DEV_AUTH_SECRET=true for local development)")
    if container is None:
        c = app.state.container
        c.startup()
        if c.settings.sync_on_startup:
            c.ingestion.sync("folder")
            if c.settings.inline_ingest:
                c.drain()

    def get_container(request: Request) -> Container:
        return request.app.state.container

    @app.exception_handler(AdmissionRejected)
    async def _rejected(_: Request, exc: AdmissionRejected) -> JSONResponse:
        status = 429 if getattr(exc, "reason", "") == "tenant_quota" else 503
        retry = getattr(exc, "retry_after_s", None)
        headers = {"Retry-After": str(max(1, int(retry)))} if retry else {}
        return JSONResponse({"detail": str(exc)}, status_code=status, headers=headers)

    @app.post("/v1/ask")
    def ask(body: AskRequest, request: Request, principal: Principal = Depends(current_principal),
            c: Container = Depends(get_container)):  # type: ignore[no-untyped-def]
        rid = _request_id(request)
        wants_stream = body.stream or "text/event-stream" in request.headers.get("accept", "")
        if wants_stream:
            events = c.answers.stream(body.question, principal, request_id=rid)
            first = next(events)  # run admission and retrieval before committing to a 200 stream

            def chained() -> Iterator[tuple[str, dict[str, Any]]]:
                yield first
                yield from events

            return StreamingResponse(_sse(chained()), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-store", "X-Request-Id": rid})
        outcome = c.answers.ask(body.question, principal, request_id=rid)
        status = {"unavailable": 503, "blocked": 400}.get(outcome.response.mode, 200)
        return JSONResponse(outcome.response.model_dump(mode="json"), status_code=status,
                            headers={"X-Request-Id": rid})

    @app.post("/v1/documents", status_code=202)
    def upload(body: UploadRequest, principal: Principal = Depends(current_principal),
               c: Container = Depends(get_container)) -> dict[str, Any]:
        raw = body.content.encode("utf-8")
        try:
            doc_id, tenant = c.ingestion.peek(raw, body.filename)
            _require_admin(c, principal, tenant)
            existing = c.registry.get(doc_id)
            if existing is not None:  # the id may belong to another tenant: no overwriting or moving it
                _require_admin(c, principal, existing.tenant)
            out = c.ingestion.submit_upload(raw, body.filename)
        except InvalidDocument as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        if c.settings.inline_ingest:
            c.drain()
        return out

    @app.delete("/v1/documents/{doc_id}", status_code=202)
    def delete(doc_id: str, principal: Principal = Depends(current_principal),
               c: Container = Depends(get_container)) -> dict[str, Any]:
        rec = c.registry.get(doc_id)
        if rec is None or rec.status == "deleted":
            raise HTTPException(status_code=404, detail="no such document")
        _require_admin(c, principal, rec.tenant)
        out = c.ingestion.delete(doc_id)
        if c.settings.inline_ingest:
            c.drain()
        return out

    @app.get("/v1/index/status")
    def status(principal: Principal = Depends(current_principal), c: Container = Depends(get_container)) -> dict[str, Any]:
        return c.status().model_dump(mode="json")

    @app.get("/healthz")
    def healthz(c: Container = Depends(get_container)) -> dict[str, Any]:
        return {"status": "ok", "index_version": c.index_set.active_version,
                "breakers": {k: v.value for k, v in c.breakers.states().items()}}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics(c: Container = Depends(get_container)) -> str:
        return c.metrics.render_prometheus()

    return app


def __getattr__(name: str) -> Any:  # lazy `app` so importing this module never builds a container
    if name == "app":
        return create_app()
    raise AttributeError(name)


__all__ = ["create_app"]
