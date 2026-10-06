# path: book/capstone/northwind-assist/northwind_assist/api/app.py
"""HTTP surface: routing, validation, status codes, SSE framing. No business rules live here.

Streaming design: the orchestrator runs in its own thread and pushes events into a queue; the
response generator only reads the queue. The Chapter 31 tracer keeps the current span in a
contextvar, and Starlette may advance a sync generator from different worker threads, so running
the whole turn on one thread is what keeps the span tree intact. If the client disconnects, the
generator is closed and the request deadline is cancelled; the turn stops at its next stage
boundary (the next model call checks the deadline before spending).
"""
from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from toolkit import ApprovalError

from ..container import Container, build_container
from ..domain.context import RequestContext
from ..domain.events import ServerEvent
from ..orchestrator import ChatRequest, Prepared, Rejected
from ..security.auth import DEV_PERSONAS, AuthError, ForbiddenTenant, issue_dev_token

STATIC = Path(__file__).resolve().parent / "static"
HEARTBEAT_S = 10.0


class DevTokenIn(BaseModel):
    persona: str = Field(pattern=r"^[a-z]{2,16}$")


class DecisionIn(BaseModel):
    note: str | None = Field(default=None, max_length=500)


class FeedbackIn(BaseModel):
    request_id: str = Field(pattern=r"^req_[0-9a-f]{16}$")
    rating: str = Field(pattern=r"^(up|down)$")
    comment: str | None = Field(default=None, max_length=1000)


class ExtractIn(BaseModel):
    text: str = Field(min_length=1, max_length=8_000)       # the input guard's size limit
    doc_type: str | None = Field(default=None, pattern=r"^(invoice|support_ticket)$")


def create_app(container: Container | None = None) -> FastAPI:
    c = container or build_container()
    app = FastAPI(title="Northwind Assist", version=c.settings.app_version)
    app.state.container = c
    feedback: list[dict[str, Any]] = []
    owners: dict[str, tuple[str, str]] = {}          # request id -> (tenant, user), for feedback ownership
    owners_lock = threading.Lock()

    # ------------------------------------------------------------------ auth
    def current(authorization: str | None = Header(default=None)) -> RequestContext:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(401, "missing bearer token", headers={"WWW-Authenticate": "Bearer"})
        try:
            return c.validator.validate(authorization.split(" ", 1)[1].strip())
        except ForbiddenTenant as exc:
            raise HTTPException(403, exc.code) from exc
        except AuthError as exc:
            raise HTTPException(401, exc.code, headers={"WWW-Authenticate": f'Bearer error="{exc.code}"'}) from exc

    def need(ctx: RequestContext, scope: str) -> None:
        if not ctx.has_scope(scope):
            raise HTTPException(403, f"missing scope {scope}")

    # ------------------------------------------------------------------ ops
    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers={"Content-Security-Policy": CSP})

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> JSONResponse:
        ready = c.kb.chunk_count() > 0
        body = {"ready": ready, "index_version": c.kb.index_version, "chunks": c.kb.chunk_count(),
                "breakers": {k: v.value for k, v in c.models.breakers.states().items()}}
        return JSONResponse(body, status_code=200 if ready else 503)

    @app.get("/v1/auth/personas")
    def personas() -> dict[str, Any]:
        if not _dev_login(c):
            raise HTTPException(404, "not found")
        return {k: {"tenant": v["tenant"], "title": v["title"], "roles": v["roles"]} for k, v in DEV_PERSONAS.items()}

    @app.post("/v1/auth/dev-token")
    def dev_token(body: DevTokenIn) -> dict[str, str]:
        if not _dev_login(c) or body.persona not in DEV_PERSONAS:
            raise HTTPException(404, "not found")
        p = DEV_PERSONAS[body.persona]
        token = issue_dev_token(c.settings, sub=body.persona, tenant=p["tenant"], groups=p["groups"], roles=p["roles"])
        return {"access_token": token, "token_type": "bearer"}

    @app.get("/v1/me")
    def me(ctx: RequestContext = Depends(current)) -> dict[str, Any]:
        return {"user_id": ctx.user_id, "tenant": ctx.tenant, "groups": sorted(ctx.groups), "roles": sorted(ctx.roles),
                "scopes": sorted(ctx.scopes)}

    # ------------------------------------------------------------------ chat
    @app.post("/v1/chat")
    def chat(body: ChatRequest, request: Request, stream: bool = Query(default=True),
             idempotency_key: str | None = Header(default=None, max_length=128),
             ctx: RequestContext = Depends(current)) -> Any:
        try:
            prepared = c.orchestrator.prepare(ctx, body, idempotency_key=idempotency_key)
        except Rejected as exc:
            headers = {"Retry-After": str(int(exc.retry_after_s or 1) or 1)}
            return JSONResponse({"error": exc.reason}, status_code=exc.status, headers=headers)
        with owners_lock:
            owners[prepared.ctx.request_id] = (ctx.tenant, ctx.user_id)
        if not stream:
            result = c.orchestrator.run(prepared)
            done = next((e.data for e in reversed(result.events) if e.event == "done"), {})
            return {**done, "events": [e.model_dump() for e in result.events]}
        # Start the turn now, not on the first read: prepare() already holds an admission slot and a
        # spend reservation, and only run() releases them. A client that disconnects before the
        # body is read would otherwise leak both.
        return StreamingResponse(_sse(prepared, _start_turn(c, prepared)), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                                          "X-Request-Id": prepared.ctx.request_id})

    @app.post("/v1/feedback")
    def give_feedback(body: FeedbackIn, ctx: RequestContext = Depends(current)) -> dict[str, Any]:
        with owners_lock:
            owner = owners.get(body.request_id)
        if owner != (ctx.tenant, ctx.user_id):
            raise HTTPException(404, "unknown request")
        row = {"request_id": body.request_id, "rating": body.rating, "comment": body.comment, "tenant": ctx.tenant,
               "ts": time.time()}
        feedback.append(row)
        with c.tracer.span("feedback", **{"response.id": body.request_id, "feedback.rating": body.rating,
                                          "tenant.id": ctx.tenant}):
            pass
        return {"recorded": True}

    app.state.feedback = feedback

    # ------------------------------------------------------------------ approvals
    @app.get("/v1/approvals")
    def approvals(ctx: RequestContext = Depends(current)) -> list[dict[str, Any]]:
        return c.tools.pending(ctx)

    @app.post("/v1/approvals/{approval_id}/approve")
    def approve(approval_id: str, body: DecisionIn, ctx: RequestContext = Depends(current)) -> dict[str, Any]:
        try:
            res = c.tools.approve(approval_id, ctx, body.note)
        except ApprovalError as exc:
            status = {"approval_not_found": 404, "requester_context_lost": 409}.get(exc.code, 403)
            raise HTTPException(status, exc.code) from exc
        return {"approval_id": approval_id, "status": res.status, "result": res.data, "error": res.error,
                "duplicate": res.duplicate}

    @app.post("/v1/approvals/{approval_id}/reject")
    def reject(approval_id: str, body: DecisionIn, ctx: RequestContext = Depends(current)) -> dict[str, Any]:
        try:
            req = c.tools.reject(approval_id, ctx, body.note)
        except ApprovalError as exc:
            raise HTTPException(404 if exc.code == "approval_not_found" else 403, exc.code) from exc
        return c.tools.view(req)

    # ------------------------------------------------------------------ memory
    @app.get("/v1/memory")
    def memory(ctx: RequestContext = Depends(current)) -> dict[str, Any]:
        return {"facts": c.memory.facts(ctx.owner()), "pending": c.memory.pending(ctx.owner())}

    @app.post("/v1/memory/{record_id}/confirm")
    def confirm(record_id: str, ctx: RequestContext = Depends(current)) -> dict[str, Any]:
        try:
            return c.memory.confirm(ctx.owner(), record_id, turn_ref=f"ui:{ctx.user_id}:{int(time.time())}").as_dict()
        except Exception as exc:  # ProfileError: no pending proposal for *this* owner
            raise HTTPException(404, "no pending proposal") from exc

    @app.post("/v1/memory/{record_id}/reject")
    def reject_memory(record_id: str, ctx: RequestContext = Depends(current)) -> dict[str, Any]:
        return {"deleted": c.memory.reject(ctx.owner(), record_id)}

    @app.delete("/v1/memory/{key}")
    def forget(key: str, ctx: RequestContext = Depends(current)) -> dict[str, Any]:
        return {"deleted": c.memory.forget(ctx.owner(), key)}

    # ------------------------------------------------------------------ extraction
    @app.post("/v1/extract")
    def extract(body: ExtractIn, ctx: RequestContext = Depends(current)) -> dict[str, Any]:
        req = ChatRequest(message="extract: document", document=body.text, doc_type=body.doc_type)
        try:
            prepared = c.orchestrator.prepare(ctx, req)
        except Rejected as exc:
            raise HTTPException(exc.status, exc.reason) from exc
        result = c.orchestrator.run(prepared)
        if result.structured is None:
            raise HTTPException(502, "extraction failed")
        return result.structured

    # ------------------------------------------------------------------ cost and admin
    @app.get("/v1/cost/daily")
    def cost_daily(day: str | None = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
                   ctx: RequestContext = Depends(current)) -> dict[str, Any]:
        need(ctx, "cost:read")
        everyone = "platform:operate" in ctx.scopes            # a tenant admin sees only its own tenant
        return c.ledger.daily_report(day, tenant=None if everyone else ctx.tenant)

    @app.get("/v1/admin/status")
    def status(ctx: RequestContext = Depends(current)) -> dict[str, Any]:
        need(ctx, "platform:operate")                          # breakers, caches and admission are shared
        return {"manifest": c.base_manifest.model_dump(mode="json"), "index": c.kb.fingerprint(),
                "breakers": c.models.breakers.snapshot(), "admission": c.resilience.admission.snapshot(),
                "caches": c.caches.stats(), "feedback": len(feedback)}

    @app.post("/v1/admin/reindex")
    def reindex(ctx: RequestContext = Depends(current)) -> dict[str, Any]:
        need(ctx, "platform:operate")                          # rebuilds every tenant's index
        before = c.kb.index_version
        after = c.kb.reindex()
        purged = c.caches.purge_tenant("retail") + c.caches.purge_tenant("logistics")
        return {"before": before, "after": after, "cache_entries_purged": purged}

    @app.delete("/v1/admin/documents/{doc_id}")
    def delete_document(doc_id: str, ctx: RequestContext = Depends(current)) -> dict[str, Any]:
        need(ctx, "admin:reindex")
        owner = c.kb.doc_tenant(doc_id)
        if owner is None or (owner != ctx.tenant and "platform:operate" not in ctx.scopes):
            # another tenant's document does not exist for a tenant admin
            raise HTTPException(404, "unknown document")
        removed = c.kb.delete(doc_id)
        purged = c.caches.purge_tenant("retail") + c.caches.purge_tenant("logistics")
        return {"doc_id": doc_id, "chunks_removed": removed, "index_version": c.kb.index_version,
                "cache_entries_purged": purged}

    return app


CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' "
       "https://fonts.googleapis.com; font-src https://fonts.gstatic.com; img-src 'self'; connect-src 'self'")


def _dev_login(c: Container) -> bool:
    return c.settings.dev_login and c.settings.auth_mode == "hs256" and c.settings.environment != "prod"


def _start_turn(c: Container, prepared: Prepared) -> queue.Queue[ServerEvent | None]:
    q: queue.Queue[ServerEvent | None] = queue.Queue()

    def work() -> None:
        try:
            c.orchestrator.run(prepared, q.put)
        except Exception as exc:  # the stream must end with an event, never with a dropped socket
            q.put(ServerEvent(event="error", data={"stage": "internal", "message": type(exc).__name__}, id=10_000))
        finally:
            q.put(None)

    threading.Thread(target=work, name=f"turn-{prepared.ctx.request_id}", daemon=True).start()
    return q


def _sse(prepared: Prepared, q: queue.Queue[ServerEvent | None]) -> Iterator[str]:
    try:
        while True:
            try:
                ev = q.get(timeout=HEARTBEAT_S)
            except queue.Empty:
                yield ": keep-alive\n\n"
                continue
            if ev is None:
                return
            yield ev.sse()
    finally:
        if prepared.ctx.deadline is not None and not prepared.ctx.deadline.expired:
            prepared.ctx.deadline.cancel("client_disconnected")


def build_app() -> FastAPI:  # uvicorn --factory northwind_assist.api.app:build_app
    return create_app()


__all__ = ["create_app", "build_app"]
