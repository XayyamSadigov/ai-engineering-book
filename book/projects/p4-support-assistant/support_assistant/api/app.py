# path: book/projects/p4-support-assistant/support_assistant/api/app.py
"""FastAPI service: chat, approvals, health.

Identity arrives in the X-User-Id header set by the authenticating gateway in front of
this service. The service must not be reachable except through that gateway. The
approval endpoints are the only way to approve: the model has no tool that can do it.
"""
# No `from __future__ import annotations`: FastAPI must resolve the local Annotated aliases.

from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from toolkit import ApprovalError, ApprovalRequest

from ..assistant import ChatResponse, NotAuthorized, SupportAssistant, UnknownUser
from ..wiring import build_container


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=64)


class DecisionRequest(BaseModel):
    note: str | None = Field(default=None, max_length=500)


class DecisionResponse(BaseModel):
    approval_id: str
    status: str
    result: dict | None = None


def create_app(assistant: SupportAssistant | None = None) -> FastAPI:
    app = FastAPI(title="Northwind support assistant (Project 4)")
    app.state.assistant = assistant or SupportAssistant(build_container())

    def get_assistant() -> SupportAssistant:
        return app.state.assistant

    def user_id(x_user_id: Annotated[str | None, Header()] = None) -> str:
        if not x_user_id:
            raise HTTPException(401, "missing X-User-Id")
        return x_user_id

    Assistant = Annotated[SupportAssistant, Depends(get_assistant)]
    User = Annotated[str, Depends(user_id)]

    def guard(fn):
        try:
            return fn()
        except UnknownUser:
            raise HTTPException(401, "unknown user") from None
        except NotAuthorized as exc:
            raise HTTPException(403, str(exc)) from None
        except ApprovalError as exc:
            status = 404 if exc.code == "approval_not_found" else 409
            raise HTTPException(status, {"code": exc.code, "message": exc.message}) from None

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatResponse)
    def chat(body: ChatRequest, a: Assistant, uid: User) -> ChatResponse:
        return guard(lambda: a.chat(uid, body.message, body.session_id))

    @app.get("/approvals", response_model=list[ApprovalRequest])
    def pending(a: Assistant, uid: User) -> list[ApprovalRequest]:
        return guard(lambda: a.pending(uid))

    @app.post("/approvals/{approval_id}/approve", response_model=DecisionResponse)
    def approve(approval_id: str, body: DecisionRequest, a: Assistant, uid: User) -> DecisionResponse:
        result = guard(lambda: a.approve(approval_id, uid, body.note))
        return DecisionResponse(approval_id=approval_id, status=result.status,
                                result={"data": result.data, "error": result.error, "duplicate": result.duplicate})

    @app.post("/approvals/{approval_id}/reject", response_model=DecisionResponse)
    def reject(approval_id: str, body: DecisionRequest, a: Assistant, uid: User) -> DecisionResponse:
        req = guard(lambda: a.reject(approval_id, uid, body.note))
        return DecisionResponse(approval_id=approval_id, status=req.status.value)

    return app


app = create_app()
