# path: book/projects/p5-incident-agent/incident_agent/api.py
"""HTTP API. Identity comes from the X-User header, standing in for your auth proxy; the server
maps it to trusted groups and tenant. Nothing the client sends can widen its own permissions.

    POST /investigations                  {"alert_id": "..."}             -> 201 Investigation
    GET  /investigations/{id}                                              -> Investigation
    POST /investigations/{id}/decision    {"approve": true, "reason": ""} -> Investigation
"""
from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .domain.models import Investigation
from .service import IncidentService, InvalidState, NotAllowed


class InvestigateIn(BaseModel):
    alert_id: str = Field(min_length=3, max_length=64)


class DecisionIn(BaseModel):
    approve: bool
    reason: str = Field(default="", max_length=500)


@lru_cache(maxsize=1)
def get_service() -> IncidentService:
    return IncidentService()


Svc = Annotated[IncidentService, Depends(get_service)]
User = Annotated[str, Header(alias="X-User")]


def create_app() -> FastAPI:
    app = FastAPI(title="Northwind incident-research agent", version="0.1.0")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/investigations", status_code=201, response_model=Investigation)
    def investigate(body: InvestigateIn, svc: Svc, x_user: User) -> Investigation:
        try:
            return svc.investigate(body.alert_id, x_user)
        except NotAllowed as exc:
            raise HTTPException(403, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, f"unknown alert {body.alert_id}") from exc

    @app.get("/investigations/{inv_id}", response_model=Investigation)
    def show(inv_id: str, svc: Svc, x_user: User) -> Investigation:
        try:
            principal = svc.principal(x_user)
            inv = svc.get(inv_id)
        except NotAllowed as exc:
            raise HTTPException(403, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, "not found") from exc
        if principal["tenant"] != inv.alert.tenant:
            raise HTTPException(404, "not found")       # do not confirm existence across tenants
        return inv

    @app.post("/investigations/{inv_id}/decision", response_model=Investigation)
    def decide(inv_id: str, body: DecisionIn, svc: Svc, x_user: User) -> Investigation:
        try:
            return svc.decide(inv_id, x_user, approve=body.approve, reason=body.reason)
        except NotAllowed as exc:
            raise HTTPException(403, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, "not found") from exc
        except InvalidState as exc:
            raise HTTPException(409, str(exc)) from exc

    return app


app = create_app()
