# path: book/projects/examples/ch32/northwind_triage/adapters/http_api.py
"""Inbound adapter: HTTP. Translates JSON to domain types and back; no business logic."""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Header
from pydantic import BaseModel

from ..application import TriageService
from ..domain import Ticket


class TriageRequest(BaseModel):
    ticket: Ticket


def create_app(service: TriageService) -> FastAPI:
    app = FastAPI(title="Northwind triage")

    @app.post("/v1/triage")
    def triage(body: TriageRequest, x_user_id: str = Header(...)) -> dict[str, Any]:
        # X-User-Id stands in for the verified identity an auth layer provides (Chapter 28's
        # RequestContext); never take it from an unauthenticated client in production, or callers
        # can pick their own flag bucket or a QA override. Units are scoped by tenant.
        result = service.triage(body.ticket, unit_id=f"{body.ticket.tenant}:{x_user_id}")
        return {
            "triage": result.triage.model_dump(mode="json"),
            "outcome": result.outcome,
            "version_fingerprint": result.manifest.fingerprint(),
        }

    @app.get("/v1/version")
    def version() -> dict[str, Any]:
        # The static part of the manifest: what this deployment is running.
        m = service.base_manifest
        return {"fingerprint": m.fingerprint(), "manifest": m.model_dump(mode="json")}

    return app
