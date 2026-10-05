# path: book/projects/p2-semantic-search/semsearch/api/app.py
"""FastAPI service: POST /search, GET /healthz.

Identity arrives in headers set by the authenticating gateway in front of this service
(X-User, X-Tenant, X-Groups). The service must not be reachable except through that
gateway; otherwise any client could claim any group. Chapter 15 replaces this with
verified tokens.
"""
from __future__ import annotations

from typing import Annotated

from aie_core.observability import get_tracer
from aie_core.settings import Settings as CoreSettings
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from ..config import SearchSettings, embedder_dimensions, make_embedder, make_store, namespace_for
from ..domain.models import SearchHit
from ..service import Principal, SearchService


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int | None = Field(default=None, ge=1, le=200)
    tags: list[str] | None = Field(default=None, max_length=20)


class SearchResponse(BaseModel):
    namespace: str
    took_ms: float
    hits: list[SearchHit]


def get_principal(
    x_user: Annotated[str | None, Header()] = None,
    x_tenant: Annotated[str | None, Header()] = None,
    x_groups: Annotated[str | None, Header()] = None,
) -> Principal:
    if not x_user or not x_tenant or not x_groups:
        # Fail closed: no identity, no search. Never default to "all".
        raise HTTPException(status_code=401, detail="missing identity headers")
    groups = tuple(g.strip() for g in x_groups.split(",") if g.strip())
    if not groups:
        raise HTTPException(status_code=401, detail="empty group list")
    return Principal(user_id=x_user, tenant=x_tenant.strip(), groups=groups)


def build_service_from_env() -> SearchService:
    core, search = CoreSettings(), SearchSettings()
    embedder = make_embedder(core, search)
    store = make_store(embedder_dimensions(embedder), search)
    return SearchService(
        store,
        embedder,
        namespace_for(embedder, search),
        default_k=search.search_default_k,
        max_k=search.search_max_k,
        tracer=get_tracer(core),
    )


def get_service(request: Request) -> SearchService:
    """The service is injected by create_app (tests) or built lazily from the environment."""
    if request.app.state.service is None:
        request.app.state.service = build_service_from_env()
    return request.app.state.service


def create_app(service: SearchService | None = None) -> FastAPI:
    app = FastAPI(title="Northwind semantic search", version="0.1.0")
    app.state.service = service

    @app.post("/search", response_model=SearchResponse)
    def search(
        body: SearchRequest,
        principal: Annotated[Principal, Depends(get_principal)],
        svc: Annotated[SearchService, Depends(get_service)],
    ) -> SearchResponse:
        result = svc.search(body.query, principal, k=body.k, tags=body.tags)
        return SearchResponse(namespace=result.namespace, took_ms=result.took_ms, hits=result.hits)

    @app.get("/healthz")
    def healthz(svc: Annotated[SearchService, Depends(get_service)]) -> dict[str, object]:
        count = svc.store.count(svc.namespace)
        # An empty active namespace is unhealthy: the service would answer every query with nothing.
        if count == 0:
            raise HTTPException(status_code=503, detail=f"namespace {svc.namespace} is empty")
        return {"status": "ok", "namespace": svc.namespace, "chunks": count, "embedding_model": svc.embedder.model}

    return app


app = create_app()

__all__ = ["create_app", "app", "get_principal", "SearchRequest", "SearchResponse"]
