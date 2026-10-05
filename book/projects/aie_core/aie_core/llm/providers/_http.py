# path: book/projects/aie_core/aie_core/llm/providers/_http.py
"""httpx plumbing shared by the real adapters: client construction and exception mapping."""
from __future__ import annotations

import json
from typing import Any

import httpx

from ..errors import (
    LLMError,
    MalformedResponseError,
    ProviderUnavailableError,
    TimeoutError,
    map_http_error,
)


def make_sync_client(base_url: str, headers: dict[str, str], timeout_s: float, transport: httpx.BaseTransport | None) -> httpx.Client:
    return httpx.Client(base_url=base_url, headers=headers, timeout=timeout_s, transport=transport)


def make_async_client(base_url: str, headers: dict[str, str], timeout_s: float, transport: httpx.AsyncBaseTransport | None) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=base_url, headers=headers, timeout=timeout_s, transport=transport)


def map_transport_error(exc: Exception, provider: str) -> LLMError:
    """Network-level failures. Timeouts and connection errors are retryable; the request was fine."""
    if isinstance(exc, httpx.TimeoutException):
        return TimeoutError(f"request timed out: {exc}", provider=provider)
    if isinstance(exc, (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError, httpx.PoolTimeout)):
        return ProviderUnavailableError(f"transport failure: {exc}", provider=provider)
    if isinstance(exc, httpx.HTTPError):
        return ProviderUnavailableError(f"http error: {exc}", provider=provider)
    raise exc


def raise_for_status(response: httpx.Response, provider: str) -> None:
    if response.is_success:
        return
    try:
        body: Any = response.json()
    except ValueError:
        body = response.text
    raise map_http_error(response.status_code, body, dict(response.headers), provider)


def parse_json_body(response: httpx.Response, provider: str) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError as exc:
        raise MalformedResponseError(f"response body is not JSON: {exc}", provider=provider, raw=response.text) from exc
    if not isinstance(data, dict):
        raise MalformedResponseError("response body is not a JSON object", provider=provider, raw=data)
    return data


def parse_json_arguments(raw: str | None, provider: str, tool_name: str) -> dict[str, Any]:
    """Tool-call arguments arrive as a JSON *string*. Empty means `{}`; anything else must parse."""
    if raw is None or raw.strip() == "":
        return {}
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise MalformedResponseError(
            f"tool call `{tool_name}` has invalid JSON arguments: {exc}", provider=provider, raw=raw
        ) from exc
    if not isinstance(value, dict):
        raise MalformedResponseError(
            f"tool call `{tool_name}` arguments must be a JSON object", provider=provider, raw=raw
        )
    return value


def timeout_for(req_timeout_s: float | None, default_timeout_s: float) -> httpx.Timeout:
    total = req_timeout_s if req_timeout_s is not None else default_timeout_s
    # Connect quickly or fail; reads may wait for the full budget (streams idle between tokens).
    return httpx.Timeout(total, connect=min(10.0, total))
