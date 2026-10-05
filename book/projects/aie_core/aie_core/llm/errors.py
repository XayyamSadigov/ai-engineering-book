# path: book/projects/aie_core/aie_core/llm/errors.py
"""Error taxonomy shared by every provider adapter.

The gateway makes decisions on *classes*, never on provider payloads: a retryable error
may be retried or routed to a fallback, a non-retryable one is raised immediately.
"""
from __future__ import annotations

import builtins
from typing import Any


class LLMError(Exception):
    """Base class. `retryable` drives retry and fallback; `retry_after_s` is a provider hint."""

    default_retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        retryable: bool | None = None,
        retry_after_s: float | None = None,
        status_code: int | None = None,
        provider: str | None = None,
        raw: Any = None,
    ) -> None:
        super().__init__(message)
        self.retryable = self.default_retryable if retryable is None else retryable
        self.retry_after_s = retry_after_s
        self.status_code = status_code
        self.provider = provider
        self.raw = raw

    def __str__(self) -> str:
        base = super().__str__()
        bits = []
        if self.provider:
            bits.append(f"provider={self.provider}")
        if self.status_code is not None:
            bits.append(f"status={self.status_code}")
        if self.retry_after_s is not None:
            bits.append(f"retry_after={self.retry_after_s:g}s")
        return f"{base} ({', '.join(bits)})" if bits else base


class RateLimitError(LLMError):
    """HTTP 429 or an equivalent quota signal. Retry after a delay, or fall back."""

    default_retryable = True


class TimeoutError(LLMError):  # noqa: A001 - name mandated by the shared API
    """The request exceeded its timeout or deadline. Retrying is safe only if the call is idempotent."""

    default_retryable = True


class ProviderUnavailableError(LLMError):
    """5xx, connection failures, overloaded signals. The request itself is fine."""

    default_retryable = True


class InvalidRequestError(LLMError):
    """4xx other than 429: bad schema, unknown model, bad auth. Retrying cannot help."""

    default_retryable = False


class ContentFilterError(LLMError):
    """The provider refused to generate or truncated for policy reasons."""

    default_retryable = False


class MalformedResponseError(LLMError):
    """The provider answered 2xx but the payload was unusable (bad JSON, missing fields)."""

    default_retryable = False


BuiltinTimeoutError = builtins.TimeoutError


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def map_http_error(
    status_code: int,
    body: Any,
    headers: dict[str, str] | None,
    provider: str,
) -> LLMError:
    """Translate an HTTP failure into the taxonomy. Shared by the OpenAI-style and Anthropic adapters."""
    headers = {k.lower(): v for k, v in (headers or {}).items()}
    message = _extract_message(body) or f"HTTP {status_code}"
    retry_after = _parse_retry_after(headers.get("retry-after"))
    common = {"status_code": status_code, "provider": provider, "raw": body}

    if status_code == 429:
        return RateLimitError(message, retry_after_s=retry_after, **common)
    if status_code == 408:
        return TimeoutError(message, retry_after_s=retry_after, **common)
    if status_code >= 500:
        return ProviderUnavailableError(message, retry_after_s=retry_after, **common)
    lowered = message.lower()
    if "content_filter" in lowered or "content filter" in lowered or "content management policy" in lowered:
        return ContentFilterError(message, **common)
    return InvalidRequestError(message, **common)


def _extract_message(body: Any) -> str | None:
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            parts = [str(err.get(k)) for k in ("type", "code", "message") if err.get(k)]
            return ": ".join(parts) if parts else None
        if isinstance(err, str):
            return err
        if "message" in body:
            return str(body["message"])
        return None
    if isinstance(body, str) and body.strip():
        return body.strip()[:500]
    return None


__all__ = [
    "LLMError",
    "RateLimitError",
    "TimeoutError",
    "ProviderUnavailableError",
    "InvalidRequestError",
    "ContentFilterError",
    "MalformedResponseError",
    "BuiltinTimeoutError",
    "map_http_error",
]
