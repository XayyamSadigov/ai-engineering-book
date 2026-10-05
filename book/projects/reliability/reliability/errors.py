# path: book/projects/reliability/reliability/errors.py
"""Reliability errors, expressed in the aie_core taxonomy.

Every error the reliability layer raises is an `LLMError` subclass, so callers keep one
decision rule everywhere: read `.retryable` and `.retry_after_s`, never parse messages.
The classes below only add meaning that the base taxonomy lacks:

- `DeadlineExceeded` is a TimeoutError that must never be retried: the budget is gone.
- `CircuitOpenError` is a fast failure from a breaker. It is retryable *later*, and its
  `retry_after_s` is the time until the breaker lets a probe through. A ModelGateway that
  receives it moves to its next client when that hint does not fit the deadline.
- `Overloaded` is a deliberate rejection by our own system (bulkhead full, quota spent,
  load shed). It maps to HTTP 429 (quota) or 503 (shed) at the API edge.
"""
from __future__ import annotations

from aie_core.llm.errors import LLMError, ProviderUnavailableError, RateLimitError
from aie_core.llm.errors import TimeoutError as LLMTimeoutError


class DeadlineExceeded(LLMTimeoutError):
    """The request's end-to-end budget is spent. Retrying cannot help."""

    default_retryable = False

    def __init__(self, message: str = "deadline exceeded", *, stage: str | None = None, **kw: object) -> None:
        super().__init__(message if stage is None else f"{message} at stage '{stage}'", **kw)  # type: ignore[arg-type]
        self.stage = stage


class CircuitOpenError(ProviderUnavailableError):
    """A breaker refused the call without contacting the dependency."""

    default_retryable = True

    def __init__(self, dependency: str, retry_after_s: float) -> None:
        super().__init__(f"circuit '{dependency}' is open", retry_after_s=retry_after_s, provider=dependency)
        self.dependency = dependency


class Overloaded(RateLimitError):
    """Our own capacity protection rejected the work. `reason` is a stable machine-readable code."""

    default_retryable = True

    def __init__(self, message: str, *, reason: str, retry_after_s: float | None = None) -> None:
        super().__init__(message, retry_after_s=retry_after_s)
        self.reason = reason


class BulkheadFullError(Overloaded):
    """The workload's concurrency pool and its waiting room are both full."""


class AdmissionRejected(Overloaded):
    """The admission controller rejected or deferred the request (quota, overload, deadline)."""


def is_retryable(exc: BaseException) -> bool:
    """One classification rule for every dependency, LLM or not.

    Errors that carry a `retryable` attribute decide for themselves. Plain connection
    failures and builtin timeouts are transient. Everything else (ValueError, KeyError,
    validation errors, bugs) is not, because retrying a deterministic failure only burns
    budget.
    """
    flag = getattr(exc, "retryable", None)
    if isinstance(flag, bool):
        return flag
    return isinstance(exc, (ConnectionError, TimeoutError))  # builtins here, not the aie_core class


def retry_after(exc: BaseException) -> float | None:
    value = getattr(exc, "retry_after_s", None)
    return float(value) if isinstance(value, (int, float)) else None


__all__ = [
    "LLMError",
    "DeadlineExceeded",
    "CircuitOpenError",
    "Overloaded",
    "BulkheadFullError",
    "AdmissionRejected",
    "is_retryable",
    "retry_after",
]
