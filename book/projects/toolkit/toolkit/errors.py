# path: book/projects/toolkit/toolkit/errors.py
"""A machine-readable error contract for tools.

The model, the loop, and the dashboards all branch on `category`. Five values are enough:
each one implies a different recovery, and adding a sixth should require a new recovery.

| category    | meaning                                  | who recovers            | retried by executor |
|-------------|------------------------------------------|-------------------------|---------------------|
| validation  | arguments are wrong                      | the model, by repairing | no                  |
| permission  | caller may not do this                   | nobody automatically    | no                  |
| not_found   | the target does not exist                | the model, by asking    | no                  |
| transient   | try again later may work                 | the executor, bounded   | yes                 |
| fatal       | broken or unknown outcome                | a human / on-call       | no                  |
"""
from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorCategory(str, Enum):
    VALIDATION = "validation"
    PERMISSION = "permission"
    NOT_FOUND = "not_found"
    TRANSIENT = "transient"
    FATAL = "fatal"


class ToolError(Exception):
    """Raise from handlers (or let the executor raise) to report a classified failure.

    Handlers of non-idempotent tools must only raise `transient` when they know the side
    effect did *not* happen (for example the connection was refused before sending).
    """

    def __init__(self, category: ErrorCategory | str, code: str, message: str, *,
                 retry_after_s: float | None = None, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.category = ErrorCategory(category)
        self.code = code
        self.message = message
        self.retry_after_s = retry_after_s
        self.details = details or {}

    @property
    def retryable(self) -> bool:
        return self.category == ErrorCategory.TRANSIENT

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"category": self.category.value, "code": self.code,
                               "message": self.message, "retryable": self.retryable}
        if self.retry_after_s is not None:
            out["retry_after_s"] = round(self.retry_after_s, 3)
        if self.details:
            out["details"] = self.details
        return out

    # Convenience constructors keep handler code short and the codes consistent.
    @classmethod
    def validation(cls, code: str, message: str, **kw: Any) -> "ToolError":
        return cls(ErrorCategory.VALIDATION, code, message, **kw)

    @classmethod
    def permission(cls, code: str, message: str, **kw: Any) -> "ToolError":
        return cls(ErrorCategory.PERMISSION, code, message, **kw)

    @classmethod
    def not_found(cls, code: str, message: str, **kw: Any) -> "ToolError":
        return cls(ErrorCategory.NOT_FOUND, code, message, **kw)

    @classmethod
    def transient(cls, code: str, message: str, **kw: Any) -> "ToolError":
        return cls(ErrorCategory.TRANSIENT, code, message, **kw)

    @classmethod
    def fatal(cls, code: str, message: str, **kw: Any) -> "ToolError":
        return cls(ErrorCategory.FATAL, code, message, **kw)


__all__ = ["ErrorCategory", "ToolError"]
