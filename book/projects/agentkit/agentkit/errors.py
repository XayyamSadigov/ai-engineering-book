# path: book/projects/agentkit/agentkit/errors.py
"""Error classes for the agent loop and the classifier that maps exceptions onto them.

The runtime decides what to do with a failure by its class, never by its message:
transient errors may be retried, validation errors go back to the model for repair,
permission errors are never retried, impossible tasks end in an honest stop, and
fatal errors stop the run because they are bugs in our code, not in the model.
"""
from __future__ import annotations

import builtins
from enum import Enum

from pydantic import ValidationError

from aie_core.llm.errors import LLMError


class ErrorClass(str, Enum):
    TRANSIENT = "transient"      # timeout, rate limit, 5xx: retry with a bound
    VALIDATION = "validation"    # bad arguments or bad output shape: repair with the error in context
    SEMANTIC = "semantic"        # valid shape, wrong content: caught by a verifier, never raised by tools
    PERMISSION = "permission"    # policy said no: never retry, explain or escalate
    IMPOSSIBLE = "impossible"    # the data or capability does not exist: stop honestly
    FATAL = "fatal"              # bug or misconfiguration in our code: stop, keep state, alert


class ToolError(Exception):
    """Base class tools may raise to state their failure class explicitly."""

    error_class: ErrorClass = ErrorClass.FATAL


class TransientToolError(ToolError):
    error_class = ErrorClass.TRANSIENT


class ToolValidationError(ToolError):
    error_class = ErrorClass.VALIDATION


class ToolPermissionError(ToolError):
    error_class = ErrorClass.PERMISSION


class ImpossibleTaskError(ToolError):
    error_class = ErrorClass.IMPOSSIBLE


def classify_error(exc: BaseException) -> ErrorClass:
    """Map an exception raised while executing a tool to an ErrorClass."""
    if isinstance(exc, ToolError):
        return exc.error_class
    if isinstance(exc, LLMError):  # a tool that itself calls a model
        return ErrorClass.TRANSIENT if exc.retryable else ErrorClass.FATAL
    if isinstance(exc, PermissionError):
        return ErrorClass.PERMISSION
    if isinstance(exc, (ValidationError, ValueError)):
        return ErrorClass.VALIDATION
    if isinstance(exc, (builtins.TimeoutError, ConnectionError)):
        return ErrorClass.TRANSIENT
    if isinstance(exc, FileNotFoundError):
        return ErrorClass.IMPOSSIBLE
    # KeyError, TypeError, AttributeError and friends are almost always bugs in the tool.
    return ErrorClass.FATAL


__all__ = [
    "ErrorClass",
    "ToolError",
    "TransientToolError",
    "ToolValidationError",
    "ToolPermissionError",
    "ImpossibleTaskError",
    "classify_error",
]
