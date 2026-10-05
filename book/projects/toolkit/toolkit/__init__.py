# path: book/projects/toolkit/toolkit/__init__.py
"""toolkit: governed tool calling for the AI Engineering book (Chapter 16).

Import surface used by later chapters (19, 20, 22, capstone):

    from toolkit import (Tool, ToolRegistry, SideEffect, ToolContext, PolicyEngine, Verdict,
                         ToolExecutor, ToolResult, ToolError, ErrorCategory, ApprovalManager,
                         InMemoryIdempotencyStore, SQLiteIdempotencyStore, InMemoryAuditLog,
                         SandboxRunner, ToolLoop)
"""
from .approval import ApprovalError, ApprovalManager, ApprovalRequest, ApprovalStatus
from .audit import AuditEvent, AuditSink, InMemoryAuditLog, JsonlAuditLog, NullAuditLog
from .errors import ErrorCategory, ToolError
from .executor import BoundTool, ExecutionContext, ToolExecutor, ToolResult, truncate_payload
from .idempotency import IdempotencyRecord, IdempotencyStore, InMemoryIdempotencyStore, SQLiteIdempotencyStore
from .loop import LoopResult, StopReason, ToolLoop
from .policy import (ArgumentRule, Decision, PolicyEngine, RateLimit, ToolContext, ToolPolicy, Verdict,
                     max_value, recipient_allowlist)
from .registry import SideEffect, Tool, ToolRegistry, args_hash, canonical_json
from .sandbox import SandboxError, SandboxLimits, SandboxResult, SandboxRunner, make_python_tool

__version__ = "0.1.0"
__all__ = [
    "SideEffect", "Tool", "ToolRegistry", "args_hash", "canonical_json",
    "ToolContext", "Verdict", "Decision", "ArgumentRule", "RateLimit", "PolicyEngine", "ToolPolicy",
    "recipient_allowlist", "max_value",
    "ApprovalStatus", "ApprovalRequest", "ApprovalError", "ApprovalManager",
    "ErrorCategory", "ToolError",
    "IdempotencyRecord", "IdempotencyStore", "InMemoryIdempotencyStore", "SQLiteIdempotencyStore",
    "AuditEvent", "AuditSink", "InMemoryAuditLog", "JsonlAuditLog", "NullAuditLog",
    "BoundTool", "ExecutionContext", "ToolResult", "ToolExecutor", "truncate_payload",
    "SandboxLimits", "SandboxResult", "SandboxError", "SandboxRunner", "make_python_tool",
    "StopReason", "LoopResult", "ToolLoop",
]
