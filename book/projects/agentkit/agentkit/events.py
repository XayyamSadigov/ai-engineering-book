# path: book/projects/agentkit/agentkit/events.py
"""Immutable events: the source of truth for an agent run.

The runtime never edits state directly. It appends events, and `state.apply` folds each
event into the current `AgentState`. Replay, resume after a crash, audit, and trajectory
tests all read the same log, so they cannot disagree with what actually happened.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from aie_core.llm.types import ToolCall, ToolSpec, Usage

from .budget import BudgetUsage, TerminationReason
from .errors import ErrorClass


class Event(BaseModel):
    """Fields every event carries. `seq` is dense per run and assigned by the runtime."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    seq: int
    step: int = 0
    at: float = Field(default_factory=time.time)


class GoalSet(Event):
    type: Literal["goal_set"] = "goal_set"
    goal: str
    system_prompt: str | None = None
    tool_specs: list[ToolSpec] = Field(default_factory=list)
    budget: dict[str, Any] = Field(default_factory=dict)
    principal: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelDecision(Event):
    """One model call and what it proposed: tool calls, or a candidate final answer."""

    type: Literal["model_decision"] = "model_decision"
    kind: Literal["tool_calls", "final"]
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    cost_usd: float = 0.0
    model: str = ""
    finish_reason: str = ""
    latency_ms: float = 0.0
    request_hash: str = ""


class ToolCallRequested(Event):
    type: Literal["tool_call_requested"] = "tool_call_requested"
    request_id: str          # unique in the run: "<step>.<index>"
    call_id: str             # the model's id, echoed back in the tool message
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    key: str                 # hash of tool name + canonical arguments; replay and loop detection use it


class ToolCallApproved(Event):
    type: Literal["tool_call_approved"] = "tool_call_approved"
    request_id: str
    tool: str
    by: Literal["policy", "approver", "human"] = "policy"
    reason: str = ""


class ToolCallDenied(Event):
    type: Literal["tool_call_denied"] = "tool_call_denied"
    request_id: str
    call_id: str
    tool: str
    reason: str
    error_class: ErrorClass = ErrorClass.PERMISSION
    by: Literal["policy", "approver", "human", "runtime"] = "policy"


class ToolResult(Event):
    type: Literal["tool_result"] = "tool_result"
    request_id: str
    call_id: str
    tool: str
    key: str
    ok: bool
    content: str                       # what the model sees, already truncated
    original_chars: int = 0
    truncated: bool = False
    data: Any = None                   # structured payload for code, not for the prompt
    artifacts: dict[str, Any] = Field(default_factory=dict)
    error_class: ErrorClass | None = None
    error: str | None = None
    notice: str = ""                   # harness warning appended to the tool message, kept separate for replay
    attempts: int = 1
    latency_ms: float = 0.0
    idempotency_key: str = ""


class BudgetUpdated(Event):
    type: Literal["budget_updated"] = "budget_updated"
    usage: BudgetUsage


class StepCompleted(Event):
    """Closes one loop iteration and records whether it produced new information."""

    type: Literal["step_completed"] = "step_completed"
    progress: bool
    detail: str = ""


class Note(Event):
    """Harness commentary. `to_model=True` notes become user messages the model reads."""

    type: Literal["note"] = "note"
    kind: str                          # e.g. "dod_rejected", "loop_warning", "approval_required"
    text: str
    to_model: bool = False
    error_class: ErrorClass | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class FinalAnswer(Event):
    type: Literal["final_answer"] = "final_answer"
    text: str
    checks: list[dict[str, Any]] = Field(default_factory=list)


class Resumed(Event):
    type: Literal["resumed"] = "resumed"
    by: str = "human"
    note: str = ""
    budget: dict[str, Any] | None = None  # set when an operator extends the budget; replay reads it


class Stopped(Event):
    type: Literal["stopped"] = "stopped"
    reason: TerminationReason
    detail: str = ""


AnyEvent = Annotated[
    Union[
        GoalSet, ModelDecision, ToolCallRequested, ToolCallApproved, ToolCallDenied, ToolResult,
        BudgetUpdated, StepCompleted, Note, FinalAnswer, Resumed, Stopped,
    ],
    Field(discriminator="type"),
]
_ADAPTER: TypeAdapter[Any] = TypeAdapter(AnyEvent)


def event_to_json(event: Event) -> str:
    return event.model_dump_json()


def event_from_json(line: str) -> Event:
    return _ADAPTER.validate_json(line)


def event_from_dict(data: dict[str, Any]) -> Event:
    return _ADAPTER.validate_python(data)


def action_key(tool: str, arguments: dict[str, Any]) -> str:
    """Stable identity of a tool call: same tool and same arguments give the same key."""
    canonical = json.dumps({"tool": tool, "args": arguments}, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "Event", "GoalSet", "ModelDecision", "ToolCallRequested", "ToolCallApproved", "ToolCallDenied",
    "ToolResult", "BudgetUpdated", "StepCompleted", "Note", "FinalAnswer", "Resumed", "Stopped",
    "AnyEvent", "event_to_json", "event_from_json", "event_from_dict", "action_key",
]
