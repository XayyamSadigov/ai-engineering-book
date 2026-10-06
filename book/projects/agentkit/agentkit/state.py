# path: book/projects/agentkit/agentkit/state.py
"""AgentState: a projection of the event log.

`apply(state, event)` is the only function that changes state. The runtime calls it after
appending each event; `derive_state(events)` calls it in a loop to rebuild state for
resume, replay, and tests. If a field cannot be derived from events, it does not belong here.
"""
from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field

from aie_core.llm.types import Message, Role, ToolCall, ToolSpec

from .budget import BudgetUsage, TerminationReason
from .errors import ErrorClass
from .events import (
    BudgetUpdated, Event, FinalAnswer, GoalSet, ModelDecision, Note, Resumed, StepCompleted, Stopped,
    ToolCallApproved, ToolCallDenied, ToolCallRequested, ToolResult,
)


class AgentStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    AWAITING_APPROVAL = "awaiting_approval"


class CallRecord(BaseModel):
    request_id: str
    call_id: str
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    key: str
    step: int
    status: Literal["requested", "approved", "denied", "done"] = "requested"
    approved_by: str | None = None
    ok: bool | None = None                                           # set by the ToolResult


class Observation(BaseModel):
    step: int
    request_id: str
    tool: str
    content: str
    ok: bool
    error_class: ErrorClass | None = None
    truncated: bool = False


class AgentState(BaseModel):
    run_id: str = ""
    goal: str = ""
    system_prompt: str | None = None
    principal: dict[str, Any] = Field(default_factory=dict)
    tool_specs: list[ToolSpec] = Field(default_factory=list)
    status: AgentStatus = AgentStatus.RUNNING
    messages: list[Message] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    usage: BudgetUsage = Field(default_factory=BudgetUsage)
    final_answer: str | None = None
    stop_reason: TerminationReason | None = None
    stop_detail: str = ""
    calls: dict[str, CallRecord] = Field(default_factory=dict)
    key_counts: dict[str, int] = Field(default_factory=dict)       # executed calls per action key
    result_hashes: set[str] = Field(default_factory=set)
    step_had_progress: bool = False
    step_open: bool = False                                          # a ModelDecision without its StepCompleted
    open_tool_calls: list[ToolCall] = Field(default_factory=list)    # the open step's batch, for crash recovery
    open_final: tuple[str, str] | None = None                        # (text, finish_reason) not yet judged
    steps_without_progress: int = 0
    consecutive_errors: int = 0
    dod_rejections: int = 0
    last_seq: int = -1

    # ------------------------------------------------------------------ queries
    @property
    def step(self) -> int:
        return self.usage.steps

    @property
    def pending_calls(self) -> list[CallRecord]:
        """Requested calls with no outcome yet, in request order."""
        return [c for c in self.calls.values() if c.status in ("requested", "approved")]

    @property
    def pending_approval(self) -> CallRecord | None:
        if self.status is not AgentStatus.AWAITING_APPROVAL:
            return None
        return next((c for c in self.calls.values() if c.status == "requested"), None)

    def tools_called(self) -> set[str]:
        """Tools that ran and succeeded; a failed call is not evidence the work was done."""
        return {c.tool for c in self.calls.values() if c.status == "done" and c.ok}

    def observation_text(self) -> str:
        return "\n".join(o.content for o in self.observations if o.ok)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def apply(state: AgentState, event: Event) -> AgentState:
    """Fold one event into the state (in place) and return it."""
    if event.seq != state.last_seq + 1:
        raise ValueError(f"event seq {event.seq} does not follow {state.last_seq} in run {state.run_id!r}")
    state.last_seq = event.seq

    if isinstance(event, GoalSet):
        state.run_id = event.run_id
        state.goal = event.goal
        state.system_prompt = event.system_prompt
        state.principal = dict(event.principal)
        state.tool_specs = list(event.tool_specs)
        if event.system_prompt:
            state.messages.append(Message.system(event.system_prompt))
        state.messages.append(Message.user(event.goal))

    elif isinstance(event, ModelDecision):
        state.usage.steps += 1
        state.usage.input_tokens += event.usage.input_tokens
        state.usage.output_tokens += event.usage.output_tokens
        state.usage.cost_usd += event.cost_usd
        state.step_had_progress = False
        state.step_open = True
        state.open_tool_calls = list(event.tool_calls)
        state.open_final = (event.text, event.finish_reason) if event.kind == "final" else None
        state.messages.append(
            Message(role=Role.ASSISTANT, content=event.text, tool_calls=list(event.tool_calls) or None)
        )

    elif isinstance(event, ToolCallRequested):
        state.calls[event.request_id] = CallRecord(
            request_id=event.request_id, call_id=event.call_id, tool=event.tool,
            arguments=dict(event.arguments), key=event.key, step=event.step,
        )

    elif isinstance(event, ToolCallApproved):
        rec = state.calls[event.request_id]
        rec.status = "approved"
        rec.approved_by = event.by

    elif isinstance(event, ToolCallDenied):
        state.calls[event.request_id].status = "denied"
        state.consecutive_errors += 1
        state.messages.append(Message.tool(event.call_id, f"DENIED ({event.error_class.value}): {event.reason}"))

    elif isinstance(event, ToolResult):
        rec = state.calls[event.request_id]
        rec.status, rec.ok = "done", event.ok
        state.usage.tool_calls += 1
        state.key_counts[event.key] = state.key_counts.get(event.key, 0) + 1
        state.observations.append(
            Observation(step=event.step, request_id=event.request_id, tool=event.tool, content=event.content,
                        ok=event.ok, error_class=event.error_class, truncated=event.truncated)
        )
        text = event.content + (f"\n[harness] {event.notice}" if event.notice else "")
        state.messages.append(Message.tool(event.call_id, text))
        if event.ok:
            state.consecutive_errors = 0
            digest = _hash(event.content)
            if digest not in state.result_hashes:
                state.result_hashes.add(digest)
                state.step_had_progress = True
            for name, value in event.artifacts.items():
                if state.artifacts.get(name) != value:
                    state.artifacts[name] = value
                    state.step_had_progress = True
        else:
            state.consecutive_errors += 1

    elif isinstance(event, StepCompleted):
        state.step_open = False
        state.steps_without_progress = 0 if event.progress else state.steps_without_progress + 1

    elif isinstance(event, BudgetUpdated):
        state.usage.elapsed_s = event.usage.elapsed_s

    elif isinstance(event, Note):
        if event.kind == "dod_rejected":
            state.dod_rejections += 1
        if event.kind in ("dod_rejected", "truncated_answer"):
            state.open_final = None
        if event.to_model:
            state.messages.append(Message.user(f"[harness:{event.kind}] {event.text}"))

    elif isinstance(event, FinalAnswer):
        state.final_answer = event.text
        state.open_final = None

    elif isinstance(event, Resumed):
        state.status = AgentStatus.RUNNING
        state.stop_reason = None
        state.stop_detail = ""

    elif isinstance(event, Stopped):
        state.stop_reason = event.reason
        state.stop_detail = event.detail
        if event.reason is TerminationReason.COMPLETED:
            state.status = AgentStatus.COMPLETED
        elif event.reason is TerminationReason.APPROVAL_REQUIRED:
            state.status = AgentStatus.AWAITING_APPROVAL
        else:
            state.status = AgentStatus.STOPPED

    return state


def derive_state(events: Iterable[Event]) -> AgentState:
    state = AgentState()
    for event in events:
        apply(state, event)
    return state


__all__ = ["AgentStatus", "CallRecord", "Observation", "AgentState", "apply", "derive_state"]
