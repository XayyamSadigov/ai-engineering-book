# path: book/projects/toolkit/toolkit/loop.py
"""A bounded tool-calling loop over any aie_core LLMClient.

The loop owns the protocol (Chapter 3): send ToolSpecs, replay the assistant message with
its tool calls verbatim, answer every call with a tool message, call again. All authority
lives in the executor; the loop never runs a tool itself. Chapter 19 grows this into a
full agent runtime with an event log, budgets, and replay.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any, Literal

from aie_core.llm.client import LLMClient
from aie_core.llm.types import CompletionRequest, Message, ToolCall, Usage
from aie_core.observability import NoopTracer, Tracer
from pydantic import BaseModel, Field

from .errors import ToolError
from .executor import ToolExecutor, ToolResult
from .policy import ToolContext
from .registry import args_hash

StopReason = Literal["final", "pending_approval", "max_rounds", "repeated_call", "fatal_tool_error"]


class LoopResult(BaseModel):
    final_text: str
    stop_reason: StopReason
    rounds: int
    messages: list[Message]
    tool_results: list[ToolResult] = Field(default_factory=list)
    pending_approvals: list[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)


class ToolLoop:
    def __init__(
        self,
        llm: LLMClient,
        executor: ToolExecutor,
        *,
        max_rounds: int = 6,
        max_calls_per_round: int = 8,
        max_identical_calls: int = 2,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        tracer: Tracer | None = None,
    ) -> None:
        self.llm = llm
        self.executor = executor
        self.max_rounds = max_rounds
        self.max_calls_per_round = max_calls_per_round
        self.max_identical_calls = max_identical_calls
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tracer = tracer or NoopTracer()

    def run(self, messages: list[Message], ctx: ToolContext, *, tool_names: list[str] | None = None,
            task_tags: list[str] | None = None, metadata: dict[str, Any] | None = None) -> LoopResult:
        history = list(messages)
        filters: dict[str, Any] = {}
        if tool_names is not None:
            filters["names"] = tool_names
        if task_tags is not None:
            filters["task_tags"] = task_tags
        specs = self.executor.visible_specs(ctx, **filters)
        allowed = {s.name for s in specs}
        results: list[ToolResult] = []
        pending: list[str] = []
        seen: Counter[str] = Counter()
        usage = Usage()

        def finish(text: str, reason: StopReason, rounds: int) -> LoopResult:
            return LoopResult(final_text=text, stop_reason=reason, rounds=rounds, messages=history,
                              tool_results=results, pending_approvals=pending, usage=usage)

        for round_no in range(1, self.max_rounds + 1):
            # After an approval request, force a text turn so the model reports the wait.
            choice = "none" if pending else "auto"
            req = CompletionRequest(messages=history, tools=specs or None, tool_choice=choice if specs else "auto",
                                    model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
                                    metadata={**(metadata or {}), "round": round_no, "user_id": ctx.user_id})
            with self.tracer.span("tool_loop.round", round=round_no, tools=len(specs)):
                completion = self.llm.complete(req)
            usage = usage + completion.usage
            calls = completion.tool_calls
            if pending or not calls:
                history.append(completion.message)
                return finish(completion.text, "pending_approval" if pending else "final", round_no)

            history.append(completion.message)  # replay verbatim, tool calls included
            stop: StopReason | None = None
            for i, call in enumerate(calls):
                if i >= self.max_calls_per_round:
                    result = self._synthetic_error(call, "too_many_calls",
                                                   f"at most {self.max_calls_per_round} tool calls per turn")
                elif call.name not in allowed:
                    # Not offered to this user/task: treat like an unknown tool, never execute.
                    result = self._synthetic_error(call, "tool_not_available",
                                                   f"tool '{call.name}' is not available in this context")
                else:
                    signature = args_hash(call.name, call.arguments)
                    seen[signature] += 1
                    if seen[signature] > self.max_identical_calls:
                        result = self._synthetic_error(call, "repeated_call",
                                                       "identical call repeated; change approach or answer")
                        stop = "repeated_call"
                    else:
                        result = self.executor.execute(call, ctx)
                results.append(result)
                history.append(result.to_message())
                if result.status == "pending_approval" and result.approval_id:
                    pending.append(result.approval_id)
                if result.error_category == "fatal" and result.error and result.error.get("code") == "outcome_unknown":
                    stop = "fatal_tool_error"
            if stop is not None:
                return finish("", stop, round_no)
        return finish("", "max_rounds", self.max_rounds)

    @staticmethod
    def _synthetic_error(call: ToolCall, code: str, message: str) -> ToolResult:
        err = ToolError.validation(code, message) if code != "tool_not_available" else ToolError.not_found(code, message)
        return ToolResult(call_id=call.id, tool_name=call.name, status="error",
                          content=json.dumps({"ok": False, "error": err.to_dict()}), error=err.to_dict())


__all__ = ["StopReason", "LoopResult", "ToolLoop"]
