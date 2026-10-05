# path: book/projects/agentkit/agentkit/budget.py
"""Budgets: hard limits on what one agent run may consume, and the usage counters they check.

A budget is checked *before* spending (can the next model call fit?) and *after* spending
(did the last step cross a line?). The second check alone lets a single large call
overshoot; the first check alone misses costs that are only known after the call.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class TerminationReason(str, Enum):
    COMPLETED = "completed"                    # final answer passed the Definition of Done
    MAX_STEPS = "max_steps"
    MAX_TOKENS = "max_tokens"
    MAX_COST = "max_cost"
    MAX_TOOL_CALLS = "max_tool_calls"
    DEADLINE = "deadline"
    REPEATED_ACTION = "repeated_action"        # identical tool call requested too many times
    NO_PROGRESS = "no_progress"                # several steps without new information
    TOOL_ERRORS = "tool_errors"                # too many consecutive failed tool calls or denials
    VERIFICATION_FAILED = "verification_failed"  # final answers kept failing the Definition of Done
    APPROVAL_REQUIRED = "approval_required"    # paused: a human must approve a pending call
    MODEL_ERROR = "model_error"                # the model call failed after the gateway's retries
    FATAL_ERROR = "fatal_error"                # a tool raised an error classified as a bug

    @property
    def is_budget(self) -> bool:
        return self in _BUDGET_REASONS

    @property
    def is_success(self) -> bool:
        return self is TerminationReason.COMPLETED


_BUDGET_REASONS = {
    TerminationReason.MAX_STEPS,
    TerminationReason.MAX_TOKENS,
    TerminationReason.MAX_COST,
    TerminationReason.MAX_TOOL_CALLS,
    TerminationReason.DEADLINE,
}


class BudgetUsage(BaseModel):
    """What a run has consumed so far. Derived from events; never edited by hand."""

    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    tool_calls: int = 0
    elapsed_s: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class Budget(BaseModel):
    """Limits for one run. `None` means unlimited; `max_steps` is always set."""

    model_config = ConfigDict(frozen=True)

    max_steps: int = Field(default=10, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_cost_usd: float | None = Field(default=None, gt=0)
    deadline_s: float | None = Field(default=None, gt=0)
    max_tool_calls: int | None = Field(default=None, ge=0)

    def exceeded(self, usage: BudgetUsage, elapsed_s: float | None = None) -> TerminationReason | None:
        """Post-spend check: has any limit been reached? Order is cheapest-to-explain first."""
        elapsed = usage.elapsed_s if elapsed_s is None else elapsed_s
        if usage.steps >= self.max_steps:
            return TerminationReason.MAX_STEPS
        if self.max_tokens is not None and usage.total_tokens >= self.max_tokens:
            return TerminationReason.MAX_TOKENS
        if self.max_cost_usd is not None and usage.cost_usd >= self.max_cost_usd:
            return TerminationReason.MAX_COST
        if self.deadline_s is not None and elapsed >= self.deadline_s:
            return TerminationReason.DEADLINE
        return None

    def admits_model_call(self, usage: BudgetUsage, estimated_tokens: int) -> bool:
        """Pre-spend check: would a call of this estimated size fit in the token budget?"""
        if self.max_tokens is None:
            return True
        return usage.total_tokens + estimated_tokens <= self.max_tokens

    def admits_tool_call(self, usage: BudgetUsage) -> bool:
        return self.max_tool_calls is None or usage.tool_calls < self.max_tool_calls

    def remaining(self, usage: BudgetUsage) -> dict[str, float | int | None]:
        return {
            "steps": self.max_steps - usage.steps,
            "tokens": None if self.max_tokens is None else self.max_tokens - usage.total_tokens,
            "cost_usd": None if self.max_cost_usd is None else round(self.max_cost_usd - usage.cost_usd, 6),
            "tool_calls": None if self.max_tool_calls is None else self.max_tool_calls - usage.tool_calls,
            "seconds": None if self.deadline_s is None else round(self.deadline_s - usage.elapsed_s, 3),
        }


__all__ = ["TerminationReason", "BudgetUsage", "Budget"]
