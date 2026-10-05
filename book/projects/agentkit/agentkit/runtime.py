# path: book/projects/agentkit/agentkit/runtime.py
"""AgentRuntime: a bounded, event-sourced agent loop over an aie_core LLMClient.

One iteration ("step") is: check limits, call the model once, then either process the
proposed tool calls (validate, detect repeats, apply policy and approval, execute, shape
the observation) or verify the candidate final answer against the Definition of Done.
Every decision is an event; state is derived from events; nothing else is mutated.
"""
from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from pydantic import BaseModel, ConfigDict

from aie_core.llm.client import LLMClient
from aie_core.llm.errors import LLMError
from aie_core.llm.gateway import PricingTable
from aie_core.llm.tokens import count_message_tokens
from aie_core.llm.types import CompletionRequest
from aie_core.observability import NoopTracer, Tracer

from .budget import Budget, TerminationReason
from .dod import DefinitionOfDone
from .errors import ErrorClass, classify_error
from .events import (
    BudgetUpdated, Event, FinalAnswer, GoalSet, ModelDecision, Note, Resumed, StepCompleted, Stopped,
    ToolCallApproved, ToolCallDenied, ToolCallRequested, ToolResult, action_key,
)
from .observations import truncate_observation
from .state import AgentState, AgentStatus, CallRecord, apply, derive_state
from .store import EventStore, InMemoryEventStore
from .tools import DefaultPolicy, Tool, ToolContext, ToolOutput, ToolPolicy, adapt_tool, validate_arguments

DEFAULT_SYSTEM_PROMPT = (
    "You are an agent working for Northwind. Use the available tools to gather evidence and act. "
    "Call tools only when they move the task forward; never repeat a call whose result you already have. "
    "When you have enough evidence, reply with the final answer and no tool calls. "
    "If the task cannot be completed with the available tools, say so and state what is missing."
)

Approver = Callable[[CallRecord, AgentState], "bool | None"]


class LoopConfig(BaseModel):
    """Loop-control knobs that are not budgets. Defaults are illustrative starting points."""

    model_config = ConfigDict(frozen=True)

    max_observation_chars: int = 4000      # per tool result, applied when the result is recorded
    max_identical_calls: int = 2           # same tool + same arguments may execute at most this often
    max_no_progress_steps: int = 3         # consecutive steps without new information
    max_consecutive_errors: int = 3        # failed or denied tool calls in a row
    max_dod_rejections: int = 2            # rejected final answers tolerated before giving up
    transient_retries: int = 1             # extra attempts for transient errors on idempotent tools
    max_tokens_per_call: int = 1024
    temperature: float = 0.0
    dod_in_prompt: bool = True             # append the Definition of Done to the system prompt


@dataclass
class RunResult:
    run_id: str
    status: AgentStatus
    stop_reason: TerminationReason | None
    detail: str
    final_answer: str | None
    state: AgentState
    events: list[Event]

    @property
    def ok(self) -> bool:
        return self.stop_reason is TerminationReason.COMPLETED

    def trajectory(self) -> list[str]:
        """Executed tool names in order; the backbone of trajectory tests."""
        return [e.tool for e in self.events if isinstance(e, ToolResult)]

    def events_of(self, kind: type[Event]) -> list[Any]:
        return [e for e in self.events if isinstance(e, kind)]


@dataclass
class _Session:
    run_id: str
    state: AgentState
    budget: Budget
    segment_start: float
    base_elapsed: float = 0.0
    events: list[Event] = field(default_factory=list)


class AgentRuntime:
    def __init__(
        self,
        llm: LLMClient,
        tools: Sequence[Any] = (),
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        budget: Budget | None = None,
        policy: ToolPolicy | None = None,
        approver: Approver | None = None,
        dod: DefinitionOfDone | None = None,
        store: EventStore | None = None,
        tracer: Tracer | None = None,
        pricing: PricingTable | None = None,
        config: LoopConfig | None = None,
        model: str | None = None,
        principal: dict[str, Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.llm = llm
        self.tools: dict[str, Tool] = {}
        for raw in tools:
            tool = adapt_tool(raw)
            if tool.name in self.tools:
                raise ValueError(f"duplicate tool name {tool.name!r}")
            self.tools[tool.name] = tool
        self.system_prompt = system_prompt
        self.budget = budget or Budget()
        self.policy: ToolPolicy = policy or DefaultPolicy()
        self.approver = approver
        self.dod = dod
        self.store: EventStore = store if store is not None else InMemoryEventStore()
        self.tracer: Tracer = tracer or NoopTracer()
        self.pricing = pricing
        self.config = config or LoopConfig()
        self.model = model
        self.principal = dict(principal or {})
        self.clock = clock

    # ------------------------------------------------------------------ public API
    def run(self, goal: str, *, run_id: str | None = None, metadata: dict[str, Any] | None = None) -> RunResult:
        run_id = run_id or uuid.uuid4().hex[:12]
        if self.store.load(run_id):
            raise ValueError(f"run {run_id!r} already exists; use resume()")
        s = _Session(run_id=run_id, state=AgentState(), budget=self.budget, segment_start=self.clock())
        visible = [t.spec for t in self._visible_tools()]
        self._emit(s, GoalSet, goal=goal, system_prompt=self._system_text(), tool_specs=visible,
                   budget=self.budget.model_dump(), principal=self.principal, metadata=metadata or {})
        return self._drive(s)

    def resume(self, run_id: str, *, approve: bool | None = None, reason: str = "",
               budget: Budget | None = None) -> RunResult:
        """Continue a run from its event log: after an approval pause, after a crash, or with a
        larger budget after a budget stop. State is rebuilt from events, never from memory."""
        events = self.store.load(run_id)
        if not events:
            raise KeyError(f"unknown run {run_id!r}")
        state = derive_state(events)
        s = _Session(run_id=run_id, state=state, budget=budget or self.budget,
                     segment_start=self.clock(), base_elapsed=state.usage.elapsed_s, events=list(events))
        if state.status is AgentStatus.COMPLETED:
            raise ValueError(f"run {run_id!r} already completed")
        if state.status is AgentStatus.AWAITING_APPROVAL:
            if approve is None:
                raise ValueError("run is awaiting approval: pass approve=True or approve=False")
            rec = state.pending_approval
            assert rec is not None
            self._emit(s, Resumed, by="human", note=reason)
            if approve:
                self._emit(s, ToolCallApproved, request_id=rec.request_id, tool=rec.tool, by="human", reason=reason)
            else:
                self._emit(s, ToolCallDenied, request_id=rec.request_id, call_id=rec.call_id, tool=rec.tool,
                           reason=reason or "rejected by a human reviewer", by="human")
        elif state.status is AgentStatus.STOPPED:
            if not (state.stop_reason and state.stop_reason.is_budget and budget is not None):
                raise ValueError(f"run stopped with {state.stop_reason}; only budget stops resume, with a new budget")
            self._emit(s, Resumed, by="operator", note=f"budget extended to {budget.model_dump()}")
        else:  # RUNNING with no Stopped event: the previous process died mid-run
            self._emit(s, Resumed, by="recovery", note="resumed after interruption")
        return self._drive(s)

    # ------------------------------------------------------------------ the loop
    def _drive(self, s: _Session) -> RunResult:
        with self.tracer.span("agent.run", run_id=s.run_id, goal=s.state.goal[:200]) as span:
            while s.state.status is AgentStatus.RUNNING:
                if s.state.pending_calls:                  # only after resume
                    self._process_calls(s)
                    if s.state.status is AgentStatus.RUNNING:
                        self._close_step(s)
                    continue
                if s.state.step_open:                      # crashed between last result and step close
                    self._close_step(s)
                    continue
                limit = self._limit_reason(s)
                if limit is not None:
                    self._stop(s, *limit)
                    break
                self._model_step(s)
            st = s.state
            span.set_attribute("status", st.status.value)
            span.set_attribute("stop_reason", st.stop_reason.value if st.stop_reason else None)
            span.set_attribute("steps", st.usage.steps)
            span.set_attribute("tool_calls", st.usage.tool_calls)
            span.set_attribute("tokens", st.usage.total_tokens)
            span.set_attribute("cost_usd", round(st.usage.cost_usd, 6))
        return RunResult(run_id=s.run_id, status=s.state.status, stop_reason=s.state.stop_reason,
                         detail=s.state.stop_detail, final_answer=s.state.final_answer, state=s.state,
                         events=list(s.events))

    def _limit_reason(self, s: _Session) -> tuple[TerminationReason, str] | None:
        st = s.state
        reason = s.budget.exceeded(st.usage, self._elapsed(s))
        if reason is not None:
            return reason, f"budget reached: {s.budget.remaining(st.usage)}"
        if st.steps_without_progress >= self.config.max_no_progress_steps:
            return TerminationReason.NO_PROGRESS, f"{st.steps_without_progress} steps without new information"
        if st.consecutive_errors >= self.config.max_consecutive_errors:
            return TerminationReason.TOOL_ERRORS, f"{st.consecutive_errors} failed or denied tool calls in a row"
        return None

    def _model_step(self, s: _Session) -> None:
        st = s.state
        step = st.step + 1
        tools = self._visible_tools()
        messages = list(st.messages)
        estimate = count_message_tokens(messages, self.model) + self.config.max_tokens_per_call
        if not s.budget.admits_model_call(st.usage, estimate):
            self._stop(s, TerminationReason.MAX_TOKENS, f"next call needs about {estimate} tokens; "
                       f"{s.budget.remaining(st.usage)['tokens']} remain")
            return
        req = CompletionRequest(
            messages=messages, model=self.model, temperature=self.config.temperature,
            max_tokens=self.config.max_tokens_per_call, tools=[t.spec for t in tools] or None,
            metadata={"run_id": s.run_id, "step": step},
        )
        with self.tracer.span("agent.step", run_id=s.run_id, step=step) as span:
            try:
                completion = self.llm.complete(req)
            except LLMError as exc:
                span.record_exception(exc)
                cls = ErrorClass.TRANSIENT if exc.retryable else ErrorClass.FATAL
                self._stop(s, TerminationReason.MODEL_ERROR, f"{cls.value}: {type(exc).__name__}: {exc}")
                return
            raw_cost = (completion.raw or {}).get("cost_usd")
            cost = float(raw_cost) if raw_cost is not None else (
                self.pricing.cost_usd(completion.model, completion.usage) if self.pricing else 0.0)
            calls = completion.tool_calls
            kind = "tool_calls" if calls else "final"
            self._emit(s, ModelDecision, step=step, kind=kind, text=completion.text, tool_calls=calls,
                       usage=completion.usage, cost_usd=cost, model=completion.model,
                       finish_reason=completion.finish_reason, latency_ms=completion.latency_ms,
                       request_hash=_request_hash(req))
            span.set_attribute("decision", kind)
            span.set_attribute("tools", [c.name for c in calls])
            span.set_attribute("input_tokens", completion.usage.input_tokens)
            span.set_attribute("output_tokens", completion.usage.output_tokens)
            span.set_attribute("cost_usd", cost)
            if calls:
                for i, call in enumerate(calls):
                    self._emit(s, ToolCallRequested, step=step, request_id=f"{step}.{i}", call_id=call.id,
                               tool=call.name, arguments=call.arguments, key=action_key(call.name, call.arguments))
                self._process_calls(s)
            else:
                self._handle_final(s, completion.text, completion.finish_reason)
            if s.state.status is AgentStatus.RUNNING:
                self._close_step(s)
            span.set_attribute("progress", s.state.steps_without_progress == 0)

    def _handle_final(self, s: _Session, text: str, finish_reason: str) -> None:
        if finish_reason == "length":
            self._emit(s, Note, kind="truncated_answer", to_model=True, error_class=ErrorClass.VALIDATION,
                       text="Your answer was cut off by the output limit. Answer again, more concisely.")
            return
        if self.dod is None:
            verdicts: list[dict[str, Any]] = []
            passed = bool(text.strip())
            feedback = "You returned neither a tool call nor an answer. Continue or answer."
        else:
            result = self.dod.verify(text, s.state)
            verdicts = [v.to_dict() for v in result.verdicts]
            passed, feedback = result.passed, result.feedback()
        if passed:
            self._emit(s, FinalAnswer, text=text, checks=verdicts)
            self._stop(s, TerminationReason.COMPLETED, "final answer accepted")
            return
        self._emit(s, Note, kind="dod_rejected", text=feedback, to_model=True,
                   error_class=ErrorClass.SEMANTIC, data={"verdicts": verdicts, "answer": text[:2000]})
        if s.state.dod_rejections > self.config.max_dod_rejections:
            self._stop(s, TerminationReason.VERIFICATION_FAILED,
                       f"{s.state.dod_rejections} final answers failed the Definition of Done")

    def _process_calls(self, s: _Session) -> None:
        for rec in list(s.state.pending_calls):
            if s.state.status is not AgentStatus.RUNNING:
                return
            tool = self.tools.get(rec.tool)
            if rec.status == "requested":
                if not self._authorize(s, rec, tool):
                    continue
                if s.state.status is not AgentStatus.RUNNING:
                    return
            if tool is None:   # approved earlier, tool removed since: refuse rather than guess
                self._deny(s, rec, f"tool '{rec.tool}' is no longer registered", ErrorClass.IMPOSSIBLE, "runtime")
                continue
            self._execute(s, rec, tool)

    def _authorize(self, s: _Session, rec: CallRecord, tool: Tool | None) -> bool:
        """Validation, loop detection, tool budget, policy, approval. True means approved."""
        st = s.state
        if tool is None or not self.policy.visible(tool, st.principal):
            names = sorted(t.name for t in self._visible_tools())
            self._deny(s, rec, f"unknown tool '{rec.tool}'; available tools: {names}", ErrorClass.VALIDATION, "runtime")
            return False
        errors = validate_arguments(tool.spec.parameters, rec.arguments)
        if errors:
            self._deny(s, rec, "invalid arguments: " + "; ".join(errors), ErrorClass.VALIDATION, "runtime")
            return False
        if st.key_counts.get(rec.key, 0) >= self.config.max_identical_calls:
            self._deny(s, rec, "identical call already executed; the result will not change", ErrorClass.VALIDATION,
                       "runtime")
            self._stop(s, TerminationReason.REPEATED_ACTION,
                       f"{rec.tool}({rec.arguments}) requested {st.key_counts[rec.key] + 1} times")
            return False
        if not s.budget.admits_tool_call(st.usage):
            self._stop(s, TerminationReason.MAX_TOOL_CALLS, f"tool-call budget of {s.budget.max_tool_calls} used")
            return False
        decision = self.policy.check(tool, rec.arguments, st.principal)
        if not decision.allowed:
            self._deny(s, rec, decision.reason or "denied by policy", ErrorClass.PERMISSION, "policy")
            return False
        if not decision.requires_approval:
            self._emit(s, ToolCallApproved, request_id=rec.request_id, tool=rec.tool, by="policy")
            return True
        verdict = self.approver(rec, st) if self.approver else None
        if verdict is None:
            self._emit(s, Note, kind="approval_required",
                       text=f"{rec.tool} awaits approval: {decision.reason}",
                       data={"request_id": rec.request_id, "tool": rec.tool, "arguments": rec.arguments})
            self._stop(s, TerminationReason.APPROVAL_REQUIRED, f"{rec.tool} needs approval")
            return False
        if verdict:
            self._emit(s, ToolCallApproved, request_id=rec.request_id, tool=rec.tool, by="approver")
            return True
        self._deny(s, rec, "rejected by approver", ErrorClass.PERMISSION, "approver")
        return False

    def _execute(self, s: _Session, rec: CallRecord, tool: Tool) -> None:
        ctx = ToolContext(run_id=s.run_id, step=rec.step, call_id=rec.call_id, request_id=rec.request_id,
                          idempotency_key=f"{s.run_id}:{rec.request_id}", principal=dict(s.state.principal))
        with self.tracer.span("agent.tool", run_id=s.run_id, step=rec.step, tool=rec.tool,
                              request_id=rec.request_id) as span:
            started = time.perf_counter()
            attempts = 0
            while True:
                attempts += 1
                try:
                    out = tool.execute(dict(rec.arguments), ctx)
                    break
                except Exception as exc:  # noqa: BLE001 - classified, recorded, never swallowed silently
                    cls = classify_error(exc)
                    if cls is ErrorClass.TRANSIENT and tool.idempotent and attempts <= self.config.transient_retries:
                        continue
                    out = ToolOutput.failure(f"{type(exc).__name__}: {exc}", cls)
                    break
            shaped = truncate_observation(out.content, self.config.max_observation_chars)
            seen = s.state.key_counts.get(rec.key, 0)
            notice = ("This exact call was already made; its result is unchanged. "
                      "Use a different approach or give your answer.") if seen else ""
            self._emit(s, ToolResult, step=rec.step, request_id=rec.request_id, call_id=rec.call_id, tool=rec.tool,
                       key=rec.key, ok=out.ok, content=shaped.text, original_chars=shaped.original_chars,
                       truncated=shaped.truncated, data=out.data, artifacts=out.artifacts,
                       error_class=out.error_class, error=out.error, notice=notice, attempts=attempts,
                       latency_ms=round((time.perf_counter() - started) * 1000, 3),
                       idempotency_key=ctx.idempotency_key)
            span.set_attribute("ok", out.ok)
            span.set_attribute("attempts", attempts)
            span.set_attribute("truncated", shaped.truncated)
            span.set_attribute("error_class", out.error_class.value if out.error_class else None)
        if out.error_class is ErrorClass.FATAL:
            self._stop(s, TerminationReason.FATAL_ERROR, f"{rec.tool}: {out.error}")

    # ------------------------------------------------------------------ helpers
    def _deny(self, s: _Session, rec: CallRecord, reason: str, cls: ErrorClass, by: str) -> None:
        self._emit(s, ToolCallDenied, request_id=rec.request_id, call_id=rec.call_id, tool=rec.tool,
                   reason=reason, error_class=cls, by=by)

    def _close_step(self, s: _Session) -> None:
        progress = s.state.step_had_progress
        self._emit(s, StepCompleted, progress=progress,
                   detail="new information" if progress else "no new information")
        self._emit(s, BudgetUpdated, usage=s.state.usage.model_copy(update={"elapsed_s": self._elapsed(s)}))

    def _stop(self, s: _Session, reason: TerminationReason, detail: str) -> None:
        self._emit(s, BudgetUpdated, usage=s.state.usage.model_copy(update={"elapsed_s": self._elapsed(s)}))
        self._emit(s, Stopped, reason=reason, detail=detail)

    def _emit(self, s: _Session, cls: type[Event], **fields: Any) -> Event:
        fields.setdefault("step", s.state.step)
        event = cls(run_id=s.run_id, seq=s.state.last_seq + 1, **fields)
        self.store.append(event)       # durable first, then projected
        apply(s.state, event)
        s.events.append(event)
        return event

    def _elapsed(self, s: _Session) -> float:
        return round(s.base_elapsed + (self.clock() - s.segment_start), 6)

    def _visible_tools(self) -> list[Tool]:
        return [t for t in self.tools.values() if self.policy.visible(t, self.principal)]

    def _system_text(self) -> str:
        if self.dod is not None and self.config.dod_in_prompt:
            return f"{self.system_prompt}\n\n{self.dod.as_prompt()}"
        return self.system_prompt


def _request_hash(req: CompletionRequest) -> str:
    payload = req.model_dump_json(exclude={"metadata"})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


__all__ = ["AgentRuntime", "LoopConfig", "RunResult", "Approver", "DEFAULT_SYSTEM_PROMPT"]
