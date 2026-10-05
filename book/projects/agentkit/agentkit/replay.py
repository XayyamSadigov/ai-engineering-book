# path: book/projects/agentkit/agentkit/replay.py
"""Replay: run a recorded trajectory again without touching the outside world.

Two pieces make this work. `RecordedToolResults` serves the original tool results by action
key, so tools are never re-executed. `RecordedLLM` serves the original model decisions in
order, so the harness itself can be re-run against old trajectories (a new verifier, a new
policy, a new truncation limit). Pass a *different* LLM instead and you get counterfactual
replay: "with this new prompt or model, would the agent have chosen differently, given the
same observations?" Calls the new planner makes that the recording never saw are misses.
"""
from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator, Sequence

from aie_core.llm.client import LLMClient
from aie_core.llm.errors import LLMError
from aie_core.llm.types import Completion, CompletionRequest, Message, Role, StreamEvent, ToolSpec

from .budget import Budget
from .dod import DefinitionOfDone
from .errors import ErrorClass
from .events import Event, GoalSet, ModelDecision, Stopped, ToolCallDenied, ToolCallRequested, ToolResult, action_key
from .runtime import AgentRuntime, LoopConfig, RunResult
from .store import EventStore, InMemoryEventStore
from .tools import PolicyDecision, SideEffect, Tool, ToolContext, ToolOutput


class ReplayExhausted(LLMError):
    """The recorded model has no decisions left: the replayed run went longer than the original."""

    default_retryable = False


class RecordedLLM:
    """An LLMClient that returns recorded ModelDecisions in order and ignores the request."""

    provider = "replay"

    def __init__(self, decisions: Sequence[ModelDecision]) -> None:
        self._decisions = list(decisions)
        self._i = 0
        self.requests: list[CompletionRequest] = []

    @classmethod
    def from_events(cls, events: Sequence[Event]) -> "RecordedLLM":
        return cls([e for e in events if isinstance(e, ModelDecision)])

    def complete(self, req: CompletionRequest) -> Completion:
        self.requests.append(req)
        if self._i >= len(self._decisions):
            raise ReplayExhausted("recorded trajectory has no more model decisions")
        d = self._decisions[self._i]
        self._i += 1
        message = Message(role=Role.ASSISTANT, content=d.text, tool_calls=list(d.tool_calls) or None)
        return Completion(message=message, usage=d.usage, finish_reason=d.finish_reason or "stop",
                          model=d.model, provider=self.provider, latency_ms=0.0, raw={"cost_usd": d.cost_usd})

    def stream(self, req: CompletionRequest) -> Iterator[StreamEvent]:
        c = self.complete(req)
        if c.text:
            yield StreamEvent(type="text_delta", text=c.text)
        for tc in c.tool_calls:
            yield StreamEvent(type="tool_call_delta", tool_call=tc)
        yield StreamEvent(type="usage", usage=c.usage)
        yield StreamEvent(type="done", finish_reason=c.finish_reason)

    async def acomplete(self, req: CompletionRequest) -> Completion:
        return self.complete(req)

    async def astream(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        for ev in self.stream(req):
            yield ev


class RecordedToolResults:
    """Recorded results by action key. Repeated keys are served in recorded order, then the last
    one again (the recording says nothing about a fourth call, and reads are assumed stable)."""

    def __init__(self, results: Sequence[ToolResult]) -> None:
        self._by_key: dict[str, list[ToolResult]] = defaultdict(list)
        for r in results:
            self._by_key[r.key].append(r)
        self._served: dict[str, int] = defaultdict(int)

    @classmethod
    def from_events(cls, events: Sequence[Event]) -> "RecordedToolResults":
        return cls([e for e in events if isinstance(e, ToolResult)])

    def lookup(self, key: str) -> ToolResult | None:
        options = self._by_key.get(key)
        if not options:
            return None
        i = min(self._served[key], len(options) - 1)
        self._served[key] += 1
        return options[i]


class ReplayTool:
    """Stands in for a real tool during replay; serves recorded results and records misses."""

    side_effect = SideEffect.READ
    requires_approval = False
    idempotent = True

    def __init__(self, spec: ToolSpec, recorded: RecordedToolResults, misses: list[dict[str, Any]]) -> None:
        self.name = spec.name
        self._spec = spec
        self._recorded = recorded
        self._misses = misses

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def execute(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        key = action_key(self.name, arguments)
        rec = self._recorded.lookup(key)
        if rec is None:
            self._misses.append({"step": ctx.step, "tool": self.name, "arguments": arguments, "key": key})
            return ToolOutput.failure("no recorded result for this call (replay miss)", ErrorClass.IMPOSSIBLE)
        return ToolOutput(content=rec.content, ok=rec.ok, data=rec.data, artifacts=dict(rec.artifacts),
                          error_class=rec.error_class, error=rec.error)


class RecordedPolicy:
    """Reproduces the original run's policy and human denials by action key; allows the rest.
    Approvals are not re-requested: nothing real executes during replay."""

    def __init__(self, events: Sequence[Event]) -> None:
        keys = {e.request_id: e.key for e in events if isinstance(e, ToolCallRequested)}
        self.denied: dict[str, str] = {
            keys[e.request_id]: e.reason for e in events
            if isinstance(e, ToolCallDenied) and e.by in ("policy", "approver", "human") and e.request_id in keys
        }

    def visible(self, tool: Tool, principal: dict[str, Any]) -> bool:
        return True

    def check(self, tool: Tool, arguments: dict[str, Any], principal: dict[str, Any]) -> PolicyDecision:
        reason = self.denied.get(action_key(tool.name, arguments))
        if reason is not None:
            return PolicyDecision(allowed=False, reason=reason)
        return PolicyDecision(allowed=True)


@dataclass(frozen=True)
class DecisionSignature:
    """What a decision *did*, stripped of ids and timing, so two runs can be compared."""

    kind: str
    calls: tuple[tuple[str, str], ...]
    text: str

    @classmethod
    def of(cls, d: ModelDecision) -> "DecisionSignature":
        calls = tuple((c.name, action_key(c.name, c.arguments)) for c in d.tool_calls)
        return cls(kind=d.kind, calls=calls, text=d.text if d.kind == "final" else "")


def decision_signatures(events: Sequence[Event]) -> list[DecisionSignature]:
    return [DecisionSignature.of(e) for e in events if isinstance(e, ModelDecision)]


@dataclass
class ReplayReport:
    original: list[DecisionSignature]
    replayed: list[DecisionSignature]
    first_divergence: int | None
    misses: list[dict[str, Any]] = field(default_factory=list)
    result: RunResult | None = None
    original_stop: Any = None

    @property
    def identical(self) -> bool:
        return self.first_divergence is None and not self.misses

    def summary(self) -> str:
        if self.identical:
            return f"identical: {len(self.original)} decisions reproduced"
        where = "none" if self.first_divergence is None else f"step {self.first_divergence + 1}"
        return f"diverged at {where}; {len(self.misses)} replay misses; " \
               f"original stop={self.original_stop}, replay stop={self.result.stop_reason if self.result else None}"


def _first_divergence(a: list[DecisionSignature], b: list[DecisionSignature]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def replay(
    events: Sequence[Event],
    llm: LLMClient | None = None,
    *,
    system_prompt: str | None = None,
    budget: Budget | None = None,
    dod: DefinitionOfDone | None = None,
    config: LoopConfig | None = None,
    store: EventStore | None = None,
    run_id: str | None = None,
) -> ReplayReport:
    """Re-run a recorded run. With `llm=None` the recorded decisions are reused (harness replay);
    with a new client the decisions are new and the observations are recorded (counterfactual)."""
    goal = next((e for e in events if isinstance(e, GoalSet)), None)
    if goal is None:
        raise ValueError("event log has no GoalSet")
    recorded = RecordedToolResults.from_events(events)
    misses: list[dict[str, Any]] = []
    tools = [ReplayTool(spec, recorded, misses) for spec in goal.tool_specs]
    reuse_prompt = system_prompt is None
    cfg = config or LoopConfig()
    if reuse_prompt:
        cfg = cfg.model_copy(update={"dod_in_prompt": False})
    runtime = AgentRuntime(
        llm or RecordedLLM.from_events(events),
        tools,
        system_prompt=(goal.system_prompt or "") if reuse_prompt else str(system_prompt),
        budget=budget or Budget(**goal.budget),
        policy=RecordedPolicy(events),
        dod=dod,
        store=store or InMemoryEventStore(),
        config=cfg,
        principal=goal.principal,
    )
    result = runtime.run(goal.goal, run_id=run_id or f"{goal.run_id}-replay-{uuid.uuid4().hex[:6]}",
                         metadata={"replay_of": goal.run_id})
    original = decision_signatures(events)
    replayed = decision_signatures(result.events)
    stops = [e for e in events if isinstance(e, Stopped)]
    return ReplayReport(original=original, replayed=replayed, first_divergence=_first_divergence(original, replayed),
                        misses=misses, result=result, original_stop=stops[-1].reason if stops else None)


__all__ = [
    "ReplayExhausted", "RecordedLLM", "RecordedToolResults", "ReplayTool", "RecordedPolicy",
    "DecisionSignature", "decision_signatures", "ReplayReport", "replay",
]
