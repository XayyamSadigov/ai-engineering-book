# path: book/projects/examples/ch25/taskevals/replay.py
"""Agent evaluation over agentkit event logs (Chapter 19).

agentkit's runtime records every run as immutable events (GoalSet, ModelDecision,
ToolCallRequested, ToolCallApproved, ToolCallDenied, ToolResult, FinalAnswer, Stopped, ...)
in an EventStore such as JsonlEventStore. Those logs are the source of truth. This module:

- exports a log to the thin `Trajectory` JSON the assertions read (`trajectory_from_events`),
  projecting the end state of the world from the successful write-tool results;
- evaluates recorded runs as they are (`recorded_run_target`), which is how you audit an agent
  already in production;
- runs counterfactual replay with `agentkit.replay(events, llm=new_client)`: recorded tool
  results are served by action key, nothing executes, and calls the new planner makes that the
  recording never saw come back as replay misses (`unrecorded` in the export).

Replay isolates the planner: observations are fixed, so a change in the trajectory comes from
the prompt, model, or policy under test. Its limit is divergence: `replay_fidelity` reports the
share of tool results served from the recording, and low fidelity means the case needs a live
run against sandboxed tools instead.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from agentkit import (
    Event,
    FinalAnswer,
    GoalSet,
    JsonlEventStore,
    ModelDecision,
    Stopped,
    TerminationReason,
    ToolCallApproved,
    ToolCallDenied,
    ToolCallRequested,
    ToolResult,
    replay,
)
from aie_core import LLMClient

from evalkit import EvalCase, FunctionEvaluator, Score, TargetResult

from .trajectory import NORTHWIND_TOOLS, Step, ToolInfo, Trajectory, TrajectoryUsage

HUMAN_APPROVERS = ("approver", "human")  # "policy" approvals are routine authorization, not sign-off
REPLAY_MISS_MARKER = "replay miss"

StateProjection = Callable[[str, dict[str, Any], Any], "tuple[str, dict[str, Any]] | None"]


def northwind_projection(tool: str, arguments: dict[str, Any], data: Any) -> tuple[str, dict[str, Any]] | None:
    """Map a successful write-tool result to a record in the world's end state."""
    if tool == "create_ticket":
        data = data if isinstance(data, dict) else {}
        # The tenant is the request's, reported by the tool; a model-supplied value never wins.
        record = {"id": data.get("ticket_id"), **arguments}
        if "tenant" in data:
            record["tenant"] = data["tenant"]
        return "tickets", record
    if tool == "draft_reply":
        return "drafts", dict(arguments)
    if tool == "send_reply":
        return "sent_replies", dict(arguments)
    return None


def _approved_keys(events: Sequence[Event]) -> set[str]:
    keys = {e.request_id: e.key for e in events if isinstance(e, ToolCallRequested)}
    return {keys[e.request_id] for e in events
            if isinstance(e, ToolCallApproved) and e.by in HUMAN_APPROVERS and e.request_id in keys}


def trajectory_from_events(
    events: Sequence[Event],
    *,
    task_id: str | None = None,
    projection: StateProjection = northwind_projection,
    tools: Mapping[str, ToolInfo] = NORTHWIND_TOOLS,
    carry_approvals_from: Sequence[Event] | None = None,
) -> Trajectory:
    """Export an agentkit event log to the evaluator's trajectory format.

    `carry_approvals_from` is the original log of a counterfactual replay: agentkit does not
    re-request approvals during replay (nothing real executes), so a replayed call to an
    approval-gated tool counts as approved only if the original run had a human approval for
    the identical action key.
    """
    goal = next((e for e in events if isinstance(e, GoalSet)), None)
    if goal is None:
        raise ValueError("event log has no GoalSet")
    carried = _approved_keys(carry_approvals_from) if carry_approvals_from is not None else set()
    requests: dict[str, ToolCallRequested] = {}
    steps = [Step(type="user_goal", content=goal.goal)]
    state: dict[str, list[dict[str, Any]]] = {}
    usage = TrajectoryUsage()
    model = ""
    stop_reason = "running"
    for e in events:
        if isinstance(e, ModelDecision):
            usage.input_tokens += e.usage.input_tokens
            usage.output_tokens += e.usage.output_tokens
            usage.cost_usd += e.cost_usd
            model = model or e.model
            if e.kind == "tool_calls" and e.text:
                steps.append(Step(type="model_decision", content=e.text))
        elif isinstance(e, ToolCallRequested):
            requests[e.request_id] = e
            steps.append(Step(type="tool_call", call_id=e.request_id, tool=e.tool, arguments=dict(e.arguments)))
            info = tools.get(e.tool)
            if e.key in carried and info is not None and info.requires_approval:
                steps.append(Step(type="approval", call_id=e.request_id, decision="approved", approver="recorded"))
        elif isinstance(e, ToolCallApproved) and e.by in HUMAN_APPROVERS:
            steps.append(Step(type="approval", call_id=e.request_id, decision="approved", approver=e.by))
        elif isinstance(e, ToolCallDenied):
            if e.by in HUMAN_APPROVERS:
                steps.append(Step(type="approval", call_id=e.request_id, decision="denied", approver=e.by))
            steps.append(Step(type="tool_result", call_id=e.request_id, status="denied", output={"error": e.reason}))
        elif isinstance(e, ToolResult):
            miss = not e.ok and REPLAY_MISS_MARKER in (e.error or e.content or "")
            status = "ok" if e.ok else ("unrecorded" if miss else "error")
            steps.append(Step(type="tool_result", call_id=e.request_id, status=status,  # type: ignore[arg-type]
                              output=e.data if e.data is not None else e.content))
            req = requests.get(e.request_id)
            if e.ok and req is not None and (rec := projection(e.tool, dict(req.arguments), e.data)) is not None:
                state.setdefault(rec[0], []).append(rec[1])
        elif isinstance(e, FinalAnswer):
            steps.append(Step(type="final_answer", content=e.text))
        elif isinstance(e, Stopped):
            stop_reason = "final_answer" if e.reason is TerminationReason.COMPLETED else e.reason.value
    if stop_reason == "running" and any(s.type == "final_answer" for s in steps):
        stop_reason = "final_answer"
    return Trajectory(
        trajectory_id=goal.run_id, task_id=task_id or goal.metadata.get("task_id", goal.run_id),
        agent_version=str(goal.metadata.get("agent_version", "unknown")), model=model or "unknown",
        goal=goal.goal, steps=steps, final_state=state, usage=usage,
        latency_ms=sum(e.latency_ms for e in events if isinstance(e, (ModelDecision, ToolResult))),
        stop_reason=stop_reason,
    )


def load_event_logs(directory: str | Path) -> dict[str, list[Event]]:
    store = JsonlEventStore(directory)
    return {run_id: store.load(run_id) for run_id in sorted(store.runs())}


def recorded_run_target(logs: Mapping[str, Sequence[Event]]) -> Callable[[EvalCase], TargetResult]:
    """Evaluate recorded runs as they happened. The case input names the log under `recording`."""

    def target(case: EvalCase) -> TargetResult:
        traj = trajectory_from_events(logs[case.input["recording"]], task_id=case.id)
        return TargetResult(output=traj.model_dump(mode="json", exclude_none=True),
                            input_tokens=traj.usage.input_tokens, output_tokens=traj.usage.output_tokens,
                            cost_usd=traj.usage.cost_usd)

    target.__name__ = "recorded-runs"
    return target


def agentkit_replay_target(
    llm: LLMClient,
    recordings: Mapping[str, Sequence[Event]],
    *,
    system_prompt: str | None = None,
    version: str = "planner",
) -> Callable[[EvalCase], TargetResult]:
    """Counterfactual replay of the case's recording with a new planner (`llm`, `system_prompt`)."""

    def target(case: EvalCase) -> TargetResult:
        original = recordings[case.input["recording"]]
        report = replay(original, llm, system_prompt=system_prompt,
                        run_id=f"{case.id}-replay")
        assert report.result is not None
        traj = trajectory_from_events(report.result.events, task_id=case.id, carry_approvals_from=original)
        traj.agent_version = version
        return TargetResult(
            output=traj.model_dump(mode="json", exclude_none=True),
            input_tokens=traj.usage.input_tokens, output_tokens=traj.usage.output_tokens,
            cost_usd=traj.usage.cost_usd,
            metadata={"replay_misses": len(report.misses), "first_divergence": report.first_divergence,
                      "replay_summary": report.summary()},
        )

    target.__name__ = f"agentkit-replay[{version}]"
    return target


def replay_fidelity_evaluator(min_fidelity: float = 0.8) -> FunctionEvaluator:
    """Share of tool results served from the recording (read from the exported trajectory)."""

    def fn(case: EvalCase, output: Any) -> Score:
        traj = Trajectory.model_validate(output)
        results = [s for s in traj.steps if s.type == "tool_result" and s.status != "denied"]
        unrecorded = sum(1 for s in results if s.status == "unrecorded")
        value = 1.0 if not results else 1.0 - unrecorded / len(results)
        return Score(name="replay_fidelity", value=value, passed=value >= min_fidelity,
                     detail=f"{unrecorded} unrecorded of {len(results)}" if unrecorded else None)

    return FunctionEvaluator("replay_fidelity", fn, version="2")


__all__ = ["HUMAN_APPROVERS", "northwind_projection", "trajectory_from_events", "load_event_logs",
           "recorded_run_target", "agentkit_replay_target", "replay_fidelity_evaluator"]
