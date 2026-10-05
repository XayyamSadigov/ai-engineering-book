# path: book/projects/examples/ch17/triage_graph.py
"""The same triage workflow as an explicit graph with conditional edges.

What changes compared with the pipeline: the approval step is a paused
state with a resumable handle, retries are a per-node policy instead of
try/except, every transition is checkpointed, and the branching rules are
one routing function you can unit-test without running any step.
"""
from __future__ import annotations

from triage_domain import (Model, Sender, TriageState, classify, draft_reply, escalate, needs_human,
                           retrieve_policy, send, validate)
from workflow_engine import END, Checkpointer, Graph, RetryPolicy, step_key

MAX_DRAFTS = 2


def route_after_validate(state: TriageState) -> str:
    assert state.report is not None
    if state.report.verdict == "ok":
        return "approval" if needs_human(state) else "send"
    if state.report.verdict == "fixable" and state.draft_attempts < MAX_DRAFTS:
        return "draft"
    return "escalate"


def route_after_approval(state: TriageState) -> str:
    return "send" if state.approved else "escalate"


def record_decision(state: TriageState, decision: dict) -> TriageState:
    state.approved = bool(decision.get("approved"))
    state.approval_note = decision.get("note")
    state.log.append(f"approval: {'yes' if state.approved else 'no'}")
    return state


def build_triage_graph(model: Model, sender: Sender,
                       checkpointer: Checkpointer | None = None,
                       model_retry: RetryPolicy = RetryPolicy(max_attempts=3),
                       tracer: object | None = None) -> Graph[TriageState]:
    g: Graph[TriageState] = Graph(TriageState, checkpointer=checkpointer, max_steps=20, tracer=tracer)
    g.add_node("classify", lambda s: classify(s, model), retry=model_retry)
    g.add_node("retrieve_policy", retrieve_policy)
    g.add_node("draft", lambda s: draft_reply(s, model), retry=model_retry)
    g.add_node("validate", lambda s: validate(s, model), retry=model_retry)
    # The approval node itself does nothing; the engine pauses before it and
    # applies the human decision through `on_decision` when resumed.
    g.add_node("approval", lambda s: s, pause_before=True, on_decision=record_decision)
    # Side effect: no retry policy, and a key that is stable across re-execution.
    g.add_node("send", lambda s: send(s, sender, key=step_key()))
    g.add_node("escalate", escalate)

    g.add_edge("classify", "retrieve_policy")
    g.add_edge("retrieve_policy", "draft")
    g.add_edge("draft", "validate")
    g.add_router("validate", route_after_validate)
    g.add_router("approval", route_after_approval)
    g.add_edge("send", END)
    g.add_edge("escalate", END)
    return g
