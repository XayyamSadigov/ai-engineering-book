# path: book/projects/examples/ch38/ledger.py
"""Long-horizon context for agents: a structured task ledger plus compacted observations.

A run of two hundred steps cannot carry two hundred raw tool results in its prompt. Chapter 5
compacts conversations; an agent needs something stricter, because its context holds the plan,
the decisions already taken, and the exact identifiers (paths, ticket ids, error names) that
later steps depend on. The design here:

- `TaskLedger`: objective, plan items with status, decisions, facts, open questions. The model
  edits it only through the typed `update_ledger` tool, so every edit is a ToolResult event
  with the new ledger as an artifact. The ledger is therefore part of the event log, and
  `ledger_from_events` rebuilds it after a crash with no extra storage.
- `harness_facts`: what the harness knows for certain from events (calls made, failures,
  denials). Computed, never written by the model.
- `CompactingLLM`: an LLMClient decorator. When a request exceeds its token budget it keeps
  the system prompt, the goal, and the last K steps verbatim, and replaces older steps with
  the ledger, the harness facts, and one digest line per old observation that preserves the
  literals. Assistant tool calls and their tool results are dropped or kept together, never
  split, so the request stays valid for every provider.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Iterator, Literal, Sequence

from pydantic import BaseModel, Field

from aie_core.llm.client import LLMClient
from aie_core.llm.tokens import count_message_tokens
from aie_core.llm.types import Completion, CompletionRequest, Message, Role, StreamEvent
from agentkit import (
    Event, EventStore, FunctionTool, GoalSet, ToolCallDenied, ToolContext, ToolOutput, ToolResult, derive_state,
)

LEDGER_ARTIFACT = "task_ledger"
Status = Literal["todo", "doing", "done", "blocked"]


class PlanItem(BaseModel):
    id: str
    text: str
    status: Status = "todo"


class LedgerUpdate(BaseModel):
    """The only way the model changes the ledger. Small, typed, validated."""

    add_items: list[str] = Field(default_factory=list)
    set_status: dict[str, Status] = Field(default_factory=dict)
    decisions: list[str] = Field(default_factory=list)
    facts: dict[str, str] = Field(default_factory=dict)
    questions: list[str] = Field(default_factory=list)
    resolve_questions: list[str] = Field(default_factory=list)


class TaskLedger(BaseModel):
    objective: str = ""
    plan: list[PlanItem] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    facts: dict[str, str] = Field(default_factory=dict)
    open_questions: list[str] = Field(default_factory=list)
    version: int = 0

    def apply(self, update: LedgerUpdate) -> "TaskLedger":
        new = self.model_copy(deep=True)
        for text in update.add_items:
            new.plan.append(PlanItem(id=f"p{len(new.plan) + 1}", text=text))
        known = {p.id: p for p in new.plan}
        unknown = sorted(set(update.set_status) - set(known))
        if unknown:
            raise ValueError(f"unknown plan item ids {unknown}; existing: {sorted(known)}")
        for item_id, status in update.set_status.items():
            known[item_id].status = status
        new.decisions.extend(update.decisions)
        new.facts.update(update.facts)
        new.open_questions = [q for q in new.open_questions if q not in set(update.resolve_questions)]
        new.open_questions.extend(q for q in update.questions if q not in new.open_questions)
        new.version += 1
        return new

    def render(self) -> str:
        lines = [f"Objective: {self.objective}", f"Plan (ledger v{self.version}):"]
        lines += [f"  [{p.status}] {p.id} {p.text}" for p in self.plan] or ["  (no plan yet)"]
        if self.decisions:
            lines.append("Decisions: " + " | ".join(self.decisions[-8:]))
        if self.facts:
            lines.append("Facts: " + "; ".join(f"{k}={v}" for k, v in sorted(self.facts.items())))
        if self.open_questions:
            lines.append("Open questions: " + " | ".join(self.open_questions))
        return "\n".join(lines)


def ledger_from_events(events: Sequence[Event]) -> TaskLedger:
    """Rehydrate the ledger from the log: the last ledger artifact wins."""
    goal = next((e.goal for e in events if isinstance(e, GoalSet)), "")
    raw = derive_state(events).artifacts.get(LEDGER_ARTIFACT)
    ledger = TaskLedger.model_validate(raw) if raw else TaskLedger()
    if not ledger.objective:
        ledger.objective = goal
    return ledger


def harness_facts(events: Sequence[Event]) -> dict[str, Any]:
    """Facts the harness can state with certainty. The model cannot edit these."""
    results = [e for e in events if isinstance(e, ToolResult)]
    calls: dict[str, int] = {}
    for r in results:
        calls[r.tool] = calls.get(r.tool, 0) + 1
    failures = [f"{r.request_id} {r.tool}: {(r.error or r.content)[:100]}" for r in results if not r.ok][-3:]
    denied = [f"{e.request_id} {e.tool}: {e.reason[:80]}" for e in events if isinstance(e, ToolCallDenied)][-3:]
    return {"tool_calls": calls, "recent_failures": failures, "recent_denials": denied}


def make_ledger_tool(store: EventStore) -> FunctionTool:
    def update_ledger(ctx: ToolContext, **changes: Any) -> ToolOutput:
        current = ledger_from_events(store.load(ctx.run_id))
        new = current.apply(LedgerUpdate.model_validate(changes))
        done = sum(p.status == "done" for p in new.plan)
        return ToolOutput(content=f"ledger v{new.version}: {done}/{len(new.plan)} plan items done",
                          artifacts={LEDGER_ARTIFACT: new.model_dump()})

    schema = LedgerUpdate.model_json_schema()
    params = {"type": "object", "properties": schema["properties"], "additionalProperties": False}
    return FunctionTool("update_ledger", "Update the task ledger: add plan items, set item status "
                        "(todo/doing/done/blocked), record decisions, facts, and open questions.",
                        params, fn=update_ledger, pass_context=True)


# --------------------------------------------------------------------------- compaction
LITERAL = re.compile(
    r"[\w./-]+\.(?:py|md|json|ya?ml|sql|toml|txt)\b"       # file paths
    r"|\b[A-Z][A-Z0-9]{1,9}-\d+\b"                           # ticket and incident ids
    r"|\[[a-z0-9][a-z0-9._-]+\]"                             # [source-id] citations
    r"|\b\w+(?:Error|Exception)\b"                           # error class names
    r"|\b\d+ (?:passed|failed|errors?)\b"                    # test counts
    r"|\bline \d+\b"
)


def digest_observation(tool: str, arguments: dict[str, Any], content: str, *, max_chars: int = 140) -> str:
    """One deterministic line per old observation: what was called, the first meaningful line,
    and every literal identifier found anywhere in the full content (up to eight)."""
    first = next((ln.strip() for ln in content.splitlines() if ln.strip()), "")
    if len(first) > max_chars:
        first = first[: max_chars - 3] + "..."
    literals: list[str] = []
    for m in LITERAL.finditer(content):
        if m.group(0) not in literals:
            literals.append(m.group(0))
    args = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
    if len(args) > 80:
        args = args[:77] + "..."
    keep = f" keep={literals[:8]}" if literals else ""
    return f"{tool}({args}) -> {first}{keep}"


def _groups(messages: Sequence[Message]) -> tuple[list[Message], list[list[Message]]]:
    """Split into head (system + goal) and step groups, each starting at an assistant message."""
    head: list[Message] = []
    groups: list[list[Message]] = []
    for m in messages:
        if m.role is Role.ASSISTANT:
            groups.append([m])
        elif groups:
            groups[-1].append(m)
        else:
            head.append(m)
    return head, groups


def _summarize_group(group: list[Message]) -> list[str]:
    assistant = group[0]
    by_id = {tc.id: tc for tc in assistant.tool_calls or []}
    lines: list[str] = []
    if not assistant.tool_calls and assistant.text.strip():
        lines.append(f"answer attempt rejected: {assistant.text.strip()[:100]}")
    for m in group[1:]:
        if m.role is Role.TOOL:
            tc = by_id.get(m.tool_call_id or "")
            lines.append(digest_observation(tc.name if tc else "tool", tc.arguments if tc else {}, m.text))
        elif m.role is Role.USER:
            lines.append(f"harness note: {m.text.strip()[:120]}")
    return lines


def compact_messages(messages: Sequence[Message], *, keep_recent_steps: int, ledger: TaskLedger | None,
                     facts: dict[str, Any] | None) -> list[Message]:
    head, groups = _groups(messages)
    if len(groups) <= keep_recent_steps:
        return list(messages)
    old, recent = groups[: len(groups) - keep_recent_steps], groups[len(groups) - keep_recent_steps:]
    parts = ["[harness:compacted] Earlier steps were compacted. The ledger and digests below are authoritative; "
             "re-read a source with a tool if you need its full text."]
    if ledger is not None:
        parts.append(ledger.render())
    if facts:
        parts.append("Harness facts: " + json.dumps(facts, sort_keys=True))
    digest = [f"- {line}" for g in old for line in _summarize_group(g)]
    parts.append("Earlier steps, oldest first:\n" + "\n".join(digest))
    return [*head, Message.user("\n\n".join(parts)), *(m for g in recent for m in g)]


@dataclass
class CompactionStat:
    run_id: str
    step: int
    tokens_before: int
    tokens_after: int
    kept_steps: int


@dataclass
class CompactingLLM:
    """Decorates any LLMClient. The runtime's own message history is untouched (it is derived
    from events); only the request sent to the model is compacted."""

    inner: LLMClient
    max_input_tokens: int
    keep_recent_steps: int = 2
    ledger_source: Callable[[str], TaskLedger | None] | None = None
    facts_source: Callable[[str], dict[str, Any] | None] | None = None
    stats: list[CompactionStat] = field(default_factory=list)

    @property
    def provider(self) -> str:
        return getattr(self.inner, "provider", "compacting")

    def _rewrite(self, req: CompletionRequest) -> CompletionRequest:
        before = count_message_tokens(req.messages, req.model)
        if before <= self.max_input_tokens:
            return req
        run_id = str(req.metadata.get("run_id", ""))
        ledger = self.ledger_source(run_id) if self.ledger_source else None
        facts = self.facts_source(run_id) if self.facts_source else None
        messages, kept = list(req.messages), self.keep_recent_steps
        while kept >= 1:
            messages = compact_messages(req.messages, keep_recent_steps=kept, ledger=ledger, facts=facts)
            if count_message_tokens(messages, req.model) <= self.max_input_tokens:
                break
            kept -= 1
        after = count_message_tokens(messages, req.model)
        self.stats.append(CompactionStat(run_id, int(req.metadata.get("step", 0)), before, after, max(kept, 1)))
        return req.model_copy(update={"messages": messages})

    def complete(self, req: CompletionRequest) -> Completion:
        return self.inner.complete(self._rewrite(req))

    def stream(self, req: CompletionRequest) -> Iterator[StreamEvent]:
        return self.inner.stream(self._rewrite(req))

    async def acomplete(self, req: CompletionRequest) -> Completion:
        return await self.inner.acomplete(self._rewrite(req))

    async def astream(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        async for ev in self.inner.astream(self._rewrite(req)):
            yield ev


__all__ = [
    "LEDGER_ARTIFACT", "PlanItem", "LedgerUpdate", "TaskLedger", "ledger_from_events", "harness_facts",
    "make_ledger_tool", "digest_observation", "compact_messages", "CompactionStat", "CompactingLLM",
]
