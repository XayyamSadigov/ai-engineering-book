# path: book/projects/examples/ch17/workflow_engine.py
"""A small, explicit workflow engine: typed state, nodes, conditional edges,
per-node retry policy, checkpoints, and pause/resume for human approval.

The engine is deliberately plain Python. Everything a library such as LangGraph
does for you is visible here in about 250 lines of code.
"""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import hashlib
import inspect
import json
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Generic, Literal, Protocol, TypeVar

from pydantic import BaseModel

END = "__end__"
S = TypeVar("S", bound=BaseModel)
Status = Literal["completed", "paused", "failed"]


# --- error classes a step may raise ---------------------------------------
class StepError(Exception):
    """Base class. `retryable` tells the engine whether a retry can help."""

    retryable: bool = False


class TransientError(StepError):
    """Infrastructure hiccup: timeout, rate limit, 5xx. Retry is reasonable."""

    retryable = True


class StepValidationError(StepError):
    """The step produced output that failed validation. Retrying the same
    input rarely helps; the graph should route, not loop blindly."""


class FatalError(StepError):
    """Do not retry, do not continue. Preserve state for a human."""


class StaleHandleError(ValueError):
    """The handle no longer matches the paused checkpoint: refuse to apply the decision."""


# --- idempotency key for side-effecting nodes -------------------------------
_STEP_KEY: contextvars.ContextVar[str] = contextvars.ContextVar("workflow_step_key")


def step_key() -> str:
    """Stable key for the node now running: `run_id:node:visit`.

    `visit` counts earlier *successful* runs of the same node in this run, so a
    node re-executed after a crash or a failed attempt gets the same key, and a
    node legitimately visited twice (a redraft loop) gets a new one.
    """
    return _STEP_KEY.get()


def state_hash(state: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()[:16]


# --- step and policy ------------------------------------------------------
class Step(Protocol[S]):
    def __call__(self, state: S) -> S | Awaitable[S]: ...


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 1
    base_delay_s: float = 0.0
    max_delay_s: float = 2.0
    retry_on: tuple[type[BaseException], ...] = (TransientError,)

    def delay(self, attempt: int) -> float:
        return min(self.max_delay_s, self.base_delay_s * (2 ** (attempt - 1)))


# --- checkpoints ----------------------------------------------------------
@dataclass(frozen=True)
class Checkpoint:
    run_id: str
    seq: int
    node: str          # node that just finished (or paused/failed before running)
    next_node: str     # where execution continues
    status: Status | Literal["running"]
    state: dict[str, Any]
    ts: float = field(default_factory=time.time)


class Checkpointer(Protocol):
    def save(self, cp: Checkpoint) -> None: ...
    def latest(self, run_id: str) -> Checkpoint | None: ...
    def history(self, run_id: str) -> list[Checkpoint]: ...


class InMemoryCheckpointer:
    def __init__(self) -> None:
        self._log: dict[str, list[Checkpoint]] = {}

    def save(self, cp: Checkpoint) -> None:
        self._log.setdefault(cp.run_id, []).append(cp)

    def latest(self, run_id: str) -> Checkpoint | None:
        log = self._log.get(run_id)
        return log[-1] if log else None

    def history(self, run_id: str) -> list[Checkpoint]:
        return list(self._log.get(run_id, []))


# --- run result -----------------------------------------------------------
@dataclass
class StepRecord:
    node: str
    attempts: int
    duration_ms: float
    error: str | None = None


@dataclass(frozen=True)
class ResumeHandle:
    run_id: str
    node: str                       # the approval node waiting for a decision
    seq: int | None = None          # seq of the paused checkpoint the human saw
    state_hash: str | None = None   # hash of the state the human saw


@dataclass
class RunResult(Generic[S]):
    run_id: str
    status: Status
    state: S
    trace: list[StepRecord]
    handle: ResumeHandle | None = None
    error: str | None = None


# --- graph ----------------------------------------------------------------
@dataclass
class _Node:
    fn: Step
    retry: RetryPolicy
    pause_before: bool
    on_decision: Callable[[Any, Any], Any] | None


class Graph(Generic[S]):
    """Directed graph of steps over one pydantic state type.

    Nodes receive the state and return a new state (or mutate and return it).
    Edges are static (`add_edge`) or conditional (`add_router`: a function of
    the state that returns the next node name or END).
    """

    def __init__(self, state_type: type[S], checkpointer: Checkpointer | None = None,
                 max_steps: int = 50, tracer: Any | None = None) -> None:
        self.state_type = state_type
        self.checkpointer = checkpointer or InMemoryCheckpointer()
        self.max_steps = max_steps
        self.tracer = tracer   # anything with .span(name, **attrs), e.g. an aie_core Tracer
        self._nodes: dict[str, _Node] = {}
        self._edges: dict[str, str | Callable[[S], str]] = {}
        self._entry: str | None = None

    # building --------------------------------------------------------
    def add_node(self, name: str, fn: Step, *, retry: RetryPolicy = RetryPolicy(),
                 pause_before: bool = False,
                 on_decision: Callable[[S, Any], S] | None = None) -> "Graph[S]":
        if name in self._nodes or name == END:
            raise ValueError(f"duplicate or reserved node name: {name}")
        self._nodes[name] = _Node(fn, retry, pause_before, on_decision)
        if self._entry is None:
            self._entry = name
        return self

    def add_edge(self, src: str, dst: str) -> "Graph[S]":
        self._check(src); self._check(dst)
        self._edges[src] = dst
        return self

    def add_router(self, src: str, router: Callable[[S], str]) -> "Graph[S]":
        self._check(src)
        self._edges[src] = router
        return self

    def set_entry(self, name: str) -> "Graph[S]":
        self._check(name)
        self._entry = name
        return self

    def _check(self, name: str) -> None:
        if name != END and name not in self._nodes:
            raise ValueError(f"unknown node: {name}")

    def _next(self, node: str, state: S) -> str:
        edge = self._edges.get(node, END)
        target = edge(state) if callable(edge) else edge
        self._check(target)
        return target

    # running ----------------------------------------------------------
    def run(self, state: S, run_id: str | None = None) -> RunResult[S]:
        return asyncio.run(self.arun(state, run_id))

    async def arun(self, state: S, run_id: str | None = None) -> RunResult[S]:
        if self._entry is None:
            raise ValueError("graph has no nodes")
        run_id = run_id or uuid.uuid4().hex[:12]
        return await self._execute(run_id, self._entry, state, seq=0, trace=[])

    def resume(self, handle: ResumeHandle, decision: Any) -> RunResult[S]:
        """Continue a paused run with a human decision."""
        return asyncio.run(self.aresume(handle, decision))

    async def aresume(self, handle: ResumeHandle, decision: Any) -> RunResult[S]:
        cp = self.checkpointer.latest(handle.run_id)
        if cp is None or cp.status != "paused" or cp.next_node != handle.node:
            raise StaleHandleError("no paused checkpoint matches this handle")
        if handle.seq is not None and cp.seq != handle.seq:
            raise StaleHandleError(f"handle is for pause seq {handle.seq}, run is paused at {cp.seq}")
        if handle.state_hash is not None and state_hash(cp.state) != handle.state_hash:
            raise StaleHandleError("paused state changed since the approver saw it")
        state = self.state_type.model_validate(cp.state)
        node = self._nodes[handle.node]
        if node.on_decision is not None:
            state = node.on_decision(state, decision)
        return await self._execute(handle.run_id, handle.node, state,
                                   seq=cp.seq + 1, trace=[], skip_pause=True)

    def resume_from_checkpoint(self, run_id: str) -> RunResult[S]:
        """Continue after a crash from the last durable checkpoint."""
        return asyncio.run(self.aresume_from_checkpoint(run_id))

    async def aresume_from_checkpoint(self, run_id: str) -> RunResult[S]:
        cp = self.checkpointer.latest(run_id)
        if cp is None:
            raise ValueError(f"no checkpoint for run {run_id}")
        if cp.status == "completed":
            return RunResult(run_id, "completed", self.state_type.model_validate(cp.state), [])
        state = self.state_type.model_validate(cp.state)
        return await self._execute(run_id, cp.next_node, state, seq=cp.seq + 1, trace=[])

    def replay(self, run_id: str) -> list[tuple[str, S]]:
        """States as they were after each node, from the checkpoint log."""
        return [(cp.node, self.state_type.model_validate(cp.state))
                for cp in self.checkpointer.history(run_id)]

    async def _execute(self, run_id: str, current: str, state: S, *, seq: int,
                       trace: list[StepRecord], skip_pause: bool = False) -> RunResult[S]:
        steps = 0
        visits = Counter(cp.node for cp in self.checkpointer.history(run_id)
                         if cp.status in ("running", "completed"))
        while current != END:
            if steps >= self.max_steps:
                self._save(run_id, seq, current, current, "failed", state)
                return RunResult(run_id, "failed", state, trace, error="max_steps exceeded")
            node = self._nodes[current]
            if node.pause_before and not skip_pause:
                cp = self._save(run_id, seq, current, current, "paused", state)
                handle = ResumeHandle(run_id, current, seq=seq, state_hash=state_hash(cp.state))
                return RunResult(run_id, "paused", state, trace, handle=handle)
            skip_pause = False
            token = _STEP_KEY.set(f"{run_id}:{current}:{visits[current]}")
            try:
                with self._span(run_id, current) as span:
                    record, state, err = await self._run_node(current, node, state)
                    nxt = current if err is not None else self._next(current, state)
                    if span is not None:
                        span.set_attribute("workflow.attempts", record.attempts)
                        span.set_attribute("workflow.status", "failed" if err else "ok")
                        span.set_attribute("workflow.next_node", nxt)
            finally:
                _STEP_KEY.reset(token)
            trace.append(record)
            if err is not None:
                self._save(run_id, seq, current, current, "failed", state)
                return RunResult(run_id, "failed", state, trace, error=f"{current}: {err}")
            self._save(run_id, seq, current, nxt, "completed" if nxt == END else "running", state)
            visits[current] += 1
            seq += 1; steps += 1; current = nxt
        return RunResult(run_id, "completed", state, trace)

    def _span(self, run_id: str, node: str) -> Any:
        if self.tracer is None:
            return contextlib.nullcontext(None)
        return self.tracer.span("workflow.node", run_id=run_id, node=node)

    async def _run_node(self, name: str, node: _Node, state: S) -> tuple[StepRecord, S, str | None]:
        started = time.perf_counter()
        attempt = 0
        while True:
            attempt += 1
            try:
                # Each attempt gets a copy: a step that mutates state and then raises
                # must not leak half-applied changes into the retry or the checkpoint.
                result = node.fn(state.model_copy(deep=True))
                if inspect.isawaitable(result):
                    result = await result
                ms = (time.perf_counter() - started) * 1000
                return StepRecord(name, attempt, ms), result, None
            except Exception as exc:  # noqa: BLE001 - we classify below
                can_retry = isinstance(exc, node.retry.retry_on) and attempt < node.retry.max_attempts
                if not can_retry:
                    ms = (time.perf_counter() - started) * 1000
                    return StepRecord(name, attempt, ms, error=repr(exc)), state, repr(exc)
                await asyncio.sleep(node.retry.delay(attempt))

    def _save(self, run_id: str, seq: int, node: str, nxt: str, status: Any, state: S) -> Checkpoint:
        cp = Checkpoint(run_id, seq, node, nxt, status, state.model_dump(mode="json"))
        self.checkpointer.save(cp)
        return cp
