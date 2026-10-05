# path: book/projects/reliability/reliability/chaos.py
"""Fault injection for tests: outages, rate limits, latency, and malformed output on a schedule.

A chaos test asserts *system* properties under injected failure, for example "no request
makes more than six model calls", "an outage of the primary provider never fails a request
that the backup can serve", or "a retrieval outage yields a PARTIAL answer, never a 500".
Faults are scheduled on the same injectable clock the system uses, so a five-minute outage
runs in microseconds and the test is deterministic.
"""
from __future__ import annotations

import random
import time
from collections import Counter
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from aie_core.llm.client import LLMClient
from aie_core.llm.errors import ProviderUnavailableError, RateLimitError
from aie_core.llm.errors import TimeoutError as LLMTimeoutError
from aie_core.llm.types import Completion, CompletionRequest, Message, StreamEvent, Usage

from .clock import Clock, Sleep

T = TypeVar("T")
FaultKind = Literal["error", "latency", "malformed"]


@dataclass
class Fault:
    kind: FaultKind
    error: Callable[[], BaseException] | None = None
    latency_s: float = 0.0
    text: str = '{"category": "net'   # truncated JSON, the most common malformed shape


@dataclass
class FaultWindow:
    start_s: float
    end_s: float
    fault: Fault
    probability: float = 1.0


def outage(start_s: float, end_s: float, *, probability: float = 1.0, dependency: str = "provider") -> FaultWindow:
    return FaultWindow(start_s, end_s, Fault("error", error=lambda: ProviderUnavailableError(
        f"injected outage of {dependency}", status_code=503, provider=dependency)), probability)


def rate_limited(start_s: float, end_s: float, *, retry_after_s: float = 2.0, probability: float = 1.0) -> FaultWindow:
    return FaultWindow(start_s, end_s, Fault("error", error=lambda: RateLimitError(
        "injected 429", status_code=429, retry_after_s=retry_after_s)), probability)


def slow(start_s: float, end_s: float, latency_s: float, *, probability: float = 1.0) -> FaultWindow:
    return FaultWindow(start_s, end_s, Fault("latency", latency_s=latency_s), probability)


def timeouts(start_s: float, end_s: float, *, probability: float = 1.0) -> FaultWindow:
    return FaultWindow(start_s, end_s, Fault("error", error=lambda: LLMTimeoutError("injected timeout")), probability)


def malformed(start_s: float, end_s: float, *, text: str = '{"category": "net', probability: float = 1.0) -> FaultWindow:
    return FaultWindow(start_s, end_s, Fault("malformed", text=text), probability)


class FaultPlan:
    """Windows are relative to the plan's creation time on `clock`. Overlaps: first match wins."""

    def __init__(self, windows: list[FaultWindow] | None = None, *, clock: Clock = time.monotonic,
                 seed: int = 0) -> None:
        self.windows = list(windows or [])
        self._clock = clock
        self._t0 = clock()
        self._rng = random.Random(seed)
        self.injected: Counter[str] = Counter()

    def add(self, window: FaultWindow) -> None:
        self.windows.append(window)

    def elapsed(self) -> float:
        return self._clock() - self._t0

    def fault_now(self) -> Fault | None:
        t = self.elapsed()
        for w in self.windows:
            if w.start_s <= t < w.end_s and self._rng.random() < w.probability:
                self.injected[w.fault.kind] += 1
                return w.fault
        return None


class ChaosLLM:
    """An LLMClient wrapper that consults a FaultPlan before delegating."""

    def __init__(self, inner: LLMClient, plan: FaultPlan, *, sleep: Sleep = time.sleep) -> None:
        self.inner = inner
        self.plan = plan
        self._sleep = sleep
        self.provider = inner.provider
        self.default_model = getattr(inner, "default_model", None)
        self.supports_response_schema = getattr(inner, "supports_response_schema", False)
        self.calls = 0

    def _before(self, req: CompletionRequest) -> Completion | None:
        self.calls += 1
        fault = self.plan.fault_now()
        if fault is None:
            return None
        if fault.kind == "error":
            assert fault.error is not None
            raise fault.error()
        if fault.kind == "latency":
            if req.timeout_s is not None and fault.latency_s >= req.timeout_s:
                self._sleep(req.timeout_s)  # the client library would give up at its timeout
                raise LLMTimeoutError(f"injected latency {fault.latency_s}s exceeded timeout {req.timeout_s:.2f}s")
            self._sleep(fault.latency_s)
            return None
        return Completion(message=Message.assistant(fault.text), usage=Usage(), finish_reason="length",
                          model=req.model or str(self.default_model), provider=self.provider, latency_ms=1.0)

    def complete(self, req: CompletionRequest) -> Completion:
        injected = self._before(req)
        return injected if injected is not None else self.inner.complete(req)

    async def acomplete(self, req: CompletionRequest) -> Completion:
        injected = self._before(req)
        return injected if injected is not None else await self.inner.acomplete(req)

    def stream(self, req: CompletionRequest) -> Iterator[StreamEvent]:
        injected = self._before(req)
        if injected is not None:
            yield StreamEvent(type="text_delta", text=injected.text)
            yield StreamEvent(type="done", finish_reason="length")
            return
        yield from self.inner.stream(req)

    async def astream(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        for ev in self.stream(req):
            yield ev


class ChaosFunction:
    """Wraps any synchronous dependency (retriever, tool backend) with the same plan semantics."""

    def __init__(self, fn: Callable[..., T], plan: FaultPlan, *, sleep: Sleep = time.sleep) -> None:
        self.fn = fn
        self.plan = plan
        self._sleep = sleep
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        fault = self.plan.fault_now()
        if fault is not None:
            if fault.kind == "error":
                assert fault.error is not None
                raise fault.error()
            if fault.kind == "latency":
                self._sleep(fault.latency_s)
            if fault.kind == "malformed":
                return fault.text
        return self.fn(*args, **kwargs)


__all__ = [
    "Fault",
    "FaultWindow",
    "FaultPlan",
    "ChaosLLM",
    "ChaosFunction",
    "outage",
    "rate_limited",
    "slow",
    "timeouts",
    "malformed",
]
