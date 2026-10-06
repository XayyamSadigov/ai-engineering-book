# path: book/projects/reliability/reliability/circuit.py
"""Circuit breakers: stop calling a dependency that is failing, and probe for recovery.

States:
  CLOSED     calls flow; outcomes go into a rolling time window of buckets.
  OPEN       calls fail fast with CircuitOpenError for `open_s` seconds.
  HALF_OPEN  up to `half_open_max_calls` probe calls are let through; all succeed -> CLOSED,
             any failure -> OPEN again (with the open period doubled, up to `max_open_s`).

The window trips on a *rate* over a minimum volume, never on a single error: with
min_calls=20 and threshold 0.5, one failed request at 3 a.m. does not open the circuit.
Slow successes can count as failures (`slow_call_s`), because for an interactive AI path a
provider that answers in 40 seconds is as broken as one that answers 503.
"""
from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

from aie_core.llm.client import LLMClient
from aie_core.llm.errors import ContentFilterError, InvalidRequestError, LLMError, MalformedResponseError
from aie_core.llm.types import Completion, CompletionRequest, StreamEvent

from .clock import Clock
from .errors import CircuitOpenError, DeadlineExceeded, Overloaded

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


def default_is_failure(exc: BaseException) -> bool:
    """Count dependency faults; do not count the caller's own mistakes or our own shedding.

    An InvalidRequestError means *we* sent a bad request; opening the circuit would punish
    every other caller for one caller's bug. A DeadlineExceeded raised by our own budget
    check says nothing about the dependency. Malformed output and content refusals are
    model behavior, handled by repair and fallback, not by availability machinery.
    """
    if isinstance(exc, (InvalidRequestError, ContentFilterError, MalformedResponseError, DeadlineExceeded, Overloaded)):
        return False
    # Cancellation (a hedge loser, a deadline, a disconnected client) and interpreter exits say
    # nothing about the dependency's health.
    if isinstance(exc, (asyncio.CancelledError, GeneratorExit, KeyboardInterrupt, SystemExit)):
        return False
    return True


@dataclass
class _Bucket:
    start: float = 0.0
    calls: int = 0
    failures: int = 0


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        *,
        failure_rate_threshold: float = 0.5,
        min_calls: int = 20,
        window_s: float = 30.0,
        buckets: int = 10,
        open_s: float = 30.0,
        max_open_s: float = 300.0,
        half_open_max_calls: int = 3,
        slow_call_s: float | None = None,
        is_failure: Callable[[BaseException], bool] = default_is_failure,
        clock: Clock = time.monotonic,
        on_state_change: Callable[[str, CircuitState, CircuitState], None] | None = None,
    ) -> None:
        self.name = name
        self.failure_rate_threshold = failure_rate_threshold
        self.min_calls = min_calls
        self.window_s = window_s
        self.bucket_s = window_s / buckets
        self.base_open_s = open_s
        self.max_open_s = max_open_s
        self.half_open_max_calls = half_open_max_calls
        self.slow_call_s = slow_call_s
        self.is_failure = is_failure
        self._clock = clock
        self._on_change = on_state_change
        self._lock = threading.RLock()
        self._buckets: list[_Bucket] = [_Bucket() for _ in range(buckets)]
        self._state = CircuitState.CLOSED
        self._opened_at = 0.0
        self._open_s = open_s
        self._probes_in_flight = 0
        self._probe_successes = 0
        self._generation = 0   # bumped on every transition; outcomes from an older one are ignored
        self.rejected = 0

    # ------------------------------------------------------------------ window
    def _bucket(self, now: float) -> _Bucket:
        idx = int(now / self.bucket_s) % len(self._buckets)
        start = now - (now % self.bucket_s)
        b = self._buckets[idx]
        if b.start != start:  # stale bucket from a previous lap of the ring: reset it
            b.start, b.calls, b.failures = start, 0, 0
        return b

    def _window_totals(self, now: float) -> tuple[int, int]:
        calls = failures = 0
        for b in self._buckets:
            if now - b.start < self.window_s:
                calls += b.calls
                failures += b.failures
        return calls, failures

    def _reset_window(self) -> None:
        for b in self._buckets:
            b.start, b.calls, b.failures = 0.0, 0, 0

    # ------------------------------------------------------------------ transitions
    def _transition(self, new: CircuitState) -> None:
        old = self._state
        if old is new:
            return
        self._state = new
        self._generation += 1
        if new is CircuitState.OPEN:
            self._opened_at = self._clock()
        if new is CircuitState.HALF_OPEN:
            self._probes_in_flight = 0
            self._probe_successes = 0
        if new is CircuitState.CLOSED:
            self._open_s = self.base_open_s
            self._reset_window()
        if self._on_change is not None:
            self._on_change(self.name, old, new)

    def _maybe_half_open(self) -> None:
        if self._state is CircuitState.OPEN and self._clock() - self._opened_at >= self._open_s:
            self._transition(CircuitState.HALF_OPEN)

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._maybe_half_open()
            return self._state

    def retry_after(self) -> float:
        with self._lock:
            if self._state is not CircuitState.OPEN:
                return 0.0
            return max(0.0, self._open_s - (self._clock() - self._opened_at))

    # ------------------------------------------------------------------ protocol
    def allow(self) -> bool:
        """Ask permission for one call. A True in HALF_OPEN reserves a probe slot."""
        return self.admit() is not None

    def admit(self) -> int | None:
        """Like `allow`, but return the generation the call was admitted in (None if rejected).
        Pass it back to `record_*` so an outcome from before a state change is not misread as
        a probe result."""
        with self._lock:
            self._maybe_half_open()
            if self._state is CircuitState.CLOSED:
                return self._generation
            if self._state is CircuitState.HALF_OPEN and self._probes_in_flight < self.half_open_max_calls:
                self._probes_in_flight += 1
                return self._generation
            self.rejected += 1
            return None

    def record_success(self, duration_s: float = 0.0, *, generation: int | None = None) -> None:
        if self.slow_call_s is not None and duration_s > self.slow_call_s:
            self._record(failed=True, generation=generation)
        else:
            self._record(failed=False, generation=generation)

    def record_failure(self, exc: BaseException | None = None, *, generation: int | None = None) -> None:
        if exc is not None and not self.is_failure(exc):
            self.release(generation=generation)  # neutral outcome: free a probe slot, record nothing
            return
        self._record(failed=True, generation=generation)

    def release(self, *, generation: int | None = None) -> None:
        """Give back a probe slot without recording an outcome (e.g., the caller's own error)."""
        with self._lock:
            if generation is not None and generation != self._generation:
                return
            if self._state is CircuitState.HALF_OPEN and self._probes_in_flight > 0:
                self._probes_in_flight -= 1

    def _record(self, *, failed: bool, generation: int | None = None) -> None:
        with self._lock:
            now = self._clock()
            if generation is not None and generation != self._generation:
                return  # admitted before a state change: a straggler, not a probe or a fresh sample
            if self._state is CircuitState.HALF_OPEN:
                self._probes_in_flight = max(0, self._probes_in_flight - 1)
                if failed:
                    self._open_s = min(self.max_open_s, self._open_s * 2)
                    self._transition(CircuitState.OPEN)
                else:
                    self._probe_successes += 1
                    if self._probe_successes >= self.half_open_max_calls:
                        self._transition(CircuitState.CLOSED)
                return
            if self._state is CircuitState.OPEN:
                return  # a straggler from before the trip; ignore
            b = self._bucket(now)
            b.calls += 1
            b.failures += int(failed)
            calls, failures = self._window_totals(now)
            if calls >= self.min_calls and failures / calls >= self.failure_rate_threshold:
                self._transition(CircuitState.OPEN)

    # ------------------------------------------------------------------ wrappers
    def _reject(self) -> CircuitOpenError:
        return CircuitOpenError(self.name, retry_after_s=self.retry_after() or self._open_s)

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        gen = self.admit()
        if gen is None:
            raise self._reject()
        started = self._clock()
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:
            self.record_failure(exc, generation=gen)
            raise
        self.record_success(self._clock() - started, generation=gen)
        return result

    async def acall(self, fn: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        gen = self.admit()
        if gen is None:
            raise self._reject()
        started = self._clock()
        try:
            result = await fn(*args, **kwargs)
        except BaseException as exc:
            self.record_failure(exc, generation=gen)
            raise
        self.record_success(self._clock() - started, generation=gen)
        return result

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._maybe_half_open()
            calls, failures = self._window_totals(self._clock())
            return {
                "name": self.name,
                "state": self._state.value,
                "calls": calls,
                "failures": failures,
                "failure_rate": round(failures / calls, 4) if calls else 0.0,
                "retry_after_s": round(self.retry_after(), 3),
                "rejected": self.rejected,
            }


class CircuitBreakerRegistry:
    """One breaker per dependency name, created on first use with shared defaults.

    Name breakers after the *failure domain*, not the call site: "llm:primary",
    "retrieval:pgvector", "tool:ticketing". Two call sites that hit the same backend must
    share a breaker, or each will have to rediscover the outage on its own.
    """

    def __init__(self, **defaults: Any) -> None:
        self._defaults = defaults
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get(self, name: str, **overrides: Any) -> CircuitBreaker:
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, **{**self._defaults, **overrides})
            return self._breakers[name]

    def states(self) -> dict[str, CircuitState]:
        with self._lock:
            names = list(self._breakers)
        return {n: self._breakers[n].state for n in names}

    def open_dependencies(self) -> set[str]:
        return {n for n, s in self.states().items() if s is CircuitState.OPEN}

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            breakers = list(self._breakers.values())
        return [b.snapshot() for b in breakers]


class CircuitBreakerClient:
    """An LLMClient guarded by a breaker. Put one around each client *inside* a ModelGateway:

        gateway = ModelGateway(primary=CircuitBreakerClient(primary, reg.get("llm:primary")),
                               fallbacks=[CircuitBreakerClient(backup, reg.get("llm:backup"))])

    When the primary's circuit is open, it raises a retryable CircuitOpenError whose
    `retry_after_s` is the remaining open time. The gateway's own rule (do not sleep past
    the deadline) then sends the request straight to the fallback with no wasted backoff.
    """

    def __init__(self, inner: LLMClient, breaker: CircuitBreaker) -> None:
        self.inner = inner
        self.breaker = breaker
        self.provider = inner.provider
        self.default_model = getattr(inner, "default_model", None)
        self.supports_response_schema = getattr(inner, "supports_response_schema", False)

    def complete(self, req: CompletionRequest) -> Completion:
        return self.breaker.call(self.inner.complete, req)

    async def acomplete(self, req: CompletionRequest) -> Completion:
        return await self.breaker.acall(self.inner.acomplete, req)

    def stream(self, req: CompletionRequest) -> Iterator[StreamEvent]:
        gen = self.breaker.admit()
        if gen is None:
            raise self.breaker._reject()
        started = self.breaker._clock()
        try:
            yield from self.inner.stream(req)
        except LLMError as exc:
            self.breaker.record_failure(exc, generation=gen)
            raise
        except BaseException as exc:
            # Same rule as call(): cancellation and GeneratorExit are neutral, other errors count.
            self.breaker.record_failure(exc, generation=gen)
            raise
        self.breaker.record_success(self.breaker._clock() - started, generation=gen)

    async def astream(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        gen = self.breaker.admit()
        if gen is None:
            raise self.breaker._reject()
        started = self.breaker._clock()
        try:
            async for ev in self.inner.astream(req):
                yield ev
        except LLMError as exc:
            self.breaker.record_failure(exc, generation=gen)
            raise
        except BaseException as exc:
            # Same rule as call(): cancellation and GeneratorExit are neutral, other errors count.
            self.breaker.record_failure(exc, generation=gen)
            raise
        self.breaker.record_success(self.breaker._clock() - started, generation=gen)


__all__ = [
    "CircuitState",
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "CircuitBreakerClient",
    "default_is_failure",
]
