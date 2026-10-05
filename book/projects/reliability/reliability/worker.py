# path: book/projects/reliability/reliability/worker.py
"""Worker loop: lease, execute, ack or nack; stop gracefully; assume every job may run twice.

At-least-once delivery means a handler can be executed again after it already succeeded
(the worker died before `ack`, or its lease expired mid-run and another worker took over).
Three habits make that harmless, and `IdempotencyStore.once` supports the third:

1. Derive every external effect's key from the job, not from the attempt: an upsert keyed
   by `job.id` or by a content hash writes the same row twice, not two rows.
2. Check the lease at checkpoints (`ctx.checkpoint()`), so a worker whose lease was lost
   stops before doing more duplicate work.
3. Wrap non-idempotent side effects (send an email, post a reply, charge a card) in
   `once(key, fn)`, keyed by `f"{job.id}:{step}"`, so the second run returns the recorded
   result instead of acting again. Chapter 16 does the same for tool calls.

Graceful shutdown: on SIGTERM the worker stops leasing, lets the current handler reach its
next checkpoint (which raises `ShutdownRequested`), and *releases* the job back to the
queue without consuming an attempt. A handler that never checkpoints is still safe: if the
orchestrator kills the process, the lease expires and the job is redelivered.
"""
from __future__ import annotations

import signal
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from aie_core.observability import NoopTracer, Tracer

from .clock import Clock, Sleep
from .deadline import Deadline
from .errors import is_retryable
from .queue import Job, JobQueue, Lease


class ShutdownRequested(Exception):
    """Raised by `ctx.checkpoint()` when the worker is draining."""


class LeaseLost(Exception):
    """Raised by `ctx.checkpoint()`/`heartbeat()` when another worker may now own the job."""


class PermanentJobError(Exception):
    """Raise from a handler for failures that no retry can fix (bad payload, missing document)."""

    retryable = False


class Handler(Protocol):
    def __call__(self, job: Job, ctx: "JobContext") -> Any: ...


class WorkOutcome(str, Enum):
    SUCCEEDED = "succeeded"
    RETRY = "retry"
    DEAD = "dead"
    RELEASED = "released"
    LEASE_LOST = "lease_lost"


@dataclass
class WorkResult:
    job_id: str
    kind: str
    outcome: WorkOutcome
    attempt: int
    error: str | None = None


@dataclass
class JobContext:
    job: Job
    lease: Lease
    queue: JobQueue
    deadline: Deadline
    visibility_timeout_s: float
    _stop: threading.Event = field(repr=False, default_factory=threading.Event)
    _queue_clock: Clock = field(repr=False, default=time.time)

    @property
    def should_stop(self) -> bool:
        return self._stop.is_set()

    def heartbeat(self) -> None:
        """Extend the lease for long jobs. Call well before `visibility_timeout_s` elapses."""
        if not self.queue.extend(self.lease, self.visibility_timeout_s):
            raise LeaseLost(f"job {self.job.id}: lease could not be extended")

    def checkpoint(self) -> None:
        if self._stop.is_set():
            raise ShutdownRequested(self.job.id)
        if self._queue_clock() >= self.lease.expires_at:
            raise LeaseLost(f"job {self.job.id}: lease expired at checkpoint")


class Worker:
    def __init__(
        self,
        queue: JobQueue,
        handlers: dict[str, Handler],
        *,
        name: str = "worker",
        visibility_timeout_s: float = 60.0,
        job_timeout_s: float | None = None,
        poll_interval_s: float = 1.0,
        classify: Callable[[BaseException], bool] = is_retryable,
        queue_clock: Clock = time.time,
        clock: Clock = time.monotonic,
        sleep: Sleep = time.sleep,
        tracer: Tracer | None = None,
    ) -> None:
        self.queue = queue
        self.handlers = handlers
        self.name = name
        self.visibility_timeout_s = visibility_timeout_s
        self.job_timeout_s = job_timeout_s if job_timeout_s is not None else visibility_timeout_s * 0.8
        self.poll_interval_s = poll_interval_s
        self.classify = classify
        self._queue_clock = queue_clock
        self._clock = clock
        self._sleep = sleep
        self.tracer = tracer or NoopTracer()
        self._stop = threading.Event()
        self.stats: Counter[str] = Counter()

    # ------------------------------------------------------------------ lifecycle
    def request_shutdown(self, *_: Any) -> None:
        self._stop.set()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def install_signal_handlers(self) -> None:
        """Call from the main thread of the worker process."""
        signal.signal(signal.SIGTERM, self.request_shutdown)
        signal.signal(signal.SIGINT, self.request_shutdown)

    def run(self, *, max_jobs: int | None = None, max_idle_polls: int | None = None) -> int:
        """Process until shutdown (or the optional limits, which tests use). Returns jobs handled."""
        handled = idle = 0
        while not self._stop.is_set():
            result = self.run_once()
            if result is None:
                idle += 1
                if max_idle_polls is not None and idle >= max_idle_polls:
                    break
                self._sleep(self.poll_interval_s)
                continue
            idle = 0
            handled += 1
            if max_jobs is not None and handled >= max_jobs:
                break
        return handled

    # ------------------------------------------------------------------ one job
    def run_once(self) -> WorkResult | None:
        if self._stop.is_set():
            return None
        lease = self.queue.lease(self.visibility_timeout_s)
        if lease is None:
            return None
        job = lease.job
        with self.tracer.span("job.process", worker=self.name, job_id=job.id, kind=job.kind,
                              attempt=job.attempts, tenant_id=job.tenant_id) as span:
            result = self._execute(job, lease)
            span.set_attribute("outcome", result.outcome.value)
            if result.error:
                span.set_attribute("error", result.error)
        self.stats[result.outcome.value] += 1
        return result

    def _execute(self, job: Job, lease: Lease) -> WorkResult:
        def done(outcome: WorkOutcome, error: str | None = None) -> WorkResult:
            return WorkResult(job_id=job.id, kind=job.kind, outcome=outcome, attempt=job.attempts, error=error)

        handler = self.handlers.get(job.kind)
        if handler is None:
            state = self.queue.nack(lease, f"no handler for kind '{job.kind}'", retryable=False)
            return done(WorkOutcome.DEAD if state else WorkOutcome.LEASE_LOST, f"unknown kind {job.kind}")

        ctx = JobContext(job=job, lease=lease, queue=self.queue,
                         deadline=Deadline.after(self.job_timeout_s, name=f"job:{job.id}", clock=self._clock),
                         visibility_timeout_s=self.visibility_timeout_s, _stop=self._stop,
                         _queue_clock=self._queue_clock)
        try:
            with ctx.deadline.scope():
                value = handler(job, ctx)
        except ShutdownRequested:
            self.queue.release(lease)
            return done(WorkOutcome.RELEASED)
        except LeaseLost as exc:
            return done(WorkOutcome.LEASE_LOST, str(exc))  # do not ack or nack: we no longer own it
        except Exception as exc:  # noqa: BLE001 - classified below
            error = f"{type(exc).__name__}: {exc}"
            state = self.queue.nack(lease, error, retryable=self.classify(exc))
            if state is None:
                return done(WorkOutcome.LEASE_LOST, error)
            return done(WorkOutcome.RETRY if state.value == "queued" else WorkOutcome.DEAD, error)
        if not self.queue.ack(lease, value):
            # The work was done, but our lease had expired: another worker may be repeating it.
            # Correctness rests on the handler being idempotent; the counter makes it visible.
            return done(WorkOutcome.LEASE_LOST, "ack after lease expiry")
        return done(WorkOutcome.SUCCEEDED)


# ============================================================================ idempotency
class IdempotencyStore(Protocol):
    def get(self, key: str) -> tuple[bool, Any]: ...
    def put(self, key: str, value: Any) -> None: ...


class InMemoryIdempotencyStore:
    """Single-process store. In production use a table with a unique key, or Redis SET NX."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> tuple[bool, Any]:
        with self._lock:
            return (key in self._data, self._data.get(key))

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._data.setdefault(key, value)


def once(store: IdempotencyStore, key: str, fn: Callable[[], Any]) -> Any:
    """Run `fn` the first time `key` is seen; afterwards return the recorded result.

    The window between `fn()` and `put()` is the irreducible at-least-once gap: a crash
    there repeats the effect. Close it with a downstream that accepts the same key
    (an Idempotency-Key header on the email or ticketing API), not with more local code.
    """
    seen, value = store.get(key)
    if seen:
        return value
    value = fn()
    store.put(key, value)
    return value


__all__ = [
    "Worker",
    "WorkOutcome",
    "WorkResult",
    "JobContext",
    "Handler",
    "ShutdownRequested",
    "LeaseLost",
    "PermanentJobError",
    "IdempotencyStore",
    "InMemoryIdempotencyStore",
    "once",
]
