# path: book/projects/reliability/reliability/deadline.py
"""Deadlines: one end-to-end budget, propagated through every call.

A timeout answers "how long may *this* call take". A deadline answers "when does the user
stop caring". The request handler creates one Deadline; every stage receives it (as an
argument or through `current_deadline()`), derives a child budget for its own work, and
checks it before starting anything expensive. When the deadline passes, or the request is
cancelled (client disconnect, barge-in, job cancelled), every descendant sees it and stops.

Across process boundaries the deadline travels as *remaining milliseconds* (`to_header`),
never as an absolute timestamp, because two machines' clocks disagree but their notion of
"800 ms from now" agrees closely enough.
"""
from __future__ import annotations

import asyncio
import contextvars
import time
from collections.abc import Awaitable, Iterator
from contextlib import contextmanager
from typing import TypeVar

from aie_core.llm.types import CompletionRequest

from .clock import Clock
from .errors import DeadlineExceeded

T = TypeVar("T")
HEADER = "X-Request-Deadline-Ms"

_current: contextvars.ContextVar["Deadline | None"] = contextvars.ContextVar("reliability_deadline", default=None)


class Deadline:
    """An absolute expiry on a monotonic clock, plus a cancellation flag shared with children."""

    __slots__ = ("expires_at", "name", "_clock", "_parent", "_cancelled", "_reason")

    def __init__(self, expires_at: float, *, name: str = "request", clock: Clock = time.monotonic,
                 parent: "Deadline | None" = None) -> None:
        self.expires_at = expires_at
        self.name = name
        self._clock = clock
        self._parent = parent
        self._cancelled = False
        self._reason: str | None = None

    # ------------------------------------------------------------------ construction
    @classmethod
    def after(cls, seconds: float, *, name: str = "request", clock: Clock = time.monotonic) -> "Deadline":
        return cls(clock() + seconds, name=name, clock=clock)

    @classmethod
    def from_header(cls, value: str | None, *, default_s: float, max_s: float | None = None,
                    name: str = "request", clock: Clock = time.monotonic) -> "Deadline":
        """Accept an upstream budget, but never more than our own ceiling (`max_s`)."""
        seconds = default_s
        if value:
            try:
                seconds = max(0.0, float(value) / 1000.0)
            except ValueError:
                seconds = default_s
        if max_s is not None:
            seconds = min(seconds, max_s)
        return cls.after(seconds, name=name, clock=clock)

    def child(self, budget_s: float | None = None, *, fraction: float | None = None,
              reserve_s: float = 0.0, name: str | None = None) -> "Deadline":
        """A sub-budget for one stage. It can only be *shorter* than the parent.

        `budget_s` caps the stage (its planned share); `fraction` takes a share of what is
        left; `reserve_s` keeps time back for work that must happen after this stage
        (post-validation, persisting the answer). Cancelling the parent cancels the child.
        """
        remaining = max(0.0, self.remaining() - reserve_s)
        if fraction is not None:
            remaining *= fraction
        if budget_s is not None:
            remaining = min(remaining, budget_s)
        return Deadline(self._clock() + remaining, name=name or self.name, clock=self._clock, parent=self)

    # ------------------------------------------------------------------ queries
    def remaining(self) -> float:
        own = self.expires_at - self._clock()
        if self._parent is not None:
            own = min(own, self._parent.remaining())
        return max(0.0, own)

    @property
    def cancelled(self) -> bool:
        return self._cancelled or (self._parent is not None and self._parent.cancelled)

    @property
    def expired(self) -> bool:
        return self.cancelled or self.remaining() <= 0.0

    @property
    def cancel_reason(self) -> str | None:
        if self._cancelled:
            return self._reason
        return self._parent.cancel_reason if self._parent is not None else None

    def cancel(self, reason: str = "cancelled") -> None:
        self._cancelled = True
        self._reason = reason

    def check(self, stage: str | None = None, *, need_s: float = 0.0) -> None:
        """Raise before doing work that cannot finish in time. `need_s` is the stage's minimum useful time."""
        if self.cancelled:
            raise DeadlineExceeded(f"request {self.cancel_reason or 'cancelled'}", stage=stage)
        left = self.remaining()
        if left <= 0.0 or left < need_s:
            raise DeadlineExceeded(f"{left * 1000:.0f} ms left, stage needs {need_s * 1000:.0f} ms", stage=stage)

    def timeout(self, cap_s: float | None = None) -> float:
        """Per-call timeout for a client library: what is left, optionally capped."""
        left = self.remaining()
        return left if cap_s is None else min(cap_s, left)

    def fits(self, seconds: float) -> bool:
        return not self.cancelled and self.remaining() > seconds

    # ------------------------------------------------------------------ propagation
    def apply(self, req: CompletionRequest) -> CompletionRequest:
        """Stamp the remaining budget on an aie_core request; the gateway takes it from there."""
        self.check("model")
        left = self.remaining()
        timeout = left if req.timeout_s is None else min(req.timeout_s, left)
        return req.model_copy(update={"timeout_s": timeout})

    def to_header(self) -> dict[str, str]:
        return {HEADER: str(int(self.remaining() * 1000))}

    @contextmanager
    def scope(self) -> Iterator["Deadline"]:
        """Make this deadline `current_deadline()` for code that is not handed it explicitly."""
        token = _current.set(self)
        try:
            yield self
        finally:
            _current.reset(token)

    async def run(self, aw: Awaitable[T], stage: str | None = None) -> T:
        """Await with the remaining budget; a timeout here cancels the awaited task."""
        self.check(stage)
        try:
            return await asyncio.wait_for(aw, timeout=self.remaining())
        except asyncio.TimeoutError as exc:  # same class as builtin TimeoutError on 3.11+
            raise DeadlineExceeded("budget ran out while waiting", stage=stage) from exc

    def __repr__(self) -> str:
        state = "cancelled" if self.cancelled else f"{self.remaining():.3f}s left"
        return f"Deadline({self.name}, {state})"


def current_deadline() -> Deadline | None:
    return _current.get()


__all__ = ["Deadline", "current_deadline", "HEADER"]
