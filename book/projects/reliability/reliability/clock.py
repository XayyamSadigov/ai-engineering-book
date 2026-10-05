# path: book/projects/reliability/reliability/clock.py
"""Injectable time. Every primitive takes `clock` (returns seconds) and, where it waits, `sleep`.

Production code uses the defaults (`time.monotonic`, `time.sleep`). Tests use `ManualClock`,
whose `sleep` advances time instead of blocking, so a test that simulates a thirty-second
circuit-breaker window runs in microseconds and never flakes.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable

Clock = Callable[[], float]
Sleep = Callable[[float], None]


class ManualClock:
    """A clock that only moves when told to. Thread-safe; `sleep` records every call."""

    def __init__(self, start: float = 1_000.0) -> None:
        self._now = start
        self._lock = threading.Lock()
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("time does not go backwards")
        with self._lock:
            self._now += seconds

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.advance(max(0.0, seconds))

    async def asleep(self, seconds: float) -> None:
        self.sleep(seconds)


monotonic: Clock = time.monotonic

__all__ = ["Clock", "Sleep", "ManualClock", "monotonic"]
