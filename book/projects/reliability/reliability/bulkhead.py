# path: book/projects/reliability/reliability/bulkhead.py
"""Bulkheads: separate concurrency pools per workload, so one workload cannot sink the rest.

Without bulkheads, a nightly evaluation run that opens 200 concurrent model calls takes
every slot of the gateway's semaphore, and an employee asking Northwind Assist a question
waits behind it. With bulkheads, "interactive", "batch", "ingestion", and "eval" each get
a fixed share; a full pool rejects its own work quickly instead of borrowing a neighbor's.

Each pool has `max_concurrent` running slots and a bounded waiting room (`max_waiting`,
`max_wait_s`). An unbounded waiting room is not a bulkhead, it is a queue that hides
overload until every waiter times out at once.
"""
from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass

from .deadline import Deadline
from .errors import BulkheadFullError


@dataclass
class BulkheadStats:
    name: str
    max_concurrent: int
    in_flight: int = 0
    waiting: int = 0
    admitted: int = 0
    rejected: int = 0

    @property
    def utilization(self) -> float:
        return self.in_flight / self.max_concurrent if self.max_concurrent else 1.0


class Bulkhead:
    """Thread-based pool. Use from synchronous code (worker threads, sync handlers)."""

    def __init__(self, name: str, max_concurrent: int, *, max_waiting: int = 0, max_wait_s: float = 0.0) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self.stats = BulkheadStats(name=name, max_concurrent=max_concurrent)
        self.max_waiting = max_waiting
        self.max_wait_s = max_wait_s
        self._cond = threading.Condition()  # waits use real time: a blocked thread needs a real timeout

    @property
    def name(self) -> str:
        return self.stats.name

    def _reject(self, why: str) -> BulkheadFullError:
        self.stats.rejected += 1
        return BulkheadFullError(f"bulkhead '{self.name}' {why}", reason=f"bulkhead_full:{self.name}",
                                 retry_after_s=1.0)

    def try_acquire(self) -> bool:
        with self._cond:
            if self.stats.in_flight < self.stats.max_concurrent:
                self.stats.in_flight += 1
                self.stats.admitted += 1
                return True
            return False

    def acquire(self, deadline: Deadline | None = None) -> None:
        with self._cond:
            if self.stats.in_flight < self.stats.max_concurrent:
                self.stats.in_flight += 1
                self.stats.admitted += 1
                return
            if self.stats.waiting >= self.max_waiting or self.max_wait_s <= 0:
                raise self._reject("is full")
            wait_s = self.max_wait_s if deadline is None else min(self.max_wait_s, deadline.remaining())
            end = time.monotonic() + wait_s
            self.stats.waiting += 1
            try:
                while self.stats.in_flight >= self.stats.max_concurrent:
                    left = end - time.monotonic()
                    if left <= 0:
                        raise self._reject(f"wait exceeded {wait_s:.2f}s")
                    self._cond.wait(timeout=left)
                self.stats.in_flight += 1
                self.stats.admitted += 1
            finally:
                self.stats.waiting -= 1

    def release(self) -> None:
        with self._cond:
            self.stats.in_flight -= 1
            self._cond.notify()

    @contextmanager
    def slot(self, deadline: Deadline | None = None) -> Iterator[None]:
        self.acquire(deadline)
        try:
            yield
        finally:
            self.release()


class AsyncBulkhead:
    """Event-loop pool with the same semantics. One instance belongs to one event loop."""

    def __init__(self, name: str, max_concurrent: int, *, max_waiting: int = 0, max_wait_s: float = 0.0) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self.stats = BulkheadStats(name=name, max_concurrent=max_concurrent)
        self.max_waiting = max_waiting
        self.max_wait_s = max_wait_s
        self._cond: asyncio.Condition | None = None

    @property
    def name(self) -> str:
        return self.stats.name

    def _condition(self) -> asyncio.Condition:
        if self._cond is None:
            self._cond = asyncio.Condition()
        return self._cond

    def _reject(self, why: str) -> BulkheadFullError:
        self.stats.rejected += 1
        return BulkheadFullError(f"bulkhead '{self.name}' {why}", reason=f"bulkhead_full:{self.name}",
                                 retry_after_s=1.0)

    async def acquire(self, deadline: Deadline | None = None) -> None:
        cond = self._condition()
        async with cond:
            if self.stats.in_flight < self.stats.max_concurrent:
                self.stats.in_flight += 1
                self.stats.admitted += 1
                return
            if self.stats.waiting >= self.max_waiting or self.max_wait_s <= 0:
                raise self._reject("is full")
            wait_s = self.max_wait_s if deadline is None else min(self.max_wait_s, deadline.remaining())
            self.stats.waiting += 1
            try:
                await asyncio.wait_for(
                    cond.wait_for(lambda: self.stats.in_flight < self.stats.max_concurrent), timeout=wait_s
                )
                self.stats.in_flight += 1
                self.stats.admitted += 1
            except asyncio.TimeoutError:
                raise self._reject(f"wait exceeded {wait_s:.2f}s") from None
            finally:
                self.stats.waiting -= 1

    async def release(self) -> None:
        cond = self._condition()
        async with cond:
            self.stats.in_flight -= 1
            cond.notify()

    @asynccontextmanager
    async def slot(self, deadline: Deadline | None = None) -> AsyncIterator[None]:
        await self.acquire(deadline)
        try:
            yield
        finally:
            await self.release()


class Bulkheads:
    """Named pools for the workloads of one process. Unknown workloads are a configuration bug."""

    def __init__(self, sizes: dict[str, int], *, max_waiting: dict[str, int] | None = None,
                 max_wait_s: dict[str, float] | None = None, use_async: bool = False) -> None:
        max_waiting = max_waiting or {}
        max_wait_s = max_wait_s or {}
        cls = AsyncBulkhead if use_async else Bulkhead
        self._pools: dict[str, Bulkhead | AsyncBulkhead] = {
            name: cls(name, size, max_waiting=max_waiting.get(name, 0), max_wait_s=max_wait_s.get(name, 0.0))
            for name, size in sizes.items()
        }

    def __getitem__(self, workload: str) -> Bulkhead | AsyncBulkhead:
        try:
            return self._pools[workload]
        except KeyError:
            raise KeyError(f"no bulkhead configured for workload '{workload}'") from None

    def stats(self) -> list[BulkheadStats]:
        return [p.stats for p in self._pools.values()]


__all__ = ["Bulkhead", "AsyncBulkhead", "Bulkheads", "BulkheadStats"]
