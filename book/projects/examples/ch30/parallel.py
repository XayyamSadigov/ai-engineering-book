# path: book/projects/examples/ch30/parallel.py
"""Run independent steps concurrently under a deadline, prefetch likely tool calls, micro-batch embeddings.

- `fan_out` starts every step at once, gives each min(its own timeout, time left), cancels
  stragglers at the deadline, and separates required steps (failure aborts) from optional ones
  (failure degrades). Latency becomes the slowest required step, not the sum.
- `Prefetcher` speculatively starts a read-only call before the model asks for it and hands over
  the result if the model's eventual request matches; unused prefetches are cancelled and counted,
  because speculation spends money.
- `MicroBatcher` coalesces concurrent single-item requests (for example query embeddings) into one
  batched call, bounded by a batch size and a maximum wait.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Generic, Hashable, Mapping, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class RequiredStepFailed(Exception):
    def __init__(self, step: str, cause: BaseException | None) -> None:
        super().__init__(f"required step {step!r} failed: {cause!r}")
        self.step = step
        self.cause = cause


@dataclass(frozen=True)
class Step:
    fn: Callable[[], Awaitable[Any]]
    timeout_s: float | None = None
    required: bool = True


@dataclass
class FanOutResult:
    results: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, BaseException] = field(default_factory=dict)
    timed_out: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    step_ms: dict[str, float] = field(default_factory=dict)

    @property
    def degraded(self) -> bool:
        return bool(self.errors or self.timed_out)


async def fan_out(steps: Mapping[str, Step], *, deadline_s: float, max_concurrency: int | None = None) -> FanOutResult:
    """Run `steps` concurrently; return when all finish or the deadline passes.

    Raises `RequiredStepFailed` if a required step errors or times out. Optional failures are
    reported in the result so the caller can answer in a degraded mode (e.g. without reranking).
    """
    loop = asyncio.get_running_loop()
    start = loop.time()
    end = start + deadline_s
    sem = asyncio.Semaphore(max_concurrency) if max_concurrency else None
    out = FanOutResult()

    async def run(name: str, step: Step) -> Any:
        async def call() -> Any:
            t0 = loop.time()
            try:
                remaining = max(0.0, end - loop.time())
                budget = remaining if step.timeout_s is None else min(step.timeout_s, remaining)
                return await asyncio.wait_for(step.fn(), timeout=budget)
            finally:
                out.step_ms[name] = (loop.time() - t0) * 1000

        if sem is None:
            return await call()
        async with sem:
            return await call()

    tasks = {name: asyncio.create_task(run(name, step), name=name) for name, step in steps.items()}
    required = {tasks[n] for n, s in steps.items() if s.required}
    pending: set[asyncio.Task[Any]] = set(tasks.values())
    try:
        while pending:
            done, pending = await asyncio.wait(pending, timeout=max(0.0, end - loop.time()),
                                               return_when=asyncio.FIRST_EXCEPTION)
            if not done:
                break                                          # the deadline passed
            if any(t in required and not t.cancelled() and t.exception() is not None for t in done):
                break                                          # a required step failed: stop the rest now
    finally:
        # Runs on deadline, on a required failure, and when the caller itself is cancelled (client
        # disconnect): a stopped HTTP call stops costing, and nothing outlives the request.
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    for name, task in tasks.items():
        if task in pending or task.cancelled():
            out.timed_out.append(name)
            continue
        exc = task.exception()
        if isinstance(exc, asyncio.TimeoutError):
            out.timed_out.append(name)  # the step's own timeout fired before the global deadline
        elif exc is not None:
            out.errors[name] = exc
        else:
            out.results[name] = task.result()
    out.elapsed_ms = (loop.time() - start) * 1000
    for name, step in steps.items():
        if step.required and name not in out.results:
            raise RequiredStepFailed(name, out.errors.get(name))
    return out


# ----------------------------------------------------------------------------- prefetch
@dataclass
class PrefetchStats:
    started: int = 0
    used: int = 0
    wasted: int = 0
    direct: int = 0  # requests that had no prefetch and ran on demand


class Prefetcher:
    """Speculative execution of read-only calls, keyed by (tool, normalized arguments).

    Only register tools that are read-only and idempotent: a speculative `send_reply` is a bug.
    """

    def __init__(self, read_only_tools: set[str]) -> None:
        self.read_only_tools = set(read_only_tools)
        self._inflight: dict[Hashable, asyncio.Task[Any]] = {}
        self.stats = PrefetchStats()

    @staticmethod
    def key(tool: str, args: Mapping[str, Any]) -> Hashable:
        return (tool, tuple(sorted((k, repr(v)) for k, v in args.items())))

    def start(self, tool: str, args: Mapping[str, Any], fn: Callable[[], Awaitable[Any]]) -> None:
        if tool not in self.read_only_tools:
            raise ValueError(f"refusing to prefetch {tool!r}: not registered as read-only")
        k = self.key(tool, args)
        if k not in self._inflight:
            self._inflight[k] = asyncio.create_task(fn())
            self.stats.started += 1

    async def take(self, tool: str, args: Mapping[str, Any], fn: Callable[[], Awaitable[Any]]) -> Any:
        """Return the prefetched result if one matches, else call `fn` now."""
        task = self._inflight.pop(self.key(tool, args), None)
        if task is not None:
            self.stats.used += 1
            return await task
        self.stats.direct += 1
        return await fn()

    async def cancel_unused(self) -> int:
        tasks = list(self._inflight.values())
        self._inflight.clear()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.stats.wasted += len(tasks)
        return len(tasks)


# ----------------------------------------------------------------------------- batching
class MicroBatcher(Generic[T, R]):
    """Coalesce concurrent `submit(item)` calls into `batch_fn(items)` calls.

    A batch is flushed when it reaches `max_batch` items or when the first item has waited
    `max_wait_ms`. The wait is the price of batching: it adds up to `max_wait_ms` to every
    request's latency, so keep it small on interactive paths.
    """

    def __init__(self, batch_fn: Callable[[list[T]], Awaitable[Sequence[R]]], *, max_batch: int = 32, max_wait_ms: float = 5.0) -> None:
        self.batch_fn = batch_fn
        self.max_batch = max_batch
        self.max_wait_s = max_wait_ms / 1000
        self._pending: list[tuple[T, asyncio.Future[R]]] = []
        self._timer: asyncio.TimerHandle | None = None
        self.batches: list[int] = []  # sizes of flushed batches, for tests and metrics
        self._tasks: set[asyncio.Task[None]] = set()

    async def submit(self, item: T) -> R:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[R] = loop.create_future()
        self._pending.append((item, fut))
        if len(self._pending) >= self.max_batch:
            self._flush()
        elif self._timer is None:
            self._timer = loop.call_later(self.max_wait_s, self._flush)
        return await fut

    def _flush(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if not self._pending:
            return
        batch, self._pending = self._pending, []
        self.batches.append(len(batch))
        task = asyncio.get_running_loop().create_task(self._run(batch))
        self._tasks.add(task)                      # keep a reference: the loop holds tasks only weakly
        task.add_done_callback(self._tasks.discard)

    async def _run(self, batch: list[tuple[T, asyncio.Future[R]]]) -> None:
        try:
            results = await self.batch_fn([item for item, _ in batch])
            if len(results) != len(batch):
                raise RuntimeError(f"batch_fn returned {len(results)} results for {len(batch)} items")
            for (_, fut), res in zip(batch, results):
                if not fut.done():
                    fut.set_result(res)
        except BaseException as exc:  # one failure fails the whole batch; callers retry individually
            for _, fut in batch:
                if not fut.done():
                    fut.set_exception(exc)


__all__ = ["Step", "FanOutResult", "fan_out", "RequiredStepFailed", "Prefetcher", "PrefetchStats", "MicroBatcher"]
