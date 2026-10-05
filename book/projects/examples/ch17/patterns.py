# path: book/projects/examples/ch17/patterns.py
"""Orchestration patterns as plain Python functions.

Each pattern is a few lines. The value is not the code; it is naming the
pattern so that a reviewer can see which one a workflow uses and what it
costs: sequence (n calls, n failure points), branch (one extra decision),
fan-out (latency of the slowest branch), map-reduce (n maps + 1 reduce),
retry (bounded attempts), fallback (second path, usually cheaper).
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Any, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def sequence(*steps: Callable[[T], T]) -> Callable[[T], T]:
    """Run steps in order, feeding each output into the next."""

    def run(state: T) -> T:
        for step in steps:
            state = step(state)
        return state

    return run


def branch(router: Callable[[T], str], routes: dict[str, Callable[[T], T]],
           default: Callable[[T], T] | None = None) -> Callable[[T], T]:
    """Pick one of several steps based on a decision function."""

    def run(state: T) -> T:
        key = router(state)
        step = routes.get(key, default)
        if step is None:
            raise KeyError(f"no route for {key!r} and no default")
        return step(state)

    return run


async def fan_out(tasks: Sequence[Callable[[], Awaitable[R]]], *,
                  max_concurrency: int = 8) -> list[R | BaseException]:
    """Run independent coroutines concurrently; return results in input order.

    Failures are returned, not raised, so the fan-in step decides what a
    partial result means for the workflow.
    """
    sem = asyncio.Semaphore(max_concurrency)

    async def guarded(task: Callable[[], Awaitable[R]]) -> R:
        async with sem:
            return await task()

    return list(await asyncio.gather(*(guarded(t) for t in tasks), return_exceptions=True))


async def map_reduce(items: Iterable[T], mapper: Callable[[T], Awaitable[R]],
                     reducer: Callable[[list[R]], Any], *, max_concurrency: int = 8) -> Any:
    """Map each item concurrently (one model call each), then reduce once.

    Raises the first mapper failure: a summary over a partial set of
    documents is a different, usually wrong, answer.
    """
    results = await fan_out([lambda i=i: mapper(i) for i in items], max_concurrency=max_concurrency)
    failures = [r for r in results if isinstance(r, BaseException)]
    if failures:
        raise failures[0]
    return reducer(results)  # type: ignore[arg-type]


def retry(fn: Callable[[], R], *, attempts: int = 3,
          retry_on: tuple[type[BaseException], ...] = (Exception,),
          sleep: Callable[[float], None] | None = None, base_delay_s: float = 0.0) -> R:
    """Call fn up to `attempts` times; only `retry_on` errors are retried."""
    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except retry_on as exc:
            last = exc
            if attempt < attempts and sleep is not None:
                sleep(base_delay_s * 2 ** (attempt - 1))
    assert last is not None
    raise last


def fallback(primary: Callable[[], R], secondary: Callable[[], R],
             on: tuple[type[BaseException], ...] = (Exception,)) -> R:
    """Try the primary path; on a listed error use the secondary path."""
    try:
        return primary()
    except on:
        return secondary()
