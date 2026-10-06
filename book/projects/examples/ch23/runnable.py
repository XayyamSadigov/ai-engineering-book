# path: book/projects/examples/ch23/runnable.py
"""What LCEL-style composition does under the hood, in plain Python.

A `Runnable` is a function with a uniform calling convention (`invoke`,
`batch`, `stream`) and composition operators. `a | b` builds a sequence;
a dict of runnables builds a fan-out. Retries are a wrapper, not magic.
Nothing here is specific to LLMs: the "framework" is about a hundred lines of glue.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Iterator, Mapping, TypeVar

In = TypeVar("In")
Out = TypeVar("Out")


@dataclass
class RunLog:
    """Every step records what it saw. This is the 'tracing' a framework
    gives you for free; here it is explicit so you can see what is logged."""

    events: list[dict[str, Any]] = field(default_factory=list)

    def record(self, name: str, inp: Any, out: Any, ms: float, attempt: int) -> None:
        self.events.append({"step": name, "input": inp, "output": out,
                            "latency_ms": round(ms, 3), "attempt": attempt})


class Runnable(Generic[In, Out]):
    name: str = "runnable"

    def invoke(self, x: In, log: RunLog | None = None) -> Out:
        raise NotImplementedError

    def batch(self, xs: list[In], log: RunLog | None = None) -> list[Out]:
        return [self.invoke(x, log) for x in xs]

    def stream(self, x: In, log: RunLog | None = None) -> Iterator[Out]:
        yield self.invoke(x, log)   # default: one chunk; real steps override

    def __or__(self, other: "Runnable[Out, Any] | Callable[[Out], Any]") -> "Sequence":
        return Sequence([self, coerce(other)])

    def with_retry(self, max_attempts: int = 3, retry_on: tuple[type[BaseException], ...] = (Exception,),
                   base_delay_s: float = 0.0) -> "Retrying":
        return Retrying(self, max_attempts, retry_on, base_delay_s)


class Lambda(Runnable[In, Out]):
    def __init__(self, fn: Callable[[In], Out], name: str | None = None) -> None:
        self.fn, self.name = fn, name or getattr(fn, "__name__", "lambda")

    def invoke(self, x: In, log: RunLog | None = None) -> Out:
        t0 = time.perf_counter()
        out = self.fn(x)
        if log is not None:
            log.record(self.name, x, out, (time.perf_counter() - t0) * 1000, attempt=1)
        return out


class Sequence(Runnable[Any, Any]):
    name = "sequence"

    def __init__(self, steps: list[Runnable]) -> None:
        self.steps = steps

    def invoke(self, x: Any, log: RunLog | None = None) -> Any:
        for step in self.steps:
            x = step.invoke(x, log)
        return x

    def __or__(self, other: Runnable | Callable) -> "Sequence":
        return Sequence([*self.steps, coerce(other)])   # flatten instead of nesting


class Parallel(Runnable[Any, dict[str, Any]]):
    """Fan-out: run every branch on the same input, collect a dict."""
    name = "parallel"

    def __init__(self, branches: Mapping[str, Runnable | Callable]) -> None:
        self.branches = {k: coerce(v) for k, v in branches.items()}

    def invoke(self, x: Any, log: RunLog | None = None) -> dict[str, Any]:
        return {k: r.invoke(x, log) for k, r in self.branches.items()}


class Retrying(Runnable[In, Out]):
    def __init__(self, inner: Runnable[In, Out], max_attempts: int,
                 retry_on: tuple[type[BaseException], ...], base_delay_s: float) -> None:
        self.inner, self.max_attempts, self.retry_on, self.base_delay_s = inner, max_attempts, retry_on, base_delay_s
        self.name = f"retry({inner.name})"

    def invoke(self, x: In, log: RunLog | None = None) -> Out:
        attempt = 0
        while True:
            attempt += 1
            t0 = time.perf_counter()
            try:
                out = self.inner.invoke(x, log)
                if log is not None and attempt > 1:
                    log.record(self.name, x, out, (time.perf_counter() - t0) * 1000, attempt)
                return out
            except self.retry_on as exc:
                if log is not None:   # every failed attempt is visible, including the last one
                    log.record(self.name, x, f"error: {exc!r}", (time.perf_counter() - t0) * 1000, attempt)
                if attempt >= self.max_attempts:
                    raise
                time.sleep(self.base_delay_s * (2 ** (attempt - 1)))


def coerce(obj: Runnable | Callable | Mapping) -> Runnable:
    if isinstance(obj, Runnable):
        return obj
    if isinstance(obj, Mapping):
        return Parallel(obj)
    if callable(obj):
        return Lambda(obj)
    raise TypeError(f"cannot coerce {type(obj).__name__} to Runnable")
