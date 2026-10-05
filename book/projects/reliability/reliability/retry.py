# path: book/projects/reliability/reliability/retry.py
"""System-level retries: one policy for non-LLM dependencies, and a budget that caps them all.

`aie_core.ModelGateway` already retries single model calls (Chapter 3). This module covers
the rest of the system (retrieval, rerankers, tool backends, webhooks) and adds the piece
a per-call policy cannot see: how many retries the *whole service* is sending.

Retry amplification: if three layers each make up to three attempts, one user request can
become 3 * 3 * 3 = 27 calls to the failing dependency, exactly when it is least able to
serve them. Two rules prevent it. Retry at one layer only, the one closest to the failure.
And give every retrying layer a `RetryBudget`: retries may be at most a fixed fraction of
recent first attempts, so in an outage the retry traffic stays a small surcharge instead of
a multiplier.
"""
from __future__ import annotations

import asyncio
import random
import threading
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from aie_core.llm.gateway import RetryPolicy

from .clock import Clock, Sleep
from .deadline import Deadline
from .errors import DeadlineExceeded, is_retryable, retry_after

T = TypeVar("T")


class RetryBudget:
    """Token-bucket retry throttle (the scheme gRPC and Finagle use).

    Every first attempt deposits `ratio` tokens; every retry withdraws one. A small floor
    (`min_retries_per_s`) lets a quiet service still retry occasionally. With ratio 0.1, a
    service doing 100 requests per second may send at most about 10 retries per second, no
    matter how many layers or how many failures.
    """

    def __init__(self, ratio: float = 0.1, *, min_retries_per_s: float = 1.0, max_tokens: float = 100.0,
                 clock: Clock = time.monotonic) -> None:
        self.ratio = ratio
        self.min_retries_per_s = min_retries_per_s
        self.max_tokens = max_tokens
        self._clock = clock
        self._tokens = 0.0
        self._floor_tokens = min_retries_per_s
        self._updated = clock()
        self._lock = threading.Lock()
        self.retries_allowed = 0
        self.retries_denied = 0

    def _refill_floor(self) -> None:
        now = self._clock()
        self._floor_tokens = min(self.min_retries_per_s,
                                 self._floor_tokens + (now - self._updated) * self.min_retries_per_s)
        self._updated = now

    def record_request(self) -> None:
        with self._lock:
            self._tokens = min(self.max_tokens, self._tokens + self.ratio)

    def try_spend(self) -> bool:
        with self._lock:
            self._refill_floor()
            if self._tokens >= 1.0 - 1e-9:  # float sums of ratio drift below whole numbers
                self._tokens -= 1.0
            elif self._floor_tokens >= 1.0:
                self._floor_tokens -= 1.0
            else:
                self.retries_denied += 1
                return False
            self.retries_allowed += 1
            return True


def _plan(policy: RetryPolicy, exc: BaseException, attempt: int, rng: random.Random,
          classify: Callable[[BaseException], bool]) -> float | None:
    if attempt >= policy.max_attempts or not classify(exc):
        return None
    return policy.delay_for(attempt, retry_after(exc), rng)


def call_with_retry(
    fn: Callable[[], T],
    *,
    policy: RetryPolicy | None = None,
    budget: RetryBudget | None = None,
    deadline: Deadline | None = None,
    classify: Callable[[BaseException], bool] = is_retryable,
    sleep: Sleep = time.sleep,
    rng: random.Random | None = None,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> T:
    """Call `fn` with bounded, jittered retries under a deadline and a shared budget.

    Stops retrying when: the error is not retryable, attempts are spent, the budget denies
    the retry, or the backoff would not fit in the deadline. The last error is re-raised
    unchanged so the caller's classification still works.
    """
    policy = policy or RetryPolicy()
    rng = rng or random.Random()
    attempt = 0
    if budget is not None:
        budget.record_request()
    while True:
        attempt += 1
        if deadline is not None:
            deadline.check("retry")
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - classification decides
            if isinstance(exc, DeadlineExceeded):
                raise
            delay = _plan(policy, exc, attempt, rng, classify)
            if delay is None:
                raise
            if deadline is not None and not deadline.fits(delay):
                raise
            if budget is not None and not budget.try_spend():
                raise
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            sleep(delay)


async def acall_with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy | None = None,
    budget: RetryBudget | None = None,
    deadline: Deadline | None = None,
    classify: Callable[[BaseException], bool] = is_retryable,
    asleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rng: random.Random | None = None,
) -> T:
    policy = policy or RetryPolicy()
    rng = rng or random.Random()
    attempt = 0
    if budget is not None:
        budget.record_request()
    while True:
        attempt += 1
        if deadline is not None:
            deadline.check("retry")
        try:
            if deadline is not None:
                return await deadline.run(fn(), stage="retry")
            return await fn()
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, DeadlineExceeded):
                raise
            delay = _plan(policy, exc, attempt, rng, classify)
            if delay is None:
                raise
            if deadline is not None and not deadline.fits(delay):
                raise
            if budget is not None and not budget.try_spend():
                raise
            await asleep(delay)


async def ahedged(
    fn: Callable[[], Awaitable[T]],
    *,
    hedge_after_s: float,
    max_hedges: int = 1,
    budget: RetryBudget | None = None,
    deadline: Deadline | None = None,
    on_hedge: Callable[[int], None] | None = None,
) -> T:
    """Hedged request: start `fn`; if no copy has succeeded `hedge_after_s` later, start another
    and return the first success, cancelling the rest.

    Hedging cuts tail latency caused by independent per-replica slowness (a slow shard, a pause,
    a cold cache), not by overload. Use it only for idempotent, read-only calls: a hedged
    `send_reply` sends two replies. Set `hedge_after_s` near the dependency's p95, so that at most
    about 5% of calls send a second copy. Hedges spend from the same `RetryBudget` as retries:
    when every call is slow, which is what overload looks like, the budget runs dry and hedging
    switches itself off instead of doubling the load on a struggling dependency.
    """
    loop = asyncio.get_running_loop()
    start = loop.time()
    if budget is not None:
        budget.record_request()
    tasks: list[asyncio.Future[T]] = [asyncio.ensure_future(fn())]
    hedges = 0
    hedging = max_hedges > 0
    last_exc: BaseException | None = None
    try:
        while True:
            if deadline is not None:
                deadline.check("hedge")
            pending = [t for t in tasks if not t.done()]
            if not pending:
                assert last_exc is not None
                raise last_exc  # every copy failed; retrying is a different decision
            timeout: float | None = None
            if hedging:
                timeout = max(0.0, start + hedge_after_s * (hedges + 1) - loop.time())
            if deadline is not None:
                remaining = deadline.remaining()
                timeout = remaining if timeout is None else min(timeout, remaining)
            done, _ = await asyncio.wait(pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                exc = t.exception()
                if exc is None:
                    return t.result()
                last_exc = exc
            if done or not hedging:
                continue
            if deadline is not None and deadline.remaining() <= 0.0:
                continue  # the check at the top of the loop raises DeadlineExceeded
            if budget is not None and not budget.try_spend():
                hedging = False  # out of budget: wait for what is already running
                continue
            hedges += 1
            if on_hedge is not None:
                on_hedge(hedges)
            tasks.append(asyncio.ensure_future(fn()))
            hedging = hedges < max_hedges
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()  # a cancelled HTTP call stops costing
        await asyncio.gather(*tasks, return_exceptions=True)


__all__ = ["RetryPolicy", "RetryBudget", "call_with_retry", "acall_with_retry", "ahedged"]
