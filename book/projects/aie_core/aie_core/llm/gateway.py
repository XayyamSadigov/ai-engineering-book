# path: book/projects/aie_core/aie_core/llm/gateway.py
"""ModelGateway: the reliability layer that wraps any LLMClient.

Responsibilities, in request order: exact-match cache lookup, admission (rate limiter,
concurrency semaphore), bounded retries with exponential backoff and full jitter under a
single deadline, fallback to the next client on retryable failures, cost accounting, and
one tracing span per attempt. Nothing here knows about any provider's wire format.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import random
import threading
import time
import weakref
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import AsyncIterator, Iterator, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel

from ..observability import NoopTracer, Span, Tracer
from .client import LLMClient
from .errors import LLMError, TimeoutError
from .tokens import count_message_tokens
from .types import Completion, CompletionRequest, StreamEvent, Usage

SPAN_NAME = "llm.complete"


# ---------------------------------------------------------------------------- retry policy
class RetryPolicy(BaseModel):
    max_attempts: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0
    jitter: bool = True
    retry_on_timeout: bool = True  # set False for non-idempotent calls (tool side effects)

    def delay_for(self, attempt: int, retry_after_s: float | None, rng: random.Random) -> float:
        """Attempt is 1-based. Full jitter: uniform(0, min(cap, base * 2**(attempt-1)))."""
        exp = min(self.max_delay_s, self.base_delay_s * (2 ** (attempt - 1)))
        delay = rng.uniform(0.0, exp) if self.jitter else exp
        if retry_after_s is not None:
            # Honor the provider's hint; jitter on top avoids every client retrying at once.
            delay = retry_after_s + (rng.uniform(0.0, min(1.0, retry_after_s * 0.1)) if self.jitter else 0.0)
        return delay

    def should_retry(self, err: LLMError, attempt: int) -> bool:
        if attempt >= self.max_attempts or not err.retryable:
            return False
        if isinstance(err, TimeoutError) and not self.retry_on_timeout:
            return False
        return True


# ---------------------------------------------------------------------------- rate limiter
class _Bucket:
    def __init__(self, per_minute: float, clock: Callable[[], float]) -> None:
        self.capacity = float(per_minute)
        self.rate = per_minute / 60.0
        self.tokens = float(per_minute)
        self.clock = clock
        self.updated = clock()

    def _refill(self) -> None:
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now

    def wait_time(self, amount: float) -> float:
        self._refill()
        amount = min(amount, self.capacity)  # a request bigger than the bucket must still pass
        if self.tokens >= amount:
            return 0.0
        return (amount - self.tokens) / self.rate

    def consume(self, amount: float) -> None:
        self.tokens -= min(amount, self.capacity)


class RateLimiter:
    """Token bucket with two dimensions, requests per minute and tokens per minute.

    Blocking `acquire` sleeps until both buckets can admit the request. Capacity equals the
    per-minute limit, so a cold limiter admits a one-minute burst; lower `burst` to smooth it.
    """

    def __init__(
        self,
        requests_per_minute: float | None = None,
        tokens_per_minute: float | None = None,
        *,
        burst: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        asleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._clock = clock
        self._sleep = sleep
        self._asleep = asleep
        self._lock = threading.Lock()
        self._req = _Bucket(requests_per_minute, clock) if requests_per_minute else None
        self._tok = _Bucket(tokens_per_minute, clock) if tokens_per_minute else None
        if burst is not None:
            for b in (self._req, self._tok):
                if b is not None:
                    b.capacity = min(b.capacity, float(burst))
                    b.tokens = min(b.tokens, b.capacity)

    def _reserve(self, tokens: int) -> float:
        """Return the wait needed; consume immediately when zero."""
        with self._lock:
            wait = 0.0
            if self._req is not None:
                wait = max(wait, self._req.wait_time(1))
            if self._tok is not None and tokens > 0:
                wait = max(wait, self._tok.wait_time(tokens))
            if wait == 0.0:
                if self._req is not None:
                    self._req.consume(1)
                if self._tok is not None and tokens > 0:
                    self._tok.consume(tokens)
            return wait

    def try_acquire(self, tokens: int = 0) -> bool:
        return self._reserve(tokens) == 0.0

    def acquire(self, tokens: int = 0, *, deadline: float | None = None) -> None:
        while True:
            wait = self._reserve(tokens)
            if wait == 0.0:
                return
            if deadline is not None and self._clock() + wait > deadline:
                raise TimeoutError(f"rate limiter wait {wait:.2f}s exceeds deadline", retryable=False)
            self._sleep(wait)

    async def aacquire(self, tokens: int = 0, *, deadline: float | None = None) -> None:
        while True:
            wait = self._reserve(tokens)
            if wait == 0.0:
                return
            if deadline is not None and self._clock() + wait > deadline:
                raise TimeoutError(f"rate limiter wait {wait:.2f}s exceeds deadline", retryable=False)
            await self._asleep(wait)


# ---------------------------------------------------------------------------- response cache
@runtime_checkable
class ResponseCache(Protocol):
    def get(self, key: str) -> Completion | None: ...

    def set(self, key: str, completion: Completion, ttl_s: float | None = None) -> None: ...


class InMemoryResponseCache:
    """LRU with per-entry TTL. Fine for one process; use Redis behind the same protocol otherwise."""

    def __init__(self, max_size: int = 1024, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.max_size = max_size
        self._clock = clock
        self._data: OrderedDict[str, tuple[Completion, float | None]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Completion | None:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self.misses += 1
                return None
            completion, expires = item
            if expires is not None and self._clock() >= expires:
                del self._data[key]
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return completion

    def set(self, key: str, completion: Completion, ttl_s: float | None = None) -> None:
        with self._lock:
            expires = self._clock() + ttl_s if ttl_s is not None else None
            self._data[key] = (completion, expires)
            self._data.move_to_end(key)
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)


def cache_key(req: CompletionRequest, model: str | None) -> str:
    """Everything that changes the answer goes into the key. `metadata` and `timeout_s` do not,
    with one exception: `metadata["cache_scope"]` (for example "tenant:retail" or
    "tenant:retail/user:emp-4471") partitions the cache when permissions are enforced outside
    the prompt, so identical requests from different scopes never share an entry."""
    material = {
        "model": model,
        "messages": [m.model_dump(mode="json", exclude_none=True) for m in req.messages],
        "tools": [t.model_dump(mode="json") for t in req.tools] if req.tools else None,
        "tool_choice": req.tool_choice,
        "response_schema": req.response_schema,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "stop": req.stop,
    }
    scope = req.metadata.get("cache_scope")
    if scope is not None:  # added only when present, so unscoped keys are unchanged
        material["scope"] = str(scope)
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------- pricing
class PricingTable:
    """USD per one million tokens, keyed by model name. All numbers you load are illustrative."""

    def __init__(self, prices: dict[str, dict[str, float]] | None = None, *, default: dict[str, float] | None = None) -> None:
        self._prices = {k: dict(v) for k, v in (prices or {}).items()}
        self._default = dict(default) if default else None

    @classmethod
    def from_json(cls, path: str) -> "PricingTable":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data.get("models", data), default=data.get("default"))

    def lookup(self, model: str) -> dict[str, float] | None:
        if model in self._prices:
            return self._prices[model]
        for name, price in self._prices.items():  # allow prefix matches such as "gpt-x" for "gpt-x-2026-01"
            if model.startswith(name):
                return price
        return self._default

    def cost_usd(self, model: str, usage: Usage) -> float:
        price = self.lookup(model)
        if price is None:
            return 0.0
        input_per_m = float(price.get("input_per_1m", 0.0))
        cached_per_m = float(price.get("cached_input_per_1m", input_per_m))
        output_per_m = float(price.get("output_per_1m", 0.0))
        uncached = max(0, usage.input_tokens - usage.cached_input_tokens)
        return (
            uncached * input_per_m + usage.cached_input_tokens * cached_per_m + usage.output_tokens * output_per_m
        ) / 1_000_000


# ---------------------------------------------------------------------------- gateway
def _default_token_estimate(req: CompletionRequest) -> int:
    """Conservative admission estimate: prompt tokens plus the whole output budget."""
    return count_message_tokens(req.messages, req.model) + req.max_tokens


def _chain_first(first: StreamEvent | None, rest: Iterator[StreamEvent]) -> Iterator[StreamEvent]:
    if first is not None:
        yield first
    yield from rest


@dataclass
class _OpenStream:
    """A stream whose first event has arrived. From here on the gateway only relays."""

    iterator: Iterator[StreamEvent] | AsyncIterator[StreamEvent]
    first: StreamEvent | None
    span: Span
    started: float
    async_sem: asyncio.Semaphore | None = None  # the slot to release, for async streams


class ModelGateway:
    provider: str

    def __init__(
        self,
        primary: LLMClient,
        fallbacks: Sequence[LLMClient] = (),
        retry: RetryPolicy | None = None,
        rate_limiter: RateLimiter | None = None,
        cache: ResponseCache | None = None,
        max_concurrency: int = 16,
        pricing: PricingTable | None = None,
        tracer: Tracer | None = None,
        default_timeout_s: float = 60.0,
        *,
        cache_ttl_s: float | None = 3600.0,
        cache_max_temperature: float = 0.0,
        require_cache_scope: bool = False,
        token_estimator: Callable[[CompletionRequest], int] = _default_token_estimate,
        sleep: Callable[[float], None] = time.sleep,
        asleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
    ) -> None:
        self.primary = primary
        self.fallbacks = list(fallbacks)
        self.retry = retry or RetryPolicy()
        self.rate_limiter = rate_limiter
        self.cache = cache
        self.pricing = pricing
        self.tracer = tracer or NoopTracer()
        self.default_timeout_s = default_timeout_s
        self.cache_ttl_s = cache_ttl_s
        self.cache_max_temperature = cache_max_temperature
        self.require_cache_scope = require_cache_scope
        self.token_estimator = token_estimator
        self._sleep = sleep
        self._asleep = asleep
        self._clock = clock
        self._rng = rng or random.Random()
        self._sem = threading.BoundedSemaphore(max_concurrency)
        self._max_concurrency = max_concurrency
        # asyncio primitives belong to one event loop, so keep one semaphore per loop.
        self._asems: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = (
            weakref.WeakKeyDictionary()
        )
        self.provider = primary.provider
        self.supports_response_schema = getattr(primary, "supports_response_schema", True)

    # ------------------------------------------------------------------ small helpers
    @property
    def clients(self) -> list[LLMClient]:
        return [self.primary, *self.fallbacks]

    def _model_for(self, client: LLMClient, req: CompletionRequest) -> str | None:
        return req.model or getattr(client, "default_model", None)

    def _deadline(self, req: CompletionRequest) -> float:
        return self._clock() + (req.timeout_s if req.timeout_s is not None else self.default_timeout_s)

    def _with_remaining(self, req: CompletionRequest, deadline: float) -> CompletionRequest:
        """Deadline propagation: each attempt may use only what is left of the whole budget."""
        remaining = deadline - self._clock()
        if remaining <= 0:
            raise TimeoutError("deadline exhausted before the attempt started", retryable=False)
        return req.model_copy(update={"timeout_s": remaining})

    def _plan_retry(self, err: LLMError, attempt: int) -> float | None:
        """Backoff delay before retrying this client, or None when the policy says stop."""
        if not self.retry.should_retry(err, attempt):
            return None
        return self.retry.delay_for(attempt, err.retry_after_s, self._rng)

    def _fits(self, delay: float, deadline: float) -> bool:
        return self._clock() + delay < deadline

    def _give_up(self, last_error: LLMError | None, deadline_hit: bool) -> LLMError:
        if last_error is None:
            return GatewayConfigurationError("no clients configured")
        if deadline_hit and not isinstance(last_error, TimeoutError):
            timeout = TimeoutError(f"deadline exceeded; last error: {last_error}", retryable=False)
            timeout.__cause__ = last_error
            return timeout
        return last_error

    def _cost(self, model: str, usage: Usage) -> float:
        return self.pricing.cost_usd(model, usage) if self.pricing else 0.0

    def _annotate(self, completion: Completion, *, cache_hit: bool, attempt: int) -> Completion:
        """Cost accounting. A cache hit costs nothing; the price the provider *would* have
        charged is reported separately as `avoided_cost_usd`, so summing `cost_usd` over
        completions gives real spend and summing `avoided_cost_usd` gives the cache's value."""
        raw = dict(completion.raw or {})
        full_cost = self._cost(completion.model, completion.usage)
        raw.update(
            {
                "cost_usd": 0.0 if cache_hit else full_cost,
                "avoided_cost_usd": full_cost if cache_hit else 0.0,
                "cache_hit": cache_hit,
                "attempt": attempt,
            }
        )
        return completion.model_copy(update={"raw": raw})

    @staticmethod
    def _fill_span(span: Span, completion: Completion, *, cache_hit: bool, queue_ms: float = 0.0) -> None:
        span.set_attribute("provider", completion.provider)
        span.set_attribute("model", completion.model)
        span.set_attribute("input_tokens", completion.usage.input_tokens)
        span.set_attribute("output_tokens", completion.usage.output_tokens)
        span.set_attribute("cached_input_tokens", completion.usage.cached_input_tokens)
        span.set_attribute("latency_ms", round(completion.latency_ms, 3))
        span.set_attribute("queue_ms", round(queue_ms, 3))
        span.set_attribute("cache_hit", cache_hit)
        span.set_attribute("cost_usd", (completion.raw or {}).get("cost_usd", 0.0))
        span.set_attribute("avoided_cost_usd", (completion.raw or {}).get("avoided_cost_usd", 0.0))
        span.set_attribute("finish_reason", completion.finish_reason)

    # ------------------------------------------------------------------ cache
    def _cacheable(self, req: CompletionRequest) -> bool:
        if self.cache is None:
            return False
        if self.require_cache_scope and req.metadata.get("cache_scope") is None:
            return False  # fail closed: an unscoped request in a multi-tenant gateway is never cached
        flag = req.metadata.get("cache")
        if flag is False:
            return False
        if flag is True:
            return True
        return req.temperature <= self.cache_max_temperature

    def _cache_lookup(self, key: str | None) -> Completion | None:
        if key is None or self.cache is None:
            return None
        hit = self.cache.get(key)
        if hit is None:
            return None
        with self.tracer.span(SPAN_NAME, attempt=0) as span:
            served = self._annotate(hit, cache_hit=True, attempt=0).model_copy(update={"latency_ms": 0.0})
            self._fill_span(span, served, cache_hit=True)
        return served

    def _cache_store(self, key: str | None, completion: Completion) -> None:
        if key is not None and self.cache is not None:
            self.cache.set(key, completion, self.cache_ttl_s)

    # ------------------------------------------------------------------ complete (sync)
    def _attempt(self, client: LLMClient, req: CompletionRequest, attempt: int, deadline: float) -> Completion:
        """One traced attempt: admission, the call, accounting. Raises LLMError on failure."""
        with self.tracer.span(SPAN_NAME, provider=client.provider, model=self._model_for(client, req), attempt=attempt) as span:
            queued = self._clock()
            attempt_req = self._with_remaining(req, deadline)
            if self.rate_limiter is not None:
                self.rate_limiter.acquire(self.token_estimator(req), deadline=deadline)
            with self._sem:
                queue_ms = (self._clock() - queued) * 1000
                completion = client.complete(attempt_req)
            completion = self._annotate(completion, cache_hit=False, attempt=attempt)
            self._fill_span(span, completion, cache_hit=False, queue_ms=queue_ms)
            return completion

    def complete(self, req: CompletionRequest) -> Completion:
        deadline = self._deadline(req)
        key = cache_key(req, self._model_for(self.primary, req)) if self._cacheable(req) else None
        cached = self._cache_lookup(key)
        if cached is not None:
            return cached

        last_error: LLMError | None = None
        deadline_hit = False
        for client in self.clients:
            attempt = 0
            while True:
                attempt += 1
                try:
                    completion = self._attempt(client, req, attempt, deadline)
                except LLMError as err:
                    last_error = err
                    if not err.retryable:
                        raise  # InvalidRequest, ContentFilter, Malformed: no retry, no fallback
                    delay = self._plan_retry(err, attempt)
                    if delay is None:
                        break  # retry budget for this client is spent; try the next one
                    if not self._fits(delay, deadline):
                        deadline_hit = True
                        break  # waiting would blow the deadline; try the next client now
                    self._sleep(delay)
                    continue
                self._cache_store(key, completion)
                return completion
            if self._clock() >= deadline:
                deadline_hit = True
                break
        raise self._give_up(last_error, deadline_hit)

    # ------------------------------------------------------------------ stream (sync)
    def _open_stream(self, client: LLMClient, req: CompletionRequest, attempt: int, deadline: float) -> _OpenStream:
        """Admission plus the first event. Any failure here is still safely retryable because
        nothing has reached the caller yet. The span is created by hand: its lifetime crosses
        the generator boundary, so a `with` block cannot own it."""
        span = Span(
            name=SPAN_NAME,
            attributes={"provider": client.provider, "model": self._model_for(client, req), "attempt": attempt, "stream": True},
        ).link_to_current()
        started = self._clock()
        try:
            attempt_req = self._with_remaining(req, deadline)
            if self.rate_limiter is not None:
                self.rate_limiter.acquire(self.token_estimator(req), deadline=deadline)
            self._sem.acquire()
            try:
                iterator = client.stream(attempt_req)
                first = next(iterator, None)
            except BaseException:
                self._sem.release()
                raise
        except BaseException as exc:
            span.record_exception(exc)
            span.finish()
            self.tracer.export(span)
            raise
        return _OpenStream(iterator=iterator, first=first, span=span, started=started)

    def _relay(self, opened: _OpenStream, client: LLMClient, req: CompletionRequest, attempt: int) -> Iterator[StreamEvent]:
        usage, finish = Usage(), "stop"
        try:
            events = opened.iterator
            assert isinstance(events, Iterator)
            for ev in _chain_first(opened.first, events):
                if ev.type == "usage" and ev.usage is not None:
                    usage = ev.usage
                elif ev.type == "done" and ev.finish_reason:
                    finish = ev.finish_reason
                yield ev
        except BaseException as exc:
            opened.span.record_exception(exc)
            raise
        finally:
            self._sem.release()
            model = self._model_for(client, req) or ""
            summary = Completion(
                message=req.messages[-1], usage=usage, finish_reason=finish, model=model, provider=client.provider,
                latency_ms=(self._clock() - opened.started) * 1000,
            )
            self._fill_span(opened.span, self._annotate(summary, cache_hit=False, attempt=attempt), cache_hit=False)
            opened.span.finish()
            self.tracer.export(opened.span)

    def stream(self, req: CompletionRequest) -> Iterator[StreamEvent]:
        """Retries and fallbacks apply only until the first event has been yielded: once bytes
        have reached the caller, a silent restart would duplicate text."""
        deadline = self._deadline(req)
        last_error: LLMError | None = None
        deadline_hit = False
        for client in self.clients:
            attempt = 0
            while True:
                attempt += 1
                try:
                    opened = self._open_stream(client, req, attempt, deadline)
                except LLMError as err:
                    last_error = err
                    if not err.retryable:
                        raise
                    delay = self._plan_retry(err, attempt)
                    if delay is None:
                        break
                    if not self._fits(delay, deadline):
                        deadline_hit = True
                        break
                    self._sleep(delay)
                    continue
                yield from self._relay(opened, client, req, attempt)
                return
            if self._clock() >= deadline:
                deadline_hit = True
                break
        raise self._give_up(last_error, deadline_hit)

    # ------------------------------------------------------------------ async
    def _async_sem(self) -> asyncio.Semaphore:
        """The async concurrency limit for the running event loop. A gateway shared by code
        that calls `asyncio.run` repeatedly (scripts, some task workers) would otherwise reuse
        a semaphore bound to a closed loop and fail under contention."""
        loop = asyncio.get_running_loop()
        sem = self._asems.get(loop)
        if sem is None:
            sem = asyncio.Semaphore(self._max_concurrency)
            self._asems[loop] = sem
        return sem

    async def _aattempt(self, client: LLMClient, req: CompletionRequest, attempt: int, deadline: float) -> Completion:
        with self.tracer.span(SPAN_NAME, provider=client.provider, model=self._model_for(client, req), attempt=attempt) as span:
            queued = self._clock()
            attempt_req = self._with_remaining(req, deadline)
            if self.rate_limiter is not None:
                await self.rate_limiter.aacquire(self.token_estimator(req), deadline=deadline)
            async with self._async_sem():
                queue_ms = (self._clock() - queued) * 1000
                completion = await client.acomplete(attempt_req)
            completion = self._annotate(completion, cache_hit=False, attempt=attempt)
            self._fill_span(span, completion, cache_hit=False, queue_ms=queue_ms)
            return completion

    async def acomplete(self, req: CompletionRequest) -> Completion:
        deadline = self._deadline(req)
        key = cache_key(req, self._model_for(self.primary, req)) if self._cacheable(req) else None
        cached = self._cache_lookup(key)
        if cached is not None:
            return cached

        last_error: LLMError | None = None
        deadline_hit = False
        for client in self.clients:
            attempt = 0
            while True:
                attempt += 1
                try:
                    completion = await self._aattempt(client, req, attempt, deadline)
                except LLMError as err:
                    last_error = err
                    if not err.retryable:
                        raise
                    delay = self._plan_retry(err, attempt)
                    if delay is None:
                        break
                    if not self._fits(delay, deadline):
                        deadline_hit = True
                        break
                    await self._asleep(delay)
                    continue
                self._cache_store(key, completion)
                return completion
            if self._clock() >= deadline:
                deadline_hit = True
                break
        raise self._give_up(last_error, deadline_hit)

    async def _aopen_stream(self, client: LLMClient, req: CompletionRequest, attempt: int, deadline: float) -> _OpenStream:
        span = Span(
            name=SPAN_NAME,
            attributes={"provider": client.provider, "model": self._model_for(client, req), "attempt": attempt, "stream": True},
        ).link_to_current()
        started = self._clock()
        sem = self._async_sem()
        try:
            attempt_req = self._with_remaining(req, deadline)
            if self.rate_limiter is not None:
                await self.rate_limiter.aacquire(self.token_estimator(req), deadline=deadline)
            await sem.acquire()
            try:
                aiterator = client.astream(attempt_req)
                first = await anext(aiterator, None)
            except BaseException:
                sem.release()
                raise
        except BaseException as exc:
            span.record_exception(exc)
            span.finish()
            self.tracer.export(span)
            raise
        return _OpenStream(iterator=aiterator, first=first, span=span, started=started, async_sem=sem)

    async def _arelay(self, opened: _OpenStream, client: LLMClient, req: CompletionRequest, attempt: int) -> AsyncIterator[StreamEvent]:
        usage, finish = Usage(), "stop"
        try:
            if opened.first is not None:
                yield opened.first
            events = opened.iterator
            assert isinstance(events, AsyncIterator)
            async for ev in events:
                if ev.type == "usage" and ev.usage is not None:
                    usage = ev.usage
                elif ev.type == "done" and ev.finish_reason:
                    finish = ev.finish_reason
                yield ev
        except BaseException as exc:
            opened.span.record_exception(exc)
            raise
        finally:
            assert opened.async_sem is not None
            opened.async_sem.release()
            model = self._model_for(client, req) or ""
            summary = Completion(
                message=req.messages[-1], usage=usage, finish_reason=finish, model=model, provider=client.provider,
                latency_ms=(self._clock() - opened.started) * 1000,
            )
            self._fill_span(opened.span, self._annotate(summary, cache_hit=False, attempt=attempt), cache_hit=False)
            opened.span.finish()
            self.tracer.export(opened.span)

    async def astream(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        deadline = self._deadline(req)
        last_error: LLMError | None = None
        deadline_hit = False
        for client in self.clients:
            attempt = 0
            while True:
                attempt += 1
                try:
                    opened = await self._aopen_stream(client, req, attempt, deadline)
                except LLMError as err:
                    last_error = err
                    if not err.retryable:
                        raise
                    delay = self._plan_retry(err, attempt)
                    if delay is None:
                        break
                    if not self._fits(delay, deadline):
                        deadline_hit = True
                        break
                    await self._asleep(delay)
                    continue
                async for ev in self._arelay(opened, client, req, attempt):
                    yield ev
                return
            if self._clock() >= deadline:
                deadline_hit = True
                break
        raise self._give_up(last_error, deadline_hit)


class GatewayConfigurationError(LLMError):
    """Raised only when a gateway has no clients at all (a configuration error)."""

    default_retryable = False


__all__ = [
    "RetryPolicy",
    "RateLimiter",
    "ResponseCache",
    "InMemoryResponseCache",
    "PricingTable",
    "ModelGateway",
    "cache_key",
    "SPAN_NAME",
]
