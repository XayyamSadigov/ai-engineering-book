# path: book/projects/reliability/reliability/queue.py
"""A job queue with the five properties async AI work needs.

1. Idempotent enqueue: the same idempotency key returns the existing job, never a twin.
2. Leases (visibility timeouts): a leased job is invisible to other workers until its lease
   expires; a dead worker's job therefore comes back on its own.
3. ack / nack: ack records success; nack re-queues with exponential backoff and jitter
   (aie_core's RetryPolicy computes the delay) or dead-letters non-retryable errors.
4. Bounded attempts: every lease counts as an attempt, including leases that expired. A job
   that crashes its worker every time (a "poison" job) reaches the dead-letter set instead
   of cycling forever.
5. Dead-letter inspection and redrive, because a DLQ nobody can read is a black hole.

Delivery is at-least-once. A worker can finish the work and die before `ack`; the job is
then delivered again. Handlers must be idempotent (see worker.py).

`InMemoryJobQueue` is for tests and single-process tools. `RedisJobQueue` keeps the same
semantics across processes, with each state change in one Lua script so it is atomic.
"""
from __future__ import annotations

import json
import random
import threading
import time
import uuid
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from aie_core.llm.gateway import RetryPolicy
from pydantic import BaseModel, Field

from .clock import Clock


class JobState(str, Enum):
    QUEUED = "queued"
    LEASED = "leased"
    SUCCEEDED = "succeeded"
    DEAD = "dead"


class Job(BaseModel):
    id: str
    kind: str
    payload: dict[str, Any] = Field(default_factory=dict)
    tenant_id: str | None = None
    idempotency_key: str | None = None
    state: JobState = JobState.QUEUED
    attempts: int = 0
    max_attempts: int = 5
    available_at: float = 0.0
    created_at: float = 0.0
    lease_expires_at: float | None = None
    last_error: str | None = None
    result: Any = None


class Lease(BaseModel):
    job: Job
    token: str
    expires_at: float


DEFAULT_BACKOFF = RetryPolicy(base_delay_s=5.0, max_delay_s=600.0, jitter=True)


@runtime_checkable
class JobQueue(Protocol):
    def enqueue(self, kind: str, payload: dict[str, Any], *, idempotency_key: str | None = None,
                tenant_id: str | None = None, max_attempts: int = 5, delay_s: float = 0.0) -> Job: ...
    def lease(self, visibility_timeout_s: float = 60.0) -> Lease | None: ...
    def extend(self, lease: Lease, visibility_timeout_s: float) -> bool: ...
    def ack(self, lease: Lease, result: Any = None) -> bool: ...
    def nack(self, lease: Lease, error: str, *, retryable: bool = True, delay_s: float | None = None) -> JobState | None: ...
    def release(self, lease: Lease) -> bool: ...
    def get(self, job_id: str) -> Job | None: ...
    def depth(self) -> int: ...
    def dead_letters(self, limit: int = 100) -> list[Job]: ...
    def redrive(self, job_id: str) -> bool: ...


# ============================================================================ in-memory
class InMemoryJobQueue:
    def __init__(self, *, clock: Clock = time.time, backoff: RetryPolicy = DEFAULT_BACKOFF,
                 rng: random.Random | None = None) -> None:
        self._clock = clock
        self._backoff = backoff
        self._rng = rng or random.Random()
        self._jobs: dict[str, Job] = {}
        self._order: dict[str, int] = {}
        self._seq = 0
        self._idem: dict[tuple[str | None, str], str] = {}
        self._tokens: dict[str, str] = {}
        self._lock = threading.RLock()

    def enqueue(self, kind: str, payload: dict[str, Any], *, idempotency_key: str | None = None,
                tenant_id: str | None = None, max_attempts: int = 5, delay_s: float = 0.0) -> Job:
        with self._lock:
            if idempotency_key is not None:
                existing = self._idem.get((tenant_id, idempotency_key))
                if existing is not None:
                    return self._jobs[existing].model_copy(deep=True)
            now = self._clock()
            job = Job(id=uuid.uuid4().hex, kind=kind, payload=payload, tenant_id=tenant_id,
                      idempotency_key=idempotency_key, max_attempts=max_attempts,
                      available_at=now + delay_s, created_at=now)
            self._jobs[job.id] = job
            self._seq += 1
            self._order[job.id] = self._seq
            if idempotency_key is not None:
                self._idem[(tenant_id, idempotency_key)] = job.id
            return job.model_copy(deep=True)

    def _reclaim_expired(self, now: float) -> None:
        for job in self._jobs.values():
            if job.state is JobState.LEASED and job.lease_expires_at is not None and job.lease_expires_at <= now:
                self._tokens.pop(job.id, None)
                job.lease_expires_at = None
                if job.attempts >= job.max_attempts:
                    job.state, job.last_error = JobState.DEAD, "lease expired on final attempt"
                else:
                    job.state, job.last_error, job.available_at = JobState.QUEUED, "lease expired", now

    def lease(self, visibility_timeout_s: float = 60.0) -> Lease | None:
        with self._lock:
            now = self._clock()
            self._reclaim_expired(now)
            ready = [j for j in self._jobs.values() if j.state is JobState.QUEUED and j.available_at <= now]
            if not ready:
                return None
            job = min(ready, key=lambda j: (j.available_at, self._order[j.id]))
            token = uuid.uuid4().hex
            job.state = JobState.LEASED
            job.attempts += 1
            job.lease_expires_at = now + visibility_timeout_s
            self._tokens[job.id] = token
            return Lease(job=job.model_copy(deep=True), token=token, expires_at=job.lease_expires_at)

    def _owned(self, lease: Lease) -> Job | None:
        job = self._jobs.get(lease.job.id)
        if job is None or job.state is not JobState.LEASED or self._tokens.get(job.id) != lease.token:
            return None
        if job.lease_expires_at is not None and job.lease_expires_at <= self._clock():
            return None  # expired, even if nobody has re-leased it yet: we no longer own it
        return job

    def extend(self, lease: Lease, visibility_timeout_s: float) -> bool:
        with self._lock:
            job = self._owned(lease)
            if job is None:
                return False
            job.lease_expires_at = self._clock() + visibility_timeout_s
            lease.expires_at = job.lease_expires_at
            return True

    def ack(self, lease: Lease, result: Any = None) -> bool:
        with self._lock:
            job = self._owned(lease)
            if job is None:
                return False  # lease lost: someone else owns the job now
            job.state, job.result, job.lease_expires_at = JobState.SUCCEEDED, result, None
            self._tokens.pop(job.id, None)
            return True

    def nack(self, lease: Lease, error: str, *, retryable: bool = True, delay_s: float | None = None) -> JobState | None:
        with self._lock:
            job = self._owned(lease)
            if job is None:
                return None
            self._tokens.pop(job.id, None)
            job.lease_expires_at, job.last_error = None, error
            if not retryable or job.attempts >= job.max_attempts:
                job.state = JobState.DEAD
                return job.state
            delay = delay_s if delay_s is not None else self._backoff.delay_for(job.attempts, None, self._rng)
            job.state, job.available_at = JobState.QUEUED, self._clock() + delay
            return job.state

    def release(self, lease: Lease) -> bool:
        with self._lock:
            job = self._owned(lease)
            if job is None:
                return False
            self._tokens.pop(job.id, None)
            job.state, job.lease_expires_at, job.available_at = JobState.QUEUED, None, self._clock()
            job.attempts = max(0, job.attempts - 1)  # shutting down is not the job's fault
            return True

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.model_copy(deep=True) if job else None

    def depth(self) -> int:
        with self._lock:
            now = self._clock()
            self._reclaim_expired(now)
            return sum(1 for j in self._jobs.values() if j.state is JobState.QUEUED and j.available_at <= now)

    def in_flight(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.state is JobState.LEASED)

    def dead_letters(self, limit: int = 100) -> list[Job]:
        with self._lock:
            self._reclaim_expired(self._clock())
            return [j.model_copy(deep=True) for j in self._jobs.values() if j.state is JobState.DEAD][:limit]

    def redrive(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state is not JobState.DEAD:
                return False
            job.state, job.attempts, job.available_at = JobState.QUEUED, 0, self._clock()
            return True


# ============================================================================ Redis
_LEASE = """
local now = tonumber(ARGV[1])
local expired = redis.call('ZRANGEBYSCORE', KEYS[2], '-inf', now)
for _, id in ipairs(expired) do
  local jk = ARGV[4] .. id
  redis.call('ZREM', KEYS[2], id)
  redis.call('HDEL', jk, 'lease_token', 'lease_expires_at')
  local attempts = tonumber(redis.call('HGET', jk, 'attempts'))
  local maxa = tonumber(redis.call('HGET', jk, 'max_attempts'))
  if attempts >= maxa then
    redis.call('HSET', jk, 'state', 'dead', 'last_error', 'lease expired on final attempt')
    redis.call('ZADD', KEYS[3], now, id)
  else
    redis.call('HSET', jk, 'state', 'queued', 'last_error', 'lease expired', 'available_at', ARGV[1])
    redis.call('ZADD', KEYS[1], now, id)
  end
end
if ARGV[5] == 'reclaim_only' then return false end
local ids = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', now, 'LIMIT', 0, 1)
if #ids == 0 then return false end
local id = ids[1]
local jk = ARGV[4] .. id
local exp = now + tonumber(ARGV[2])
redis.call('ZREM', KEYS[1], id)
redis.call('ZADD', KEYS[2], exp, id)
redis.call('HINCRBY', jk, 'attempts', 1)
redis.call('HSET', jk, 'state', 'leased', 'lease_token', ARGV[3], 'lease_expires_at', tostring(exp))
return id
"""

_RELEASE_IDEM = """
if redis.call('HGET', KEYS[1], ARGV[1]) == ARGV[2] then return redis.call('HDEL', KEYS[1], ARGV[1]) end
return 0
"""

_ENQUEUE = """
if ARGV[3] ~= '' then
  local existing = redis.call('HGET', KEYS[1], ARGV[3])
  if existing then return existing end
  redis.call('HSET', KEYS[1], ARGV[3], ARGV[1])
end
for i = 5, #ARGV, 2 do redis.call('HSET', ARGV[2], ARGV[i], ARGV[i + 1]) end
redis.call('ZADD', KEYS[2], ARGV[4], ARGV[1])
return ARGV[1]
"""

_OWNED = """
local function owned(jk, token, now)
  if redis.call('HGET', jk, 'lease_token') ~= token then return false end
  local exp = tonumber(redis.call('HGET', jk, 'lease_expires_at'))
  return exp ~= nil and exp > tonumber(now)
end
"""

_ACK = _OWNED + """
if not owned(ARGV[2], ARGV[3], ARGV[5]) then return 0 end
redis.call('ZREM', KEYS[1], ARGV[1])
redis.call('HDEL', ARGV[2], 'lease_token', 'lease_expires_at')
redis.call('HSET', ARGV[2], 'state', 'succeeded', 'result', ARGV[4])
return 1
"""

_NACK = _OWNED + """
if not owned(ARGV[2], ARGV[3], ARGV[7]) then return 'lost' end
redis.call('ZREM', KEYS[2], ARGV[1])
redis.call('HDEL', ARGV[2], 'lease_token', 'lease_expires_at')
local attempts = tonumber(redis.call('HGET', ARGV[2], 'attempts'))
local maxa = tonumber(redis.call('HGET', ARGV[2], 'max_attempts'))
if ARGV[5] == '0' or attempts >= maxa then
  redis.call('HSET', ARGV[2], 'state', 'dead', 'last_error', ARGV[4])
  redis.call('ZADD', KEYS[3], ARGV[7], ARGV[1])
  return 'dead'
end
redis.call('HSET', ARGV[2], 'state', 'queued', 'last_error', ARGV[4], 'available_at', ARGV[6])
redis.call('ZADD', KEYS[1], ARGV[6], ARGV[1])
return 'queued'
"""

_RELEASE = _OWNED + """
if not owned(ARGV[2], ARGV[3], ARGV[4]) then return 0 end
redis.call('ZREM', KEYS[2], ARGV[1])
redis.call('HDEL', ARGV[2], 'lease_token', 'lease_expires_at')
local attempts = tonumber(redis.call('HGET', ARGV[2], 'attempts'))
if attempts > 0 then redis.call('HINCRBY', ARGV[2], 'attempts', -1) end
redis.call('HSET', ARGV[2], 'state', 'queued', 'available_at', ARGV[4])
redis.call('ZADD', KEYS[1], ARGV[4], ARGV[1])
return 1
"""

_EXTEND = _OWNED + """
if not owned(ARGV[2], ARGV[3], ARGV[5]) then return 0 end
redis.call('ZADD', KEYS[1], 'XX', ARGV[4], ARGV[1])
redis.call('HSET', ARGV[2], 'lease_expires_at', ARGV[4])
return 1
"""

_REDRIVE = """
if redis.call('ZREM', KEYS[1], ARGV[1]) == 0 then return 0 end
redis.call('HSET', ARGV[2], 'state', 'queued', 'attempts', 0, 'available_at', ARGV[3])
redis.call('ZADD', KEYS[2], ARGV[3], ARGV[1])
return 1
"""


def _s(value: Any) -> str | None:
    if value is None:
        return None
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


class RedisJobQueue:
    """Keys (all under one hash tag so a Redis Cluster keeps them in one slot):

        {ns}:ready   ZSET  job id -> available_at (delayed and ready jobs together)
        {ns}:leased  ZSET  job id -> lease expiry
        {ns}:dead    ZSET  job id -> time dead-lettered
        {ns}:idem    HASH  "tenant|key" -> job id
        {ns}:job:ID  HASH  job fields

    Times come from `clock` (wall time by default) and are passed into the scripts, so all
    workers must have reasonably synchronized clocks; keep visibility timeouts much larger
    than plausible skew. Succeeded jobs keep their hash for `result_ttl_s` so that
    idempotent re-submissions can still find them.
    """

    def __init__(self, redis: Any, namespace: str = "northwind:jobs", *, clock: Clock = time.time,
                 backoff: RetryPolicy = DEFAULT_BACKOFF, rng: random.Random | None = None,
                 result_ttl_s: int = 7 * 24 * 3600) -> None:
        self.r = redis
        ns = "{" + namespace + "}"
        self.k_ready, self.k_leased, self.k_dead, self.k_idem = f"{ns}:ready", f"{ns}:leased", f"{ns}:dead", f"{ns}:idem"
        self.job_prefix = f"{ns}:job:"
        self._clock = clock
        self._backoff = backoff
        self._rng = rng or random.Random()
        self.result_ttl_s = result_ttl_s
        self._lease = redis.register_script(_LEASE)
        self._enqueue = redis.register_script(_ENQUEUE)
        self._release_idem = redis.register_script(_RELEASE_IDEM)
        self._ack = redis.register_script(_ACK)
        self._nack = redis.register_script(_NACK)
        self._release = redis.register_script(_RELEASE)
        self._extend = redis.register_script(_EXTEND)
        self._redrive = redis.register_script(_REDRIVE)

    @classmethod
    def from_url(cls, url: str, **kw: Any) -> "RedisJobQueue":
        import redis  # optional dependency

        return cls(redis.Redis.from_url(url), **kw)

    def _jk(self, job_id: str) -> str:
        return self.job_prefix + job_id

    def _load(self, job_id: str) -> Job | None:
        raw = self.r.hgetall(self._jk(job_id))
        if not raw:
            return None
        d = {_s(k): _s(v) for k, v in raw.items()}
        return Job(
            id=d["id"], kind=d["kind"], payload=json.loads(d.get("payload") or "{}"),
            tenant_id=d.get("tenant_id") or None, idempotency_key=d.get("idempotency_key") or None,
            state=JobState(d["state"]), attempts=int(d.get("attempts") or 0),
            max_attempts=int(d.get("max_attempts") or 5), available_at=float(d.get("available_at") or 0),
            created_at=float(d.get("created_at") or 0),
            lease_expires_at=float(d["lease_expires_at"]) if d.get("lease_expires_at") else None,
            last_error=d.get("last_error") or None,
            result=json.loads(d["result"]) if d.get("result") else None,
        )

    def enqueue(self, kind: str, payload: dict[str, Any], *, idempotency_key: str | None = None,
                tenant_id: str | None = None, max_attempts: int = 5, delay_s: float = 0.0) -> Job:
        now = self._clock()
        job_id = uuid.uuid4().hex
        fields = {
            "id": job_id, "kind": kind, "payload": json.dumps(payload), "tenant_id": tenant_id or "",
            "idempotency_key": idempotency_key or "", "state": JobState.QUEUED.value, "attempts": "0",
            "max_attempts": str(max_attempts), "available_at": repr(now + delay_s), "created_at": repr(now),
        }
        idem = f"{tenant_id or ''}|{idempotency_key}" if idempotency_key else ""
        args: list[str] = [job_id, self._jk(job_id), idem, repr(now + delay_s)]
        for k, v in fields.items():
            args += [k, v]
        returned = _s(self._enqueue(keys=[self.k_idem, self.k_ready], args=args))
        if returned and returned != job_id and self._load(returned) is None:
            # The key's job expired with its result TTL: the key is free again, not a duplicate.
            # Compare-and-delete: free the key only if it still points at the expired job, so two
            # enqueuers racing here cannot each free it and create two new jobs.
            self._release_idem(keys=[self.k_idem], args=[idem, returned])
            returned = _s(self._enqueue(keys=[self.k_idem, self.k_ready], args=args))
        job = self._load(returned or job_id)
        assert job is not None
        return job

    def lease(self, visibility_timeout_s: float = 60.0) -> Lease | None:
        token = uuid.uuid4().hex
        job_id = _s(self._lease(keys=[self.k_ready, self.k_leased, self.k_dead],
                                args=[repr(self._clock()), repr(visibility_timeout_s), token, self.job_prefix, "lease"]))
        if not job_id:
            return None
        job = self._load(job_id)
        assert job is not None and job.lease_expires_at is not None
        return Lease(job=job, token=token, expires_at=job.lease_expires_at)

    def _reclaim(self) -> None:
        self._lease(keys=[self.k_ready, self.k_leased, self.k_dead],
                    args=[repr(self._clock()), "0", "", self.job_prefix, "reclaim_only"])

    def extend(self, lease: Lease, visibility_timeout_s: float) -> bool:
        exp = self._clock() + visibility_timeout_s
        ok = int(self._extend(keys=[self.k_leased], args=[lease.job.id, self._jk(lease.job.id), lease.token, repr(exp),
                                                           repr(self._clock())]))
        if ok:
            lease.expires_at = exp
        return bool(ok)

    def ack(self, lease: Lease, result: Any = None) -> bool:
        ok = int(self._ack(keys=[self.k_leased],
                           args=[lease.job.id, self._jk(lease.job.id), lease.token, json.dumps(result),
                                 repr(self._clock())]))
        if ok:
            self.r.expire(self._jk(lease.job.id), self.result_ttl_s)
        return bool(ok)

    def nack(self, lease: Lease, error: str, *, retryable: bool = True, delay_s: float | None = None) -> JobState | None:
        now = self._clock()
        delay = delay_s if delay_s is not None else self._backoff.delay_for(lease.job.attempts, None, self._rng)
        res = _s(self._nack(keys=[self.k_ready, self.k_leased, self.k_dead],
                            args=[lease.job.id, self._jk(lease.job.id), lease.token, error[:2000],
                                  "1" if retryable else "0", repr(now + delay), repr(now)]))
        return None if res == "lost" else JobState(res)

    def release(self, lease: Lease) -> bool:
        return bool(int(self._release(keys=[self.k_ready, self.k_leased],
                                      args=[lease.job.id, self._jk(lease.job.id), lease.token, repr(self._clock())])))

    def get(self, job_id: str) -> Job | None:
        return self._load(job_id)

    def depth(self) -> int:
        self._reclaim()
        return int(self.r.zcount(self.k_ready, "-inf", repr(self._clock())))

    def in_flight(self) -> int:
        return int(self.r.zcard(self.k_leased))

    def dead_letters(self, limit: int = 100) -> list[Job]:
        self._reclaim()
        ids = [_s(i) for i in self.r.zrange(self.k_dead, 0, limit - 1)]
        return [j for j in (self._load(i) for i in ids if i) if j is not None]

    def redrive(self, job_id: str) -> bool:
        return bool(int(self._redrive(keys=[self.k_dead, self.k_ready],
                                      args=[job_id, self._jk(job_id), repr(self._clock())])))


__all__ = ["JobState", "Job", "Lease", "JobQueue", "InMemoryJobQueue", "RedisJobQueue", "DEFAULT_BACKOFF"]
