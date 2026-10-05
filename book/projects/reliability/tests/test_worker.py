# path: book/projects/reliability/tests/test_worker.py
from aie_core.llm.errors import ProviderUnavailableError
from aie_core.observability import InMemoryTracer

from reliability import (
    InMemoryIdempotencyStore,
    InMemoryJobQueue,
    JobState,
    PermanentJobError,
    ShutdownRequested,
    Worker,
    WorkOutcome,
    once,
)
from reliability.queue import RetryPolicy

BACKOFF = RetryPolicy(base_delay_s=1.0, max_delay_s=1.0, jitter=False)


def test_success_and_tracing(clock):
    q = InMemoryJobQueue(clock=clock)
    tracer = InMemoryTracer()
    job = q.enqueue("echo", {"x": 1})
    w = Worker(q, {"echo": lambda j, ctx: j.payload["x"] + 1}, queue_clock=clock, clock=clock, tracer=tracer)
    result = w.run_once()
    assert result.outcome is WorkOutcome.SUCCEEDED
    assert q.get(job.id).result == 2
    span = tracer.find("job.process")[0]
    assert span.attributes["outcome"] == "succeeded" and span.attributes["attempt"] == 1


def test_transient_errors_retry_permanent_errors_dead_letter(clock):
    q = InMemoryJobQueue(clock=clock, backoff=BACKOFF)
    flaky = q.enqueue("flaky", {})
    bad = q.enqueue("bad", {})

    def flaky_handler(job, ctx):
        raise ProviderUnavailableError("503")

    def bad_handler(job, ctx):
        raise PermanentJobError("missing document")

    w = Worker(q, {"flaky": flaky_handler, "bad": bad_handler}, queue_clock=clock, clock=clock)
    outcomes = {w.run_once().kind: None for _ in range(2)}
    assert set(outcomes) == {"flaky", "bad"}
    assert q.get(flaky.id).state is JobState.QUEUED
    assert q.get(bad.id).state is JobState.DEAD


def test_unknown_kind_is_dead_lettered(clock):
    q = InMemoryJobQueue(clock=clock)
    job = q.enqueue("mystery", {})
    assert Worker(q, {}, queue_clock=clock, clock=clock).run_once().outcome is WorkOutcome.DEAD
    assert q.get(job.id).state is JobState.DEAD


def test_graceful_shutdown_releases_job_at_checkpoint(clock):
    q = InMemoryJobQueue(clock=clock)
    job = q.enqueue("long", {"chunks": 5})
    w = Worker(q, {}, queue_clock=clock, clock=clock)

    def long_handler(j, ctx):
        for i in range(j.payload["chunks"]):
            if i == 2:
                w.request_shutdown()                # SIGTERM arrives mid-job
            ctx.checkpoint()
        return "done"

    w.handlers["long"] = long_handler
    assert w.run_once().outcome is WorkOutcome.RELEASED
    assert w.run_once() is None                     # draining: no new leases
    after = q.get(job.id)
    assert after.state is JobState.QUEUED and after.attempts == 0


def test_lost_lease_detected_at_checkpoint_and_on_ack(clock):
    q = InMemoryJobQueue(clock=clock)
    q.enqueue("slow", {})
    q.enqueue("slow_no_checkpoint", {})

    def slow(j, ctx):
        clock.advance(61)                            # longer than the 60 s lease, no heartbeat
        ctx.checkpoint()

    def slow_no_checkpoint(j, ctx):
        clock.advance(61)
        return "finished anyway"

    w = Worker(q, {"slow": slow, "slow_no_checkpoint": slow_no_checkpoint}, visibility_timeout_s=60,
               queue_clock=clock, clock=clock)
    assert w.run_once().outcome is WorkOutcome.LEASE_LOST
    r = w.run_once()
    assert r.outcome is WorkOutcome.LEASE_LOST and r.error == "ack after lease expiry"


def test_heartbeat_keeps_long_job_alive(clock):
    q = InMemoryJobQueue(clock=clock)
    job = q.enqueue("long", {})

    def long_job(j, ctx):
        for _ in range(4):
            clock.advance(40)
            ctx.heartbeat()
            ctx.checkpoint()
        return "ok"

    w = Worker(q, {"long": long_job}, visibility_timeout_s=60, queue_clock=clock, clock=clock)
    assert w.run_once().outcome is WorkOutcome.SUCCEEDED
    assert q.get(job.id).attempts == 1


def test_redelivered_job_does_not_repeat_side_effect(clock):
    """At-least-once delivery plus once(): the reply is sent exactly once across two executions."""
    q = InMemoryJobQueue(clock=clock)
    store = InMemoryIdempotencyStore()
    sent: list[str] = []
    q.enqueue("send_reply", {"ticket_id": "T-9"})

    def handler(j, ctx):
        status = once(store, f"{j.id}:send_reply", lambda: sent.append(j.payload["ticket_id"]) or "sent")
        if j.attempts == 1:
            clock.advance(61)                        # first worker stalls past its lease after sending
        return status

    w = Worker(q, {"send_reply": handler}, visibility_timeout_s=60, queue_clock=clock, clock=clock)
    assert w.run_once().outcome is WorkOutcome.LEASE_LOST
    assert w.run_once().outcome is WorkOutcome.SUCCEEDED
    assert sent == ["T-9"]


def test_run_loop_processes_until_idle(clock):
    q = InMemoryJobQueue(clock=clock)
    for i in range(3):
        q.enqueue("echo", {"i": i})
    w = Worker(q, {"echo": lambda j, ctx: j.payload["i"]}, queue_clock=clock, clock=clock, sleep=clock.sleep)
    assert w.run(max_idle_polls=1) == 3
    assert w.stats["succeeded"] == 3


def test_shutdown_requested_is_exported():
    assert issubclass(ShutdownRequested, Exception)
