# path: book/projects/examples/ch28/reliability_bridge.py
"""Run Chapter 28's job model on Chapter 29's durable queue and worker.

Chapter 28's ``JobQueuePort`` moves bare job ids (``enqueue``/``dequeue``) and keeps every fact
about a job in ``JobRepoPort`` (the ``jobs`` table). That is enough to show the job model, and it
is also why the port is not a drop-in for ``reliability.JobQueue``: a bare ``dequeue`` has no
lease and no ack, so a worker that dies right after dequeuing loses the job. Chapter 29's queue
owns delivery state (lease, ack, nack with backoff, dead letters), and its ``Worker`` decides
retries from the exception class.

The bridge splits the two responsibilities instead of forcing one interface onto the other:

- ``JobRepoPort`` stays the API-visible record (state, attempts, result, error), so the routers,
  ``JobService.get``/``cancel``, and the job event stream do not change;
- ``ReliabilityJobQueue`` implements the submission half of ``JobQueuePort`` (``enqueue``,
  ``depth``) on top of ``reliability.JobQueue`` and refuses ``dequeue``, because delivery belongs
  to ``reliability.Worker``;
- ``bridge_handler`` adapts a Chapter 28 async handler into a Chapter 29 handler and mirrors each
  attempt's outcome into the repo;
- ``reconcile_dead_letters`` marks repo rows ``failed`` for jobs the queue dead-lettered without a
  handler ever reporting back (a worker that crashed on the final attempt).

The skeleton's ``Worker`` is replaced, not wrapped. Error semantics change with it:
``reliability.is_retryable`` treats the ``ValueError`` of a malformed payload as deterministic and
dead-letters the job after one attempt, where the skeleton retried it ``max_attempts`` times.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable

from reliability import Job as DeliveryJob
from reliability import JobContext, JobQueue, PermanentJobError, Worker, is_retryable

from api_skeleton import TERMINAL_STATES, Job, JobHandler, JobRepoPort, JobState, WebhookPort

KIND = "ch28.job"


class ReliabilityJobQueue:
    """Submission side of Chapter 28's ``JobQueuePort``, backed by a ``reliability.JobQueue``."""

    def __init__(self, queue: JobQueue, repo: JobRepoPort) -> None:
        self.queue = queue
        self.repo = repo

    def enqueue(self, job_id: str) -> None:
        job = self.repo.get(job_id)
        # The repo's job id is the delivery idempotency key: enqueueing one job twice delivers it once.
        self.queue.enqueue(
            KIND,
            {"job_id": job_id},
            idempotency_key=job_id,
            tenant_id=job.tenant_id if job else None,
            max_attempts=job.max_attempts if job else 3,
        )

    def dequeue(self) -> str | None:
        raise NotImplementedError(
            "delivery goes through reliability.Worker: a bare dequeue can be neither leased nor acked"
        )

    def depth(self) -> int:
        return self.queue.depth()


def _notify(webhook: WebhookPort | None, job: Job) -> None:
    if webhook is not None and job.callback_url and job.state in TERMINAL_STATES:
        asyncio.run(webhook.notify(job.callback_url, {"job_id": job.id, "state": job.state.value}))


def bridge_handler(
    repo: JobRepoPort,
    handlers: dict[str, JobHandler],
    webhook: WebhookPort | None = None,
    *,
    classify: Callable[[BaseException], bool] = is_retryable,
) -> Callable[[DeliveryJob, JobContext], Any]:
    """Wrap Chapter 28 handlers for ``reliability.Worker`` and mirror outcomes into the repo.

    The worker owns the decision (ack, nack with backoff, dead letter); the wrapper only records
    what the API must show. It re-raises every handler error so the worker can classify it.
    """

    def handle(delivery: DeliveryJob, ctx: JobContext) -> Any:
        job = repo.get(str(delivery.payload.get("job_id")))
        if job is None:
            raise PermanentJobError(f"job {delivery.payload.get('job_id')} is not in the jobs table")
        if job.state in TERMINAL_STATES:
            # Cancelled before delivery, or a redelivery after success: ack without running again.
            return {"skipped": job.state.value}
        job.state = JobState.RUNNING
        job.attempts = delivery.attempts
        repo.save(job)
        try:
            # Bound the run by the delivery's deadline, so a hung handler cannot outlive its lease
            # and be leased (and run) a second time alongside itself. Work that finished is recorded
            # even if a shutdown arrived meanwhile: discarding it would only force a rerun.
            result = asyncio.run(asyncio.wait_for(handlers[job.type](job), timeout=ctx.deadline.remaining()))
        except Exception as exc:
            current = repo.get(job.id)
            if current is not None and current.state is JobState.CANCELLED:
                raise PermanentJobError(f"job {job.id} was cancelled while running") from exc
            job.error = f"{type(exc).__name__}: {exc}"
            final = not classify(exc) or delivery.attempts >= delivery.max_attempts
            job.state = JobState.FAILED if final else JobState.QUEUED
            repo.save(job)
            _notify(webhook, job)
            raise
        current = repo.get(job.id)
        if current is not None and current.state is JobState.CANCELLED:
            return {"cancelled_during_run": True}  # cancellation is a state: do not overwrite it
        job.result, job.state = result, JobState.SUCCEEDED
        repo.save(job)
        _notify(webhook, job)
        return result

    return handle


def make_worker(
    queue: JobQueue,
    repo: JobRepoPort,
    handlers: dict[str, JobHandler],
    webhook: WebhookPort | None = None,
    **worker_kwargs: Any,
) -> Worker:
    """A Chapter 29 worker that runs Chapter 28 jobs. One `classify` decides retries for both the
    queue and the repo, so the two can never disagree about whether a job will run again."""
    classify = worker_kwargs.pop("classify", is_retryable)
    return Worker(queue, {KIND: bridge_handler(repo, handlers, webhook, classify=classify)},
                  classify=classify, **worker_kwargs)


def reconcile_dead_letters(queue: JobQueue, repo: JobRepoPort, webhook: WebhookPort | None = None) -> list[str]:
    """Mark repo jobs ``failed`` when the queue dead-lettered them behind the handler's back.

    A worker that crashes on the final attempt never reaches the wrapper's ``except`` branch; the
    queue dead-letters the job when the lease expires, and the repo would show ``running`` forever.
    Run this on a schedule (or from the dead-letter dashboard) and alert on what it returns.
    """
    fixed: list[str] = []
    for dead in queue.dead_letters(limit=1000):
        job = repo.get(str(dead.payload.get("job_id")))
        if job is None or job.state in TERMINAL_STATES:
            continue
        job.state = JobState.FAILED
        job.attempts = dead.attempts
        job.error = dead.last_error or "dead-lettered"
        repo.save(job)
        _notify(webhook, job)
        fixed.append(job.id)
    return fixed


__all__ = ["KIND", "ReliabilityJobQueue", "bridge_handler", "make_worker", "reconcile_dead_letters"]
