# path: book/projects/reliability/examples/worker_main.py
"""Run a Northwind Assist worker process.

    REDIS_URL=redis://localhost:6379/0 python -m examples.worker_main          # real queue
    python -m examples.worker_main --demo                                      # in-memory demo

Two handlers show the idempotency habits from worker.py. `ingest_document` upserts by
content hash, so a redelivered job rewrites the same row. `send_reply` performs a
non-idempotent side effect, so it goes through `once()` keyed by job id.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
from typing import Any

from reliability import (
    InMemoryIdempotencyStore,
    InMemoryJobQueue,
    Job,
    JobContext,
    PermanentJobError,
    RedisJobQueue,
    Worker,
    once,
)

log = logging.getLogger("northwind.worker")
DOCUMENTS: dict[str, dict[str, Any]] = {}          # stands in for the chunk table (Chapter 15)
SENT: list[str] = []                                 # stands in for the email/ticketing API
IDEMPOTENCY = InMemoryIdempotencyStore()


def ingest_document(job: Job, ctx: JobContext) -> dict[str, Any]:
    text = job.payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise PermanentJobError("payload has no text")      # poison: dead-letter, do not retry
    digest = hashlib.sha256(text.encode()).hexdigest()
    for i, chunk in enumerate(text.split("\n\n")):
        ctx.checkpoint()                                    # stop promptly on shutdown or lost lease
        DOCUMENTS[f"{digest}:{i}"] = {"tenant": job.tenant_id, "text": chunk}   # upsert by content key
    return {"document": digest}


def send_reply(job: Job, ctx: JobContext) -> dict[str, Any]:
    ctx.checkpoint()

    def effect() -> str:
        SENT.append(job.payload["ticket_id"])
        return f"sent:{job.payload['ticket_id']}"

    return {"status": once(IDEMPOTENCY, f"{job.id}:send_reply", effect)}


HANDLERS = {"ingest_document": ingest_document, "send_reply": send_reply}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    visibility = float(os.getenv("WORKER_VISIBILITY_TIMEOUT_S", "120"))
    poll = float(os.getenv("WORKER_POLL_INTERVAL_S", "1.0"))
    url = os.getenv("REDIS_URL")
    queue = RedisJobQueue.from_url(url) if url and not args.demo else InMemoryJobQueue()
    if args.demo:
        queue.enqueue("ingest_document", {"text": "VPN runbook\n\nStep 1: restart the client."},
                      tenant_id="logistics", idempotency_key="doc-42")
        queue.enqueue("send_reply", {"ticket_id": "T-1001"}, tenant_id="retail", idempotency_key="reply-1001")
    worker = Worker(queue, HANDLERS, name=os.getenv("HOSTNAME", "worker"), visibility_timeout_s=visibility,
                    poll_interval_s=poll)
    worker.install_signal_handlers()
    handled = worker.run(max_idle_polls=1 if args.demo else None)
    log.info("worker stopped after %d jobs: %s", handled, dict(worker.stats))


if __name__ == "__main__":
    main()
