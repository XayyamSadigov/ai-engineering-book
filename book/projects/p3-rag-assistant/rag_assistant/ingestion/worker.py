# path: book/projects/p3-rag-assistant/rag_assistant/ingestion/worker.py
"""The ingestion worker process: reliability.Worker over the job queue with our handlers.

    rag-assistant-worker            # run until SIGTERM; drains the current job, then exits

Scale out by running more worker processes: leases make each job invisible to the others,
and the handlers are idempotent, so a duplicate delivery after a lease expiry is harmless.
"""
from __future__ import annotations

import logging

from reliability import Worker

from ..config import AssistantSettings


def build_worker(container, *, poll_interval_s: float | None = None, sleep=None) -> Worker:  # type: ignore[no-untyped-def]
    s = container.settings
    kwargs = {}
    if sleep is not None:
        kwargs["sleep"] = sleep
    return Worker(container.queue, container.handlers.handlers(), name="rag-ingest",
                  visibility_timeout_s=s.worker_visibility_timeout_s,
                  poll_interval_s=s.worker_poll_interval_s if poll_interval_s is None else poll_interval_s,
                  tracer=container.tracer, **kwargs)


def main() -> int:
    from ..wiring import build_container

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    container = build_container(AssistantSettings())
    worker = build_worker(container)
    worker.install_signal_handlers()
    handled = worker.run()
    logging.getLogger("rag_assistant.worker").info("worker stopped after %d jobs: %s", handled, dict(worker.stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
