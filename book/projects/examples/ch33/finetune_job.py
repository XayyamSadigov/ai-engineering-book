# path: book/projects/examples/ch33/finetune_job.py
"""Provider-neutral fine-tuning job interface (Chapter 33).

Hosted fine-tuning APIs all expose the same three-step flow: upload a JSONL file, start a job
that references it, poll the job until it yields a model id. ``FineTuneProvider`` captures that
flow as a Protocol. ``FakeProvider`` implements it in memory so the pipeline is testable
offline; ``OpenAICompatibleFineTuneProvider`` shows how the same Protocol maps onto an
OpenAI-style REST surface via ``httpx`` (tests inject a mock transport, so no network).

Nothing here is specific to one vendor. Endpoint names in the OpenAI-compatible adapter are
"for example"; several hosted and self-hosted servers accept that shape.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

import httpx
from aie_core.llm.errors import LLMError, ProviderUnavailableError, TimeoutError, map_http_error
from pydantic import BaseModel, Field

JobStatus = Literal["validating", "queued", "running", "succeeded", "failed", "cancelled"]
TERMINAL: frozenset[str] = frozenset({"succeeded", "failed", "cancelled"})


class Hyperparameters(BaseModel):
    """The knobs hosted APIs commonly expose. Open-weights trainers add LoRA rank, alpha, targets."""

    n_epochs: int = Field(default=3, ge=1)
    learning_rate_multiplier: float | None = Field(default=None, gt=0)
    batch_size: int | None = Field(default=None, ge=1)


class FineTuneJob(BaseModel):
    id: str
    status: JobStatus
    base_model: str
    training_file_id: str
    validation_file_id: str | None = None
    hyperparameters: Hyperparameters = Field(default_factory=Hyperparameters)
    fine_tuned_model: str | None = None  # populated on success
    created_at: datetime
    error: str | None = None
    trained_tokens: int | None = None


class JobEvent(BaseModel):
    at: datetime
    level: Literal["info", "warn", "error"]
    message: str


class FineTuneProvider(Protocol):
    """What every fine-tuning backend must offer. Keep it this small on purpose."""

    def upload_training_file(self, path: Path) -> str: ...

    def create_job(
        self,
        training_file_id: str,
        base_model: str,
        hyperparameters: Hyperparameters | None = None,
        validation_file_id: str | None = None,
        suffix: str | None = None,
    ) -> FineTuneJob: ...

    def get_job(self, job_id: str) -> FineTuneJob: ...

    def cancel_job(self, job_id: str) -> FineTuneJob: ...

    def list_events(self, job_id: str) -> list[JobEvent]: ...


# --------------------------------------------------------------------------------------
# JSONL validation (shared by every provider; vendors apply the same rules server-side)
# --------------------------------------------------------------------------------------


class ValidationError(ValueError):
    pass


def validate_chat_jsonl(path: Path, min_examples: int = 10) -> dict[str, Any]:
    """Reject files a hosted API would reject, before paying for the upload.

    Rules: valid JSON per line; a ``messages`` list; roles only system/user/assistant; the last
    message must be from the assistant (that is where the loss is computed); non-empty content.
    Returns summary statistics for the data card and the job record.
    """
    n = 0
    chars = 0
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"line {lineno}: invalid JSON ({exc.msg})") from exc
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValidationError(f"line {lineno}: missing or empty 'messages'")
            for m in messages:
                if m.get("role") not in {"system", "user", "assistant"}:
                    raise ValidationError(f"line {lineno}: bad role {m.get('role')!r}")
                if not isinstance(m.get("content"), str) or not m["content"].strip():
                    raise ValidationError(f"line {lineno}: empty content for role {m.get('role')!r}")
                chars += len(m["content"])
            if messages[-1]["role"] != "assistant":
                raise ValidationError(f"line {lineno}: last message must be from the assistant")
            n += 1
    if n < min_examples:
        raise ValidationError(f"only {n} examples; need at least {min_examples}")
    return {"examples": n, "approx_tokens": chars // 4, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


# --------------------------------------------------------------------------------------
# Fake provider: deterministic, in-memory, used by tests and by the chapter walkthrough
# --------------------------------------------------------------------------------------


class FakeProvider:
    """Simulates a hosted fine-tuning backend.

    Each ``get_job`` call advances the job one step along validating -> queued -> running ->
    succeeded, so a polling loop terminates after a known number of polls. ``fail_at`` makes the
    job fail at a given status to exercise error handling.
    """

    def __init__(self, polls_per_stage: int = 1, fail_at: JobStatus | None = None) -> None:
        self.files: dict[str, dict[str, Any]] = {}
        self.jobs: dict[str, FineTuneJob] = {}
        self.events: dict[str, list[JobEvent]] = {}
        self._polls: dict[str, int] = {}
        self._suffixes: dict[str, str] = {}
        self.polls_per_stage = polls_per_stage
        self.fail_at = fail_at
        self.calls: list[str] = []

    def upload_training_file(self, path: Path) -> str:
        self.calls.append("upload")
        stats = validate_chat_jsonl(path)
        file_id = "file-" + stats["sha256"][:12]
        self.files[file_id] = {"path": str(path), **stats}
        return file_id

    def create_job(self, training_file_id: str, base_model: str, hyperparameters: Hyperparameters | None = None,
                   validation_file_id: str | None = None, suffix: str | None = None) -> FineTuneJob:
        self.calls.append("create")
        if training_file_id not in self.files:
            raise ValidationError(f"unknown training file {training_file_id}")
        job_id = "ftjob-" + hashlib.sha1(f"{training_file_id}{base_model}{len(self.jobs)}".encode()).hexdigest()[:10]
        job = FineTuneJob(id=job_id, status="validating", base_model=base_model, training_file_id=training_file_id,
                          validation_file_id=validation_file_id, hyperparameters=hyperparameters or Hyperparameters(),
                          created_at=datetime.now(timezone.utc))
        self.jobs[job_id] = job
        self._polls[job_id] = 0
        self._suffixes[job_id] = suffix or "custom"
        self.events[job_id] = [JobEvent(at=job.created_at, level="info", message="job created")]
        return job

    def get_job(self, job_id: str) -> FineTuneJob:
        self.calls.append("poll")
        job = self.jobs[job_id]
        if job.status in TERMINAL:
            return job
        self._polls[job_id] += 1
        if self._polls[job_id] % self.polls_per_stage:
            return job
        order: list[JobStatus] = ["validating", "queued", "running", "succeeded"]
        nxt = order[order.index(job.status) + 1]
        if self.fail_at is not None and nxt == self.fail_at:
            job.status = "failed"
            job.error = f"simulated failure during {self.fail_at}"
            self.events[job_id].append(JobEvent(at=datetime.now(timezone.utc), level="error", message=job.error))
            return job
        job.status = nxt
        self.events[job_id].append(JobEvent(at=datetime.now(timezone.utc), level="info", message=f"status -> {nxt}"))
        if nxt == "succeeded":
            stats = self.files[job.training_file_id]
            job.trained_tokens = stats["approx_tokens"] * job.hyperparameters.n_epochs
            job.fine_tuned_model = f"ft:{job.base_model}:{self._suffixes[job_id]}:{job.id[-6:]}"
        return job

    def cancel_job(self, job_id: str) -> FineTuneJob:
        self.calls.append("cancel")
        job = self.jobs[job_id]
        if job.status not in TERMINAL:
            job.status = "cancelled"
        return job

    def list_events(self, job_id: str) -> list[JobEvent]:
        return list(self.events.get(job_id, []))


# --------------------------------------------------------------------------------------
# OpenAI-compatible adapter: same Protocol, REST underneath. No network in tests.
# --------------------------------------------------------------------------------------


class OpenAICompatibleFineTuneProvider:
    """Maps the Protocol onto the ``/files`` and ``/fine_tuning/jobs`` style endpoints.

    The mapping is the whole point of the class: status strings differ per vendor, so
    ``_STATUS`` normalizes them into our ``JobStatus``. Pass ``transport`` to test offline.
    """

    _STATUS: dict[str, JobStatus] = {
        "validating_files": "validating", "queued": "queued", "running": "running",
        "succeeded": "succeeded", "failed": "failed", "cancelled": "cancelled",
    }

    def __init__(self, base_url: str, api_key: str, transport: httpx.BaseTransport | None = None, timeout_s: float = 30.0) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_s,
                                    headers={"Authorization": f"Bearer {api_key}"}, transport=transport)

    def _send(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        """One HTTP call, with failures mapped into the aie_core error taxonomy so callers can
        branch on ``retryable`` exactly as they do for completion calls."""
        try:
            resp = self._client.request(method, url, **kwargs)
        except httpx.TimeoutException as exc:
            raise TimeoutError(f"fine-tuning API timed out: {exc}", provider="fine-tune") from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(f"fine-tuning API unreachable: {exc}", provider="fine-tune") from exc
        if not resp.is_success:
            try:
                body: Any = resp.json()
            except ValueError:
                body = resp.text
            raise map_http_error(resp.status_code, body, dict(resp.headers), "fine-tune")
        return resp.json()

    def upload_training_file(self, path: Path) -> str:
        validate_chat_jsonl(path)
        with path.open("rb") as fh:
            data = self._send("POST", "/files", data={"purpose": "fine-tune"}, files={"file": (path.name, fh, "application/jsonl")})
        return data["id"]

    def create_job(self, training_file_id: str, base_model: str, hyperparameters: Hyperparameters | None = None,
                   validation_file_id: str | None = None, suffix: str | None = None) -> FineTuneJob:
        hp = hyperparameters or Hyperparameters()
        body: dict[str, Any] = {"training_file": training_file_id, "model": base_model,
                                "hyperparameters": hp.model_dump(exclude_none=True)}
        if validation_file_id:
            body["validation_file"] = validation_file_id
        if suffix:
            body["suffix"] = suffix
        return self._parse(self._send("POST", "/fine_tuning/jobs", json=body))

    def get_job(self, job_id: str) -> FineTuneJob:
        return self._parse(self._send("GET", f"/fine_tuning/jobs/{job_id}"))

    def cancel_job(self, job_id: str) -> FineTuneJob:
        return self._parse(self._send("POST", f"/fine_tuning/jobs/{job_id}/cancel"))

    def list_events(self, job_id: str) -> list[JobEvent]:
        data = self._send("GET", f"/fine_tuning/jobs/{job_id}/events")
        return [JobEvent(at=datetime.fromtimestamp(e["created_at"], tz=timezone.utc),
                         level=e.get("level", "info"), message=e["message"]) for e in data.get("data", [])]

    def _parse(self, raw: dict[str, Any]) -> FineTuneJob:
        hp_raw = raw.get("hyperparameters") or {}
        hp = Hyperparameters(
            n_epochs=hp_raw.get("n_epochs", 3) if isinstance(hp_raw.get("n_epochs"), int) else 3,
            learning_rate_multiplier=hp_raw.get("learning_rate_multiplier") if isinstance(hp_raw.get("learning_rate_multiplier"), (int, float)) else None,
            batch_size=hp_raw.get("batch_size") if isinstance(hp_raw.get("batch_size"), int) else None,
        )
        err = raw.get("error")
        return FineTuneJob(
            id=raw["id"], status=self._STATUS.get(raw["status"], "running"), base_model=raw["model"],
            training_file_id=raw["training_file"], validation_file_id=raw.get("validation_file"),
            hyperparameters=hp, fine_tuned_model=raw.get("fine_tuned_model"),
            created_at=datetime.fromtimestamp(raw["created_at"], tz=timezone.utc),
            error=(err.get("message") if isinstance(err, dict) else err) or None,
            trained_tokens=raw.get("trained_tokens"),
        )


# --------------------------------------------------------------------------------------
# The pipeline: upload -> job -> poll -> model id, with a record you can store
# --------------------------------------------------------------------------------------


class FineTuneRecord(BaseModel):
    """What you register alongside the model id. Without this, rollback and audit are guesswork."""

    fine_tuned_model: str
    base_model: str
    job_id: str
    training_file_id: str
    training_file_sha256: str
    data_card_sha256: str | None
    hyperparameters: Hyperparameters
    trained_tokens: int | None
    finished_at: datetime


class FineTuneFailed(RuntimeError):
    def __init__(self, job: FineTuneJob, events: list[JobEvent]) -> None:
        self.job = job
        self.events = events
        super().__init__(f"job {job.id} ended with status {job.status}: {job.error or 'no error message'}")


class FineTunePollTimeout(FineTuneFailed):
    """Polling gave up while the job was still running. The job is *not* cancelled: it keeps
    training (and billing) at the provider. Resume with ``run_fine_tune(..., resume_job_id=...)``
    or cancel it explicitly; never start a second job for the same dataset."""


def _check_against_card(card_path: Path | None, train_path: Path, val_path: Path | None) -> None:
    """With a data card, the train file must be the one the card describes, and neither the train
    nor the val file may be the frozen test set."""
    if card_path is None or not card_path.exists():
        return
    files = json.loads(card_path.read_text()).get("files", {})
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()   # noqa: E731
    train_sha = digest(train_path)
    if "train" in files and files["train"]["sha256"] != train_sha:
        raise ValueError(f"{train_path} does not match the data card's train file; rebuild or pass the right file")
    test_sha = files.get("test", {}).get("sha256")
    for path in (train_path, val_path):
        if path is not None and test_sha and digest(path) == test_sha:
            raise ValueError(f"{path} is the frozen test set; it must never be uploaded for training")


def run_fine_tune(
    provider: FineTuneProvider,
    train_path: Path,
    base_model: str,
    *,
    val_path: Path | None = None,
    data_card_path: Path | None = None,
    hyperparameters: Hyperparameters | None = None,
    suffix: str | None = None,
    poll_interval_s: float = 30.0,
    max_polls: int = 2000,
    sleep: Callable[[float], None] = time.sleep,
    on_status: Callable[[FineTuneJob], None] | None = None,
    resume_job_id: str | None = None,
    max_consecutive_poll_errors: int = 5,
) -> FineTuneRecord:
    """Drive a job to completion. ``sleep`` is injectable so tests run instantly.

    A job runs for hours, so the poll loop must outlive transient API failures: retryable errors
    (429, 5xx, timeouts) are absorbed up to ``max_consecutive_poll_errors`` in a row, while
    non-retryable ones (bad key, unknown job) raise at once. Pass ``resume_job_id`` after a crash
    to keep polling the job that already exists instead of uploading and paying for a second one.
    """
    _check_against_card(data_card_path, train_path, val_path)   # right files, and never the test set
    train_stats = validate_chat_jsonl(train_path)
    if val_path is not None:
        validate_chat_jsonl(val_path)          # before any upload, so a bad val file orphans nothing
    if resume_job_id is not None:
        job = provider.get_job(resume_job_id)
        if job.base_model != base_model:
            raise ValueError(f"job {job.id} trains {job.base_model!r}, not {base_model!r}; resume with the original inputs")
        train_id = job.training_file_id
    else:
        train_id = provider.upload_training_file(train_path)
        val_id = provider.upload_training_file(val_path) if val_path else None
        job = provider.create_job(train_id, base_model, hyperparameters, validation_file_id=val_id, suffix=suffix)
    if on_status:
        on_status(job)  # the caller persists job.id here: it is the only handle for resuming
    last_status: str | None = job.status
    poll_errors = 0
    for _ in range(max_polls):
        try:
            job = provider.get_job(job.id)
            poll_errors = 0
        except LLMError as err:
            poll_errors += 1
            if not err.retryable or poll_errors > max_consecutive_poll_errors:
                raise
            sleep(err.retry_after_s or poll_interval_s)
            continue
        if job.status != last_status and on_status:
            on_status(job)
        last_status = job.status
        if job.status in TERMINAL:
            break
        sleep(poll_interval_s)
    if job.status not in TERMINAL:
        raise FineTunePollTimeout(job, provider.list_events(job.id))
    if job.status != "succeeded" or not job.fine_tuned_model:
        raise FineTuneFailed(job, provider.list_events(job.id))
    card_hash = hashlib.sha256(data_card_path.read_bytes()).hexdigest() if data_card_path and data_card_path.exists() else None
    return FineTuneRecord(
        fine_tuned_model=job.fine_tuned_model, base_model=job.base_model, job_id=job.id, training_file_id=train_id,
        training_file_sha256=train_stats["sha256"], data_card_sha256=card_hash,
        hyperparameters=job.hyperparameters, trained_tokens=job.trained_tokens,
        finished_at=datetime.now(timezone.utc),
    )
