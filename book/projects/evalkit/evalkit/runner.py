# path: book/projects/evalkit/evalkit/runner.py
"""Run a target over a dataset, capture everything, score it, and keep the lineage.

The runner separates three things that teams often blur:

1. *Generation*: call the system under test (the "target") once per case, with bounded
   concurrency, and record output, latency, tokens, cost, trace id, and errors.
2. *Scoring*: apply evaluators to (case, output). Scoring can be repeated later with new
   evaluators without paying for generation again (`score_run`).
3. *Lineage*: a `Run` carries the versions of target, prompt, model, dataset, and every
   evaluator. A score without those versions cannot support a release decision.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import math
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from aie_core.llm.types import Completion
from aie_core.observability import NoopTracer, Tracer
from pydantic import BaseModel, Field

from .cases import Dataset, EvalCase


# ============================================================================ scores and evaluators
class Score(BaseModel):
    """One named measurement of one output. `value` is in [0, 1] by convention."""

    name: str
    value: float
    passed: bool | None = None
    detail: Any = None


EvaluatorOutput = Score | Sequence[Score] | float | int | bool


@runtime_checkable
class Evaluator(Protocol):
    """Anything with a name, a version, and `__call__(case, output)`.

    `metric_names` lists every Score name the evaluator can emit; the runner uses it to
    record failures for cases whose target errored. It defaults to `[name]`.
    """

    name: str
    version: str

    def __call__(self, case: EvalCase, output: Any) -> EvaluatorOutput: ...


class FunctionEvaluator:
    """Wrap a plain function `(case, output) -> float | bool | Score | list[Score]`."""

    def __init__(
        self,
        name: str,
        fn: Callable[[EvalCase, Any], EvaluatorOutput],
        *,
        version: str = "1",
        pass_threshold: float | None = 1.0,
        metric_names: Sequence[str] | None = None,
    ) -> None:
        self.name = name
        self.fn = fn
        self.version = version
        self.pass_threshold = pass_threshold
        self.metric_names = list(metric_names or [name])

    def __call__(self, case: EvalCase, output: Any) -> list[Score]:
        return coerce_scores(self.name, self.fn(case, output), self.pass_threshold)


def evaluator(
    name: str | None = None,
    *,
    version: str = "1",
    pass_threshold: float | None = 1.0,
    metric_names: Sequence[str] | None = None,
) -> Callable[[Callable[[EvalCase, Any], EvaluatorOutput]], FunctionEvaluator]:
    """Decorator form of `FunctionEvaluator`."""

    def wrap(fn: Callable[[EvalCase, Any], EvaluatorOutput]) -> FunctionEvaluator:
        return FunctionEvaluator(
            name or fn.__name__, fn, version=version, pass_threshold=pass_threshold, metric_names=metric_names
        )

    return wrap


def coerce_scores(name: str, raw: EvaluatorOutput, pass_threshold: float | None = 1.0) -> list[Score]:
    if isinstance(raw, Score):
        return [raw]
    if isinstance(raw, bool):
        return [Score(name=name, value=1.0 if raw else 0.0, passed=raw)]
    if isinstance(raw, (int, float)):
        value = float(raw)
        passed = None if pass_threshold is None else value >= pass_threshold
        return [Score(name=name, value=value, passed=passed)]
    if isinstance(raw, Sequence) and all(isinstance(s, Score) for s in raw):
        return list(raw)
    raise TypeError(f"evaluator {name!r} returned unsupported type {type(raw).__name__}")


def _metric_names(ev: Evaluator) -> list[str]:
    return list(getattr(ev, "metric_names", None) or [ev.name])


# ============================================================================ target results
class TargetResult(BaseModel):
    """What a target may return when it wants to report cost, tokens, or its own trace id."""

    output: Any
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    trace_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def _coerce_target_result(value: Any) -> TargetResult:
    if isinstance(value, TargetResult):
        return value
    if isinstance(value, Completion):
        raw = value.raw or {}
        output: Any = value.text
        if not output and value.tool_calls:
            output = [tc.model_dump() for tc in value.tool_calls]
        return TargetResult(
            output=output,
            cost_usd=float(raw.get("cost_usd", 0.0) or 0.0),
            input_tokens=value.usage.input_tokens,
            output_tokens=value.usage.output_tokens,
            metadata={"model": value.model, "provider": value.provider, "finish_reason": value.finish_reason},
        )
    return TargetResult(output=value)


Target = Callable[[EvalCase], Any] | Callable[[EvalCase], Awaitable[Any]]


# ============================================================================ run records
class CaseResult(BaseModel):
    case_id: str
    repeat: int = 0
    output: Any = None
    error: str | None = None
    error_type: str | None = None
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    trace_id: str | None = None
    scores: dict[str, float | None] = Field(default_factory=dict)
    passed: dict[str, bool | None] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)
    evaluator_errors: dict[str, str] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunVersions(BaseModel):
    """Everything that, if changed, could change the score."""

    target: str
    prompt: str | None = None
    model: str | None = None
    dataset: str = ""  # dataset fingerprint, filled by the runner
    evaluators: dict[str, str] = Field(default_factory=dict)  # name -> version, filled by the runner
    extra: dict[str, str] = Field(default_factory=dict)  # index version, tool schema version, ...


class Run(BaseModel):
    run_id: str
    created_at: datetime
    versions: RunVersions
    dataset_name: str
    dataset_version: str
    dataset_hash: str
    concurrency: int = 1
    repeats: int = 1
    wall_time_s: float = 0.0
    results: list[CaseResult] = Field(default_factory=list)

    # ------------------------------------------------------------------ access
    def by_case(self) -> dict[str, list[CaseResult]]:
        out: dict[str, list[CaseResult]] = {}
        for r in self.results:
            out.setdefault(r.case_id, []).append(r)
        return out

    def metric_names(self) -> list[str]:
        names: dict[str, None] = {}
        for r in self.results:
            names.update(dict.fromkeys(r.scores))
        return list(names)

    def case_scores(self, metric: str) -> dict[str, float]:
        """Per-case score, averaged over repeats. Cases where the metric is missing are omitted."""
        out: dict[str, float] = {}
        for case_id, rows in self.by_case().items():
            vals = [r.scores[metric] for r in rows if r.scores.get(metric) is not None]
            if vals:
                out[case_id] = sum(vals) / len(vals)  # type: ignore[arg-type]
        return out

    def mean(self, metric: str) -> float:
        vals = list(self.case_scores(metric).values())
        return sum(vals) / len(vals) if vals else math.nan

    def pass_rate(self, metric: str) -> float:
        flags = [r.passed[metric] for r in self.results if r.passed.get(metric) is not None]
        return sum(1 for f in flags if f) / len(flags) if flags else math.nan

    def failing_cases(self, metric: str) -> list[str]:
        return sorted({r.case_id for r in self.results if r.passed.get(metric) is False})

    def flaky_cases(self, metric: str) -> list[str]:
        """Cases whose pass/fail verdict differs across repeats: the nondeterminism you ship."""
        out = []
        for case_id, rows in self.by_case().items():
            verdicts = {r.passed.get(metric) for r in rows if r.passed.get(metric) is not None}
            if len(verdicts) > 1:
                out.append(case_id)
        return sorted(out)

    @property
    def errors(self) -> list[CaseResult]:
        return [r for r in self.results if r.error is not None]

    @property
    def error_rate(self) -> float:
        return len(self.errors) / len(self.results) if self.results else 0.0

    @property
    def evaluator_error_count(self) -> int:
        return sum(len(r.evaluator_errors) for r in self.results)

    def latency_percentile(self, p: float) -> float:
        """Nearest-rank percentile of per-call latency in ms, p in [0, 100]."""
        xs = sorted(r.latency_ms for r in self.results)
        if not xs:
            return math.nan
        k = max(0, min(len(xs) - 1, math.ceil(p / 100 * len(xs)) - 1))
        return xs[k]

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.results)

    @property
    def cost_per_case_usd(self) -> float:
        return self.total_cost_usd / len(self.results) if self.results else 0.0

    # ------------------------------------------------------------------ persistence
    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load_json(cls, path: str | Path) -> "Run":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


# ============================================================================ execution
def _score_case(
    case: EvalCase,
    result: CaseResult,
    evaluators: Sequence[Evaluator],
    error_score: float,
) -> None:
    for ev in evaluators:
        if result.error is not None:
            # A crashed target is a failed case, never a missing one: averages must include it.
            for name in _metric_names(ev):
                result.scores[name] = error_score
                result.passed[name] = False
                result.details[name] = "target_error"
            continue
        try:
            for s in coerce_scores(ev.name, ev(case, result.output)):
                result.scores[s.name] = s.value
                result.passed[s.name] = s.passed
                if s.detail is not None:
                    result.details[s.name] = s.detail
        except Exception as exc:  # an evaluator bug must be visible, not a silent zero
            result.evaluator_errors[ev.name] = f"{type(exc).__name__}: {exc}"
            for name in _metric_names(ev):
                result.scores[name] = None
                result.passed[name] = None


def _new_result(case: EvalCase, repeat: int) -> CaseResult:
    return CaseResult(case_id=case.id, repeat=repeat, tags=list(case.tags))


def _fill(result: CaseResult, tr: TargetResult, latency_ms: float, span_id: str) -> None:
    result.output = tr.output
    result.latency_ms = latency_ms
    result.cost_usd = tr.cost_usd
    result.input_tokens = tr.input_tokens
    result.output_tokens = tr.output_tokens
    result.trace_id = tr.trace_id or span_id
    result.metadata = tr.metadata


def _run_one_sync(
    target: Callable[[EvalCase], Any],
    case: EvalCase,
    repeat: int,
    run_id: str,
    tracer: Tracer,
    evaluators: Sequence[Evaluator],
    error_score: float,
) -> CaseResult:
    result = _new_result(case, repeat)
    with tracer.span("eval.case", run_id=run_id, case_id=case.id, repeat=repeat) as span:
        start = time.perf_counter()
        try:
            tr = _coerce_target_result(target(case))
            _fill(result, tr, (time.perf_counter() - start) * 1000, span.span_id)
        except Exception as exc:
            result.latency_ms = (time.perf_counter() - start) * 1000
            result.error, result.error_type = str(exc), type(exc).__name__
            result.trace_id = span.span_id
            span.set_attribute("error.type", result.error_type)
    _score_case(case, result, evaluators, error_score)
    return result


def _make_run(
    dataset: Dataset, versions: RunVersions, evaluators: Sequence[Evaluator], concurrency: int, repeats: int
) -> Run:
    v = versions.model_copy(
        update={
            "dataset": dataset.fingerprint,
            "evaluators": {**versions.evaluators, **{ev.name: str(ev.version) for ev in evaluators}},
        }
    )
    return Run(
        run_id=uuid.uuid4().hex[:12],
        created_at=datetime.now(timezone.utc),
        versions=v,
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        dataset_hash=dataset.content_hash,
        concurrency=concurrency,
        repeats=repeats,
    )


def run_target(
    target: Target,
    dataset: Dataset,
    *,
    versions: RunVersions | None = None,
    evaluators: Sequence[Evaluator] = (),
    concurrency: int = 8,
    repeats: int = 1,
    tracer: Tracer | None = None,
    error_score: float = 0.0,
    on_result: Callable[[CaseResult], None] | None = None,
) -> Run:
    """Run `target(case)` for every case (`repeats` times each) and score the outputs.

    Sync targets run on a thread pool of size `concurrency`; async targets are dispatched to
    `arun_target`. Results keep dataset order regardless of completion order.
    """
    if inspect.iscoroutinefunction(target):
        return asyncio.run(
            arun_target(
                target,
                dataset,
                versions=versions,
                evaluators=evaluators,
                concurrency=concurrency,
                repeats=repeats,
                tracer=tracer,
                error_score=error_score,
                on_result=on_result,
            )
        )
    if concurrency < 1 or repeats < 1:
        raise ValueError("concurrency and repeats must be >= 1")
    versions = versions or RunVersions(target=getattr(target, "__name__", "target"))
    tracer = tracer or NoopTracer()
    run = _make_run(dataset, versions, evaluators, concurrency, repeats)
    jobs = [(case, rep) for case in dataset for rep in range(repeats)]
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(_run_one_sync, target, case, rep, run.run_id, tracer, evaluators, error_score)  # type: ignore[arg-type]
            for case, rep in jobs
        ]
        for fut in futures:
            res = fut.result()
            run.results.append(res)
            if on_result:
                on_result(res)
    run.wall_time_s = time.perf_counter() - start
    return run


async def arun_target(
    target: Target,
    dataset: Dataset,
    *,
    versions: RunVersions | None = None,
    evaluators: Sequence[Evaluator] = (),
    concurrency: int = 8,
    repeats: int = 1,
    tracer: Tracer | None = None,
    error_score: float = 0.0,
    on_result: Callable[[CaseResult], None] | None = None,
) -> Run:
    """Async variant: the target is awaited under a semaphore; evaluators run in threads."""
    if concurrency < 1 or repeats < 1:
        raise ValueError("concurrency and repeats must be >= 1")
    versions = versions or RunVersions(target=getattr(target, "__name__", "target"))
    tracer = tracer or NoopTracer()
    run = _make_run(dataset, versions, evaluators, concurrency, repeats)
    sem = asyncio.Semaphore(concurrency)

    async def one(case: EvalCase, rep: int) -> CaseResult:
        async with sem:
            result = _new_result(case, rep)
            with tracer.span("eval.case", run_id=run.run_id, case_id=case.id, repeat=rep) as span:
                start = time.perf_counter()
                try:
                    value = target(case)
                    if inspect.isawaitable(value):
                        value = await value
                    _fill(result, _coerce_target_result(value), (time.perf_counter() - start) * 1000, span.span_id)
                except Exception as exc:
                    result.latency_ms = (time.perf_counter() - start) * 1000
                    result.error, result.error_type = str(exc), type(exc).__name__
                    result.trace_id = span.span_id
        await asyncio.to_thread(_score_case, case, result, evaluators, error_score)
        if on_result:
            on_result(result)
        return result

    start = time.perf_counter()
    run.results = list(await asyncio.gather(*(one(c, r) for c in dataset for r in range(repeats))))
    run.wall_time_s = time.perf_counter() - start
    return run


def score_run(
    run: Run,
    dataset: Dataset,
    evaluators: Sequence[Evaluator],
    *,
    error_score: float = 0.0,
    replace: bool = False,
) -> Run:
    """Re-score stored outputs with (new) evaluators, without calling the target again.

    The dataset must be the one the run was generated on (checked by content hash).
    With `replace=False` the new scores are merged into the existing ones.
    """
    if dataset.content_hash != run.dataset_hash:
        raise ValueError(
            f"run {run.run_id} was generated on dataset hash {run.dataset_hash[:12]}, "
            f"not {dataset.content_hash[:12]}"
        )
    new = run.model_copy(deep=True)
    new.run_id = uuid.uuid4().hex[:12]
    evs = {**({} if replace else new.versions.evaluators), **{ev.name: str(ev.version) for ev in evaluators}}
    new.versions = new.versions.model_copy(update={"evaluators": evs})
    for r in new.results:
        if replace:
            r.scores, r.passed, r.details, r.evaluator_errors = {}, {}, {}, {}
        _score_case(dataset.get(r.case_id), r, evaluators, error_score)
    return new


__all__ = [
    "Score",
    "Evaluator",
    "EvaluatorOutput",
    "FunctionEvaluator",
    "evaluator",
    "coerce_scores",
    "TargetResult",
    "Target",
    "CaseResult",
    "RunVersions",
    "Run",
    "run_target",
    "arun_target",
    "score_run",
]
