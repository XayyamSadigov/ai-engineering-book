# path: book/projects/examples/ch04/prompts/regression.py
"""Prompt regression harness: cases (JSONL) -> run a prompt version against an LLMClient ->
deterministic assertions plus an optional judge -> a report comparing two versions.

Chapter 24 owns the general evaluation toolkit and its statistics; Chapter 25 owns the CI
gate. This module is the narrow tool a prompt author runs before opening a pull request.
"""
from __future__ import annotations

import json
import re
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from aie_core.llm.client import LLMClient
from aie_core.llm.errors import LLMError
from aie_core.observability import Tracer

from .registry import PromptVersion
from .template import PromptRenderError
from .tracing import traced_complete

AssertionType = Literal[
    "json_valid", "schema_valid", "equals", "one_of", "contains", "not_contains",
    "regex", "max_chars", "subset", "quote_in_variable",
]


class Assertion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: AssertionType
    path: str | None = None  # dotted path into the parsed JSON output: "citations.0"
    value: Any = None
    values: list[Any] | None = None
    pattern: str | None = None
    max: int | None = None
    variable: str | None = None


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    variables: dict[str, Any]
    assertions: list[Assertion] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    reference: dict[str, Any] = Field(default_factory=dict)  # material for a judge, never sent to the prompt


class AssertionResult(BaseModel):
    type: str
    passed: bool
    detail: str = ""


class JudgeVerdict(BaseModel):
    score: float
    passed: bool
    rationale: str = ""


Judge = Callable[[Case, str], JudgeVerdict]


class CaseRun(BaseModel):
    attempt: int
    output: str = ""
    passed: bool = False
    assertions: list[AssertionResult] = Field(default_factory=list)
    judge: JudgeVerdict | None = None
    error: str | None = None  # infrastructure or judge failure: not evidence about the prompt
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


class CaseResult(BaseModel):
    case_id: str
    tags: list[str]
    runs: list[CaseRun]

    @property
    def pass_rate(self) -> float:
        scored = [r for r in self.runs if r.error is None]
        return sum(r.passed for r in scored) / len(scored) if scored else 0.0

    @property
    def errored(self) -> bool:
        return all(r.error is not None for r in self.runs)

    @property
    def passed(self) -> bool:
        """Strict: every scored run passed. A case that passes 2 of 3 times is not passing."""
        return not self.errored and self.pass_rate == 1.0

    @property
    def flaky(self) -> bool:
        return 0.0 < self.pass_rate < 1.0


class SuiteResult(BaseModel):
    prompt_id: str
    version: str
    content_hash: str
    repeats: int
    cases: list[CaseResult]

    def by_id(self) -> dict[str, CaseResult]:
        return {c.case_id: c for c in self.cases}

    @property
    def pass_rate(self) -> float:
        scored = [c for c in self.cases if not c.errored]
        return sum(c.passed for c in scored) / len(scored) if scored else 0.0

    def _runs(self) -> list[CaseRun]:
        return [r for c in self.cases for r in c.runs if r.error is None]

    @property
    def mean_input_tokens(self) -> float:
        runs = self._runs()
        return statistics.fmean(r.input_tokens for r in runs) if runs else 0.0

    @property
    def mean_output_tokens(self) -> float:
        runs = self._runs()
        return statistics.fmean(r.output_tokens for r in runs) if runs else 0.0

    @property
    def p95_latency_ms(self) -> float:
        lat = sorted(r.latency_ms for r in self._runs())
        return lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else 0.0


# ---------------------------------------------------------------- loading
def load_cases(path: str | Path) -> list[Case]:
    cases, seen = [], set()
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        case = Case.model_validate_json(line)
        if case.id in seen:
            raise ValueError(f"{path}:{n}: duplicate case id {case.id}")
        seen.add(case.id)
        cases.append(case)
    return cases


# ---------------------------------------------------------------- JSON helpers
_MISSING = object()


def get_path(data: Any, path: str | None) -> Any:
    if not path:
        return data
    cur = data
    for part in path.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return _MISSING
    return cur


_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,), "array": (list,), "string": (str,), "boolean": (bool,),
    "integer": (int,), "number": (int, float), "null": (type(None),),
}


def schema_errors(value: Any, schema: Mapping[str, Any], where: str = "$") -> list[str]:
    """A deliberately small JSON Schema subset (type, enum, const, required, properties,
    additionalProperties, items, minItems, maxItems, maxLength, minimum, maximum). Chapter 6
    validates with pydantic; this is enough for a regression gate without a dependency."""
    errors: list[str] = []
    expected = schema.get("type")
    if expected is not None:
        types = expected if isinstance(expected, list) else [expected]
        ok = any(isinstance(value, _JSON_TYPES[t]) and not (t in ("integer", "number") and isinstance(value, bool)) for t in types)
        if not ok:
            return [f"{where}: expected {expected}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{where}: {value!r} not in enum")
    if "const" in schema and value != schema["const"]:
        errors.append(f"{where}: expected const {schema['const']!r}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in value:
                errors.append(f"{where}: missing required {req!r}")
        for key, sub in value.items():
            if key in props:
                errors += schema_errors(sub, props[key], f"{where}.{key}")
            elif schema.get("additionalProperties") is False:
                errors.append(f"{where}: unexpected property {key!r}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{where}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{where}: more than {schema['maxItems']} items")
        if isinstance(schema.get("items"), Mapping):
            for i, item in enumerate(value):
                errors += schema_errors(item, schema["items"], f"{where}[{i}]")
    if isinstance(value, str) and "maxLength" in schema and len(value) > schema["maxLength"]:
        errors.append(f"{where}: longer than {schema['maxLength']} chars")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{where}: below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{where}: above maximum {schema['maximum']}")
    return errors


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


# ---------------------------------------------------------------- assertions
def check(a: Assertion, output: str, parsed: Any, case: Case, version: PromptVersion) -> AssertionResult:
    def res(ok: bool, detail: str = "") -> AssertionResult:
        return AssertionResult(type=a.type, passed=ok, detail="" if ok else detail)

    if a.type in ("contains", "not_contains", "regex", "max_chars"):
        if a.type == "contains":
            return res(str(a.value).lower() in output.lower(), f"missing {a.value!r}")
        if a.type == "not_contains":
            return res(str(a.value).lower() not in output.lower(), f"found forbidden {a.value!r}")
        if a.type == "regex":
            return res(re.search(a.pattern or "", output) is not None, f"no match for /{a.pattern}/")
        return res(len(output) <= (a.max or 0), f"{len(output)} chars > {a.max}")

    if parsed is _MISSING:
        return res(False, "output is not valid JSON")
    if a.type == "json_valid":
        return res(True)
    if a.type == "schema_valid":
        if version.output_schema is None:
            return res(False, "prompt declares no output_schema")
        errs = schema_errors(parsed, version.output_schema)
        return res(not errs, "; ".join(errs[:3]))

    got = get_path(parsed, a.path)
    if got is _MISSING:
        return res(False, f"path {a.path!r} not found")
    if a.type == "equals":
        return res(got == a.value, f"{a.path}={got!r}, expected {a.value!r}")
    if a.type == "one_of":
        return res(got in (a.values or []), f"{a.path}={got!r} not in {a.values}")
    if a.type == "subset":
        items = got if isinstance(got, list) else [got]
        extra = [x for x in items if x not in (a.values or [])]
        return res(not extra, f"{a.path} has values outside the allowed set: {extra}")
    if a.type == "quote_in_variable":
        # A verifiable intermediate artifact: the quoted evidence must really be in the input.
        scope = case.variables if a.variable is None else case.variables.get(a.variable, "")
        source = " ".join(_norm(s) for s in _strings(scope))
        quote = _norm(str(got))
        return res(bool(quote) and quote in source, f"{a.path}={got!r} is not a quote from the input")
    raise ValueError(f"unhandled assertion type {a.type}")


def _parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return _MISSING


# ---------------------------------------------------------------- running
def run_case(
    version: PromptVersion,
    client: LLMClient,
    case: Case,
    *,
    attempt: int = 0,
    judge: Judge | None = None,
    tracer: Tracer | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> CaseRun:
    try:
        rendered = version.render(case.variables)
    except PromptRenderError as exc:
        # The case does not fit the prompt's declared variables: a suite bug, reported loudly.
        return CaseRun(attempt=attempt, error=f"render: {exc}")
    try:
        completion = traced_complete(client, rendered, tracer, **dict(overrides or {}))
    except LLMError as exc:
        return CaseRun(attempt=attempt, error=f"llm: {type(exc).__name__}: {exc}")
    output = completion.text
    parsed = _parse_json(output)
    results = [check(a, output, parsed, case, version) for a in case.assertions]
    run = CaseRun(
        attempt=attempt,
        output=output,
        assertions=results,
        input_tokens=completion.usage.input_tokens,
        output_tokens=completion.usage.output_tokens,
        latency_ms=completion.latency_ms,
    )
    passed = all(r.passed for r in results)
    # The judge is expensive and noisy: only ask it about outputs that passed the cheap gates.
    if judge is not None and passed:
        try:
            run.judge = judge(case, output)
        except Exception as exc:  # noqa: BLE001 - a broken judge is an error, not a verdict
            run.error = f"judge: {type(exc).__name__}: {exc}"
            return run
        passed = run.judge.passed
    run.passed = passed
    return run


def run_suite(
    version: PromptVersion,
    client: LLMClient,
    cases: Sequence[Case],
    *,
    repeats: int = 1,
    judge: Judge | None = None,
    tracer: Tracer | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> SuiteResult:
    """Run every case `repeats` times. Repeats matter whenever temperature > 0 or the
    provider is not bit-for-bit deterministic: one pass is a sample, not a property."""
    results = []
    for case in cases:
        runs = [
            run_case(version, client, case, attempt=i, judge=judge, tracer=tracer, overrides=overrides)
            for i in range(repeats)
        ]
        results.append(CaseResult(case_id=case.id, tags=case.tags, runs=runs))
    return SuiteResult(
        prompt_id=version.spec.id,
        version=version.spec.version,
        content_hash=version.content_hash,
        repeats=repeats,
        cases=results,
    )


# ---------------------------------------------------------------- comparing
class GateDecision(BaseModel):
    ok: bool
    reasons: list[str]


class Comparison(BaseModel):
    prompt_id: str
    baseline: str
    candidate: str
    baseline_pass_rate: float
    candidate_pass_rate: float
    regressions: list[str]  # passed in baseline, fails in candidate
    fixes: list[str]
    still_failing: list[str]
    flaky: list[str]  # candidate cases that pass on some repeats only
    errored: list[str]  # candidate cases with no scored run: infrastructure, not prompt
    critical_regressions: list[str]
    input_tokens_delta_pct: float
    output_tokens_delta_pct: float
    case_count: int

    def gate(
        self,
        *,
        max_pass_rate_drop: float = 0.0,
        max_input_token_growth_pct: float = 50.0,
        max_regressions: int | None = None,
    ) -> GateDecision:
        """Default policy: block on any critical regression, on a net pass-rate drop, on
        errored cases, and on prompt growth beyond the token budget. Non-critical regressions
        are reported for review; set `max_regressions=0` to block on those too."""
        reasons = []
        if max_regressions is not None and len(self.regressions) > max_regressions:
            reasons.append(f"{len(self.regressions)} regressions (allowed {max_regressions})")
        if self.critical_regressions:
            reasons.append(f"critical cases regressed: {', '.join(self.critical_regressions)}")
        if self.baseline_pass_rate - self.candidate_pass_rate > max_pass_rate_drop:
            reasons.append(f"pass rate fell {self.baseline_pass_rate:.0%} -> {self.candidate_pass_rate:.0%}")
        if self.errored:
            reasons.append(f"{len(self.errored)} cases errored; rerun before judging the prompt")
        if self.input_tokens_delta_pct > max_input_token_growth_pct:
            reasons.append(f"input tokens grew {self.input_tokens_delta_pct:.0f}% (budget {max_input_token_growth_pct:.0f}%)")
        return GateDecision(ok=not reasons, reasons=reasons)


def _pct(new: float, old: float) -> float:
    return 0.0 if old == 0 else (new - old) / old * 100


def compare(baseline: SuiteResult, candidate: SuiteResult, *, critical_tag: str = "critical") -> Comparison:
    if baseline.prompt_id != candidate.prompt_id:
        raise ValueError("compare versions of the same prompt id")
    base, cand = baseline.by_id(), candidate.by_id()
    if set(base) != set(cand):
        raise ValueError("baseline and candidate must run the same case ids")
    regressions, fixes, still, flaky, errored = [], [], [], [], []
    for cid in base:
        b, c = base[cid], cand[cid]
        if c.errored:
            errored.append(cid)
            continue
        if c.flaky:
            flaky.append(cid)
        if b.passed and not c.passed:
            regressions.append(cid)
        elif not b.passed and c.passed:
            fixes.append(cid)
        elif not b.passed and not c.passed:
            still.append(cid)
    critical = [cid for cid in regressions if critical_tag in cand[cid].tags]
    return Comparison(
        prompt_id=baseline.prompt_id,
        baseline=baseline.version,
        candidate=candidate.version,
        baseline_pass_rate=baseline.pass_rate,
        candidate_pass_rate=candidate.pass_rate,
        regressions=regressions,
        fixes=fixes,
        still_failing=still,
        flaky=flaky,
        errored=errored,
        critical_regressions=critical,
        input_tokens_delta_pct=_pct(candidate.mean_input_tokens, baseline.mean_input_tokens),
        output_tokens_delta_pct=_pct(candidate.mean_output_tokens, baseline.mean_output_tokens),
        case_count=len(base),
    )


def render_report(cmp: Comparison, candidate: SuiteResult, gate: GateDecision | None = None) -> str:
    gate = gate or cmp.gate()
    lines = [
        f"# Prompt regression: {cmp.prompt_id} {cmp.baseline} -> {cmp.candidate}",
        "",
        "| metric | baseline | candidate |",
        "|---|---|---|",
        f"| pass rate ({cmp.case_count} cases) | {cmp.baseline_pass_rate:.0%} | {cmp.candidate_pass_rate:.0%} |",
        f"| input tokens per call, change | | {cmp.input_tokens_delta_pct:+.0f}% |",
        f"| output tokens per call, change | | {cmp.output_tokens_delta_pct:+.0f}% |",
        "",
        f"Fixed: {', '.join(cmp.fixes) or 'none'}",
        f"Regressed: {', '.join(cmp.regressions) or 'none'}",
        f"Still failing: {', '.join(cmp.still_failing) or 'none'}",
        f"Flaky: {', '.join(cmp.flaky) or 'none'}",
        f"Errored: {', '.join(cmp.errored) or 'none'}",
        "",
        f"Gate: {'PASS' if gate.ok else 'FAIL'}",
    ]
    lines += [f"- {r}" for r in gate.reasons]
    by_id = candidate.by_id()
    for cid in cmp.regressions + cmp.still_failing:
        run = next((r for r in by_id[cid].runs if not r.passed), None)
        if run is None:
            continue
        failed = [f"{a.type}: {a.detail}" for a in run.assertions if not a.passed]
        if run.judge and not run.judge.passed:
            failed.append(f"judge score {run.judge.score}: {run.judge.rationale}")
        lines.append(f"- {cid}: {'; '.join(failed) or run.error or 'failed'}")
    return "\n".join(lines) + "\n"


__all__ = [
    "Assertion", "Case", "AssertionResult", "JudgeVerdict", "Judge", "CaseRun", "CaseResult",
    "SuiteResult", "GateDecision", "Comparison", "load_cases", "get_path", "schema_errors",
    "check", "run_case", "run_suite", "compare", "render_report",
]
