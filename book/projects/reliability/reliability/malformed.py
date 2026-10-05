# path: book/projects/reliability/reliability/malformed.py
"""Malformed model output inside a chain: validate, repair once, fall back, salvage, default.

A 2xx response that does not parse is not a transport failure, so transport retries are
the wrong tool (Chapter 3). `aie_core.complete_structured` already does validate-and-repair
for one call. A *chain* needs more: a step that still fails after its repair must not take
the whole run down if its output can be recovered some other way. The ladder, cheapest
useful option first:

  1. validate                       -> Outcome.OK
  2. repair once with the error     -> Outcome.REPAIRED     (one extra call, same model)
  3. fallback model, no repair      -> Outcome.FALLBACK     (a different failure distribution)
  4. salvage locally                -> Outcome.SALVAGED     (close truncated JSON; no call)
     or keep the fields that do validate -> Outcome.PARTIAL
  5. a safe default                 -> Outcome.DEFAULT
  6. give up, but return, not raise -> Outcome.FAILED

The step's caller reads `outcome` and decides; a chain records it per step so a degraded
result is visible in traces instead of masquerading as a normal one.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Generic, TypeVar

from aie_core.llm.client import LLMClient
from aie_core.llm.errors import MalformedResponseError
from aie_core.llm.structured import complete_structured, extract_json
from aie_core.llm.types import Completion, CompletionRequest, StreamEvent
from pydantic import BaseModel, TypeAdapter, ValidationError

M = TypeVar("M", bound=BaseModel)


class Outcome(str, Enum):
    OK = "ok"
    REPAIRED = "repaired"
    FALLBACK = "fallback"
    SALVAGED = "salvaged"
    PARTIAL = "partial"
    DEFAULT = "default"
    FAILED = "failed"

    @property
    def degraded(self) -> bool:
        return self not in (Outcome.OK, Outcome.REPAIRED)


@dataclass
class Recovered(Generic[M]):
    value: M | None
    outcome: Outcome
    partial: dict[str, Any] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    calls: int = 0

    @property
    def usable(self) -> bool:
        return self.value is not None

    def raise_if_failed(self) -> M:
        if self.value is None:
            raise MalformedResponseError("; ".join(self.errors) or "no usable structured output")
        return self.value


class _Recorder:
    """Pass-through client that counts calls and keeps the last raw text for salvage."""

    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner
        self.provider = inner.provider
        self.default_model = getattr(inner, "default_model", None)
        self.supports_response_schema = getattr(inner, "supports_response_schema", False)
        self.calls = 0
        self.last: Completion | None = None

    def complete(self, req: CompletionRequest) -> Completion:
        self.calls += 1
        self.last = self.inner.complete(req)
        return self.last

    async def acomplete(self, req: CompletionRequest) -> Completion:
        self.calls += 1
        self.last = await self.inner.acomplete(req)
        return self.last

    def stream(self, req: CompletionRequest) -> Iterator[StreamEvent]:  # pragma: no cover - unused
        return self.inner.stream(req)

    def astream(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:  # pragma: no cover - unused
        return self.inner.astream(req)


def close_truncated_json(text: str) -> str:
    """Best-effort repair of JSON cut off by an output-token limit: close strings and brackets.

    Drops a trailing comma or a dangling `"key":` with no value. It cannot invent missing
    values, so the result still goes through schema validation.
    """
    stack: list[str] = []
    in_string = escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    out = text + ('"' if in_string else "")
    stripped = out.rstrip()
    if stripped.endswith(","):
        stripped = stripped[:-1]
    if stripped.endswith(":"):  # dangling key: remove `, "key":` entirely
        stripped = stripped[:-1].rstrip()
    if stack and stack[-1] == "}" and stripped.endswith('"'):
        start = stripped.rfind('"', 0, len(stripped) - 1)
        before = stripped[:start].rstrip()
        if before.endswith(("{", ",")):  # the last string is a key with no value yet
            stripped = before.rstrip(",")
    return stripped + "".join(reversed(stack))


def _load_lenient(text: str) -> dict[str, Any] | None:
    candidate = extract_json(text)
    for attempt in (candidate, close_truncated_json(candidate)):
        try:
            data = json.loads(attempt)
        except ValueError:
            continue
        if isinstance(data, dict):
            return data
    return None


def salvage_partial(text: str, schema: type[M]) -> tuple[dict[str, Any], list[str]]:
    """Keep every top-level field that validates on its own; report required fields still missing."""
    data = _load_lenient(text) or {}
    valid: dict[str, Any] = {}
    for name, info in schema.model_fields.items():
        key = info.alias or name
        if key not in data:
            continue
        try:
            annotated = Annotated[(info.annotation, *info.metadata)] if info.metadata else info.annotation
            valid[name] = TypeAdapter(annotated).validate_python(data[key])  # keeps ge/le/pattern
        except ValidationError:
            continue
    missing = [n for n, info in schema.model_fields.items() if info.is_required() and n not in valid]
    return valid, missing


def complete_with_recovery(
    client: LLMClient,
    req: CompletionRequest,
    schema: type[M],
    *,
    repair_attempts: int = 1,
    fallback_client: LLMClient | None = None,
    default: M | None = None,
    allow_partial: bool = True,
) -> Recovered[M]:
    """Run the ladder. Transport errors (rate limits, outages) still raise: they belong to the
    gateway and to the chain's failure policy, not to output recovery."""
    errors: list[str] = []
    texts: list[str] = []
    calls = 0

    primary = _Recorder(client)
    try:
        value, _ = complete_structured(primary, req, schema, max_repair_attempts=repair_attempts)
        return Recovered(value=value, outcome=Outcome.OK if primary.calls == 1 else Outcome.REPAIRED,  # type: ignore[arg-type]
                         calls=primary.calls)
    except MalformedResponseError as exc:
        errors.append(f"primary: {exc}")
        calls += primary.calls
        if primary.last is not None:
            texts.append(primary.last.text)

    if fallback_client is not None:
        backup = _Recorder(fallback_client)
        try:
            value, _ = complete_structured(backup, req, schema, max_repair_attempts=0)
            return Recovered(value=value, outcome=Outcome.FALLBACK, errors=errors, calls=calls + backup.calls)  # type: ignore[arg-type]
        except MalformedResponseError as exc:
            errors.append(f"fallback: {exc}")
            calls += backup.calls
            if backup.last is not None:
                texts.insert(0, backup.last.text)

    best: dict[str, Any] = {}
    best_missing: list[str] = [n for n, i in schema.model_fields.items() if i.is_required()]
    for text in texts:
        partial, missing = salvage_partial(text, schema)
        if not missing:
            try:
                return Recovered(value=schema.model_validate(partial), outcome=Outcome.SALVAGED,
                                 partial=partial, errors=errors, calls=calls)
            except ValidationError as exc:  # cross-field validators can still fail
                errors.append(f"salvage: {exc.error_count()} errors")
        if len(partial) > len(best):
            best, best_missing = partial, missing

    if allow_partial and best and default is not None:
        merged = {**default.model_dump(), **best}
        try:
            return Recovered(value=schema.model_validate(merged), outcome=Outcome.PARTIAL, partial=best,
                             missing=best_missing, errors=errors, calls=calls)
        except ValidationError:
            pass
    if default is not None:
        return Recovered(value=default, outcome=Outcome.DEFAULT, partial=best, missing=best_missing,
                         errors=errors, calls=calls)
    return Recovered(value=None, outcome=Outcome.PARTIAL if best else Outcome.FAILED, partial=best,
                     missing=best_missing, errors=errors, calls=calls)


__all__ = ["Outcome", "Recovered", "complete_with_recovery", "salvage_partial", "close_truncated_json"]
