# path: book/projects/evalkit/evalkit/metrics/deterministic.py
"""Deterministic metrics: pure functions, no model calls, same answer every time.

If correctness can be checked with code, check it with code. Every function here is cheap,
reproducible, and explainable in one sentence, which is exactly what an LLM judge is not.
"""
from __future__ import annotations

import json
import math
import re
import string
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ValidationError

_PUNCT = re.compile(f"[{re.escape(string.punctuation)}]")
_WS = re.compile(r"\s+")
_NUM = re.compile(r"[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")


class PRF(BaseModel):
    """Precision, recall, F1 plus the counts they came from (always report the counts)."""

    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def prf_from_counts(tp: int, fp: int, fn: int, *, zero_division: float = 1.0) -> PRF:
    """P/R/F1 with explicit conventions for empty denominators.

    For set metrics (the default, `zero_division=1.0`): nothing predicted and nothing expected
    is a perfect score; predicting nothing when something was expected gives precision 1 (no
    wrong claims) and recall 0. Classification per-label metrics pass `zero_division=0.0`, so a
    class the model never predicts scores precision 0 instead of inflating the macro average.
    """
    precision = tp / (tp + fp) if tp + fp else zero_division
    recall = tp / (tp + fn) if tp + fn else zero_division
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PRF(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn)


# ---------------------------------------------------------------------------- text
def normalize_text(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace. The usual normalization for matching."""
    return _WS.sub(" ", _PUNCT.sub(" ", text.lower())).strip()


def exact_match(prediction: Any, expected: Any, *, normalize: bool = True) -> float:
    """1.0 when prediction equals expected. Strings are normalized unless `normalize=False`."""
    if isinstance(prediction, str) and isinstance(expected, str) and normalize:
        return float(normalize_text(prediction) == normalize_text(expected))
    return float(prediction == expected)


def contains(text: str, phrases: Sequence[str], *, mode: str = "all", normalize: bool = True) -> tuple[bool, list[str]]:
    """Check required phrases. Returns (ok, missing) for mode "all", (ok, found) for mode "any"."""
    hay = normalize_text(text) if normalize else text
    norm = [(p, normalize_text(p) if normalize else p) for p in phrases]
    present = [p for p, n in norm if n and n in hay]
    if mode == "all":
        missing = [p for p, _ in norm if p not in present]
        return (not missing, missing)
    if mode == "any":
        return (bool(present), present)
    raise ValueError("mode must be 'all' or 'any'")


def forbids(text: str, phrases: Sequence[str], *, normalize: bool = True) -> tuple[bool, list[str]]:
    """Check that none of the forbidden phrases appear. Returns (ok, violations)."""
    hay = normalize_text(text) if normalize else text
    hits = [p for p in phrases if (normalize_text(p) if normalize else p) in hay]
    return (not hits, hits)


# ---------------------------------------------------------------------------- numbers
def parse_number(value: Any) -> float | None:
    """Best-effort number extraction: 1,234.50 -> 1234.5; returns None when no number is present."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = _NUM.search(value)
        if m:
            try:
                return float(m.group(0).replace(",", ""))
            except ValueError:
                return None
    return None


def numeric_close(prediction: Any, expected: Any, *, abs_tol: float = 0.0, rel_tol: float = 0.0) -> bool:
    """True when both parse as numbers and |p - e| <= max(abs_tol, rel_tol * |e|)."""
    p, e = parse_number(prediction), parse_number(expected)
    if p is None or e is None:
        return False
    return math.isclose(p, e, abs_tol=abs_tol, rel_tol=rel_tol) or p == e


# ---------------------------------------------------------------------------- sets
def set_precision_recall(predicted: Iterable[Any], expected: Iterable[Any]) -> PRF:
    """Set overlap metrics: cited source ids, extracted entities, tools called, labels assigned."""
    p, e = set(predicted), set(expected)
    return prf_from_counts(len(p & e), len(p - e), len(e - p))


# ---------------------------------------------------------------------------- structured fields
class FieldScores(PRF):
    per_field: dict[str, str]  # field -> "tp" | "fp" | "fn" | "fp+fn" | "tn"


def _field_equal(a: Any, b: Any, normalize: bool, numeric_tol: float) -> bool:
    if isinstance(a, str) and isinstance(b, str):
        return normalize_text(a) == normalize_text(b) if normalize else a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return math.isclose(float(a), float(b), abs_tol=numeric_tol)
    return a == b


def field_prf(
    predicted: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    fields: Sequence[str] | None = None,
    normalize: bool = True,
    numeric_tol: float = 0.0,
) -> FieldScores:
    """Field-level precision/recall/F1 for extraction.

    For each field: correct non-null value is a TP; a non-null prediction where the expected
    value is null is an FP (invented field); a missing prediction for a non-null expected value
    is an FN; a wrong non-null value counts as both an FP and an FN, because it is a wrong
    claim *and* a missed fact. Both null is a true negative and does not enter the counts.
    """
    names = list(fields) if fields is not None else sorted(set(predicted) | set(expected))
    tp = fp = fn = 0
    per: dict[str, str] = {}
    for f in names:
        p, e = predicted.get(f), expected.get(f)
        p_null, e_null = p in (None, "", []), e in (None, "", [])
        if p_null and e_null:
            per[f] = "tn"
        elif p_null:
            fn += 1
            per[f] = "fn"
        elif e_null:
            fp += 1
            per[f] = "fp"
        elif _field_equal(p, e, normalize, numeric_tol):
            tp += 1
            per[f] = "tp"
        else:
            fp += 1
            fn += 1
            per[f] = "fp+fn"
    base = prf_from_counts(tp, fp, fn)
    return FieldScores(**base.model_dump(), per_field=per)


# ---------------------------------------------------------------------------- JSON schema
_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "null": (type(None),),
}


def _type_ok(value: Any, t: str) -> bool:
    if t in ("integer", "number") and isinstance(value, bool):
        return False
    return isinstance(value, _JSON_TYPES.get(t, (object,)))


def _validate(value: Any, schema: Mapping[str, Any], path: str, errors: list[str]) -> None:
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in enum {schema['enum']}")
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected const {schema['const']!r}")
    t = schema.get("type")
    if t is not None:
        types = [t] if isinstance(t, str) else list(t)
        if not any(_type_ok(value, x) for x in types):
            errors.append(f"{path}: expected {t}, got {type(value).__name__}")
            return
    if "anyOf" in schema:
        if not any(not _collect(value, s, path) for s in schema["anyOf"]):
            errors.append(f"{path}: matches none of anyOf")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in value:
                errors.append(f"{path}: missing required field '{req}'")
        for k, v in value.items():
            if k in props:
                _validate(v, props[k], f"{path}.{k}", errors)
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: unexpected field '{k}'")
    if isinstance(value, list):
        if "items" in schema:
            for i, item in enumerate(value):
                _validate(item, schema["items"], f"{path}[{i}]", errors)
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: more than {schema['maxItems']} items")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} > maximum {schema['maximum']}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than {schema['maxLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path}: does not match pattern {schema['pattern']!r}")


def _collect(value: Any, schema: Mapping[str, Any], path: str) -> list[str]:
    errs: list[str] = []
    _validate(value, schema, path, errs)
    return errs


def json_schema_valid(output: Any, schema: Mapping[str, Any] | type[BaseModel]) -> tuple[bool, list[str]]:
    """Validate `output` (JSON text or already-parsed value) against a schema.

    A pydantic model class is validated with pydantic. A dict schema is checked with a small
    built-in validator covering the subset structured-output schemas use in practice: type,
    enum, const, required, properties, additionalProperties, items, min/max, lengths, pattern,
    anyOf. `$ref` and conditionals are not supported; use a full JSON Schema library for those.
    """
    value = output
    if isinstance(output, str):
        try:
            value = json.loads(output)
        except json.JSONDecodeError as exc:
            return False, [f"$: not valid JSON ({exc.msg})"]
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        try:
            schema.model_validate(value)
            return True, []
        except ValidationError as exc:
            return False, [f"$.{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
    errors = _collect(value, schema, "$")
    return (not errors, errors)


__all__ = [
    "PRF",
    "FieldScores",
    "prf_from_counts",
    "normalize_text",
    "exact_match",
    "contains",
    "forbids",
    "parse_number",
    "numeric_close",
    "set_precision_recall",
    "field_prf",
    "json_schema_valid",
]
