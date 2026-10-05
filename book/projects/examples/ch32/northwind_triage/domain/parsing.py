# path: book/projects/examples/ch32/northwind_triage/domain/parsing.py
"""Turn model text into a TriageDecision, or raise ParseError. Never anything else.

The contract that property tests check:
  * for ANY input string, parse_triage returns a TriageDecision or raises ParseError;
  * a valid decision rendered as JSON and wrapped in prose or code fences parses back to itself;
  * normalization is idempotent.
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from .models import Category, Priority, TriageDecision


class ParseError(ValueError):
    """Model output could not be turned into a valid decision."""


def extract_json_object(text: str) -> dict[str, Any]:
    """Return the first top-level JSON object embedded in text.

    Models wrap JSON in prose ("Here is the result:") and in ``` fences. A regex cannot
    balance braces or respect string escapes, so this is a small scanner: find a '{',
    track depth outside strings, and try json.loads on each balanced candidate.
    """
    if not isinstance(text, str):
        raise ParseError(f"expected str, got {type(text).__name__}")
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
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
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        value = json.loads(candidate)
                    except (json.JSONDecodeError, RecursionError):
                        break
                    if isinstance(value, dict):
                        return value
                    break
        start = text.find("{", start + 1)
    raise ParseError("no JSON object found in model output")


def normalize_priority(value: Any) -> str:
    """Accept 'P2', 'p2', '2', 2, ' P2 ' and return 'P2'. Idempotent."""
    if isinstance(value, bool):
        raise ParseError("priority must not be a boolean")
    raw = str(value).strip().upper()
    if raw.startswith("P"):
        raw = raw[1:]
    if raw not in {"1", "2", "3", "4"}:
        raise ParseError(f"invalid priority {value!r}")
    return f"P{raw}"


def normalize_category(value: Any) -> str:
    """Accept 'Security Report', 'security-report', 'SECURITY_REPORT'. Idempotent."""
    if not isinstance(value, str):
        raise ParseError(f"category must be a string, got {type(value).__name__}")
    raw = value.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return Category(raw).value
    except ValueError as exc:
        raise ParseError(f"unknown category {value!r}") from exc


def normalize_confidence(value: Any) -> float:
    if isinstance(value, bool):
        raise ParseError("confidence must not be a boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ParseError(f"invalid confidence {value!r}") from exc
    if number != number or not 0.0 <= number <= 1.0:  # NaN or out of range
        raise ParseError(f"confidence out of range: {value!r}")
    return number


def parse_triage(text: str) -> TriageDecision:
    obj = extract_json_object(text)
    try:
        return TriageDecision(
            category=Category(normalize_category(obj.get("category"))),
            priority=Priority(normalize_priority(obj.get("priority"))),
            confidence=normalize_confidence(obj.get("confidence", 0.0)),
            rationale=str(obj.get("rationale", ""))[:500],
        )
    except ValidationError as exc:  # defensive: normalizers should make this unreachable
        raise ParseError(str(exc)) from exc


def render_decision(decision: TriageDecision) -> str:
    """Canonical JSON form; the inverse of parse_triage for valid decisions."""
    return json.dumps(
        {
            "category": decision.category.value,
            "priority": decision.priority.value,
            "confidence": decision.confidence,
            "rationale": decision.rationale,
        },
        ensure_ascii=False,
    )
