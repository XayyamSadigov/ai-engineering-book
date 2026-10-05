# path: book/projects/examples/ch18/schema_check.py
"""A deliberately small JSON Schema subset validator for tool arguments.

Both sides use it: the server, because it must never trust a client, and the host,
because it must never trust the model. Supported keywords: type (object at the top,
string / integer / number / boolean for properties), properties, required,
additionalProperties: false, enum, minLength, maxLength, pattern, minimum, maximum.
A production system uses a full validator (for example the `jsonschema` package); the
point here is that validation happens in code, outside the model, on both sides.
"""
from __future__ import annotations

import re
from typing import Any

_PY_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
}


def _check_value(name: str, spec: dict[str, Any], value: Any) -> list[str]:
    errors: list[str] = []
    expected = spec.get("type")
    if expected in _PY_TYPES:
        ok = isinstance(value, _PY_TYPES[expected])
        if expected in ("integer", "number") and isinstance(value, bool):
            ok = False  # bool is a subclass of int in Python; JSON Schema disagrees
        if not ok:
            return [f"{name}: expected {expected}, got {type(value).__name__}"]
    if "enum" in spec and value not in spec["enum"]:
        errors.append(f"{name}: must be one of {spec['enum']}")
    if isinstance(value, str):
        if "minLength" in spec and len(value) < spec["minLength"]:
            errors.append(f"{name}: shorter than {spec['minLength']}")
        if "maxLength" in spec and len(value) > spec["maxLength"]:
            errors.append(f"{name}: longer than {spec['maxLength']}")
        if "pattern" in spec and not re.search(spec["pattern"], value):
            errors.append(f"{name}: does not match {spec['pattern']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in spec and value < spec["minimum"]:
            errors.append(f"{name}: below minimum {spec['minimum']}")
        if "maximum" in spec and value > spec["maximum"]:
            errors.append(f"{name}: above maximum {spec['maximum']}")
    return errors


def validate_arguments(schema: dict[str, Any], args: Any) -> list[str]:
    """Return a list of human-readable errors; an empty list means valid."""
    if not isinstance(args, dict):
        return ["arguments: expected an object"]
    props: dict[str, Any] = schema.get("properties", {})
    errors = [f"{r}: required" for r in schema.get("required", []) if r not in args]
    if schema.get("additionalProperties") is False:
        errors += [f"{k}: unexpected argument" for k in args if k not in props]
    for key, value in args.items():
        if key in props:
            errors += _check_value(key, props[key], value)
    return errors


__all__ = ["validate_arguments"]
