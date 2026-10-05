# path: book/projects/aie_core/aie_core/llm/structured.py
"""Structured output with a repair loop.

Strategy: ask for the schema natively when the client supports it, parse the text as JSON
(tolerating code fences), validate with pydantic, and on failure send the validation error
back as a user message so the model can correct itself. Give up after a bounded number of
repairs and raise MalformedResponseError; the caller decides what to do with the failure.
"""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from .client import LLMClient
from .errors import MalformedResponseError
from .types import Completion, CompletionRequest, Message

_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)

SCHEMA_INSTRUCTION = (
    "Respond with a single JSON object and nothing else. It must validate against this JSON Schema:\n{schema}"
)
REPAIR_INSTRUCTION = (
    "Your previous answer did not match the required schema. Error:\n{error}\n"
    "Reply again with only the corrected JSON object."
)


def extract_json(text: str) -> str:
    """Strip Markdown fences and leading prose so `json.loads` sees the object itself."""
    stripped = text.strip()
    m = _FENCE_RE.match(stripped)
    if m:
        return m.group(1).strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return stripped
    # last resort: take the outermost braces
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


class TruncatedOutputError(MalformedResponseError):
    """The model hit `max_tokens` before finishing the object. Re-asking with the same budget
    cannot help, so the repair loop stops immediately; raise `max_tokens` or shrink the schema."""


def parse_structured(completion: Completion, schema: type[BaseModel]) -> BaseModel:
    """Validate a completion against `schema`. Accepts JSON text or a single tool call.
    A truncated completion is rejected even when the cut-off text happens to parse."""
    if completion.truncated:
        raise TruncatedOutputError(
            f"structured output truncated at max_tokens after {completion.usage.output_tokens} output tokens",
            provider=completion.provider,
            raw={"truncated": True, "text": completion.text},
        )
    if not completion.text.strip() and completion.tool_calls:
        payload: Any = completion.tool_calls[0].arguments
    else:
        payload = json.loads(extract_json(completion.text))
    return schema.model_validate(payload)


def _prepare(client: LLMClient, req: CompletionRequest, schema: type[BaseModel]) -> CompletionRequest:
    schema_json = schema.model_json_schema()
    if getattr(client, "supports_response_schema", False):
        return req.model_copy(update={"response_schema": schema_json})
    instruction = SCHEMA_INSTRUCTION.format(schema=json.dumps(schema_json, ensure_ascii=False))
    messages = list(req.messages)
    if messages and messages[0].role.value == "system":
        messages[0] = Message.system(messages[0].text + "\n\n" + instruction)
    else:
        messages.insert(0, Message.system(instruction))
    return req.model_copy(update={"messages": messages})


def complete_structured(
    client: LLMClient,
    req: CompletionRequest,
    schema: type[BaseModel],
    max_repair_attempts: int = 2,
) -> tuple[BaseModel, Completion]:
    current = _prepare(client, req, schema)
    last_error: Exception | None = None
    for attempt in range(max_repair_attempts + 1):
        completion = client.complete(current)
        try:
            return parse_structured(completion, schema), completion
        except TruncatedOutputError:
            raise  # repairing at the same max_tokens would truncate again
        except (ValueError, ValidationError) as exc:  # json.JSONDecodeError is a ValueError
            last_error = exc
            if attempt == max_repair_attempts:
                break
            current = current.model_copy(
                update={
                    "messages": [
                        *current.messages,
                        Message.assistant(completion.text or json.dumps(
                            completion.tool_calls[0].arguments if completion.tool_calls else {}
                        )),
                        Message.user(REPAIR_INSTRUCTION.format(error=_short(str(exc)))),
                    ]
                }
            )
    raise MalformedResponseError(
        f"structured output failed validation after {max_repair_attempts + 1} attempts: {_short(str(last_error))}",
        provider=getattr(client, "provider", None),
        raw=str(last_error),
    )


async def acomplete_structured(
    client: LLMClient,
    req: CompletionRequest,
    schema: type[BaseModel],
    max_repair_attempts: int = 2,
) -> tuple[BaseModel, Completion]:
    current = _prepare(client, req, schema)
    last_error: Exception | None = None
    for attempt in range(max_repair_attempts + 1):
        completion = await client.acomplete(current)
        try:
            return parse_structured(completion, schema), completion
        except TruncatedOutputError:
            raise
        except (ValueError, ValidationError) as exc:
            last_error = exc
            if attempt == max_repair_attempts:
                break
            current = current.model_copy(
                update={
                    "messages": [
                        *current.messages,
                        Message.assistant(completion.text),
                        Message.user(REPAIR_INSTRUCTION.format(error=_short(str(exc)))),
                    ]
                }
            )
    raise MalformedResponseError(
        f"structured output failed validation after {max_repair_attempts + 1} attempts: {_short(str(last_error))}",
        provider=getattr(client, "provider", None),
        raw=str(last_error),
    )


def _short(text: str, limit: int = 1500) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


__all__ = ["complete_structured", "acomplete_structured", "parse_structured", "extract_json", "TruncatedOutputError"]
