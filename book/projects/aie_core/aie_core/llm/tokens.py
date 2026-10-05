# path: book/projects/aie_core/aie_core/llm/tokens.py
"""Token counting for budgets and rate limiting.

Exact counts come from the provider's `usage` field after the fact. Before the call we can
only estimate: tiktoken when installed (exact for OpenAI tokenizers, a decent proxy for
others), otherwise a characters-per-token heuristic. Treat every number here as an
estimate and leave headroom.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from typing import Any

from .types import Message

CHARS_PER_TOKEN = 4.0  # illustrative average for English prose; code and non-Latin text differ
MESSAGE_OVERHEAD_TOKENS = 4  # role, separators, and framing per message (approximation)
REPLY_PRIMING_TOKENS = 3  # the assistant header the model is primed with

_DISABLE_TIKTOKEN = False


@lru_cache(maxsize=16)
def _encoding_for(model: str | None) -> Any | None:
    if _DISABLE_TIKTOKEN:
        return None
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        if model:
            try:
                return tiktoken.encoding_for_model(model)
            except KeyError:
                pass
        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # offline without a cached vocabulary, or an unexpected tiktoken failure
        return None


def heuristic_count(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


def count_tokens(text: str, model: str | None = None) -> int:
    enc = _encoding_for(model)
    if enc is None:
        return heuristic_count(text)
    return len(enc.encode(text, disallowed_special=()))


def count_message_tokens(messages: list[Message], model: str | None = None) -> int:
    total = REPLY_PRIMING_TOKENS
    for m in messages:
        total += MESSAGE_OVERHEAD_TOKENS
        total += count_tokens(m.text, model)
        if m.name:
            total += count_tokens(m.name, model)
        for tc in m.tool_calls or []:
            total += count_tokens(tc.name, model)
            total += count_tokens(json.dumps(tc.arguments, ensure_ascii=False), model)
    return total


__all__ = ["count_tokens", "count_message_tokens", "heuristic_count", "CHARS_PER_TOKEN"]
