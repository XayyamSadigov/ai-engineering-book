# path: book/projects/agentkit/agentkit/observations.py
"""Observation shaping: what part of a tool result the model actually reads.

Every observation is replayed on every later step, so a 40 KB log dump at step 2 is paid
for again at steps 3, 4, and 5. Truncation is applied once, when the result is recorded,
and the event keeps the original size so the trace shows what was cut.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Truncated:
    text: str
    original_chars: int
    truncated: bool


def truncate_observation(text: str, max_chars: int, *, head_ratio: float = 0.7) -> Truncated:
    """Keep the head and the tail and say how much was removed.

    Head-and-tail beats head-only for logs and stack traces, where the cause is often at the
    end. Tools that know their structure (search results, tables) should paginate or
    summarize themselves instead of relying on this generic cut.
    """
    original = len(text)
    if max_chars <= 0 or original <= max_chars:
        return Truncated(text=text, original_chars=original, truncated=False)
    marker_budget = 80
    keep = max(0, max_chars - marker_budget)
    head = int(keep * head_ratio)
    tail = keep - head
    removed = original - head - tail
    marker = f"\n...[{removed} characters truncated by harness; refine the query to see more]...\n"
    body = text[:head] + marker + (text[-tail:] if tail else "")
    return Truncated(text=body, original_chars=original, truncated=True)


__all__ = ["Truncated", "truncate_observation"]
