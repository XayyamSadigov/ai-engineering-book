# path: book/projects/examples/ch05/context/layout.py
"""Cache-friendly layout checks.

Prefix caching (provider prompt caching, or prefix reuse in a serving engine, Chapter 34)
only pays when consecutive requests share an identical token prefix. These helpers make
that property testable: how much of two prompts is shared, which stable items contain
volatile text, and what fraction of a request stream kept its prefix.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from aie_core.llm.tokens import count_tokens
from aie_core.llm.types import Message

from .builder import BuildResult
from .items import ContextItem

VOLATILE_PATTERNS: dict[str, re.Pattern[str]] = {
    "timestamp": re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}"),
    "date": re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    "uuid": re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I),
    "request_id": re.compile(r"\b(?:request|trace|session)[_ -]?id\s*[:=]\s*\S+", re.I),
}


def serialize(messages: Sequence[Message]) -> str:
    """A provider-neutral linearization; real tokenization differs, but prefixes behave the same."""
    return "".join(f"<|{m.role.value}|>{m.text}" for m in messages)


def shared_prefix_tokens(
    a: Sequence[Message], b: Sequence[Message], counter: Callable[[str], int] = count_tokens
) -> int:
    sa, sb = serialize(a), serialize(b)
    n = 0
    for ca, cb in zip(sa, sb):
        if ca != cb:
            break
        n += 1
    return counter(sa[:n])


def lint_stable_items(items: Sequence[ContextItem]) -> list[str]:
    """Warn about volatile content inside items that are supposed to be identical every request."""
    warnings: list[str] = []
    for item in items:
        if not item.stable:
            continue
        for name, pattern in VOLATILE_PATTERNS.items():
            if pattern.search(item.content):
                warnings.append(f"{item.source_id}: {name} in stable {item.kind} breaks prefix caching")
    return warnings


class PrefixStabilityTracker:
    """Feed it every BuildResult; it reports how often the stable prefix repeated.

    This is an upper bound on what a prefix cache could hit. The real hit rate comes from the
    provider's cached-token usage field (Usage.cached_input_tokens) and must be tracked too.
    """

    def __init__(self) -> None:
        self.seen: set[str] = set()
        self.requests = 0
        self.repeats = 0

    def observe(self, result: BuildResult) -> bool:
        self.requests += 1
        hit = result.prefix_hash in self.seen
        self.repeats += hit
        self.seen.add(result.prefix_hash)
        return hit

    @property
    def repeat_rate(self) -> float:
        return self.repeats / self.requests if self.requests else 0.0


__all__ = ["serialize", "shared_prefix_tokens", "lint_stable_items", "PrefixStabilityTracker", "VOLATILE_PATTERNS"]
