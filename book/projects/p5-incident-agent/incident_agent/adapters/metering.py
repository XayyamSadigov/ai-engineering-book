# path: book/projects/p5-incident-agent/incident_agent/adapters/metering.py
"""Counts every model call of an investigation and enforces a hard ceiling across all roles."""
from __future__ import annotations

import re
import threading
from collections import Counter
from typing import Any, AsyncIterator, Iterator

from aie_core.llm.client import LLMClient
from aie_core.llm.errors import LLMError
from aie_core.llm.types import Completion, CompletionRequest, StreamEvent

_ROLE = re.compile(r"\[role:([a-z_:\-]+)\]")


class CallBudgetExceeded(LLMError):
    default_retryable = False


def role_of(req: CompletionRequest) -> str:
    system = next((m.text for m in req.messages if m.role.value == "system"), "")
    found = _ROLE.search(system)
    if found:
        return found.group(1)
    return "judge" if "evaluation judge" in system else "unknown"


class _Meter:
    """Counters shared by every client of one investigation, so the ceiling covers all roles."""

    def __init__(self, max_calls: int) -> None:
        self.max_calls = max_calls
        self.calls: Counter[str] = Counter()
        self.tokens = 0
        self.lock = threading.Lock()


class MeteredLLM:
    def __init__(self, inner: LLMClient, max_calls: int, *, meter: _Meter | None = None) -> None:
        self.inner = inner
        self.meter = meter or _Meter(max_calls)
        self.max_calls = self.meter.max_calls
        self.provider = getattr(inner, "provider", "unknown")
        self.supports_response_schema = getattr(inner, "supports_response_schema", False)

    def share(self, inner: LLMClient) -> "MeteredLLM":
        """Meter another client (a separate judge model) against the same ceiling and totals."""
        return MeteredLLM(inner, self.max_calls, meter=self.meter)

    @property
    def calls(self) -> Counter[str]:
        return self.meter.calls

    @property
    def tokens(self) -> int:
        return self.meter.tokens

    def complete(self, req: CompletionRequest) -> Completion:
        m = self.meter
        with m.lock:
            if sum(m.calls.values()) >= m.max_calls:
                raise CallBudgetExceeded(f"investigation model-call budget of {m.max_calls} exhausted")
            m.calls[role_of(req)] += 1
        c = self.inner.complete(req)
        with m.lock:
            m.tokens += c.usage.input_tokens + c.usage.output_tokens
        return c

    async def acomplete(self, req: CompletionRequest) -> Completion:
        return self.complete(req)

    def stream(self, req: CompletionRequest) -> Iterator[StreamEvent]:
        return self.inner.stream(req)

    def astream(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        return self.inner.astream(req)

    def snapshot(self) -> dict[str, Any]:
        return {"model_calls": sum(self.calls.values()), "by_role": dict(self.calls), "tokens": self.tokens}


__all__ = ["MeteredLLM", "CallBudgetExceeded", "role_of"]
