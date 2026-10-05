# path: book/projects/p1-extraction-api/extraction_api/application/metering.py
"""Count every model call a document costs, including the ones a repair loop hides.

`complete_structured` returns only the final completion, so its usage understates the real
cost of a document that needed two schema repairs. Wrapping the client per document gives
an exact bill: calls, input tokens, output tokens.
"""
from __future__ import annotations

import threading
from typing import AsyncIterator, Iterator

from aie_core import Completion, CompletionRequest, LLMClient, StreamEvent, Usage


class MeteredClient:
    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.provider = getattr(inner, "provider", "unknown")
        self.calls = 0
        self.usage = Usage()
        self.calls_by_task: dict[str, int] = {}
        self._lock = threading.Lock()

    @property
    def supports_response_schema(self) -> bool:
        # complete_structured asks this to choose native schema mode vs prompt+parse
        return bool(getattr(self._inner, "supports_response_schema", False))

    def _record(self, req: CompletionRequest, completion: Completion) -> None:
        task = str(req.metadata.get("task", "unknown"))
        with self._lock:
            self.calls += 1
            self.usage = self.usage + completion.usage
            self.calls_by_task[task] = self.calls_by_task.get(task, 0) + 1

    def complete(self, req: CompletionRequest) -> Completion:
        completion = self._inner.complete(req)
        self._record(req, completion)
        return completion

    async def acomplete(self, req: CompletionRequest) -> Completion:
        completion = await self._inner.acomplete(req)
        self._record(req, completion)
        return completion

    def stream(self, req: CompletionRequest) -> Iterator[StreamEvent]:  # not used for extraction
        return self._inner.stream(req)

    def astream(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        return self._inner.astream(req)


__all__ = ["MeteredClient"]
