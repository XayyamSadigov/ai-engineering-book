# path: book/projects/p5-incident-agent/incident_agent/adapters/cassette.py
"""Record every model completion of a run to JSONL, keyed by a hash of the request; replay them later.

agentkit's replay() re-runs one agent's event log. A whole investigation also makes model calls
outside any agent loop (planner, writer, judge), so its regression test needs this cassette:
record once against a real or scripted model, then re-run the full pipeline offline and assert
the plan, the trajectory, and the report are unchanged. A request the cassette has never seen
is a miss, raised as a non-retryable error: the pipeline asked something new.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import AsyncIterator, Iterator

from aie_core.llm.client import LLMClient
from aie_core.llm.errors import LLMError
from aie_core.llm.types import Completion, CompletionRequest, StreamEvent


class CassetteMiss(LLMError):
    default_retryable = False


def request_key(req: CompletionRequest) -> str:
    payload = req.model_dump_json(exclude={"metadata", "timeout_s"})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


class RecordingLLM:
    def __init__(self, inner: LLMClient, path: Path) -> None:
        self.inner, self.path = inner, path
        self.provider = getattr(inner, "provider", "unknown")
        self.supports_response_schema = getattr(inner, "supports_response_schema", False)
        path.parent.mkdir(parents=True, exist_ok=True)

    def complete(self, req: CompletionRequest) -> Completion:
        c = self.inner.complete(req)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": request_key(req), "completion": c.model_dump(mode="json")}) + "\n")
        return c

    async def acomplete(self, req: CompletionRequest) -> Completion:
        return self.complete(req)

    def stream(self, req: CompletionRequest) -> Iterator[StreamEvent]:
        return self.inner.stream(req)

    def astream(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        return self.inner.astream(req)


class ReplayLLM:
    provider = "cassette"

    def __init__(self, path: Path, *, supports_response_schema: bool = True) -> None:
        self.supports_response_schema = supports_response_schema
        self._by_key: dict[str, list[Completion]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            self._by_key.setdefault(row["key"], []).append(Completion.model_validate(row["completion"]))
        self._served: dict[str, int] = {}
        self.misses: list[str] = []

    def complete(self, req: CompletionRequest) -> Completion:
        key = request_key(req)
        options = self._by_key.get(key)
        if not options:
            self.misses.append(key)
            raise CassetteMiss(f"no recorded completion for request {key}")
        i = min(self._served.get(key, 0), len(options) - 1)
        self._served[key] = i + 1
        return options[i]

    async def acomplete(self, req: CompletionRequest) -> Completion:
        return self.complete(req)

    def stream(self, req: CompletionRequest) -> Iterator[StreamEvent]:
        raise NotImplementedError("cassette replay serves complete() only")

    def astream(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError("cassette replay serves complete() only")


__all__ = ["RecordingLLM", "ReplayLLM", "CassetteMiss", "request_key"]
