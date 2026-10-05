# path: book/projects/aie_core/aie_core/llm/providers/fake.py
"""A scripted LLM for tests. No network, deterministic, records every request."""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, AsyncIterator, Iterator, Sequence

from ..tokens import count_message_tokens, count_tokens
from ..types import Completion, CompletionRequest, Message, Role, StreamEvent, ToolCall, Usage

ScriptedResponse = str | dict[str, Any] | Completion | Sequence[ToolCall] | Exception
Handler = Callable[[CompletionRequest], Completion | str | dict[str, Any] | Sequence[ToolCall]]


class FakeLLM:
    provider: str
    supports_response_schema: bool = True

    def __init__(
        self,
        responses: Sequence[ScriptedResponse] | None = None,
        *,
        handler: Handler | None = None,
        provider: str = "fake",
        model: str = "fake-model",
        latency_ms: float = 1.0,
        chunk_size: int = 8,
    ) -> None:
        if responses is None and handler is None:
            responses = ["ok"]
        self._responses: list[ScriptedResponse] = list(responses or [])
        self._handler = handler
        self.provider = provider
        self.model = model
        self.default_model = model
        self.latency_ms = latency_ms
        self.chunk_size = chunk_size
        self.requests: list[CompletionRequest] = []

    # ----------------------------------------------------------------- scripting
    def _next(self, req: CompletionRequest) -> Completion:
        self.requests.append(req)
        if self._handler is not None:
            return self._coerce(self._handler(req), req)
        if not self._responses:
            raise RuntimeError("FakeLLM has no scripted responses left")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return self._coerce(item, req)

    def _coerce(self, item: Any, req: CompletionRequest) -> Completion:
        model = req.model or self.model
        if isinstance(item, Completion):
            return item.model_copy(update={"provider": item.provider or self.provider, "model": item.model or model})
        if isinstance(item, str):
            message = Message.assistant(item)
            finish = "stop"
        elif isinstance(item, dict):
            message = Message.assistant(json.dumps(item, ensure_ascii=False))
            finish = "stop"
        elif isinstance(item, (list, tuple)) and all(isinstance(x, ToolCall) for x in item):
            message = Message(role=Role.ASSISTANT, content="", tool_calls=list(item))
            finish = "tool_calls"
        else:
            raise TypeError(f"unsupported scripted response: {type(item)!r}")
        usage = Usage(
            input_tokens=count_message_tokens(req.messages),
            output_tokens=count_tokens(message.text)
            + sum(count_tokens(json.dumps(tc.arguments)) for tc in message.tool_calls or []),
        )
        return Completion(
            message=message,
            usage=usage,
            finish_reason=finish,
            model=model,
            provider=self.provider,
            latency_ms=self.latency_ms,
        )

    # ------------------------------------------------------------------ interface
    def complete(self, req: CompletionRequest) -> Completion:
        return self._next(req)

    def stream(self, req: CompletionRequest) -> Iterator[StreamEvent]:
        completion = self._next(req)
        text = completion.text
        for i in range(0, len(text), self.chunk_size):
            yield StreamEvent(type="text_delta", text=text[i : i + self.chunk_size])
        for tc in completion.tool_calls:
            yield StreamEvent(type="tool_call_delta", tool_call=tc)
        yield StreamEvent(type="usage", usage=completion.usage)
        yield StreamEvent(type="done", finish_reason=completion.finish_reason)

    async def acomplete(self, req: CompletionRequest) -> Completion:
        return self._next(req)

    async def astream(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        for ev in self.stream(req):
            yield ev

    # ----------------------------------------------------------------- test helpers
    @property
    def last_request(self) -> CompletionRequest:
        return self.requests[-1]

    def remaining(self) -> int:
        return len(self._responses)


__all__ = ["FakeLLM"]
