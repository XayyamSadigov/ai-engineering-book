# path: book/projects/aie_core/aie_core/llm/client.py
"""The one interface the rest of the book programs against."""
from __future__ import annotations

from typing import AsyncIterator, Iterable, Iterator, Protocol, runtime_checkable

from .types import Completion, CompletionRequest, Message, Role, StreamEvent, ToolCall, Usage


@runtime_checkable
class LLMClient(Protocol):
    provider: str

    def complete(self, req: CompletionRequest) -> Completion: ...

    def stream(self, req: CompletionRequest) -> Iterator[StreamEvent]: ...

    async def acomplete(self, req: CompletionRequest) -> Completion: ...

    def astream(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]: ...


def collect_stream(
    events: Iterable[StreamEvent],
    *,
    model: str = "",
    provider: str = "",
    latency_ms: float = 0.0,
) -> Completion:
    """Fold a finished stream back into a Completion (used by tests and by callers that
    stream for UX but still want the usual object at the end)."""
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    usage = Usage()
    finish_reason = "stop"
    for ev in events:
        if ev.type == "text_delta" and ev.text:
            text_parts.append(ev.text)
        elif ev.type == "tool_call_delta" and ev.tool_call is not None:
            tool_calls.append(ev.tool_call)
        elif ev.type == "usage" and ev.usage is not None:
            usage = ev.usage
        elif ev.type == "done":
            finish_reason = ev.finish_reason or ("tool_calls" if tool_calls else "stop")
        elif ev.type == "error":
            raise RuntimeError(ev.error or "stream error")
    message = Message(
        role=Role.ASSISTANT,
        content="".join(text_parts),
        tool_calls=tool_calls or None,
    )
    return Completion(
        message=message,
        usage=usage,
        finish_reason=finish_reason,
        model=model,
        provider=provider,
        latency_ms=latency_ms,
    )


__all__ = ["LLMClient", "collect_stream"]
