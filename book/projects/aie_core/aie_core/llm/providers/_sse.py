# path: book/projects/aie_core/aie_core/llm/providers/_sse.py
"""Minimal Server-Sent Events parser.

SSE is line-oriented: fields `event:` and `data:` accumulate until a blank line, which
dispatches one event. Multi-line `data:` fields are joined with newlines. Lines starting
with `:` are comments (providers use them as keep-alives).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Iterable, Iterator


@dataclass
class SSEEvent:
    event: str | None = None
    data: str = ""
    id: str | None = None
    _data_lines: list[str] = field(default_factory=list, repr=False)

    def finish(self) -> "SSEEvent":
        self.data = "\n".join(self._data_lines)
        return self


class _Accumulator:
    def __init__(self) -> None:
        self.current = SSEEvent()
        self.has_content = False

    def feed(self, line: str) -> SSEEvent | None:
        line = line.rstrip("\r")
        if line == "":
            if not self.has_content:
                return None
            ev = self.current.finish()
            self.current = SSEEvent()
            self.has_content = False
            return ev
        if line.startswith(":"):
            return None
        name, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        self.has_content = True
        if name == "event":
            self.current.event = value
        elif name == "data":
            self.current._data_lines.append(value)
        elif name == "id":
            self.current.id = value
        return None

    def flush(self) -> SSEEvent | None:
        if self.has_content:
            ev = self.current.finish()
            self.current = SSEEvent()
            self.has_content = False
            return ev
        return None


def iter_sse(lines: Iterable[str]) -> Iterator[SSEEvent]:
    acc = _Accumulator()
    for line in lines:
        ev = acc.feed(line)
        if ev is not None:
            yield ev
    tail = acc.flush()
    if tail is not None:
        yield tail


async def aiter_sse(lines: AsyncIterator[str]) -> AsyncIterator[SSEEvent]:
    acc = _Accumulator()
    async for line in lines:
        ev = acc.feed(line)
        if ev is not None:
            yield ev
    tail = acc.flush()
    if tail is not None:
        yield tail


__all__ = ["SSEEvent", "iter_sse", "aiter_sse"]
