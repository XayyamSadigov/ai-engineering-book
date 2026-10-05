# path: book/capstone/northwind-assist/northwind_assist/domain/events.py
"""The typed event vocabulary of a chat turn (Chapter 28's SSE contract, extended).

meta      versions, route, model and degrade level, sent before any expensive work
citation  one resolved evidence source, sent before the first sentence that cites it
delta     validated answer text (whole sentences, never raw tokens)
tool      one governed tool call and its outcome
approval  a side effect waiting for a human, with the exact arguments it will run with
memory    a profile fact proposed for the user's confirmation
notice    degraded mode, withheld sentences, guardrail redactions
structured a schema-valid extraction result
done      final status, answer, citations, usage, cost and lineage
error     stage and message; the stream ends after it
"""
from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

EventName = Literal["meta", "citation", "delta", "tool", "approval", "memory", "notice", "structured", "done",
                    "error"]
EVENT_NAMES: tuple[str, ...] = ("meta", "citation", "delta", "tool", "approval", "memory", "notice", "structured",
                                "done", "error")


class ServerEvent(BaseModel):
    event: EventName
    data: dict[str, Any] = Field(default_factory=dict)
    id: int = 0

    def sse(self) -> str:
        payload = json.dumps(self.data, ensure_ascii=False, default=str)
        return f"id: {self.id}\nevent: {self.event}\ndata: {payload}\n\n"


__all__ = ["ServerEvent", "EventName", "EVENT_NAMES"]
