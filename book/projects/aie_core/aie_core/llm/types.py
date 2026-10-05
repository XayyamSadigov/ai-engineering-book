# path: book/projects/aie_core/aie_core/llm/types.py
"""Provider-neutral request and response types.

Every provider adapter translates *to* these types on the way in and *from* them on
the way out, so application code never sees a vendor payload.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ContentPart(BaseModel):
    """One piece of multimodal content. Text-only callers never build these by hand."""

    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: str | None = None


class ToolCall(BaseModel):
    """A request from the model to run a tool. `arguments` is already parsed JSON."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    role: Role
    content: str | list[ContentPart] = ""
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    @classmethod
    def system(cls, text: str) -> "Message":
        return cls(role=Role.SYSTEM, content=text)

    @classmethod
    def user(cls, text: str) -> "Message":
        return cls(role=Role.USER, content=text)

    @classmethod
    def assistant(cls, text: str) -> "Message":
        return cls(role=Role.ASSISTANT, content=text)

    @classmethod
    def tool(cls, tool_call_id: str, content: str) -> "Message":
        return cls(role=Role.TOOL, content=content, tool_call_id=tool_call_id)

    @property
    def text(self) -> str:
        """Content flattened to a string (image parts are dropped)."""
        if isinstance(self.content, str):
            return self.content
        return "".join(p.text or "" for p in self.content if p.type == "text")


class ToolSpec(BaseModel):
    """What the model is told about a tool. `parameters` is a JSON Schema object."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
        )


class CompletionRequest(BaseModel):
    messages: list[Message]
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1024
    tools: list[ToolSpec] | None = None
    tool_choice: Literal["auto", "none", "required"] | str = "auto"
    response_schema: dict[str, Any] | None = None  # JSON Schema for structured output
    stop: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timeout_s: float | None = None


class Completion(BaseModel):
    message: Message
    usage: Usage = Field(default_factory=Usage)
    finish_reason: str = "stop"
    model: str = ""
    provider: str = ""
    latency_ms: float = 0.0
    raw: dict[str, Any] | None = None

    @property
    def text(self) -> str:
        return self.message.text

    @property
    def tool_calls(self) -> list[ToolCall]:
        return list(self.message.tool_calls or [])

    @property
    def truncated(self) -> bool:
        """True when generation stopped because `max_tokens` ran out (`finish_reason == "length"`).
        The text is cut mid-way; structured output parsed from it is unreliable even if it happens
        to be valid JSON."""
        return self.finish_reason == "length"


class StreamEvent(BaseModel):
    """One incremental event from a streaming completion.

    `text_delta` carries a text fragment. `tool_call_delta` carries one *complete* tool
    call: argument JSON arrives in fragments that cannot be validated until the call is
    closed, so adapters buffer and emit the call once. `usage` arrives at the end when the
    provider reports it. `done` is always the last event of a successful stream.
    """

    type: Literal["text_delta", "tool_call_delta", "usage", "done", "error"]
    text: str | None = None
    tool_call: ToolCall | None = None
    usage: Usage | None = None
    error: str | None = None
    finish_reason: str | None = None


__all__ = [
    "Role",
    "ContentPart",
    "ToolCall",
    "Message",
    "ToolSpec",
    "Usage",
    "CompletionRequest",
    "Completion",
    "StreamEvent",
]
