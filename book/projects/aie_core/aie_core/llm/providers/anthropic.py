# path: book/projects/aie_core/aie_core/llm/providers/anthropic.py
"""Adapter for the Anthropic Messages wire format.

Differences from the OpenAI dialect that this file absorbs so callers never see them:
system prompt is a top-level field, content is a list of typed blocks, tool results are
user-role blocks, structured output is emulated through a forced tool call, and the
stream is a typed event protocol rather than a stream of partial messages.
"""
from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator, Iterator

import httpx

from ..errors import ContentFilterError, MalformedResponseError
from ..types import (
    Completion,
    CompletionRequest,
    ContentPart,
    Message,
    Role,
    StreamEvent,
    ToolCall,
    Usage,
)
from ._http import (
    make_async_client,
    make_sync_client,
    map_transport_error,
    parse_json_arguments,
    parse_json_body,
    raise_for_status,
    timeout_for,
)
from ._sse import aiter_sse, iter_sse

STRUCTURED_TOOL_NAME = "emit_structured_output"

_STOP_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}


class AnthropicClient:
    provider: str = "anthropic"
    # Structured output is emulated, so the flag is still True: callers may pass a schema.
    supports_response_schema: bool = True

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str | None = None,
        *,
        base_url: str = "https://api.anthropic.com",
        api_version: str = "2023-06-01",
        timeout_s: float = 60.0,
        extra_headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.default_model = default_model
        self.timeout_s = timeout_s
        headers = {"Content-Type": "application/json", "anthropic-version": api_version}
        if api_key:
            headers["x-api-key"] = api_key
        headers.update(extra_headers or {})
        self._client = make_sync_client(base_url, headers, timeout_s, transport)
        self._aclient = make_async_client(base_url, headers, timeout_s, async_transport)

    # ------------------------------------------------------------------ request mapping
    def build_payload(self, req: CompletionRequest, *, stream: bool = False) -> dict[str, Any]:
        model = req.model or self.default_model
        if not model:
            raise ValueError("no model: set CompletionRequest.model or default_model")
        system_parts = [m.text for m in req.messages if m.role == Role.SYSTEM]
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "messages": self._encode_messages([m for m in req.messages if m.role != Role.SYSTEM]),
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if req.stop:
            payload["stop_sequences"] = req.stop
        tools = [
            {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in (req.tools or [])
        ]
        tool_choice: dict[str, Any] | None = None
        if req.response_schema is not None:
            tools.append(
                {
                    "name": STRUCTURED_TOOL_NAME,
                    "description": "Return the final answer as structured data matching the schema.",
                    "input_schema": req.response_schema,
                }
            )
            tool_choice = {"type": "tool", "name": STRUCTURED_TOOL_NAME}
        elif tools:
            tool_choice = self._encode_tool_choice(req.tool_choice)
            if tool_choice is None:  # "none": do not offer tools at all
                tools = []
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if stream:
            payload["stream"] = True
        return payload

    @staticmethod
    def _encode_tool_choice(choice: str) -> dict[str, Any] | None:
        if choice == "auto":
            return {"type": "auto"}
        if choice == "required":
            return {"type": "any"}
        if choice == "none":
            return None
        return {"type": "tool", "name": choice}

    @staticmethod
    def _encode_content(content: str | list[ContentPart]) -> list[dict[str, Any]]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}] if content else []
        blocks: list[dict[str, Any]] = []
        for p in content:
            if p.type == "text":
                blocks.append({"type": "text", "text": p.text or ""})
            else:
                blocks.append({"type": "image", "source": {"type": "url", "url": p.image_url or ""}})
        return blocks

    def _encode_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Anthropic requires strict user/assistant alternation and wants all tool results that
        answer one assistant turn inside a single user message, so adjacent same-role blocks merge."""
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == Role.TOOL:
                role = "user"
                blocks: list[dict[str, Any]] = [
                    {"type": "tool_result", "tool_use_id": m.tool_call_id or "", "content": m.text}
                ]
            elif m.role == Role.ASSISTANT:
                role = "assistant"
                blocks = self._encode_content(m.content)
                for tc in m.tool_calls or []:
                    blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
            else:
                role = "user"
                blocks = self._encode_content(m.content)
            if not blocks:
                continue
            if out and out[-1]["role"] == role:
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": role, "content": blocks})
        return out

    # ----------------------------------------------------------------- response mapping
    def parse_completion(self, data: dict[str, Any], latency_ms: float, *, structured: bool) -> Completion:
        stop_reason = data.get("stop_reason") or "end_turn"
        finish_reason = _STOP_REASONS.get(stop_reason, stop_reason)
        if finish_reason == "content_filter":
            raise ContentFilterError("completion refused by provider", provider=self.provider, raw=data)
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in data.get("content") or []:
            if block.get("type") == "text":
                text_parts.append(block.get("text") or "")
            elif block.get("type") == "tool_use":
                args = block.get("input")
                if not isinstance(args, dict):
                    raise MalformedResponseError("tool_use input is not an object", provider=self.provider, raw=data)
                if structured and block.get("name") == STRUCTURED_TOOL_NAME:
                    text_parts.append(json.dumps(args, ensure_ascii=False))
                    finish_reason = "stop"
                else:
                    tool_calls.append(ToolCall(id=block.get("id") or "", name=block.get("name") or "", arguments=args))
        message = Message(role=Role.ASSISTANT, content="".join(text_parts), tool_calls=tool_calls or None)
        return Completion(
            message=message,
            usage=self._parse_usage(data.get("usage")),
            finish_reason=finish_reason,
            model=data.get("model") or "",
            provider=self.provider,
            latency_ms=latency_ms,
            raw=data,
        )

    @staticmethod
    def _parse_usage(usage: dict[str, Any] | None) -> Usage:
        if not usage:
            return Usage()
        cached = int(usage.get("cache_read_input_tokens") or 0)
        created = int(usage.get("cache_creation_input_tokens") or 0)
        # Anthropic reports uncached, cache-read and cache-write tokens separately; we normalize
        # to "all input tokens" plus "how many of them were cache reads", like the OpenAI dialect.
        return Usage(
            input_tokens=int(usage.get("input_tokens") or 0) + cached + created,
            output_tokens=int(usage.get("output_tokens") or 0),
            cached_input_tokens=cached,
        )

    # -------------------------------------------------------------------- sync API
    def complete(self, req: CompletionRequest) -> Completion:
        payload = self.build_payload(req)
        started = time.perf_counter()
        try:
            resp = self._client.post("/v1/messages", json=payload, timeout=timeout_for(req.timeout_s, self.timeout_s))
        except httpx.HTTPError as exc:
            raise map_transport_error(exc, self.provider) from exc
        raise_for_status(resp, self.provider)
        latency_ms = (time.perf_counter() - started) * 1000
        return self.parse_completion(
            parse_json_body(resp, self.provider), latency_ms, structured=req.response_schema is not None
        )

    def stream(self, req: CompletionRequest) -> Iterator[StreamEvent]:
        payload = self.build_payload(req, stream=True)
        state = _AnthropicStreamState(self.provider, structured=req.response_schema is not None)
        try:
            with self._client.stream(
                "POST", "/v1/messages", json=payload, timeout=timeout_for(req.timeout_s, self.timeout_s)
            ) as resp:
                if not resp.is_success:
                    resp.read()
                raise_for_status(resp, self.provider)
                for sse in iter_sse(resp.iter_lines()):
                    yield from state.handle(sse.event, sse.data)
        except httpx.HTTPError as exc:
            raise map_transport_error(exc, self.provider) from exc
        yield from state.finish()

    # ------------------------------------------------------------------- async API
    async def acomplete(self, req: CompletionRequest) -> Completion:
        payload = self.build_payload(req)
        started = time.perf_counter()
        try:
            resp = await self._aclient.post(
                "/v1/messages", json=payload, timeout=timeout_for(req.timeout_s, self.timeout_s)
            )
        except httpx.HTTPError as exc:
            raise map_transport_error(exc, self.provider) from exc
        raise_for_status(resp, self.provider)
        latency_ms = (time.perf_counter() - started) * 1000
        return self.parse_completion(
            parse_json_body(resp, self.provider), latency_ms, structured=req.response_schema is not None
        )

    async def astream(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        payload = self.build_payload(req, stream=True)
        state = _AnthropicStreamState(self.provider, structured=req.response_schema is not None)
        try:
            async with self._aclient.stream(
                "POST", "/v1/messages", json=payload, timeout=timeout_for(req.timeout_s, self.timeout_s)
            ) as resp:
                if not resp.is_success:
                    await resp.aread()
                raise_for_status(resp, self.provider)
                async for sse in aiter_sse(resp.aiter_lines()):
                    for ev in state.handle(sse.event, sse.data):
                        yield ev
        except httpx.HTTPError as exc:
            raise map_transport_error(exc, self.provider) from exc
        for ev in state.finish():
            yield ev

    def close(self) -> None:
        self._client.close()

    async def aclose(self) -> None:
        await self._aclient.aclose()


class _AnthropicStreamState:
    """Typed event protocol: content blocks open, receive deltas, and close by index."""

    def __init__(self, provider: str, *, structured: bool) -> None:
        self.provider = provider
        self.structured = structured
        self.blocks: dict[int, dict[str, Any]] = {}
        self.input_tokens = 0
        self.cached_tokens = 0
        self.output_tokens = 0
        self.finish_reason: str | None = None
        self.done = False

    def handle(self, event: str | None, data: str) -> Iterator[StreamEvent]:
        if not data:
            return
        try:
            payload = json.loads(data)
        except ValueError as exc:
            raise MalformedResponseError(f"bad SSE chunk: {exc}", provider=self.provider, raw=data) from exc
        kind = event or payload.get("type")
        if kind == "message_start":
            usage = (payload.get("message") or {}).get("usage") or {}
            self.cached_tokens = int(usage.get("cache_read_input_tokens") or 0)
            self.input_tokens = int(usage.get("input_tokens") or 0) + self.cached_tokens
        elif kind == "content_block_start":
            idx = int(payload.get("index", 0))
            block = payload.get("content_block") or {}
            self.blocks[idx] = {"type": block.get("type"), "id": block.get("id"), "name": block.get("name"), "json": ""}
        elif kind == "content_block_delta":
            idx = int(payload.get("index", 0))
            delta = payload.get("delta") or {}
            if delta.get("type") == "text_delta" and delta.get("text"):
                yield StreamEvent(type="text_delta", text=delta["text"])
            elif delta.get("type") == "input_json_delta":
                self.blocks.setdefault(idx, {"type": "tool_use", "id": None, "name": None, "json": ""})
                self.blocks[idx]["json"] += delta.get("partial_json") or ""
        elif kind == "content_block_stop":
            idx = int(payload.get("index", 0))
            block = self.blocks.pop(idx, None)
            if block and block["type"] == "tool_use":
                name = block["name"] or ""
                args = parse_json_arguments(block["json"], self.provider, name)
                if self.structured and name == STRUCTURED_TOOL_NAME:
                    yield StreamEvent(type="text_delta", text=json.dumps(args, ensure_ascii=False))
                else:
                    yield StreamEvent(type="tool_call_delta", tool_call=ToolCall(id=block["id"] or "", name=name, arguments=args))
        elif kind == "message_delta":
            delta = payload.get("delta") or {}
            if delta.get("stop_reason"):
                self.finish_reason = _STOP_REASONS.get(delta["stop_reason"], delta["stop_reason"])
                if self.finish_reason == "content_filter":
                    raise ContentFilterError("stream refused by provider", provider=self.provider, raw=payload)
            usage = payload.get("usage") or {}
            if usage.get("output_tokens") is not None:
                self.output_tokens = int(usage["output_tokens"])
        elif kind == "error":
            err = payload.get("error") or {}
            yield StreamEvent(type="error", error=str(err.get("message") or err))
        # ping / message_stop need no action

    def finish(self) -> Iterator[StreamEvent]:
        if self.done:
            return
        self.done = True
        yield StreamEvent(
            type="usage",
            usage=Usage(input_tokens=self.input_tokens, output_tokens=self.output_tokens, cached_input_tokens=self.cached_tokens),
        )
        finish = self.finish_reason or "stop"
        if self.structured and finish == "tool_calls":
            finish = "stop"
        yield StreamEvent(type="done", finish_reason=finish)


__all__ = ["AnthropicClient", "STRUCTURED_TOOL_NAME"]
