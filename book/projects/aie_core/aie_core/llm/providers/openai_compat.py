# path: book/projects/aie_core/aie_core/llm/providers/openai_compat.py
"""Adapter for the OpenAI Chat Completions wire format.

Many servers speak this dialect (hosted OpenAI, vLLM, Ollama, LM Studio, various
gateways), which is why the class is named for the *format*, not the vendor.
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


class OpenAICompatibleClient:
    provider: str
    supports_response_schema: bool = True
    # Some servers expect "max_completion_tokens" instead; override per instance when needed.
    max_tokens_field: str = "max_tokens"

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        default_model: str | None = None,
        *,
        provider: str = "openai",
        timeout_s: float = 60.0,
        extra_headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
        strict_schemas: bool = False,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.provider = provider
        # Server-specific request fields the neutral request has no slot for, for example
        # {"logprobs": True, "top_logprobs": 3} on engines that return token log-probabilities.
        # They never override a field the adapter sets itself.
        self.extra_body = dict(extra_body or {})
        self.default_model = default_model
        self.timeout_s = timeout_s
        self.strict_schemas = strict_schemas
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        headers.update(extra_headers or {})
        self._client = make_sync_client(base_url, headers, timeout_s, transport)
        self._aclient = make_async_client(base_url, headers, timeout_s, async_transport)

    # ------------------------------------------------------------------ request mapping
    def build_payload(self, req: CompletionRequest, *, stream: bool = False) -> dict[str, Any]:
        model = req.model or self.default_model
        if not model:
            raise ValueError("no model: set CompletionRequest.model or default_model")
        payload: dict[str, Any] = {
            "model": model,
            "messages": [self._encode_message(m) for m in req.messages],
            "temperature": req.temperature,
            self.max_tokens_field: req.max_tokens,
        }
        if req.stop:
            payload["stop"] = req.stop
        if req.tools:
            payload["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in req.tools
            ]
            payload["tool_choice"] = self._encode_tool_choice(req.tool_choice)
        if req.response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": str(req.response_schema.get("title", "response")),
                    "schema": req.response_schema,
                    "strict": self.strict_schemas,
                },
            }
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        for key, value in self.extra_body.items():
            payload.setdefault(key, value)
        return payload

    @staticmethod
    def _encode_tool_choice(choice: str) -> Any:
        if choice in ("auto", "none", "required"):
            return choice
        return {"type": "function", "function": {"name": choice}}

    @staticmethod
    def _encode_content(content: str | list[ContentPart]) -> Any:
        if isinstance(content, str):
            return content
        parts: list[dict[str, Any]] = []
        for p in content:
            if p.type == "text":
                parts.append({"type": "text", "text": p.text or ""})
            else:
                parts.append({"type": "image_url", "image_url": {"url": p.image_url or ""}})
        return parts

    def _encode_message(self, m: Message) -> dict[str, Any]:
        if m.role == Role.TOOL:
            return {"role": "tool", "tool_call_id": m.tool_call_id or "", "content": m.text}
        out: dict[str, Any] = {"role": m.role.value, "content": self._encode_content(m.content)}
        if m.name:
            out["name"] = m.name
        if m.role == Role.ASSISTANT and m.tool_calls:
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                }
                for tc in m.tool_calls
            ]
            if out["content"] == "":
                out["content"] = None
        return out

    # ----------------------------------------------------------------- response mapping
    def parse_completion(self, data: dict[str, Any], latency_ms: float) -> Completion:
        choices = data.get("choices") or []
        if not choices:
            raise MalformedResponseError("response has no choices", provider=self.provider, raw=data)
        choice = choices[0]
        finish_reason = choice.get("finish_reason") or "stop"
        if finish_reason == "content_filter":
            raise ContentFilterError("completion blocked by content filter", provider=self.provider, raw=data)
        msg = choice.get("message") or {}
        tool_calls = [
            ToolCall(
                id=tc.get("id") or f"call_{i}",
                name=tc["function"]["name"],
                arguments=parse_json_arguments(tc["function"].get("arguments"), self.provider, tc["function"]["name"]),
            )
            for i, tc in enumerate(msg.get("tool_calls") or [])
        ]
        message = Message(
            role=Role.ASSISTANT,
            content=msg.get("content") or "",
            tool_calls=tool_calls or None,
        )
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
        details = usage.get("prompt_tokens_details") or {}
        return Usage(
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            cached_input_tokens=int(details.get("cached_tokens") or 0),
        )

    # -------------------------------------------------------------------- sync API
    def complete(self, req: CompletionRequest) -> Completion:
        payload = self.build_payload(req)
        started = time.perf_counter()
        try:
            resp = self._client.post("/chat/completions", json=payload, timeout=timeout_for(req.timeout_s, self.timeout_s))
        except httpx.HTTPError as exc:
            raise map_transport_error(exc, self.provider) from exc
        raise_for_status(resp, self.provider)
        latency_ms = (time.perf_counter() - started) * 1000
        return self.parse_completion(parse_json_body(resp, self.provider), latency_ms)

    def stream(self, req: CompletionRequest) -> Iterator[StreamEvent]:
        payload = self.build_payload(req, stream=True)
        state = _OpenAIStreamState(self.provider)
        try:
            with self._client.stream(
                "POST", "/chat/completions", json=payload, timeout=timeout_for(req.timeout_s, self.timeout_s)
            ) as resp:
                if not resp.is_success:
                    resp.read()
                raise_for_status(resp, self.provider)
                for sse in iter_sse(resp.iter_lines()):
                    yield from state.handle(sse.data)
        except httpx.HTTPError as exc:
            raise map_transport_error(exc, self.provider) from exc
        yield from state.finish()

    # ------------------------------------------------------------------- async API
    async def acomplete(self, req: CompletionRequest) -> Completion:
        payload = self.build_payload(req)
        started = time.perf_counter()
        try:
            resp = await self._aclient.post(
                "/chat/completions", json=payload, timeout=timeout_for(req.timeout_s, self.timeout_s)
            )
        except httpx.HTTPError as exc:
            raise map_transport_error(exc, self.provider) from exc
        raise_for_status(resp, self.provider)
        latency_ms = (time.perf_counter() - started) * 1000
        return self.parse_completion(parse_json_body(resp, self.provider), latency_ms)

    async def astream(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        payload = self.build_payload(req, stream=True)
        state = _OpenAIStreamState(self.provider)
        try:
            async with self._aclient.stream(
                "POST", "/chat/completions", json=payload, timeout=timeout_for(req.timeout_s, self.timeout_s)
            ) as resp:
                if not resp.is_success:
                    await resp.aread()
                raise_for_status(resp, self.provider)
                async for sse in aiter_sse(resp.aiter_lines()):
                    for ev in state.handle(sse.data):
                        yield ev
        except httpx.HTTPError as exc:
            raise map_transport_error(exc, self.provider) from exc
        for ev in state.finish():
            yield ev

    def close(self) -> None:
        self._client.close()

    async def aclose(self) -> None:
        await self._aclient.aclose()


class _OpenAIStreamState:
    """Accumulates chunked deltas. Tool-call argument fragments are buffered per index and
    emitted as one complete ToolCall when the stream closes them."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.pending: dict[int, dict[str, Any]] = {}
        self.finish_reason: str | None = None
        self.usage: Usage | None = None
        self.done = False

    def handle(self, data: str) -> Iterator[StreamEvent]:
        if not data or data.strip() == "[DONE]":
            return
        try:
            chunk = json.loads(data)
        except ValueError as exc:
            raise MalformedResponseError(f"bad SSE chunk: {exc}", provider=self.provider, raw=data) from exc
        if "error" in chunk and not chunk.get("choices"):
            msg = str((chunk.get("error") or {}).get("message") or chunk["error"])
            yield StreamEvent(type="error", error=msg)
            return
        if chunk.get("usage"):
            self.usage = OpenAICompatibleClient._parse_usage(chunk["usage"])
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                yield StreamEvent(type="text_delta", text=delta["content"])
            for tc in delta.get("tool_calls") or []:
                idx = int(tc.get("index", 0))
                slot = self.pending.setdefault(idx, {"id": None, "name": None, "args": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["args"] += fn["arguments"]
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
                if self.finish_reason == "content_filter":
                    raise ContentFilterError("stream blocked by content filter", provider=self.provider, raw=chunk)
                yield from self._flush_tool_calls()

    def _flush_tool_calls(self) -> Iterator[StreamEvent]:
        for idx in sorted(self.pending):
            slot = self.pending[idx]
            name = slot["name"] or ""
            yield StreamEvent(
                type="tool_call_delta",
                tool_call=ToolCall(
                    id=slot["id"] or f"call_{idx}",
                    name=name,
                    arguments=parse_json_arguments(slot["args"], self.provider, name),
                ),
            )
        self.pending.clear()

    def finish(self) -> Iterator[StreamEvent]:
        if self.done:
            return
        self.done = True
        yield from self._flush_tool_calls()
        if self.usage is not None:
            yield StreamEvent(type="usage", usage=self.usage)
        yield StreamEvent(type="done", finish_reason=self.finish_reason or "stop")


__all__ = ["OpenAICompatibleClient"]
