# path: book/projects/aie_core/tests/test_anthropic_provider.py
import json

import httpx
import pytest

from aie_core.llm.errors import ContentFilterError, ProviderUnavailableError, RateLimitError
from aie_core.llm.providers import STRUCTURED_TOOL_NAME, AnthropicClient
from aie_core.llm.types import CompletionRequest, Message, Role, ToolCall, ToolSpec

from ._helpers import RecordingTransport, sse_body

TOOL = ToolSpec(name="search_tickets", description="Search", parameters={"type": "object", "properties": {"q": {"type": "string"}}})


def make_client(handler):
    transport = RecordingTransport(handler)
    return AnthropicClient(api_key="ak", default_model="claude-x", base_url="http://test", transport=transport), transport


def ok(content, stop_reason="end_turn", usage=None):
    return httpx.Response(
        200,
        json={
            "id": "msg", "model": "claude-x-2026", "type": "message", "role": "assistant",
            "content": content, "stop_reason": stop_reason,
            "usage": usage or {"input_tokens": 20, "output_tokens": 5, "cache_read_input_tokens": 12},
        },
    )


def test_request_mapping_system_tools_and_tool_results():
    client, transport = make_client(lambda r: ok([{"type": "text", "text": "done"}]))
    req = CompletionRequest(
        messages=[
            Message.system("A"),
            Message.system("B"),
            Message.user("find tickets"),
            Message(role=Role.ASSISTANT, content="Looking.", tool_calls=[
                ToolCall(id="tu_1", name="search_tickets", arguments={"q": "vpn"}),
                ToolCall(id="tu_2", name="search_tickets", arguments={"q": "sso"}),
            ]),
            Message.tool("tu_1", "[1]"),
            Message.tool("tu_2", "[2]"),
        ],
        tools=[TOOL],
        tool_choice="required",
        stop=["END"],
    )
    client.complete(req)
    sent = transport.last_json()
    assert transport.requests[-1].headers["x-api-key"] == "ak"
    assert transport.requests[-1].headers["anthropic-version"]
    assert transport.requests[-1].url.path == "/v1/messages"
    assert sent["system"] == "A\n\nB"
    assert sent["stop_sequences"] == ["END"] and sent["max_tokens"] == 1024
    assert sent["tools"] == [{"name": "search_tickets", "description": "Search", "input_schema": TOOL.parameters}]
    assert sent["tool_choice"] == {"type": "any"}
    roles = [m["role"] for m in sent["messages"]]
    assert roles == ["user", "assistant", "user"], "tool results must merge into one user turn"
    assistant = sent["messages"][1]["content"]
    assert assistant[0] == {"type": "text", "text": "Looking."}
    assert assistant[1] == {"type": "tool_use", "id": "tu_1", "name": "search_tickets", "input": {"q": "vpn"}}
    results = sent["messages"][2]["content"]
    assert [b["tool_use_id"] for b in results] == ["tu_1", "tu_2"] and results[0]["type"] == "tool_result"


def test_tool_choice_none_drops_tools():
    client, transport = make_client(lambda r: ok([{"type": "text", "text": "x"}]))
    client.complete(CompletionRequest(messages=[Message.user("hi")], tools=[TOOL], tool_choice="none"))
    assert "tools" not in transport.last_json() and "tool_choice" not in transport.last_json()


def test_response_mapping_text_tool_use_usage():
    content = [{"type": "text", "text": "Let me check."}, {"type": "tool_use", "id": "tu_9", "name": "search_tickets", "input": {"q": "vpn"}}]
    client, _ = make_client(lambda r: ok(content, stop_reason="tool_use"))
    c = client.complete(CompletionRequest(messages=[Message.user("hi")], tools=[TOOL]))
    assert c.text == "Let me check." and c.finish_reason == "tool_calls"
    assert c.tool_calls == [ToolCall(id="tu_9", name="search_tickets", arguments={"q": "vpn"})]
    assert c.usage.input_tokens == 32 and c.usage.cached_input_tokens == 12 and c.usage.output_tokens == 5
    assert c.provider == "anthropic" and c.model == "claude-x-2026"


def test_structured_output_is_emulated_with_forced_tool():
    schema = {"type": "object", "properties": {"category": {"type": "string"}}, "required": ["category"]}

    def handler(r):
        sent = json.loads(r.content)
        assert sent["tool_choice"] == {"type": "tool", "name": STRUCTURED_TOOL_NAME}
        assert sent["tools"][-1]["input_schema"] == schema
        return ok([{"type": "tool_use", "id": "tu_s", "name": STRUCTURED_TOOL_NAME, "input": {"category": "billing"}}], stop_reason="tool_use")

    client, _ = make_client(handler)
    c = client.complete(CompletionRequest(messages=[Message.user("classify")], response_schema=schema))
    assert json.loads(c.text) == {"category": "billing"}
    assert c.tool_calls == [] and c.finish_reason == "stop"


def test_max_tokens_stop_reason_maps_to_length():
    client, _ = make_client(lambda r: ok([{"type": "text", "text": "trunc"}], stop_reason="max_tokens"))
    assert client.complete(CompletionRequest(messages=[Message.user("hi")])).finish_reason == "length"


def test_refusal_maps_to_content_filter():
    client, _ = make_client(lambda r: ok([], stop_reason="refusal"))
    with pytest.raises(ContentFilterError):
        client.complete(CompletionRequest(messages=[Message.user("hi")]))


def test_error_mapping_overloaded_and_rate_limit():
    client, _ = make_client(lambda r: httpx.Response(529, json={"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}))
    with pytest.raises(ProviderUnavailableError) as info:
        client.complete(CompletionRequest(messages=[Message.user("hi")]))
    assert "overloaded" in str(info.value).lower()
    client, _ = make_client(lambda r: httpx.Response(429, json={"error": {"message": "rl"}}, headers={"retry-after": "7"}))
    with pytest.raises(RateLimitError) as info:
        client.complete(CompletionRequest(messages=[Message.user("hi")]))
    assert info.value.retry_after_s == 7.0


def test_streaming_text_and_tool_use_events():
    events = [
        ("message_start", {"type": "message_start", "message": {"usage": {"input_tokens": 11, "cache_read_input_tokens": 4}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi "}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "there"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("content_block_start", {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "tu_7", "name": "search_tickets", "input": {}}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"q": "v'}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": 'pn"}'}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    client, transport = make_client(lambda r: httpx.Response(200, content=sse_body(events)))
    out = list(client.stream(CompletionRequest(messages=[Message.user("hi")], tools=[TOOL])))
    assert transport.last_json()["stream"] is True
    assert [e.type for e in out] == ["text_delta", "text_delta", "tool_call_delta", "usage", "done"]
    assert out[2].tool_call == ToolCall(id="tu_7", name="search_tickets", arguments={"q": "vpn"})
    assert out[3].usage.input_tokens == 15 and out[3].usage.cached_input_tokens == 4 and out[3].usage.output_tokens == 9
    assert out[4].finish_reason == "tool_calls"


def test_streaming_structured_output_becomes_text():
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
    events = [
        ("message_start", {"type": "message_start", "message": {"usage": {"input_tokens": 1}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "t", "name": STRUCTURED_TOOL_NAME, "input": {}}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"a": 1}'}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 2}}),
    ]
    client, _ = make_client(lambda r: httpx.Response(200, content=sse_body(events)))
    out = list(client.stream(CompletionRequest(messages=[Message.user("hi")], response_schema=schema)))
    assert out[0].type == "text_delta" and json.loads(out[0].text) == {"a": 1}
    assert out[-1].finish_reason == "stop"


async def test_async_complete():
    transport = httpx.MockTransport(lambda r: ok([{"type": "text", "text": "async"}]))
    client = AnthropicClient(api_key="k", default_model="m", base_url="http://test", async_transport=transport)
    c = await client.acomplete(CompletionRequest(messages=[Message.user("hi")]))
    assert c.text == "async"
