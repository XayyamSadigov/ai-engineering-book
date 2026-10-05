# path: book/projects/aie_core/tests/test_openai_provider.py
import json

import httpx
import pytest

from aie_core.llm.client import collect_stream
from aie_core.llm.errors import (
    ContentFilterError,
    InvalidRequestError,
    MalformedResponseError,
    ProviderUnavailableError,
    RateLimitError,
    TimeoutError,
)
from aie_core.llm.providers import OpenAICompatibleClient
from aie_core.llm.types import CompletionRequest, ContentPart, Message, Role, ToolCall, ToolSpec

from ._helpers import RecordingTransport, sse_body

TOOL = ToolSpec(
    name="lookup_employee",
    description="Find an employee by id",
    parameters={"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]},
)


def make_client(handler, **kwargs) -> tuple[OpenAICompatibleClient, RecordingTransport]:
    transport = RecordingTransport(handler)
    client = OpenAICompatibleClient(
        base_url="http://test/v1", api_key="sk-test", default_model="m1", transport=transport, **kwargs
    )
    return client, transport


def ok_response(content="Hello", tool_calls=None, finish="stop", usage=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    body = {
        "id": "x",
        "model": "m1-2026",
        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": 6}},
    }
    return httpx.Response(200, json=body)


def test_request_mapping_messages_tools_schema():
    client, transport = make_client(lambda r: ok_response())
    req = CompletionRequest(
        messages=[
            Message.system("sys"),
            Message(role=Role.USER, content=[ContentPart(type="text", text="look"), ContentPart(type="image_url", image_url="http://img")]),
            Message(role=Role.ASSISTANT, content="", tool_calls=[ToolCall(id="c1", name="lookup_employee", arguments={"id": 7})]),
            Message.tool("c1", '{"name": "Ada"}'),
        ],
        tools=[TOOL],
        tool_choice="lookup_employee",
        response_schema={"title": "Answer", "type": "object", "properties": {"x": {"type": "string"}}},
        stop=["END"],
        max_tokens=77,
    )
    client.complete(req)
    sent = transport.last_json()
    assert transport.requests[-1].headers["authorization"] == "Bearer sk-test"
    assert transport.requests[-1].url.path == "/v1/chat/completions"
    assert sent["model"] == "m1" and sent["max_tokens"] == 77 and sent["stop"] == ["END"]
    assert sent["messages"][0] == {"role": "system", "content": "sys"}
    assert sent["messages"][1]["content"][1] == {"type": "image_url", "image_url": {"url": "http://img"}}
    assistant = sent["messages"][2]
    assert assistant["content"] is None
    assert assistant["tool_calls"][0]["function"] == {"name": "lookup_employee", "arguments": '{"id": 7}'}
    assert sent["messages"][3] == {"role": "tool", "tool_call_id": "c1", "content": '{"name": "Ada"}'}
    assert sent["tools"][0]["function"]["name"] == "lookup_employee"
    assert sent["tool_choice"] == {"type": "function", "function": {"name": "lookup_employee"}}
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["response_format"]["json_schema"]["name"] == "Answer"
    assert "stream" not in sent


def test_response_mapping_text_and_usage():
    client, _ = make_client(lambda r: ok_response())
    c = client.complete(CompletionRequest(messages=[Message.user("hi")]))
    assert c.text == "Hello" and c.finish_reason == "stop"
    assert c.model == "m1-2026" and c.provider == "openai"
    assert (c.usage.input_tokens, c.usage.output_tokens, c.usage.cached_input_tokens) == (10, 4, 6)
    assert c.latency_ms >= 0 and c.raw["id"] == "x"


def test_response_mapping_tool_calls_parses_arguments():
    tc = [{"id": "call_a", "type": "function", "function": {"name": "lookup_employee", "arguments": '{"id": 42}'}}]
    client, _ = make_client(lambda r: ok_response(content=None, tool_calls=tc, finish="tool_calls"))
    c = client.complete(CompletionRequest(messages=[Message.user("hi")], tools=[TOOL]))
    assert c.finish_reason == "tool_calls"
    assert c.tool_calls == [ToolCall(id="call_a", name="lookup_employee", arguments={"id": 42})]
    assert c.text == ""


def test_malformed_tool_arguments_raise():
    tc = [{"id": "call_a", "type": "function", "function": {"name": "f", "arguments": "{not json"}}]
    client, _ = make_client(lambda r: ok_response(content=None, tool_calls=tc, finish="tool_calls"))
    with pytest.raises(MalformedResponseError):
        client.complete(CompletionRequest(messages=[Message.user("hi")]))


def test_missing_model_is_a_programming_error():
    client = OpenAICompatibleClient(base_url="http://test/v1", transport=RecordingTransport(lambda r: ok_response()))
    with pytest.raises(ValueError):
        client.complete(CompletionRequest(messages=[Message.user("hi")]))


@pytest.mark.parametrize(
    "status,body,headers,expected,retryable",
    [
        (429, {"error": {"message": "slow down", "type": "rate_limit"}}, {"retry-after": "2.5"}, RateLimitError, True),
        (500, {"error": {"message": "boom"}}, {}, ProviderUnavailableError, True),
        (503, "upstream down", {}, ProviderUnavailableError, True),
        (400, {"error": {"message": "bad schema", "code": "invalid_request"}}, {}, InvalidRequestError, False),
        (401, {"error": {"message": "bad key"}}, {}, InvalidRequestError, False),
        (400, {"error": {"message": "blocked", "code": "content_filter"}}, {}, ContentFilterError, False),
        (408, {"error": {"message": "slow"}}, {}, TimeoutError, True),
    ],
)
def test_http_error_mapping(status, body, headers, expected, retryable):
    def handler(r):
        return httpx.Response(status, json=body, headers=headers) if isinstance(body, dict) else httpx.Response(status, text=body, headers=headers)

    client, _ = make_client(handler)
    with pytest.raises(expected) as info:
        client.complete(CompletionRequest(messages=[Message.user("hi")]))
    err = info.value
    assert err.retryable is retryable and err.status_code == status and err.provider == "openai"
    if headers.get("retry-after"):
        assert err.retry_after_s == 2.5


def test_timeout_and_connect_errors_map_to_taxonomy():
    def timeout(r):
        raise httpx.ReadTimeout("slow", request=r)

    client, _ = make_client(timeout)
    with pytest.raises(TimeoutError) as info:
        client.complete(CompletionRequest(messages=[Message.user("hi")]))
    assert info.value.retryable

    def connect(r):
        raise httpx.ConnectError("refused", request=r)

    client, _ = make_client(connect)
    with pytest.raises(ProviderUnavailableError):
        client.complete(CompletionRequest(messages=[Message.user("hi")]))


def test_content_filter_finish_reason_raises():
    client, _ = make_client(lambda r: ok_response(content="", finish="content_filter"))
    with pytest.raises(ContentFilterError):
        client.complete(CompletionRequest(messages=[Message.user("hi")]))


def test_non_json_body_is_malformed():
    client, _ = make_client(lambda r: httpx.Response(200, text="<html>oops</html>"))
    with pytest.raises(MalformedResponseError):
        client.complete(CompletionRequest(messages=[Message.user("hi")]))


def test_request_timeout_uses_request_budget():
    client, transport = make_client(lambda r: ok_response())
    client.complete(CompletionRequest(messages=[Message.user("hi")], timeout_s=3.5))
    # httpx stores the per-request timeout on the request extensions
    assert transport.requests[-1].extensions["timeout"]["read"] == 3.5


def chunk(delta, finish=None, usage=None):
    body = {"id": "x", "model": "m1", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage:
        body["usage"] = usage
    return body


def test_streaming_text_deltas_and_usage():
    events = [
        (None, chunk({"role": "assistant", "content": ""})),
        (None, chunk({"content": "Hel"})),
        (None, chunk({"content": "lo"})),
        (None, chunk({}, finish="stop")),
        (None, {"id": "x", "model": "m1", "choices": [], "usage": {"prompt_tokens": 8, "completion_tokens": 2}}),
        (None, "[DONE]"),
    ]
    client, transport = make_client(lambda r: httpx.Response(200, content=sse_body(events), headers={"content-type": "text/event-stream"}))
    out = list(client.stream(CompletionRequest(messages=[Message.user("hi")])))
    assert transport.last_json()["stream"] is True
    assert transport.last_json()["stream_options"] == {"include_usage": True}
    kinds = [e.type for e in out]
    assert kinds == ["text_delta", "text_delta", "usage", "done"]
    assert "".join(e.text for e in out if e.type == "text_delta") == "Hello"
    assert out[-2].usage.input_tokens == 8 and out[-1].finish_reason == "stop"


def test_streaming_tool_call_fragments_are_assembled():
    events = [
        (None, chunk({"tool_calls": [{"index": 0, "id": "call_z", "type": "function", "function": {"name": "lookup_employee", "arguments": ""}}]})),
        (None, chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"id"'}}]})),
        (None, chunk({"tool_calls": [{"index": 0, "function": {"arguments": ": 42}"}}]})),
        (None, chunk({}, finish="tool_calls")),
        (None, "[DONE]"),
    ]
    client, _ = make_client(lambda r: httpx.Response(200, content=sse_body(events)))
    out = list(client.stream(CompletionRequest(messages=[Message.user("hi")], tools=[TOOL])))
    calls = [e.tool_call for e in out if e.type == "tool_call_delta"]
    assert calls == [ToolCall(id="call_z", name="lookup_employee", arguments={"id": 42})]
    assert out[-1].type == "done" and out[-1].finish_reason == "tool_calls"
    assert collect_stream(out).tool_calls[0].name == "lookup_employee"


def test_streaming_http_error_before_first_byte_maps():
    client, _ = make_client(lambda r: httpx.Response(429, json={"error": {"message": "slow"}}, headers={"retry-after": "1"}))
    with pytest.raises(RateLimitError):
        list(client.stream(CompletionRequest(messages=[Message.user("hi")])))


def test_streaming_inline_error_event():
    events = [(None, chunk({"content": "par"})), (None, {"error": {"message": "server exploded"}})]
    client, _ = make_client(lambda r: httpx.Response(200, content=sse_body(events)))
    out = list(client.stream(CompletionRequest(messages=[Message.user("hi")])))
    assert [e.type for e in out] == ["text_delta", "error", "done"]
    assert out[1].error == "server exploded"


async def test_async_complete_and_stream():
    def handler(r):
        payload = json.loads(r.content)
        if payload.get("stream"):
            return httpx.Response(200, content=sse_body([(None, chunk({"content": "ok"})), (None, chunk({}, finish="stop")), (None, "[DONE]")]))
        return ok_response(content="async!")

    transport = httpx.MockTransport(handler)
    client = OpenAICompatibleClient(base_url="http://test/v1", default_model="m1", async_transport=transport)
    c = await client.acomplete(CompletionRequest(messages=[Message.user("hi")]))
    assert c.text == "async!"
    events = [e async for e in client.astream(CompletionRequest(messages=[Message.user("hi")]))]
    assert [e.type for e in events] == ["text_delta", "done"]
    await client.aclose()


def test_extra_body_adds_server_fields_without_overriding_core_ones_and_raw_keeps_them():
    def handler(request):
        resp = ok_response(content="vpn")
        body = json.loads(resp.content)
        body["choices"][0]["logprobs"] = {"content": [{"token": "vpn", "logprob": -0.05}]}
        return httpx.Response(200, json=body)

    client, transport = make_client(handler, extra_body={"logprobs": True, "top_logprobs": 2, "model": "ignored"})
    completion = client.complete(CompletionRequest(messages=[Message.user("classify")]))
    sent = transport.last_json()
    assert sent["logprobs"] is True and sent["top_logprobs"] == 2
    assert sent["model"] == "m1"  # a core field is never overridden by extra_body
    assert completion.raw["choices"][0]["logprobs"]["content"][0]["logprob"] == -0.05
