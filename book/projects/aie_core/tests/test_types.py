# path: book/projects/aie_core/tests/test_types.py
from aie_core.llm.client import collect_stream
from aie_core.llm.types import (
    Completion,
    CompletionRequest,
    ContentPart,
    Message,
    Role,
    StreamEvent,
    ToolCall,
    Usage,
)


def test_message_helpers_set_roles():
    assert Message.system("s").role is Role.SYSTEM
    assert Message.user("u").role is Role.USER
    assert Message.assistant("a").role is Role.ASSISTANT
    t = Message.tool("call_1", "result")
    assert t.role is Role.TOOL and t.tool_call_id == "call_1" and t.text == "result"


def test_message_text_flattens_parts():
    m = Message(role=Role.USER, content=[ContentPart(type="text", text="a"), ContentPart(type="image_url", image_url="x"), ContentPart(type="text", text="b")])
    assert m.text == "ab"


def test_completion_convenience_properties():
    c = Completion(message=Message(role=Role.ASSISTANT, content="", tool_calls=[ToolCall(id="1", name="f", arguments={"a": 1})]))
    assert c.text == ""
    assert c.tool_calls[0].name == "f"
    assert Completion(message=Message.assistant("hi")).tool_calls == []


def test_request_defaults_are_deterministic():
    req = CompletionRequest(messages=[Message.user("x")])
    assert req.temperature == 0.0 and req.tool_choice == "auto" and req.metadata == {}
    # metadata default must not be shared between instances
    req.metadata["k"] = 1
    assert CompletionRequest(messages=[]).metadata == {}


def test_usage_addition():
    u = Usage(input_tokens=1, output_tokens=2) + Usage(input_tokens=3, cached_input_tokens=4)
    assert (u.input_tokens, u.output_tokens, u.cached_input_tokens) == (4, 2, 4)


def test_collect_stream_rebuilds_completion():
    events = [
        StreamEvent(type="text_delta", text="Hel"),
        StreamEvent(type="text_delta", text="lo"),
        StreamEvent(type="tool_call_delta", tool_call=ToolCall(id="c1", name="lookup_employee", arguments={"id": 7})),
        StreamEvent(type="usage", usage=Usage(input_tokens=5, output_tokens=3)),
        StreamEvent(type="done", finish_reason="tool_calls"),
    ]
    c = collect_stream(events, model="m", provider="p")
    assert c.text == "Hello" and c.tool_calls[0].arguments == {"id": 7}
    assert c.usage.output_tokens == 3 and c.finish_reason == "tool_calls" and c.model == "m"


def test_truncated_property():
    assert Completion(message=Message.assistant("x"), finish_reason="length").truncated is True
    assert Completion(message=Message.assistant("x"), finish_reason="stop").truncated is False
