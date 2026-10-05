# path: book/projects/aie_core/tests/test_structured.py
import pytest
from pydantic import BaseModel, Field

from aie_core.llm.errors import MalformedResponseError
from aie_core.llm.providers import FakeLLM
from aie_core.llm.structured import acomplete_structured, complete_structured, extract_json
from aie_core.llm.types import Completion, CompletionRequest, Message, Role, ToolCall


class TicketClassification(BaseModel):
    category: str = Field(pattern="^(billing|access|hardware)$")
    priority: int = Field(ge=1, le=4)


REQ = CompletionRequest(messages=[Message.system("Classify tickets."), Message.user("VPN is down for the whole team")])


def test_extract_json_strips_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json('```\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json('Sure! Here it is: {"a": 1} hope that helps') == '{"a": 1}'
    assert extract_json('{"a": 1}') == '{"a": 1}'


def test_happy_path_uses_response_schema():
    llm = FakeLLM(responses=[{"category": "access", "priority": 2}])
    obj, completion = complete_structured(llm, REQ, TicketClassification)
    assert obj == TicketClassification(category="access", priority=2)
    sent = llm.last_request
    assert sent.response_schema is not None and sent.response_schema["title"] == "TicketClassification"
    assert completion.text.startswith("{")


def test_repair_loop_feeds_validation_error_back():
    llm = FakeLLM(responses=['```json\n{"category": "network", "priority": 2}\n```', {"category": "access", "priority": 9}, {"category": "access", "priority": 1}])
    obj, _ = complete_structured(llm, REQ, TicketClassification, max_repair_attempts=2)
    assert obj.priority == 1 and len(llm.requests) == 3
    second = llm.requests[1].messages
    assert second[-2].role is Role.ASSISTANT and "network" in second[-2].text
    assert second[-1].role is Role.USER and "did not match" in second[-1].text and "category" in second[-1].text
    assert len(llm.requests[2].messages) == len(REQ.messages) + 4


def test_gives_up_after_budget():
    llm = FakeLLM(responses=["not json", "still not json", "nope"])
    with pytest.raises(MalformedResponseError) as info:
        complete_structured(llm, REQ, TicketClassification, max_repair_attempts=2)
    assert len(llm.requests) == 3 and "3 attempts" in str(info.value)


def test_zero_repairs_means_single_call():
    llm = FakeLLM(responses=["garbage", {"category": "billing", "priority": 1}])
    with pytest.raises(MalformedResponseError):
        complete_structured(llm, REQ, TicketClassification, max_repair_attempts=0)
    assert len(llm.requests) == 1


def test_falls_back_to_prompt_instruction_when_schema_unsupported():
    llm = FakeLLM(responses=[{"category": "billing", "priority": 3}])
    llm.supports_response_schema = False
    complete_structured(llm, REQ, TicketClassification)
    sent = llm.last_request
    assert sent.response_schema is None
    assert sent.messages[0].role is Role.SYSTEM and "JSON Schema" in sent.messages[0].text and "Classify tickets." in sent.messages[0].text


def test_accepts_tool_call_payload():
    def handler(req):
        return Completion(message=Message(role=Role.ASSISTANT, content="", tool_calls=[ToolCall(id="t", name="emit", arguments={"category": "hardware", "priority": 4})]), finish_reason="tool_calls")

    obj, _ = complete_structured(FakeLLM(handler=handler), REQ, TicketClassification)
    assert obj.category == "hardware"


async def test_async_variant_repairs():
    llm = FakeLLM(responses=["{bad", {"category": "billing", "priority": 2}])
    obj, _ = await acomplete_structured(llm, REQ, TicketClassification)
    assert obj.priority == 2 and len(llm.requests) == 2
    echo = llm.requests[1].messages[-2]
    assert echo.role is Role.ASSISTANT and echo.text == "{bad"  # the bad answer is echoed back verbatim


def test_truncated_output_is_detected_and_not_repaired():
    from aie_core.llm.structured import TruncatedOutputError

    def handler(req):
        return Completion(message=Message.assistant('{"category": "billing", "priority": 2'), finish_reason="length")

    llm = FakeLLM(handler=handler)
    with pytest.raises(TruncatedOutputError) as info:
        complete_structured(llm, REQ, TicketClassification, max_repair_attempts=3)
    assert len(llm.requests) == 1  # no repair round trips at the same max_tokens
    assert isinstance(info.value, MalformedResponseError) and info.value.raw["truncated"] is True


def test_truncated_flag_rejects_even_valid_json():
    def handler(req):
        return Completion(message=Message.assistant('{"category": "billing", "priority": 2}'), finish_reason="length")

    with pytest.raises(MalformedResponseError):
        complete_structured(FakeLLM(handler=handler), REQ, TicketClassification)
