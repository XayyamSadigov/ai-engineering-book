# path: book/projects/aie_core/tests/test_tokens.py
import pytest

from aie_core.llm import tokens
from aie_core.llm.types import Message, Role, ToolCall


@pytest.fixture
def heuristic(monkeypatch):
    monkeypatch.setattr(tokens, "_DISABLE_TIKTOKEN", True)
    tokens._encoding_for.cache_clear()
    yield
    tokens._encoding_for.cache_clear()


def test_heuristic_is_chars_over_four_rounded_up(heuristic):
    assert tokens.count_tokens("") == 0
    assert tokens.count_tokens("abcd") == 1
    assert tokens.count_tokens("abcde") == 2
    assert tokens.count_tokens("x" * 400) == 100


def test_message_tokens_add_overhead(heuristic):
    msgs = [Message.system("x" * 40), Message.user("y" * 40)]
    expected = tokens.REPLY_PRIMING_TOKENS + 2 * tokens.MESSAGE_OVERHEAD_TOKENS + 10 + 10
    assert tokens.count_message_tokens(msgs) == expected


def test_tool_calls_and_names_are_counted(heuristic):
    base = tokens.count_message_tokens([Message.user("hi")])
    with_name = tokens.count_message_tokens([Message(role=Role.USER, content="hi", name="alice")])
    with_call = tokens.count_message_tokens([Message(role=Role.ASSISTANT, content="", tool_calls=[ToolCall(id="1", name="lookup_employee", arguments={"id": 12345})])])
    assert with_name > base and with_call > base


def test_count_tokens_is_positive_and_monotone_in_any_mode():
    short = tokens.count_tokens("Northwind refund policy")
    long = tokens.count_tokens("Northwind refund policy " * 20)
    assert 0 < short < long


def test_tiktoken_path_when_available():
    pytest.importorskip("tiktoken")
    tokens._encoding_for.cache_clear()
    enc = tokens._encoding_for(None)
    if enc is None:
        pytest.skip("tiktoken vocabulary not cached offline")
    assert tokens.count_tokens("hello world") == 2
