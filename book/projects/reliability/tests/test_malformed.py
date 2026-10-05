# path: book/projects/reliability/tests/test_malformed.py
import json
from typing import Literal

from aie_core import CompletionRequest, FakeLLM, Message
from pydantic import BaseModel, Field

from reliability import Outcome, close_truncated_json, complete_with_recovery, salvage_partial


class Ticket(BaseModel):
    category: Literal["network", "access", "billing", "other"]
    priority: int = Field(ge=1, le=4)
    summary: str


GOOD = {"category": "network", "priority": 2, "summary": "VPN drops"}
REQ = CompletionRequest(messages=[Message.user("VPN keeps dropping")])


def test_ok_on_first_try():
    r = complete_with_recovery(FakeLLM(responses=[GOOD]), REQ, Ticket)
    assert r.outcome is Outcome.OK and r.value.category == "network" and r.calls == 1


def test_repaired_after_one_reask():
    llm = FakeLLM(responses=['{"category": "netwrk", "priority": 2, "summary": "x"}', GOOD])
    r = complete_with_recovery(llm, REQ, Ticket, repair_attempts=1)
    assert r.outcome is Outcome.REPAIRED and r.calls == 2
    assert "did not match" in llm.requests[1].messages[-1].text


def test_fallback_model_after_failed_repair():
    primary = FakeLLM(responses=["not json", "still not json"])
    backup = FakeLLM(responses=[GOOD])
    r = complete_with_recovery(primary, REQ, Ticket, fallback_client=backup)
    assert r.outcome is Outcome.FALLBACK and r.calls == 3


def test_salvages_truncated_json_without_another_call():
    truncated = json.dumps(GOOD)[:-2]                # cut off by the output limit: '..."VPN dro'
    llm = FakeLLM(responses=[truncated, truncated])
    r = complete_with_recovery(llm, REQ, Ticket)
    assert r.outcome is Outcome.SALVAGED and r.value.summary.startswith("VPN dro")


def test_partial_fields_merged_into_default():
    broken = '{"category": "billing", "priority": 9, "summary": '
    llm = FakeLLM(responses=[broken, broken])
    default = Ticket(category="other", priority=3, summary="unclassified")
    r = complete_with_recovery(llm, REQ, Ticket, default=default)
    assert r.outcome is Outcome.PARTIAL
    assert r.value.category == "billing" and r.value.priority == 3   # invalid priority dropped
    assert set(r.missing) == {"priority", "summary"}


def test_failed_returns_instead_of_raising():
    r = complete_with_recovery(FakeLLM(responses=["nope", "nope"]), REQ, Ticket)
    assert r.outcome is Outcome.FAILED and not r.usable and r.errors


def test_close_truncated_json_shapes():
    assert json.loads(close_truncated_json('{"a": 1, "b": [1, 2')) == {"a": 1, "b": [1, 2]}
    assert json.loads(close_truncated_json('{"a": 1, "b":')) == {"a": 1}
    assert json.loads(close_truncated_json('{"a": 1, "ke')) == {"a": 1}
    assert json.loads(close_truncated_json('{"a": "hel')) == {"a": "hel"}


def test_salvage_partial_reports_missing():
    valid, missing = salvage_partial('{"category": "access"}', Ticket)
    assert valid == {"category": "access"} and missing == ["priority", "summary"]
