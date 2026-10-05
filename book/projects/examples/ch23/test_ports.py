# path: book/projects/examples/ch23/test_ports.py
from __future__ import annotations

from pathlib import Path

import pytest

from ports import (
    AnswerService,
    FrameworkRetrieverAdapter,
    FrameworkRetrieverLike,
    LLMClient,
    Passage,
    RecordingLLM,
    ReplayLLM,
    Retriever,
)

DOCS = [
    {"page_content": "Parental leave is 16 weeks for all employees.",
     "metadata": {"id": "hr-01", "source": "hr/parental-leave.md", "score": 0.9}},
    {"page_content": "VPN outages are escalated to it-oncall.",
     "metadata": {"id": "it-07", "source": "it/runbooks/vpn.md", "score": 0.7}},
]


class ScriptedLLM:
    def __init__(self, reply: str) -> None:
        self.reply, self.prompts = reply, []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def test_adapter_translates_framework_vocabulary_into_domain_passages() -> None:
    adapter = FrameworkRetrieverAdapter(FrameworkRetrieverLike(DOCS))
    assert isinstance(adapter, Retriever)        # structural check against the port
    out = adapter.retrieve("parental leave policy", k=4)
    assert out == [Passage(id="hr-01", text=DOCS[0]["page_content"], source="hr/parental-leave.md", score=0.9)]


def test_domain_service_never_sees_framework_types() -> None:
    llm = ScriptedLLM("Parental leave is 16 weeks [hr-01].")
    svc = AnswerService(FrameworkRetrieverAdapter(FrameworkRetrieverLike(DOCS)), llm)
    ans = svc.answer("how long is parental leave")
    assert ans.citations == ["hr-01"] and not ans.abstained
    assert "[hr-01]" in llm.prompts[0] and "page_content" not in llm.prompts[0]


def test_abstains_without_evidence_and_on_insufficient() -> None:
    svc = AnswerService(FrameworkRetrieverAdapter(FrameworkRetrieverLike(DOCS)), ScriptedLLM("x"))
    assert svc.answer("quantum chromodynamics").abstained
    svc2 = AnswerService(FrameworkRetrieverAdapter(FrameworkRetrieverLike(DOCS)), ScriptedLLM("INSUFFICIENT"))
    assert svc2.answer("parental leave for contractors").abstained


def test_record_then_replay_fixture_round_trip(tmp_path: Path) -> None:
    fixture = tmp_path / "answers.json"
    retriever = FrameworkRetrieverAdapter(FrameworkRetrieverLike(DOCS))

    live = RecordingLLM(ScriptedLLM("VPN outages go to it-oncall [it-07]."), fixture)
    recorded = AnswerService(retriever, live).answer("who handles vpn outages")

    replayed = AnswerService(retriever, ReplayLLM(fixture)).answer("who handles vpn outages")
    assert replayed == recorded
    assert isinstance(ReplayLLM(fixture), LLMClient)


def test_replay_fails_loudly_when_prompt_text_drifts(tmp_path: Path) -> None:
    fixture = tmp_path / "answers.json"
    retriever = FrameworkRetrieverAdapter(FrameworkRetrieverLike(DOCS))
    RecordingLLM(ScriptedLLM("ok [it-07]"), fixture).complete("old prompt")
    svc = AnswerService(retriever, ReplayLLM(fixture))
    with pytest.raises(LookupError, match="prompt text changed"):
        svc.answer("who handles vpn outages")
