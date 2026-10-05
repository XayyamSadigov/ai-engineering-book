# path: book/projects/guardrails/tests/test_moderation.py
from __future__ import annotations

from aie_core.llm.providers import FakeLLM

from guardrails import (Action, GuardrailPipeline, KeywordModerator, LLMModerator, ModerationCheck, Moderator,
                        Stage, Subject)


def test_keyword_moderator_word_boundaries():
    m = KeywordModerator()
    assert m.moderate("I will hurt you").scores["violence"] == 1.0
    assert m.moderate("kill the stuck process and improve your skill").top()[1] == 0.0


def test_protocol_conformance():
    assert isinstance(KeywordModerator(), Moderator)
    assert isinstance(LLMModerator(FakeLLM()), Moderator)


def test_llm_moderator_parses_structured_scores():
    llm = FakeLLM(responses=[{"harassment": 0.1, "violence": 0.92, "rationale": "threat"}])
    res = LLMModerator(llm).moderate("x")
    assert res.top() == ("violence", 0.92) and res.provider.startswith("llm:")
    assert "<untrusted_data" in llm.requests[0].messages[1].text


def test_policy_block_flag_route(ctx):
    llm = FakeLLM(responses=[{"violence": 0.95}, {"harassment": 0.6}, {"self_harm": 0.7}, {}])
    check = ModerationCheck(LLMModerator(llm))
    s = Subject(Stage.INPUT, text="x")
    assert check.evaluate(s, ctx).action is Action.BLOCK
    assert check.evaluate(s, ctx).action is Action.FLAG
    routed = check.evaluate(s, ctx)
    assert routed.action is Action.FLAG and routed.metadata["route"] == "self_harm"
    assert check.evaluate(s, ctx).action is Action.ALLOW


def test_moderation_outage_fails_open(ctx):
    check = ModerationCheck(LLMModerator(FakeLLM(responses=[RuntimeError("down")])))
    res = GuardrailPipeline([check]).check_output("hello", ctx)
    assert res.allowed and res.errors
