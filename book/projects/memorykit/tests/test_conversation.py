# path: book/projects/memorykit/tests/test_conversation.py
"""Conversation memory: window, compaction with verified fact extraction, redaction, promotion."""
from __future__ import annotations

import pytest

from aie_core.llm.providers import FakeLLM
from aie_core.llm.types import CompletionRequest

from memorykit import ConversationMemory, Decision, Source, UserProfileMemory

TURNS = [
    ("user", "My laptop docking station stopped working, ticket INC-4821."),
    ("assistant", "I see your shipping site is Lisbon warehouse."),
    ("user", "Yes please reply in Spanish. My phone is +34 612 345 678."),
    ("assistant", "Noted. A replacement docking station ships Friday."),
    ("user", "Great, thanks."),
]

EXTRACTED = {"facts": [
    {"key": "ticket_id", "value": "INC-4821", "turn_index": 0},
    {"key": "shipping_site", "value": "Lisbon warehouse", "turn_index": 1},   # said by the assistant
    {"key": "preferred_language", "value": "Spanish", "turn_index": 2},
    {"key": "phone", "value": "+34 612 345 678", "turn_index": 2},
    {"key": "asset_tag", "value": "NW-77812", "turn_index": 0},              # invented
    {"key": "ticket_ref", "value": "INC 4821", "turn_index": 0},             # paraphrased
]}


class Script:
    def __init__(self) -> None:
        self.summary_inputs: list[str] = []

    def __call__(self, req: CompletionRequest):
        system = req.messages[0].text
        if "extract exact facts" in system:
            return EXTRACTED
        self.summary_inputs.append(req.messages[-1].text)
        return f"Summary #{len(self.summary_inputs)}: docking station broken, replacement arranged."


def run(script: Script) -> ConversationMemory:
    mem = ConversationMemory(FakeLLM(handler=script), session_id="s1", window_turns=2, trigger_turns=4)
    for role, content in TURNS:
        mem.add(role, content)
    return mem


def test_no_compaction_below_trigger():
    script = Script()
    mem = ConversationMemory(FakeLLM(handler=script), session_id="s1", window_turns=2, trigger_turns=4)
    for role, content in TURNS[:4]:
        assert mem.add(role, content) is False
    assert mem.compactions == 0 and len(mem.window()) == 4 and mem.state_block() == ""


def test_compaction_keeps_window_and_verified_exact_facts():
    script = Script()
    mem = run(script)
    assert mem.compactions == 1 and mem.watermark == 3
    assert [m.text for m in mem.window()] == [TURNS[3][1], TURNS[4][1]]
    assert set(mem.facts) == {"ticket_id", "shipping_site", "preferred_language", "phone"}
    assert mem.facts["ticket_id"].source == Source.USER_STATED
    assert mem.facts["shipping_site"].source == Source.MODEL_INFERRED
    assert mem.facts["ticket_id"].provenance == "turn:s1#0"
    block = mem.state_block()
    assert "ticket_id = INC-4821" in block and "Summary #1" in block
    # The summarizer is told which keys hold exact values, and gets only the folded turns.
    assert "INC-4821" in script.summary_inputs[0] and "Great, thanks" not in script.summary_inputs[0]


def test_redaction_rewrites_log_facts_and_summary():
    script = Script()
    mem = run(script)
    changed = mem.redact("+34 612 345 678")
    assert changed == 1
    assert all("612" not in t.content for t in mem.turns)
    assert "phone" not in mem.facts
    assert len(script.summary_inputs) == 2 and "612" not in script.summary_inputs[-1]
    assert "612" not in mem.state_block()


def test_promotion_goes_through_profile_policy(store, clock, ana):
    mem = run(Script())
    profile = UserProfileMemory(store, clock=clock)
    out = mem.promote_to_profile(profile, ana, allowed_keys={"preferred_language", "shipping_site", "phone"})
    assert set(out) == {"preferred_language", "shipping_site", "phone"}  # ticket_id is session state
    assert out["preferred_language"].decision == Decision.ACCEPT
    assert out["shipping_site"].decision == Decision.PENDING
    assert out["phone"].decision == Decision.REJECT and out["phone"].reasons == ["pii_not_allowed:phone"]
    assert profile.facts(ana)["preferred_language"].provenance == ["turn:s1#2"]


def test_trigger_must_exceed_window():
    with pytest.raises(ValueError):
        ConversationMemory(FakeLLM(), session_id="s", window_turns=6, trigger_turns=6)


class InventingScript(Script):
    """A summarizer that writes an identifier found in none of the folded turns."""

    def __call__(self, req: CompletionRequest):
        out = super().__call__(req)
        return out if isinstance(out, dict) else "Escalated as INC-9999; replacement arranged."


def test_summary_with_an_invented_identifier_is_rejected_and_nothing_is_dropped():
    mem = run(InventingScript())
    assert mem.compactions == 0 and mem.watermark == 0 and mem.summary == ""
    assert mem.rejected == ["novel_literals:INC-9999"]
    assert len(mem.window()) == len(TURNS)                 # the turns stay verbatim instead
    assert mem.facts["ticket_id"].value == "INC-4821"     # verified facts are still extracted
    assert "INC-9999" not in mem.state_block()


def test_redaction_drops_a_rebuilt_summary_that_fails_the_guard():
    script = Script()
    mem = run(script)
    mem.llm = FakeLLM(handler=InventingScript())          # the rebuild invents an identifier
    mem.redact("+34 612 345 678")
    assert mem.summary == "" and mem.rejected == ["novel_literals:INC-9999"]
    assert "612" not in mem.state_block()
