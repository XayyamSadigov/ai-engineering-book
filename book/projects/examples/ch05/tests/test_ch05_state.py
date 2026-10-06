# path: book/projects/examples/ch05/tests/test_ch05_state.py
from __future__ import annotations

import pytest

from aie_core.llm.providers import FakeLLM
from context import (
    BudgetPolicy, ContextBuilder, ConversationState, LLMSummarizer, RequestScope, Trust,
)
from context.state import Fact, InMemoryStateStore, StaleStateError, Turn, literals


def words(text: str) -> int:
    return len(text.split())


class RecordingSummarizer:
    def __init__(self, output: str) -> None:
        self.output = output
        self.calls: list[tuple[str, list[int], list[str]]] = []

    def summarize(self, previous_summary: str, turns: list[Turn], facts: list[Fact]) -> str:
        self.calls.append((previous_summary, [t.index for t in turns], [f.key for f in facts]))
        return self.output


def chatty_state(n: int = 10, keep_last: int = 4) -> ConversationState:
    s = ConversationState(keep_last=keep_last, trigger_tokens=50, counter=words)
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        s.add_turn(role, f"turn {i} talks about the VPN problem at some length with details and more details")
    return s


def test_compaction_folds_old_turns_and_keeps_last_n_verbatim():
    s = chatty_state()
    summ = RecordingSummarizer("User cannot connect to VPN; tried restart and reinstall.")
    report = s.compact(summ)
    assert report.accepted and report.reason == "ok"
    assert report.folded_turns == [0, 1, 2, 3, 4, 5]
    assert [t.index for t in s.active_turns()] == [6, 7, 8, 9]
    assert report.tokens_after < report.tokens_before
    assert len(s.log) == 10, "the log is the source of truth and is never truncated"


def test_under_trigger_does_nothing():
    s = ConversationState(trigger_tokens=10_000, counter=words)
    s.add_turn("user", "hello")
    assert s.compact(RecordingSummarizer("x y")).reason == "under_trigger"


def test_exact_facts_are_never_summarized_and_always_rendered_verbatim():
    s = chatty_state()
    s.remember("refund_amount", "1,250.00 USD", "amount", source_turn=3)
    s.remember("ticket", "INC-4821", "identifier", trust=Trust.TRUSTED)
    summ = RecordingSummarizer("User asked for a refund (see refund_amount) on the VPN license.")
    s.compact(summ)
    prev, folded, fact_keys = summ.calls[0]
    assert fact_keys == ["refund_amount", "ticket"]  # the recording fake keeps only fact keys
    items = s.to_items()
    facts = [i for i in items if i.kind == "fact"]
    assert all(i.pinned for i in facts)
    rendered = "\n".join(i.content for i in facts)
    assert "1,250.00 USD" in rendered and "INC-4821" in rendered
    assert {i.trust for i in facts} == {Trust.TRUSTED, Trust.UNTRUSTED}


def test_pinned_turn_survives_compaction_verbatim():
    s = ConversationState(keep_last=2, trigger_tokens=10, counter=words)
    s.add_turn("user", "Goal: migrate my mailbox before Friday without losing calendar entries", pinned=True)
    for i in range(6):
        s.add_turn("assistant" if i % 2 == 0 else "user", f"step {i} discussion with plenty of words to fold away")
    report = s.compact(RecordingSummarizer("Discussed mailbox migration steps."))
    assert report.accepted
    assert 0 not in report.folded_turns
    assert s.active_turns()[0].content.startswith("Goal: migrate my mailbox")
    assert any(i.pinned and i.kind == "turn" for i in s.to_items())


def test_summary_inventing_a_number_is_rejected():
    s = chatty_state()
    report = s.compact(RecordingSummarizer("User was promised a refund of 300 dollars under INC-9999."))
    assert not report.accepted and report.reason == "novel_literals"
    assert report.novel_literals == ["300", "INC-9999"]
    assert s.summary == "" and len(s.active_turns()) == 10, "state is unchanged on rejection"


def test_summary_longer_than_source_is_rejected():
    s = chatty_state(n=6, keep_last=4)
    verbose = " ".join(["words"] * 200)
    report = s.compact(RecordingSummarizer(verbose), force=True)
    assert report.reason == "no_gain"


def test_incremental_compaction_passes_previous_summary_and_rebuild_starts_fresh():
    s = chatty_state(n=10, keep_last=4)
    summ = RecordingSummarizer("VPN issue, restart tried.")
    s.compact(summ)
    for i in range(10, 16):
        s.add_turn("user", f"turn {i} more VPN talk with enough words to trigger compaction again")
    s.compact(summ)
    assert summ.calls[1][0] == "VPN issue, restart tried."
    assert summ.calls[1][1] == [6, 7, 8, 9, 10, 11]
    rebuild = RecordingSummarizer("VPN issue across sessions.")
    report = s.rebuild_summary(rebuild)
    assert report.accepted
    assert rebuild.calls[0][0] == "" and rebuild.calls[0][1] == list(range(12))


def test_rehydration_returns_original_turns_behind_the_summary():
    s = chatty_state()
    s.compact(RecordingSummarizer("VPN trouble."))
    original = s.turns_between(2, 3)
    assert [t.index for t in original] == [2, 3]
    assert original[0].content.startswith("turn 2")
    summary_item = next(i for i in s.to_items() if i.kind == "summary")
    assert summary_item.metadata["covers"] == [0, 5]


def test_llm_summarizer_sends_keys_not_values_and_folded_turns_only():
    s = chatty_state()
    s.remember("asset_tag", "NW-LT-22917", "identifier")
    llm = FakeLLM(responses=["User has a VPN problem on the device (see asset_tag)."])
    report = s.compact(LLMSummarizer(llm))
    assert report.accepted
    req = llm.last_request
    assert req.temperature == 0.0 and req.metadata["purpose"] == "context.compaction"
    user = req.messages[-1].text
    assert "asset_tag" in user and "NW-LT-22917" not in user
    assert "turn 5" in user and "turn 6" not in user


def test_literals_extracts_ids_and_numbers():
    assert literals("Ticket INC-4821 refund $1,250.00 in 2 days") == {"INC-4821", "1250.00"}


def test_state_items_flow_through_builder_in_order():
    s = chatty_state()
    s.remember("ticket", "INC-4821", "identifier", trust=Trust.TRUSTED)
    s.compact(RecordingSummarizer("VPN trouble since Monday."))
    b = ContextBuilder(BudgetPolicy(context_window=4000, output_reserve=500), counter=words)
    result = b.build(s.to_items(), RequestScope(user_id="u", tenant="retail"))
    system_text = result.messages[0].text
    assert system_text.index("Summary of turns") < system_text.index("INC-4821")
    roles = [m.role.value for m in result.messages[1:]]
    assert roles == ["user", "assistant", "user", "assistant"]


def test_snapshot_round_trip_renders_identical_items():
    s = chatty_state()
    s.remember("ticket", "INC-4821", "identifier", trust=Trust.TRUSTED)
    s.compact(RecordingSummarizer("User cannot connect to VPN; see ticket."))
    restored = ConversationState.from_snapshot(s.snapshot(), counter=words)
    assert [i.model_dump() for i in restored.to_items()] == [i.model_dump() for i in s.to_items()]
    assert restored.version == s.version > 0


def test_retried_turn_with_same_message_id_is_not_appended_twice():
    s = ConversationState(counter=words)
    first = s.add_turn("user", "is my laptop eligible?", message_id="m-1")
    again = s.add_turn("user", "is my laptop eligible?", message_id="m-1")
    assert first is again and len(s.log) == 1 and s.version == 1


def test_store_rejects_a_stale_concurrent_write():
    store = InMemoryStateStore()
    a = store.load("sess-1", counter=words)
    a.add_turn("user", "first message")
    store.save("sess-1", a)

    worker_1 = store.load("sess-1", counter=words)
    worker_2 = store.load("sess-1", counter=words)
    worker_1.add_turn("assistant", "answer from worker 1")
    store.save("sess-1", worker_1)
    worker_2.add_turn("assistant", "answer from worker 2")
    with pytest.raises(StaleStateError):
        store.save("sess-1", worker_2)  # loaded v1, store is at v2: reload and re-apply instead
    reloaded = store.load("sess-1", counter=words)
    assert [t.content for t in reloaded.log] == ["first message", "answer from worker 1"]
