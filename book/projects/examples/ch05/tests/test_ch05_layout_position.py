# path: book/projects/examples/ch05/tests/test_ch05_layout_position.py
from __future__ import annotations

from aie_core.llm.providers import FakeLLM
from context import (
    BudgetPolicy, ContextBuilder, ContextItem, ConversationState, PrefixStabilityTracker,
    RequestScope, Trust, lint_stable_items, shared_prefix_tokens,
)
from context.experiments.position import (
    DEFAULT_NEEDLE, SimulatedPositionalReader, build_haystack_prompt, load_distractors, run_position_sweep,
)


def words(text: str) -> int:
    return len(text.split())


SCOPE = RequestScope(user_id="u", tenant="retail", groups=["all"])
SYSTEM = ContextItem(kind="instructions", content="You are Northwind Assist. " * 30, source_id="prompt:v1",
                     trust=Trust.TRUSTED, pinned=True)


def build(question: str, state: ConversationState | None = None):
    items = [SYSTEM, ContextItem(kind="query", content=question, source_id="user:q")]
    if state:
        items += state.to_items()
    return ContextBuilder(BudgetPolicy(context_window=8000, output_reserve=500), counter=words).build(items, SCOPE)


def test_different_requests_share_the_stable_prefix():
    a, b = build("How do I reset my VPN token?"), build("What is the PTO carryover cap?")
    assert a.prefix_hash == b.prefix_hash
    assert shared_prefix_tokens(a.messages, b.messages, words) >= 150


def test_state_changes_do_not_move_the_stable_prefix_hash():
    s = ConversationState(keep_last=1, trigger_tokens=1, counter=words)
    s.add_turn("user", "first question about vpn tokens")
    before = build("q", s)
    s.add_turn("assistant", "an answer")
    s.add_turn("user", "follow up")
    s.compact(type("S", (), {"summarize": lambda self, p, t, f: "vpn"})())
    after = build("q", s)
    assert before.prefix_hash == after.prefix_hash
    assert before.messages[0].text != after.messages[0].text  # the state tail changed, the prefix did not


def test_lint_flags_volatile_content_in_stable_items():
    bad = ContextItem(kind="instructions", content="Today is 2026-10-04T09:00. request_id: abc123",
                      source_id="prompt:bad", trust=Trust.TRUSTED)
    warnings = lint_stable_items([bad, SYSTEM])
    assert any("timestamp" in w for w in warnings) and any("request_id" in w for w in warnings)
    assert not lint_stable_items([SYSTEM])


def test_prefix_tracker_counts_repeats():
    tracker = PrefixStabilityTracker()
    for q in ["a", "b", "c"]:
        tracker.observe(build(q))
    assert tracker.repeat_rate == 2 / 3


def test_haystack_places_needle_exactly_where_asked():
    distractors = load_distractors()[:9]
    for index in (0, 4, 9):
        messages, _ = build_haystack_prompt(DEFAULT_NEEDLE, distractors, index, counter=words)
        user = messages[-1].text
        blocks = user.split("</untrusted_data>")[:-1]
        assert len(blocks) == 10
        assert DEFAULT_NEEDLE.answer in blocks[index]


def test_sweep_detects_a_planted_u_curve():
    reader = SimulatedPositionalReader(DEFAULT_NEEDLE.answer, edge=1.0, middle=0.2)
    result = run_position_sweep(FakeLLM(handler=reader), n_docs=11, trials=30, counter=words)
    acc = {r.position: r.accuracy for r in result.rows}
    assert acc[0.0] == 1.0 and acc[1.0] == 1.0
    assert acc[0.5] < 0.5
    assert result.spread() > 0.5


def test_sweep_reports_flat_curve_for_position_blind_reader():
    reader = SimulatedPositionalReader(DEFAULT_NEEDLE.answer, edge=1.0, middle=1.0)
    result = run_position_sweep(FakeLLM(handler=reader), n_docs=11, trials=5, counter=words)
    assert result.spread() == 0.0


def test_sweep_is_deterministic_and_uses_same_distractors_per_position():
    llm1 = FakeLLM(handler=SimulatedPositionalReader(DEFAULT_NEEDLE.answer))
    llm2 = FakeLLM(handler=SimulatedPositionalReader(DEFAULT_NEEDLE.answer))
    r1 = run_position_sweep(llm1, n_docs=9, trials=4, seed=7, counter=words)
    r2 = run_position_sweep(llm2, n_docs=9, trials=4, seed=7, counter=words)
    assert r1 == r2
    assert len({r.mean_prompt_tokens for r in r1.rows}) == 1, "only position varies, not prompt size"
    assert all(req.temperature == 0.0 for req in llm1.requests)
