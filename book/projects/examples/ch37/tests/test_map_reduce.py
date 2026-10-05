# path: book/projects/examples/ch37/tests/test_map_reduce.py
from __future__ import annotations

import pytest

from aie_core.llm.providers import FakeLLM
from map_reduce import (
    BudgetExceeded,
    InputEnvironment,
    MapReduceBudget,
    MapReduceSummarizer,
    RecursiveReader,
    corpus_as_one_text,
    extractive_summarizer,
    split_pieces,
)


@pytest.fixture(scope="module")
def text() -> str:
    return corpus_as_one_text()


def test_pieces_cover_the_input_in_order(text: str):
    pieces = split_pieces(text, 800)
    assert pieces[0][0] == 0
    assert all(a[1] <= b[0] for a, b in zip(pieces, pieces[1:]))


def test_hierarchical_reduce_with_provenance(text: str):
    budget = MapReduceBudget(piece_tokens=800, summary_tokens=60, reduce_input_tokens=200, max_llm_calls=80)
    llm = FakeLLM(handler=extractive_summarizer())
    result = MapReduceSummarizer(llm, budget).run(text)
    assert len(result.levels) >= 3 and result.levels[-1] == 1  # more than one reduce level
    assert result.llm_calls == sum(result.levels) == len(llm.requests)
    assert len(result.spans) == result.levels[0]  # the final summary knows every piece it came from
    assert all(r.max_tokens == 60 for r in llm.requests)  # output budget sent on every call


def test_estimate_is_a_conservative_upper_bound(text: str):
    budget = MapReduceBudget(piece_tokens=800, summary_tokens=60, reduce_input_tokens=200, max_llm_calls=80)
    s = MapReduceSummarizer(FakeLLM(handler=extractive_summarizer()), budget)
    n = len(split_pieces(text, 800))
    result = s.run(text)
    # the estimate assumes every summary fills its output budget, so real runs need fewer calls
    assert result.llm_calls <= s.estimate_calls(n) <= 2 * result.llm_calls


def test_preflight_refuses_a_job_over_budget_without_calling_the_model(text: str):
    llm = FakeLLM(handler=extractive_summarizer())
    with pytest.raises(BudgetExceeded, match="estimated"):
        MapReduceSummarizer(llm, MapReduceBudget(piece_tokens=300, max_llm_calls=10)).run(text)
    assert llm.requests == []


def test_runaway_output_is_truncated_to_the_summary_budget(text: str):
    verbose = FakeLLM(handler=lambda req: "word " * 2000)
    result = MapReduceSummarizer(verbose, MapReduceBudget(summary_tokens=50, reduce_input_tokens=400)).run(text[:20000])
    assert len(result.summary.split()) <= 60


def test_environment_operations_are_bounded_and_use_global_line_numbers(text: str):
    env = InputEnvironment(text)
    hit = env.grep("INC-2025-1142")[0]
    line_no = int(hit.split(":", 1)[0])
    sub = env.slice(line_no - 5, line_no + 5)
    assert sub.span == (line_no - 5, line_no + 5)
    assert sub.grep("INC-2025-1142")[0].startswith(f"{line_no}:")
    assert len(env.read(0, 10_000, max_tokens=100).split()) < 150


def test_recursive_reader_greps_reads_and_answers(text: str):
    env = InputEnvironment(text)
    line_no = int(env.grep(r"\*\*Duration\*\*")[0].split(":", 1)[0])
    script = [
        {"action": "grep", "pattern": r"\*\*Duration\*\*"},
        {"action": "read", "start": line_no - 2, "end": line_no + 2},
        {"action": "answer", "answer": "3 hours 12 minutes"},
    ]
    llm = FakeLLM(responses=script)
    ans = RecursiveReader(llm).answer("How long did INC-2025-1142 last?", env)
    assert ans.status == "answered" and ans.answer == "3 hours 12 minutes"
    assert "3 hours 12 minutes" in llm.requests[-1].messages[-1].text  # the read result was in context
    assert "PayBridge" not in llm.requests[0].messages[-1].text  # the model never saw the raw input up front


def test_recursion_respects_depth_and_shares_the_call_budget(text: str):
    env = InputEnvironment(text)
    always_recurse = FakeLLM(handler=lambda req: {"action": "recurse", "start": 0, "end": 500, "question": "dig"})
    reader = RecursiveReader(always_recurse, max_depth=2, max_steps=3, max_llm_calls=9)
    ans = reader.answer("anything", env)
    assert ans.status == "budget_exhausted"
    assert len(always_recurse.requests) <= 9
    assert max(r.metadata["depth"] for r in always_recurse.requests) == 2
    assert any("depth_limited" in t for t in ans.trace)
