# path: book/projects/p6-research-team/tests/test_baseline_and_benchmark.py
from __future__ import annotations

from research_team import SingleAgent
from research_team.eval.benchmark import run, summarize, verdict
from research_team.eval.scoring import load_questions, score
from research_team.render import parse_cited_lines
from research_team.scripted import ScriptedPolicy

from .conftest import TRAVEL_Q, llm


def test_single_agent_answers_with_grounded_citations(corpus, principal):
    report = SingleAgent(llm(), corpus).ask(TRAVEL_Q, principal)
    assert report.status == "complete"
    assert len(report.children) == 1 and report.children[0].ok
    assert parse_cited_lines(report.answer)


def test_sequential_single_agent_context_grows_batched_does_not_need_to(corpus, principal):
    seq = SingleAgent(llm(ScriptedPolicy(batch_tool_calls=False)), corpus).ask(TRAVEL_Q, principal)
    bat = SingleAgent(llm(ScriptedPolicy(batch_tool_calls=True)), corpus).ask(TRAVEL_Q, principal)
    assert bat.usage.model_calls == 3 < seq.usage.model_calls
    assert bat.usage.input_tokens < seq.usage.input_tokens     # every extra step resends the whole transcript


def test_verify_step_removes_fabricated_claims_from_single_agent(corpus, principal):
    q = "What is the nightly hotel cap for international travel?"
    plain = SingleAgent(llm(ScriptedPolicy(fabricate_every=1)), corpus).ask(q, principal)
    checked = SingleAgent(llm(ScriptedPolicy(fabricate_every=1)), corpus, verify=True).ask(q, principal)
    questions = {x.id: x for x in load_questions()}
    assert score(questions["BQ-7"], plain, corpus, principal).unsupported_shipped > 0
    assert score(questions["BQ-7"], checked, corpus, principal).unsupported_shipped == 0
    assert checked.rejected_claims


def test_benchmark_runs_offline_and_reports_a_verdict(corpus):
    qs = [q for q in load_questions() if q.id in ("BQ-2", "BQ-7")]
    scores = run(["single", "single+verify", "team"], qs, corpus, live=False, fabricate_every=4,
                 latency_scale=0.0, log_dir=None)
    assert len(scores) == 6
    summary = summarize(scores, ["single", "single+verify", "team"])
    lines = verdict(summary)
    assert any("team vs single+verify" in line for line in lines)
    assert all(0 <= s.rubric <= 4 for s in scores)
    team = [s for s in scores if s.architecture == "team"]
    assert all(s.agents >= 4 for s in team)                     # planner, researcher(s), verifier, synthesizer
