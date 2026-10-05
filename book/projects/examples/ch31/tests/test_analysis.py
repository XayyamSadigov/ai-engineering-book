# path: book/projects/examples/ch31/tests/test_analysis.py
"""The incident walk-through as assertions: each playbook step finds what the chapter claims."""
import json
import math
from pathlib import Path

from analysis import (bootstrap_diff_ci, compare_versions, cost_by_tenant, evidence_path, latency_breakdown,
                      percentile, retrieved_not_cited, trajectory_issues, trajectory_view, triage_by_stage)
from incident_sim import DEPLOY_AT, END_AT, write_incident
from join_signals import join, main as join_main, read_jsonl, sample_for_review
from metrics import METRICS, compute, evaluate, is_known, load_rules
from semconv import Attr, ErrorClass, SpanName
from trace_store import SpanRecord, TraceStore

HERE = Path(__file__).resolve().parents[1]


def windows(store):
    return (store.filter(since=DEPLOY_AT - 24 * 3600, until=DEPLOY_AT, route="rag.answer"),
            store.filter(since=DEPLOY_AT, until=END_AT, route="rag.answer"))


def test_trees_are_complete(store):
    assert len(store) == 1080
    assert all(t.root is not None and t.root.name == SpanName.REQUEST for t in store)
    assert store.completeness() == {Attr.TENANT: 0.0, Attr.ROUTE: 0.0, Attr.PROMPT_VERSION: 0.0,
                                    Attr.INDEX_VERSION: 0.0, "orphaned": 0.0}


def test_orphans_detected():
    rows = [SpanRecord("t", "a", None, "request", 0, 1, 1000),
            SpanRecord("t", "b", "missing", "retrieval.search", 0.1, 0.2, 100)]
    tree = TraceStore(rows).get("t")
    assert tree.root.span_id == "a" and [s.span_id for s in tree.orphans] == ["b"] and not tree.complete


def test_filters(store):
    assert all(t.tenant == "retail" for t in store.filter(tenant="retail"))
    assert all(t.versions[Attr.INDEX_VERSION] == "idx-2026-10-01"
               for t in store.filter(versions={Attr.INDEX_VERSION: "idx-2026-10-01"}))
    loops = store.filter(error_class=ErrorClass.LOOP)
    assert len(loops) >= 1 and all("loop" in t.error_classes for t in loops)
    assert all(t.gold_ids for t in store.filter(labeled=True))


def test_joins_match_everything(sim):
    st = TraceStore.from_spans(sim.spans)
    stats = join(st, sim.evals, sim.feedback)["stats"]
    assert stats["evals_unmatched"] == 0 and stats["feedback_unmatched"] == 0
    assert stats["feedback_matched"] == len(sim.feedback) > 100


def test_unmatched_feedback_is_reported(sim):
    st = TraceStore.from_spans(sim.spans)
    st.attach_feedback([{"response_id": "resp-does-not-exist", "value": -1}])
    assert len(st.unmatched_feedback) == 1


def test_alerts_fire_on_quality_not_on_latency_or_cost(store):
    results = evaluate(load_rules(HERE / "alerts.yaml"), store, now=END_AT)
    fired = {(r.rule, r.group) for r in results if r.fired}
    assert ("answer_quality_drop", "retail") in fired
    assert ("context_truncation_high", "all") in fired
    names = {r for r, _ in fired}
    assert not names & {"request_p95_slo", "cost_per_request_jump", "cross_tenant_retrieval", "telemetry_gaps"}


def test_no_alerts_before_deploy(store):
    results = evaluate(load_rules(HERE / "alerts.yaml"), store, now=DEPLOY_AT)
    assert not [r for r in results if r.fired and r.rule != "agent_loops"]


def test_every_alert_metric_is_defined():
    for rule in load_rules(HERE / "alerts.yaml"):
        assert is_known(rule.metric)
    assert is_known("errors.class_rate:loop") and not is_known("errors.class_rate:nope")
    assert "quality.eval_pass_rate" in METRICS


def test_prompt_canary_is_not_the_cause(store):
    _, after = windows(store)
    cmp = compare_versions(after, Attr.PROMPT_VERSION, "7", "8")
    point, lo, hi = cmp.pass_rate_diff
    assert lo < 0 < hi                       # no detectable quality difference
    assert cmp.b.mean_output_tokens > cmp.a.mean_output_tokens  # v8 is wordier: a cost effect


def test_stage_triage_points_at_context_build(store):
    before, after = windows(store)
    report = triage_by_stage(before, after)
    assert report.first_failing_stage == "in_context"
    assert abs(report.baseline["retrieved"] - report.candidate["retrieved"]) < 0.06
    assert "first failing stage" in report.render()


def test_retrieved_not_cited_finds_packer_drops(store):
    _, after = windows(store)
    findings = retrieved_not_cited(after)
    assert findings and sum(f.dropped_by_packer for f in findings) / len(findings) > 0.6
    assert all(f.retrieved_rank and f.retrieved_rank >= 4 for f in findings if f.dropped_by_packer)


def test_evidence_path_is_cumulative(store):
    for t in store.filter(labeled=True):
        p = evidence_path(t)
        assert p["cited"] <= p["in_context"] <= p["reranked"] <= p["retrieved"]


def test_latency_breakdown_and_cost(store):
    _, after = windows(store)
    lb = latency_breakdown(after)
    assert lb[SpanName.LLM_ATTEMPT]["share_p50"] > 0.8
    assert lb[SpanName.REQUEST]["p95"] >= lb[SpanName.REQUEST]["p50"]
    costs = cost_by_tenant(after)
    assert set(costs) == {"retail", "logistics"}
    for row in costs.values():
        assert row["cost_per_success"] >= row["cost_per_request"] > 0


def test_trajectory_view_flags_loop(store):
    tree = store.filter(error_class=ErrorClass.LOOP).traces()[0]
    issues = trajectory_issues(tree)
    assert any(i.startswith("loop: search_tickets") for i in issues)
    assert any("max_steps" in i for i in issues)
    assert "step 6: search_tickets" in trajectory_view(tree)


def test_side_effect_without_approval_flagged():
    rows = [SpanRecord("t", "r", None, "request", 0, 1, 1000, attributes={Attr.TENANT: "x"}),
            SpanRecord("t", "s", "r", SpanName.TOOL, 0.1, 0.2, 100,
                       attributes={Attr.TOOL_NAME: "send_reply", Attr.TOOL_SIDE_EFFECT: True,
                                   Attr.TOOL_APPROVAL: "pending", Attr.TOOL_STATUS: "ok"})]
    assert any("without approval" in i for i in trajectory_issues(TraceStore(rows).get("t")))


def test_stats_helpers():
    assert percentile([1, 2, 3, 4], 50) == 2.5 and math.isnan(percentile([], 50))
    p, lo, hi = bootstrap_diff_ci([0.0] * 50, [1.0] * 50)
    assert p == 1.0 and lo == hi == 1.0


def test_metric_class_rate(store):
    assert compute("errors.class_rate:loop", store) > 0


def test_sample_for_review_is_stratified(store):
    picked = sample_for_review(store, 40, seed=1)
    assert len(picked) == len(set(picked)) == 40
    tenants = {store.get(t).tenant for t in picked}
    assert tenants == {"retail", "logistics"}


def test_end_to_end_files_and_join_cli(tmp_path):
    paths = write_incident(tmp_path / "inc", seed=7)
    out = tmp_path / "joined.jsonl"
    assert join_main(["--traces", str(paths["traces"]), "--evals", str(paths["evals"]),
                      "--feedback", str(paths["feedback"]), "--out", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 1080
    labeled = [r for r in rows if r["gold_ids"]]
    assert labeled and all(r["eval_passed"] is not None for r in labeled)
    assert any(r["feedback_lag_s"] for r in rows)
    # the JSONL path gives the same answer as the in-memory path
    st = TraceStore.from_jsonl(paths["traces"])
    join(st, read_jsonl(paths["evals"]), read_jsonl(paths["feedback"]))
    before, after = windows(st)
    assert triage_by_stage(before, after).first_failing_stage == "in_context"
