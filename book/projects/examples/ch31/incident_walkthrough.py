# path: book/projects/examples/ch31/incident_walkthrough.py
"""Run the debugging playbook end to end on the simulated incident and print each step.

    python incident_walkthrough.py            # simulate in memory
    python incident_walkthrough.py data/incident   # or load JSONL written by incident_sim.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from analysis import (compare_versions, cost_by_tenant, latency_breakdown, render_latency, retrieved_not_cited,
                      summarize, trajectory_view, triage_by_stage)
from incident_sim import DEPLOY_AT, END_AT, simulate
from join_signals import join, read_jsonl
from metrics import evaluate, load_rules
from semconv import Attr, ErrorClass
from trace_store import TraceStore

HERE = Path(__file__).parent


def load(source: str | None) -> TraceStore:
    if source:
        d = Path(source)
        store = TraceStore.from_jsonl(d / "traces.jsonl")
        join(store, read_jsonl(d / "eval_results.jsonl"), read_jsonl(d / "feedback.jsonl"))
        return store
    result = simulate()
    store = TraceStore.from_spans(result.spans)
    join(store, result.evals, result.feedback)
    return store


def main(argv: list[str]) -> None:
    store = load(argv[1] if len(argv) > 1 else None)
    before = store.filter(since=DEPLOY_AT - 24 * 3600, until=DEPLOY_AT)
    after = store.filter(since=DEPLOY_AT, until=END_AT)

    print("== step 1: what fired ==")
    results = evaluate(load_rules(HERE / "alerts.yaml"), store, now=END_AT)
    for a in results:
        if a.fired:
            print(f"FIRED {a.severity:<6} {a.rule:<26} group={a.group:<10} {a.reason}  (n={a.samples})")
    quiet = sorted({a.rule for a in results} - {a.rule for a in results if a.fired})
    print("quiet:", ", ".join(quiet))

    print("\n== step 2: scope by tenant and route ==")
    for tenant in ("retail", "logistics"):
        for route in ("rag.answer", "agent.incident"):
            b = summarize(before.filter(tenant=tenant, route=route))
            a = summarize(after.filter(tenant=tenant, route=route))
            if b.n and b.labeled:
                print(f"{tenant:<10}{route:<16} pass {b.eval_pass_rate:.2f} -> {a.eval_pass_rate:.2f}   "
                      f"neg.fb {b.negative_feedback_rate:.2f} -> {a.negative_feedback_rate:.2f}   n={b.n}/{a.n}")
            elif b.n:
                print(f"{tenant:<10}{route:<16} unlabeled; neg.fb {b.negative_feedback_rate:.2f} -> "
                      f"{a.negative_feedback_rate:.2f}   n={b.n}/{a.n}")

    print("\n== step 3: version diff inside the window (prompt canary) ==")
    print(compare_versions(after.filter(route="rag.answer"), Attr.PROMPT_VERSION, "7", "8").render())

    print("\n== step 4: stage triage, labeled traces, before vs after ==")
    print(triage_by_stage(before.filter(route="rag.answer"), after.filter(route="rag.answer")).render())

    print("\n== step 5: evidence retrieved but not cited (after) ==")
    findings = retrieved_not_cited(after)
    dropped = sum(f.dropped_by_packer for f in findings)
    print(f"{len(findings)} traces; {dropped} had gold dropped by the context packer")
    for f in findings[:4]:
        print(f"  {f.trace_id[:12]} gold={f.gold} rank={f.retrieved_rank} in_context={f.in_context} "
              f"dropped={f.dropped_by_packer} cited={f.cited}")
    example = store.get(next(f.trace_id for f in findings if f.dropped_by_packer))
    print(example.render([Attr.INDEX_VERSION, Attr.CONTEXT_TOKENS, Attr.CONTEXT_DROPPED, Attr.LLM_INPUT_TOKENS]))

    print("\n== step 6: latency and cost, before vs after ==")
    print("before:\n" + render_latency(latency_breakdown(before.filter(route="rag.answer"))))
    print("after:\n" + render_latency(latency_breakdown(after.filter(route="rag.answer"))))
    for label, st in (("before", before), ("after", after)):
        for tenant, row in cost_by_tenant(st).items():
            print(f"{label:<7}{tenant:<10} cost/req {row['cost_per_request']:.5f}  cost/success {row['cost_per_success']:.5f}"
                  f"  llm calls/req {row['llm_calls_per_request']:.2f}")

    print("\n== step 7: one agent trajectory flagged as a loop ==")
    looping = store.filter(error_class=ErrorClass.LOOP).traces()
    if looping:
        print(trajectory_view(looping[0]))

    print("\n== telemetry completeness ==")
    print({k: round(v, 4) for k, v in store.completeness().items()})


if __name__ == "__main__":
    main(sys.argv)
