# path: book/projects/examples/ch17/compare.py
"""Run the pipeline and the graph on the same tickets with a model whose
latency is simulated, and report orchestration overhead separately from
model time. Run: python compare.py
"""
from __future__ import annotations

import json
import time

from triage_domain import FakeModel, TriageState
from triage_graph import build_triage_graph
from triage_pipeline import PipelineMetrics, run_pipeline
from workflow_engine import InMemoryCheckpointer

OK = json.dumps({"verdict": "ok", "reasons": []})
TICKETS = [TriageState(ticket_id=f"T-{i}", tenant="logistics", text="parcel late") for i in range(20)]


def fresh_model(latency_ms: float) -> FakeModel:
    return FakeModel({"classify": ["shipping"], "draft": ["We will re-ship."], "validate": [OK]},
                     simulated_latency_ms=latency_ms)


def bench_pipeline(latency_ms: float) -> dict[str, float]:
    model = fresh_model(latency_ms)
    metrics = PipelineMetrics()
    t0 = time.perf_counter()
    for t in TICKETS:
        run_pipeline(t.model_copy(), model, approver=lambda s: True, sender=lambda *_: None, metrics=metrics)
    wall = (time.perf_counter() - t0) * 1000
    model_ms = sum(c.duration_ms for c in model.calls)
    return {"wall_ms": wall, "model_ms": model_ms, "overhead_ms": wall - model_ms,
            "calls": len(model.calls), "checkpoints": 0}


def bench_graph(latency_ms: float) -> dict[str, float]:
    model = fresh_model(latency_ms)
    cps = InMemoryCheckpointer()
    g = build_triage_graph(model, lambda *_: None, checkpointer=cps)
    t0 = time.perf_counter()
    for t in TICKETS:
        g.run(t.model_copy(), run_id=t.ticket_id)
    wall = (time.perf_counter() - t0) * 1000
    model_ms = sum(c.duration_ms for c in model.calls)
    cps_n = sum(len(cps.history(t.ticket_id)) for t in TICKETS)
    return {"wall_ms": wall, "model_ms": model_ms, "overhead_ms": wall - model_ms,
            "calls": len(model.calls), "checkpoints": cps_n}


if __name__ == "__main__":
    for latency in (0.0, 5.0):
        print(f"\nsimulated model latency {latency:.0f} ms per call, {len(TICKETS)} tickets")
        print(f"{'impl':10} {'wall_ms':>9} {'model_ms':>9} {'overhead_ms':>12} {'calls':>6} {'ckpts':>6}")
        for name, fn in (("pipeline", bench_pipeline), ("graph", bench_graph)):
            r = fn(latency)
            print(f"{name:10} {r['wall_ms']:9.1f} {r['model_ms']:9.1f} {r['overhead_ms']:12.1f} "
                  f"{r['calls']:6d} {r['checkpoints']:6d}")
