# path: book/projects/examples/ch31/join_signals.py
"""Join offline eval results and online user feedback to trace ids.

Labels arrive late and from different systems: an evalkit run writes `CaseResult` rows that
carry `trace_id`; the product writes feedback events that carry the client-visible
`response_id` (the client never sees trace ids); a labeling queue writes human verdicts days
later. This script resolves all of them to traces and emits one joined row per trace, which
is what dashboards, version comparisons, and fine-tuning exports (Chapter 33) consume.

    python join_signals.py --traces traces.jsonl --evals eval_results.jsonl \
        --feedback feedback.jsonl --out joined.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

from semconv import Attr
from trace_store import TraceStore, TraceTree


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def normalize_eval(row: dict[str, Any], *, source: str = "offline") -> dict[str, Any]:
    """Accept evalkit CaseResult rows (scores/passed dicts) or flat eval rows."""
    passed = row.get("passed")
    if isinstance(passed, dict):  # evalkit: {"metric": bool | None}
        verdicts = [v for v in passed.values() if v is not None]
        passed = all(verdicts) if verdicts else None
    scores = row.get("scores")
    score = row.get("score")
    if score is None and isinstance(scores, dict):
        numeric = [v for v in scores.values() if isinstance(v, (int, float))]
        score = sum(numeric) / len(numeric) if numeric else None
    meta = row.get("metadata") or {}
    return {
        "trace_id": row.get("trace_id"),
        "case_id": row.get("case_id"),
        "name": row.get("name") or row.get("evaluator") or "eval",
        "score": score,
        "passed": passed,
        "source": row.get("source") or source,
        "gold_ids": row.get("gold_ids") or meta.get("gold_ids"),
        "ts": row.get("ts"),
    }


def joined_row(t: TraceTree) -> dict[str, Any]:
    first_feedback_ts = min((f.get("ts") for f in t.feedback if f.get("ts") is not None), default=None)
    return {
        "trace_id": t.trace_id,
        "start": t.start,
        "tenant": t.tenant,
        "route": t.get(Attr.ROUTE),
        "traffic": t.get(Attr.TRAFFIC),
        "versions": t.versions,
        "duration_ms": round(t.duration_ms, 1),
        "cost_usd": round(t.cost_usd, 8),
        "error_classes": sorted(t.error_classes),
        "eval_passed": t.eval_passed,
        "eval_scores": {e["name"]: e.get("score") for e in t.evals},
        "gold_ids": t.gold_ids,
        "feedback": t.feedback_value,
        "feedback_reason": t.feedback[-1].get("reason") if t.feedback else None,
        "feedback_lag_s": round(first_feedback_ts - t.start, 1) if first_feedback_ts is not None else None,
    }


def join(store: TraceStore, evals: Iterable[dict[str, Any]], feedback: Iterable[dict[str, Any]]) -> dict[str, Any]:
    eval_rows = [normalize_eval(r) for r in evals]
    matched_evals = store.attach_evals(eval_rows)
    feedback_rows = list(feedback)
    matched_feedback = store.attach_feedback(feedback_rows)
    rows = [joined_row(t) for t in store]
    return {
        "rows": rows,
        "stats": {
            "traces": len(rows),
            "evals": len(eval_rows), "evals_matched": matched_evals,
            "feedback": len(feedback_rows), "feedback_matched": matched_feedback,
            # unmatched labels mean traces were dropped, sampled out, or ids were not propagated
            "feedback_unmatched": len(feedback_rows) - matched_feedback,
            "evals_unmatched": len(eval_rows) - matched_evals,
        },
    }


def sample_for_review(store: TraceStore, n: int, *, strata_key: str = Attr.TENANT, seed: int = 0,
                      negative_share: float = 0.5) -> list[str]:
    """Pick traces for human review: half from thumbs-down/errored traces, the rest stratified
    across `strata_key` so a small tenant is not drowned out by a large one."""
    rng = random.Random(seed)
    traces = store.traces()
    flagged = [t for t in traces if (t.feedback_value or 0) < 0 or t.error_classes]
    rng.shuffle(flagged)
    picked = [t.trace_id for t in flagged[: int(n * negative_share)]]
    strata: dict[str, list[TraceTree]] = defaultdict(list)
    for t in traces:
        if t.trace_id not in picked:
            strata[str(t.get(strata_key))].append(t)
    for group in strata.values():
        rng.shuffle(group)
    keys = sorted(strata)
    while len(picked) < n and any(strata.values()):
        for k in keys:
            if strata[k] and len(picked) < n:
                picked.append(strata[k].pop().trace_id)
    return picked


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--traces", required=True, nargs="+")
    ap.add_argument("--evals", nargs="*", default=[])
    ap.add_argument("--feedback", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    store = TraceStore.from_jsonl(*args.traces)
    evals = [r for p in args.evals for r in read_jsonl(p)]
    feedback = [r for p in args.feedback for r in read_jsonl(p)]
    result = join(store, evals, feedback)
    with Path(args.out).open("w", encoding="utf-8") as f:
        for row in result["rows"]:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(result["stats"], indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
