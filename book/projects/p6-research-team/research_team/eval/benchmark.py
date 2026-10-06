# path: book/projects/p6-research-team/research_team/eval/benchmark.py
"""Compare the research team against single-agent baselines on the eight benchmark questions.

    python -m research_team.eval.benchmark                         # offline, scripted policy
    python -m research_team.eval.benchmark --fabricate-every 0     # no injected hallucinations
    python -m research_team.eval.benchmark --live                  # model from LLM_PROVIDER/LLM_MODEL

Configurations:
  single          one AgentRuntime, one context, one tool step per facet (ReAct style)
  single-batched  same agent issuing all searches, then all reads, as parallel tool calls
  single+verify   single, then the verifier agent and guard as a fixed second step (a workflow)
  team            planner, parallel researchers, verifier, synthesizer

The report ends with a verdict computed from the numbers, including when the team does not pay off.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

from aie_core.llm.gateway import PricingTable

from ..baseline import SingleAgent
from ..contracts import AnswerReport
from ..corpus import Corpus
from ..scripted import offline_llm
from ..team import ResearchTeam
from .scoring import Question, Score, load_questions, score

PRINCIPAL = {"user_id": "emp-1042", "groups": ["all"], "tenant": "retail"}
# Illustrative prices per million tokens; not any provider's actual price list.
ILLUSTRATIVE_PRICING = PricingTable(default={"input_per_1m": 1.0, "output_per_1m": 4.0})
CONFIGS = ("single", "single-batched", "single+verify", "team")


def build(config: str, *, corpus: Corpus, live: bool, fabricate_every: int, latency_scale: float,
          log_dir: str | None) -> Callable[[str], AnswerReport]:
    if live:
        from aie_core import make_llm_client
        llm: Any = make_llm_client()
    else:
        llm = offline_llm(fabricate_every=fabricate_every, batch_tool_calls=config == "single-batched",
                          latency_scale=latency_scale)
    kw = dict(pricing=ILLUSTRATIVE_PRICING, log_dir=log_dir)
    if config == "team":
        team = ResearchTeam(llm, corpus, **kw)
        return lambda q: team.ask(q, PRINCIPAL)
    agent = SingleAgent(llm, corpus, verify=config == "single+verify", **kw)
    if config == "single-batched":
        agent.architecture = "single-batched"
    return lambda q: agent.ask(q, PRINCIPAL)


def run(configs: list[str], questions: list[Question], corpus: Corpus, **kw: Any) -> list[Score]:
    scores: list[Score] = []
    for config in configs:
        ask = build(config, corpus=corpus, **kw)
        for q in questions:
            report = ask(q.question)
            s = score(q, report, corpus, PRINCIPAL)
            scores.append(s.model_copy(update={"architecture": config}))
    return scores


def _mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def summarize(scores: list[Score], configs: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for config in configs:
        out[config] = {}
        for kind in ("all", "cross", "control"):
            rows = [s for s in scores if s.architecture == config and (kind == "all" or s.kind == kind)]
            out[config][kind] = {
                "n": len(rows),
                "rubric": round(_mean([s.rubric for s in rows]), 2),
                "doc_recall": round(_mean([s.doc_recall for s in rows]), 2),
                "fact_recall": round(_mean([s.fact_recall for s in rows]), 2),
                "citation_validity": round(_mean([s.citation_validity for s in rows]), 3),
                "unsupported_shipped": sum(s.unsupported_shipped for s in rows),
                "tokens": round(_mean([s.tokens for s in rows])),
                "cost_usd": round(_mean([s.cost_usd for s in rows]), 5),
                "model_calls": round(_mean([s.model_calls for s in rows]), 1),
                "wall_ms": round(_mean([s.wall_ms for s in rows]), 1),
                "duplicate_claims": sum(s.duplicate_claims for s in rows),
            }
    return out


def verdict(summary: dict[str, dict[str, dict[str, float]]], *, min_rubric_gain: float = 0.5,
            max_token_ratio: float = 1.5) -> list[str]:
    """Rules, not vibes: the team pays off against a baseline on a question kind only if it uses at
    most `max_token_ratio` times the baseline's tokens and either gains at least `min_rubric_gain`
    on the rubric or is at least 25% faster at equal quality."""
    lines = []
    if "team" not in summary:
        return lines
    for base in [c for c in summary if c != "team"]:
        for kind in ("cross", "control"):
            t, b = summary["team"][kind], summary[base][kind]
            if not t["n"] or not b["n"]:
                continue
            gain = t["rubric"] - b["rubric"]
            tok = t["tokens"] / b["tokens"] if b["tokens"] else float("inf")
            speed = b["wall_ms"] / t["wall_ms"] if t["wall_ms"] else 1.0
            pays = tok <= max_token_ratio and (gain >= min_rubric_gain or (gain >= 0 and speed >= 1.25))
            lines.append(f"team vs {base} on {kind} questions: rubric {gain:+.2f}, tokens x{tok:.2f}, "
                         f"speed x{speed:.2f} -> {'PAYS OFF' if pays else 'DOES NOT PAY OFF'}")
    if {"single", "single+verify"} <= set(summary):
        for kind in ("cross", "control"):
            s, sv, t = (summary[c][kind]["rubric"] for c in ("single", "single+verify", "team"))
            if t - s > 0:
                share = min(1.0, max(0.0, (sv - s) / (t - s)))
                lines.append(f"attribution on {kind} questions: verification alone (single+verify) recovers "
                             f"{share:.0%} of the team's rubric gain over single")
    return lines


def render(summary: dict[str, dict[str, dict[str, float]]], verdicts: list[str]) -> str:
    cols = ["rubric", "doc_recall", "fact_recall", "citation_validity", "unsupported_shipped", "tokens",
            "cost_usd", "model_calls", "wall_ms", "duplicate_claims"]
    lines = []
    for kind in ("all", "cross", "control"):
        lines.append(f"\n{kind} questions")
        lines.append("| config | " + " | ".join(cols) + " |")
        lines.append("|---" * (len(cols) + 1) + "|")
        for config, by_kind in summary.items():
            row = by_kind[kind]
            lines.append(f"| {config} | " + " | ".join(str(row[c]) for c in cols) + " |")
    lines.append("\nVerdict")
    lines.extend(f"- {v}" for v in verdicts)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--live", action="store_true", help="use the model configured by LLM_PROVIDER/LLM_MODEL")
    ap.add_argument("--fabricate-every", type=int, default=4,
                    help="offline only: alter a number in about 1 of N numeric claims (0 disables)")
    ap.add_argument("--latency-scale", type=float, default=1.0, help="offline only: simulated model latency")
    ap.add_argument("--log-dir", default=None, help="write per-agent JSONL event logs here")
    ap.add_argument("--out", default=None, help="write per-question scores and the summary as JSON")
    args = ap.parse_args(argv)
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = set(configs) - set(CONFIGS)
    if unknown:
        ap.error(f"unknown configs: {sorted(unknown)}")
    corpus = Corpus.from_shared_data()
    scores = run(configs, load_questions(), corpus, live=args.live, fabricate_every=args.fabricate_every,
                 latency_scale=args.latency_scale, log_dir=args.log_dir)
    summary = summarize(scores, configs)
    verdicts = verdict(summary)
    mode = "live" if args.live else f"offline scripted, fabricate_every={args.fabricate_every}"
    print(f"Benchmark ({mode}); prices and latencies are illustrative.")
    print(render(summary, verdicts))
    if args.out:
        Path(args.out).write_text(json.dumps({"scores": [s.model_dump() for s in scores], "summary": summary,
                                              "verdict": verdicts}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
