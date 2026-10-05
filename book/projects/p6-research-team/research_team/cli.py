# path: book/projects/p6-research-team/research_team/cli.py
"""Command line entry point.

    python -m research_team ask "What do I need to do before travelling abroad with a company laptop?"
    python -m research_team ask --single "..."        # the baseline, for comparison
    python -m research_team trace .runs/<trace_id>    # the coordination tree and each agent's outcome
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from agentkit import JsonlEventStore, ModelDecision, Stopped, ToolResult

from .baseline import SingleAgent
from .config import P6Settings
from .corpus import Corpus
from .ledger import TeamLog
from .team import ResearchTeam

DEFAULT_PRINCIPAL = {"user_id": "emp-1042", "groups": ["all"], "tenant": "retail"}


def _llm(settings: P6Settings) -> Any:
    if settings.offline:
        from .scripted import offline_llm
        return offline_llm()
    from aie_core import make_llm_client
    return make_llm_client()


def cmd_ask(args: argparse.Namespace, settings: P6Settings) -> int:
    corpus = Corpus.from_shared_data(settings.shared_data_dir)
    principal = json.loads(args.principal) if args.principal else DEFAULT_PRINCIPAL
    llm = _llm(settings)
    if args.single or args.verify:
        system: Any = SingleAgent(llm, corpus, verify=args.verify, log_dir=settings.log_dir)
    else:
        system = ResearchTeam(llm, corpus, config=settings.team_config(), log_dir=settings.log_dir)
    report = system.ask(args.question, principal)
    if args.json:
        print(report.model_dump_json(indent=1))
        return 0 if report.status != "failed" else 1
    print(report.answer)
    print(f"\n[{report.architecture}] status={report.status} trace={report.trace_id} "
          f"tokens={report.usage.total_tokens} model_calls={report.usage.model_calls} "
          f"agents={sum(1 for c in report.children if c.run_id)} rejected={len(report.rejected_claims)} "
          f"wall_ms={report.wall_ms:.0f}")
    for note in report.notes:
        print(f"note: {note}")
    print(f"logs: {Path(settings.log_dir) / report.trace_id}")
    return 0 if report.status != "failed" else 1


def cmd_trace(args: argparse.Namespace, settings: P6Settings) -> int:
    run_dir = Path(args.run_dir)
    store = JsonlEventStore(run_dir / "agents")
    team_log = run_dir / "team.jsonl"
    if not team_log.exists():
        for run_id in store.runs():            # single-agent runs have no team log
            _print_agent(store, run_id, "  ")
        return 0
    for e in TeamLog.load(team_log):
        if e.kind == "task_dispatched":
            print(f"{e.seq:>3} dispatch {e.task_id} role={e.data['role']} tokens={e.data['budget']['max_tokens']}")
        elif e.kind == "spawn_refused":
            print(f"{e.seq:>3} REFUSED  {e.task_id} reason={e.data['reason']}")
        elif e.kind == "task_finished":
            print(f"{e.seq:>3} finished {e.task_id} status={e.data['status']} stop={e.data['stop_reason']}")
            if e.run_id:
                _print_agent(store, e.run_id, "      ")
        elif e.kind in ("verified", "conflict", "duplicate_work", "team_finished", "plan_accepted"):
            print(f"{e.seq:>3} {e.kind} {json.dumps(e.data, default=str)[:160]}")
    return 0


def _print_agent(store: JsonlEventStore, run_id: str, indent: str) -> None:
    events = store.load(run_id)
    tools = [e.tool for e in events if isinstance(e, ToolResult)]
    calls = sum(1 for e in events if isinstance(e, ModelDecision))
    stop = next((e for e in events if isinstance(e, Stopped)), None)
    print(f"{indent}{run_id}: {calls} model calls, tools={tools}, "
          f"stop={stop.reason.value if stop else 'running'}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research_team", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask", help="answer a question")
    a.add_argument("question")
    a.add_argument("--single", action="store_true", help="single-agent baseline")
    a.add_argument("--verify", action="store_true", help="single agent plus verification step")
    a.add_argument("--principal", help='JSON, e.g. {"groups": ["all"], "tenant": "logistics"}')
    a.add_argument("--json", action="store_true")
    t = sub.add_parser("trace", help="print the coordination tree of a run directory")
    t.add_argument("run_dir")
    args = ap.parse_args(argv)
    settings = P6Settings()
    return cmd_ask(args, settings) if args.cmd == "ask" else cmd_trace(args, settings)


if __name__ == "__main__":
    sys.exit(main())
