# path: book/projects/p5-incident-agent/incident_agent/cli.py
"""Command line: investigate an alert, show an investigation, approve or reject publication, replay a step.

    p5 investigate ALR-2026-0914-01 --user oncall-logistics
    p5 show <investigation-id>
    p5 approve <investigation-id> --user ic-logistics --reason "checked against dashboards"
    p5 reject <investigation-id> --user ic-logistics --reason "cause not confirmed"
    p5 replay <investigation-id>          # re-run every recorded step from its event log, no tools executed
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence, TextIO

from agentkit import replay

from .service import IncidentService, InvalidState, NotAllowed


def _summary(inv, out: TextIO) -> None:
    print(f"{inv.id}  status={inv.status.value}  {inv.detail}".rstrip(), file=out)
    print("trajectory: " + " -> ".join(f"{t}({a})" for t, a in inv.trajectory()), file=out)
    if inv.deviations:
        print(f"replans: {len(inv.plans) - 1}; deviations: {inv.deviations}", file=out)
    for r in inv.rounds:
        judge = f" judge={r.judge['score']}/5" if r.judge else ""
        print(f"round {r.round}: score={r.score:.2f} dod_problems={len(r.dod_problems)}{judge}", file=out)
    print(f"usage: {json.dumps(inv.usage)}", file=out)


def main(argv: Sequence[str] | None = None, *, service: IncidentService | None = None,
         out: TextIO = sys.stdout) -> int:
    parser = argparse.ArgumentParser(prog="p5", description="Northwind incident-research agent")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("investigate")
    p.add_argument("alert_id")
    p.add_argument("--user", default="oncall-logistics")
    p = sub.add_parser("show")
    p.add_argument("investigation_id")
    p.add_argument("--report", action="store_true")
    for name in ("approve", "reject"):
        p = sub.add_parser(name)
        p.add_argument("investigation_id")
        p.add_argument("--user", default="ic-logistics")
        p.add_argument("--reason", default="")
    p = sub.add_parser("replay")
    p.add_argument("investigation_id")
    args = parser.parse_args(argv)
    svc = service or IncidentService()
    try:
        if args.cmd == "investigate":
            inv = svc.investigate(args.alert_id, args.user)
            _summary(inv, out)
            if inv.report:
                print("\n" + inv.report, file=out)
            return 0 if inv.status.value in ("awaiting_approval", "published") else 1
        if args.cmd == "show":
            inv = svc.get(args.investigation_id)
            _summary(inv, out)
            if args.report and inv.report:
                print("\n" + inv.report, file=out)
            return 0
        if args.cmd in ("approve", "reject"):
            inv = svc.decide(args.investigation_id, args.user, approve=args.cmd == "approve", reason=args.reason)
            print(f"{inv.id}  status={inv.status.value}  message={inv.posted_message_id}", file=out)
            return 0
        inv = svc.get(args.investigation_id)
        for step in inv.steps:
            report = replay(svc.event_store.load(step.run_id))
            print(f"{step.run_id}: {report.summary()}", file=out)
        return 0
    except KeyError as exc:
        print(f"not found: {exc}", file=out)
        return 2
    except (NotAllowed, InvalidState) as exc:
        print(f"refused: {exc}", file=out)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
