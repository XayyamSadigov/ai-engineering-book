# path: book/projects/examples/ch04/prompts_cli.py
"""Command line for the prompt registry: verify, lock, render, compare.

    python prompts_cli.py verify                       # load every prompt, check prompts.lock
    python prompts_cli.py lock                         # rewrite prompts.lock after adding versions
    python prompts_cli.py render ticket.classify prod --vars vars.json
    python prompts_cli.py compare ticket.classify --baseline prod --candidate 1.2.0 \
        --cases cases/ticket_classify.jsonl --repeats 1 --max-token-growth 400

With LLM_PROVIDER=fake (the default) `compare` uses the simulated router in demo_model.py.
With a real provider configured it uses aie_core.make_llm_client(), i.e. a ModelGateway.
Exit code 0 means verified / gate passed, 1 means a problem was found.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from aie_core.observability import get_tracer
from aie_core.settings import Settings, make_llm_client

from prompts import PromptRegistry, compare, load_cases, render_report, run_suite

HERE = Path(__file__).parent
PROMPT_DIR = Path(os.environ.get("PROMPT_DIR", HERE / "prompt_files"))
LOCK_PATH = Path(os.environ.get("PROMPT_LOCK", HERE / "prompts.lock"))


def cmd_verify(registry: PromptRegistry, _: argparse.Namespace) -> int:
    lock = json.loads(LOCK_PATH.read_text()) if LOCK_PATH.exists() else {}
    problems = registry.verify_lock(lock)
    unlocked = sorted(set(registry.lock()) - set(lock))
    for p in problems:
        print(f"ERROR {p}")
    for key in unlocked:
        print(f"NOTE  {key} is new; run `prompts_cli.py lock` and commit prompts.lock")
    print(f"{sum(len(registry.versions(i)) for i in registry.ids())} prompt versions loaded")
    return 1 if problems else 0


def cmd_lock(registry: PromptRegistry, _: argparse.Namespace) -> int:
    old = json.loads(LOCK_PATH.read_text()) if LOCK_PATH.exists() else {}
    problems = registry.verify_lock(old)
    if problems:  # never let `lock` launder a mutated published version
        for p in problems:
            print(f"ERROR {p}")
        return 1
    LOCK_PATH.write_text(json.dumps(registry.lock(), indent=2, sort_keys=True) + "\n")
    print(f"wrote {LOCK_PATH.name}")
    return 0


def cmd_render(registry: PromptRegistry, args: argparse.Namespace) -> int:
    variables = json.loads(Path(args.vars).read_text()) if args.vars else {}
    rendered = registry.get(args.prompt_id, args.version).render(variables)
    print(f"# {rendered.ref}  hash={rendered.ref.content_hash[:16]}")
    for m in rendered.messages:
        print(f"\n=== {m.role.value} ===\n{m.text}")
    return 0


def cmd_compare(registry: PromptRegistry, args: argparse.Namespace) -> int:
    settings = Settings()
    if settings.llm_provider == "fake":
        from demo_model import simulated_router

        client = simulated_router()
        print("LLM_PROVIDER=fake: using the simulated router; results illustrate the harness only\n")
    else:
        client = make_llm_client(settings)
    tracer = get_tracer(settings)
    cases = load_cases(args.cases)
    base_v = registry.get(args.prompt_id, args.baseline)
    cand_v = registry.get(args.prompt_id, args.candidate)
    base = run_suite(base_v, client, cases, repeats=args.repeats, tracer=tracer)
    cand = run_suite(cand_v, client, cases, repeats=args.repeats, tracer=tracer)
    cmp = compare(base, cand)
    gate = cmp.gate(max_pass_rate_drop=args.max_drop, max_input_token_growth_pct=args.max_token_growth)
    print(render_report(cmp, cand, gate))
    return 0 if gate.ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    sub.add_parser("lock")
    r = sub.add_parser("render")
    r.add_argument("prompt_id")
    r.add_argument("version", nargs="?", default="latest")
    r.add_argument("--vars", help="JSON file with template variables")
    c = sub.add_parser("compare")
    c.add_argument("prompt_id")
    c.add_argument("--baseline", default="prod")
    c.add_argument("--candidate", required=True)
    c.add_argument("--cases", required=True)
    c.add_argument("--repeats", type=int, default=1)
    c.add_argument("--max-drop", type=float, default=0.0)
    c.add_argument("--max-token-growth", type=float, default=50.0)
    args = parser.parse_args(argv)
    registry = PromptRegistry.from_directory(PROMPT_DIR)
    handler = {"verify": cmd_verify, "lock": cmd_lock, "render": cmd_render, "compare": cmd_compare}[args.command]
    return handler(registry, args)


if __name__ == "__main__":
    sys.exit(main())
