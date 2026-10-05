# path: book/projects/examples/ch25/ci/run_suite.py
"""Run the fast evaluation suites and write one evalkit Run JSON per suite.

    python ci/run_suite.py --system candidate --out eval-out/runs
    python ci/run_suite.py --system baseline --out ci/baselines      # refresh committed baselines

`--system` selects the stand-in behavior (baseline, candidate, regressed). In a real project the
target comes from the code under test and this flag disappears: the pipeline always evaluates
the commit it runs on, and the baseline is the Run JSON stored by the last release.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taskevals.suites import SUITES, run_suite  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--system", default="candidate", choices=["baseline", "candidate", "regressed"])
    ap.add_argument("--suite", default="all", choices=["all", *SUITES])
    ap.add_argument("--out", type=Path, default=Path("eval-out/runs"))
    args = ap.parse_args(argv)
    names = list(SUITES) if args.suite == "all" else [args.suite]
    args.out.mkdir(parents=True, exist_ok=True)
    for name in names:
        run = run_suite(name, args.system)
        path = run.save_json(args.out / f"{name}.json")
        print(f"{name:15s} cases={len(run.by_case()):3d} errors={len(run.errors)} "
              f"evaluator_errors={run.evaluator_error_count} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
