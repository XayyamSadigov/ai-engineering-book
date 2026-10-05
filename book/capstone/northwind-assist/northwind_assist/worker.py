# path: book/capstone/northwind-assist/northwind_assist/worker.py
"""The asynchronous tier (Chapter 28): `northwind-assist-worker --role ingest|eval`.

ingest  Project 3's ingestion worker, unchanged: leases jobs from the Redis queue, parses,
        chunks, annotates authority, embeds, writes pgvector and BM25 snapshots, bumps cache
        generations. The API (NA_KNOWLEDGE_BACKEND=p3) reads the same registry and indexes.
eval    one offline evaluation and release-gate run (the nightly drift check, Chapter 32);
        exit code 0 pass, 1 gate failure, 2 setup error, so a scheduler can alert on it.
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", choices=["ingest", "eval"], default="ingest")
    ap.add_argument("--out", default="/data/eval")
    args = ap.parse_args(argv)
    if args.role == "ingest":
        from rag_assistant.ingestion.worker import main as ingest_main  # noqa: PLC0415

        return ingest_main()
    from .evaluation.run_eval import main as eval_main  # noqa: PLC0415

    return eval_main(["--out", args.out])


if __name__ == "__main__":
    sys.exit(main())
