# path: book/projects/p2-semantic-search/semsearch/cli.py
"""Command line: ingest, search, eval, ann-check, bench-ann, bench-filter, serve."""
from __future__ import annotations

import argparse
import json
import sys
from statistics import mean

from aie_core.observability import get_tracer
from aie_core.settings import Settings as CoreSettings

from .config import SearchSettings, embedder_dimensions, make_embedder, make_store, namespace_for
from .domain.metrics import overlap_at_k
from .eval.run_eval import default_gold_path, evaluate, load_gold
from .ingest.loader import load_corpus
from .ingest.pipeline import ingest
from .service import Principal, SearchService


def _service() -> SearchService:
    core, search = CoreSettings(), SearchSettings()
    embedder = make_embedder(core, search)
    store = make_store(embedder_dimensions(embedder), search)
    return SearchService(
        store, embedder, namespace_for(embedder, search),
        default_k=search.search_default_k, max_k=search.search_max_k, tracer=get_tracer(core),
    )


def cmd_ingest(args: argparse.Namespace) -> int:
    search = SearchSettings()
    svc = _service()
    docs = load_corpus(args.docs or search.docs_dir)
    report = ingest(docs, svc.store, svc.embedder, svc.namespace, max_chars=search.chunk_max_chars, prune=args.prune, tracer=svc.tracer)
    print(report.summary())
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    svc = _service()
    principal = Principal(user_id="cli", tenant=args.tenant, groups=tuple(g for g in args.groups.split(",") if g))
    result = svc.search(args.query, principal, k=args.k, tags=args.tags.split(",") if args.tags else None)
    for rank, h in enumerate(result.hits, start=1):
        first_line = h.text.splitlines()[0][:80]
        print(f"{rank:>2}. {h.score:.3f}  {h.doc_id}#{h.ordinal}  [{h.tenant}]  {first_line}")
    print(f"({len(result.hits)} hits in {result.took_ms} ms, namespace {result.namespace})")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    svc = _service()
    path = args.gold or str(default_gold_path())
    report = evaluate(svc, load_gold(path), gold_path=path)
    print(json.dumps(report.to_dict(), indent=2) if args.json else report.format())
    # A leak fails the run regardless of recall: this is the CI gate.
    return 1 if report.leaks else 0


def cmd_ann_check(args: argparse.Namespace) -> int:
    """Measure the store's ANN recall against its own exact scan, using gold questions as queries."""
    svc = _service()
    gold = load_gold(args.gold)
    overlaps = []
    for row in gold:
        q = svc.embedder.embed_query(row.question)
        flt = svc.authorization_filter(Principal(user_id="ann", tenant=row.tenant, groups=tuple(row.user_groups)))
        approx = [h.id for h in svc.store.search(svc.namespace, q, args.k, flt)]
        exact = [h.id for h in svc.store.search(svc.namespace, q, args.k, flt, exact=True)]
        overlaps.append(overlap_at_k(approx, exact, args.k))
    print(f"ANN recall@{args.k} vs exact over {len(overlaps)} queries: {mean(overlaps):.3f} (min {min(overlaps):.3f})")
    return 0


def cmd_bench_ann(args: argparse.Namespace) -> int:
    from .bench import ann_sweep, format_sweep

    rows, exact_ms = ann_sweep(n=args.n, dims=args.dims, n_lists=args.lists)
    print(format_sweep(rows, exact_ms))
    return 0


def cmd_bench_filter(args: argparse.Namespace) -> int:
    from .bench import filter_experiment, format_filter

    print(format_filter(filter_experiment(n=args.n, candidates=args.candidates)))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("semsearch.api.app:app", host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="semsearch", description="Northwind semantic search (Project 2)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="load, chunk, embed, and index documents")
    s.add_argument("--docs", help="directory of .md files (default DOCS_DIR)")
    s.add_argument("--prune", action="store_true", help="delete indexed documents missing from the source")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("search", help="run one filtered query")
    s.add_argument("query")
    s.add_argument("--tenant", default="shared")
    s.add_argument("--groups", default="all")
    s.add_argument("--tags", default="")
    s.add_argument("--k", type=int, default=None)
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("eval", help="recall@k / MRR / ACL leakage against a gold file")
    s.add_argument("--gold", default=None)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("ann-check", help="ANN recall of the configured store versus its exact scan")
    s.add_argument("--gold", default=None)
    s.add_argument("--k", type=int, default=10)
    s.set_defaults(func=cmd_ann_check)

    s = sub.add_parser("bench-ann", help="IVF recall/latency sweep on synthetic vectors")
    s.add_argument("--n", type=int, default=20000)
    s.add_argument("--dims", type=int, default=64)
    s.add_argument("--lists", type=int, default=128)
    s.set_defaults(func=cmd_bench_ann)

    s = sub.add_parser("bench-filter", help="post-filter vs pre-filter under selective filters")
    s.add_argument("--n", type=int, default=20000)
    s.add_argument("--candidates", type=int, default=40)
    s.set_defaults(func=cmd_bench_filter)

    s = sub.add_parser("serve", help="run the FastAPI service")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
