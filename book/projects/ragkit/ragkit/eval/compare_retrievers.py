# path: book/projects/ragkit/ragkit/eval/compare_retrievers.py
"""Compare retrieval configurations on the shared gold set.

    python -m ragkit.eval.compare_retrievers                      # offline, deterministic
    python -m ragkit.eval.compare_retrievers --by-tag --k 1 3 5 10
    python -m ragkit.eval.compare_retrievers --embeddings settings --reranker llm   # real providers

Configurations: bm25, dense, hybrid (RRF), hybrid+rerank. Each gold question is run under its own
principal (user_groups, tenant). Metrics are document-level over the top-k chunks:

- hit@k: at least one required doc appears in the top k chunks.
- recall@k: fraction of required docs that appear in the top k chunks.
- MRR: reciprocal rank of the first chunk from a required doc (0 if none within max k).
- cand_recall: recall within the reranker's input (rerank_k fused candidates), the ceiling a
  reranker can reach. Reported for pipeline configurations.
- leaks: questions tagged `forbidden-doc` whose restricted doc appeared anywhere in the results.

Questions tagged `forbidden-doc` are scored inverted: they pass (hit = recall = MRR = 1) only when
no required doc is retrieved, because the user may not read it and the correct behavior is to
abstain. Chapter 14 builds the general evaluation harness; this script is the Chapter 12 bench.

Offline dense retrieval uses FakeEmbeddings in vocabulary mode over the corpus's shared content
words (document frequency >= 2, no digit-bearing tokens). That is a deliberate stand-in for a
dense model's known weakness: it matches topical vocabulary and blurs rare identifiers.
Numbers it produces are illustrative of mechanisms, not of any real embedding model.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from aie_core.embeddings import EmbeddingClient, FakeEmbeddings

from ..chunking import MarkdownSectionChunker
from ..documents import Chunk, Document
from ..pipeline import chunk_documents, load_documents
from ..retrieval.bm25 import BM25Index, BM25Tokenizer
from ..retrieval.common import indexed_text
from ..retrieval.dense import DenseRetriever
from ..retrieval.pipeline import RetrievalPipeline, stage_candidates
from ..retrieval.rerank import CrossEncoderReranker, LexicalOverlapReranker, LLMReranker
from ..retrieval.types import Principal, Reranker, RetrievalQuery, Retriever

SHARED_DATA = Path(__file__).resolve().parents[3] / "shared-data"


# ----------------------------------------------------------------------------- corpus and gold
def load_corpus(docs_dir: Path = SHARED_DATA / "docs", max_tokens: int = 300) -> tuple[list[Document], list[Chunk]]:
    """Chapter 11's default: section-aware chunks of up to 300 tokens."""
    report = load_documents(docs_dir, root=docs_dir.parent)
    docs = report.documents
    return docs, chunk_documents(docs, MarkdownSectionChunker(max_tokens))


@dataclass
class GoldQuestion:
    id: str
    question: str
    required_doc_ids: list[str]
    user_groups: list[str]
    tenant: str
    tags: list[str] = field(default_factory=list)

    @property
    def forbidden(self) -> bool:
        return "forbidden-doc" in self.tags

    @property
    def principal(self) -> Principal:
        return Principal(user_id=f"gold-{self.id}", tenant=self.tenant, groups=self.user_groups)


def load_gold(path: Path = SHARED_DATA / "eval" / "retrieval_gold.jsonl") -> list[GoldQuestion]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out.append(GoldQuestion(**{k: row[k] for k in GoldQuestion.__dataclass_fields__ if k in row}))
    return out


def semantic_vocabulary(chunks: Iterable[Chunk], min_df: int = 2) -> list[str]:
    """Content words shared by at least `min_df` chunks, without digits or stopwords."""
    tok = BM25Tokenizer(fold_plurals=False)
    df: Counter[str] = Counter()
    for c in chunks:
        words = {w for w in tok.tokenize(indexed_text(c)) if w.isalpha() and len(w) > 2}
        df.update(words)
    return sorted(w for w, n in df.items() if n >= min_df)


# ----------------------------------------------------------------------------- configurations
@dataclass
class Bench:
    docs: list[Document]
    chunks: list[Chunk]
    bm25: BM25Index
    dense: DenseRetriever


def build_bench(embeddings: EmbeddingClient | None = None, docs_dir: Path = SHARED_DATA / "docs") -> Bench:
    docs, chunks = load_corpus(docs_dir)
    bm25 = BM25Index()
    bm25.add(chunks)
    emb = embeddings or FakeEmbeddings(vocabulary=semantic_vocabulary(chunks))
    dense = DenseRetriever(emb)
    dense.index(chunks)
    return Bench(docs, chunks, bm25, dense)


def configurations(bench: Bench, reranker: Reranker, *, candidate_k: int = 30, rerank_k: int = 20,
                   final_k: int = 10) -> dict[str, Retriever | RetrievalPipeline]:
    both = {"bm25": bench.bm25, "dense": bench.dense}
    kw = dict(candidate_k=candidate_k, rerank_k=rerank_k, final_k=final_k)
    return {
        "bm25": bench.bm25,
        "dense": bench.dense,
        "hybrid": RetrievalPipeline(both, **kw),
        "hybrid+rerank": RetrievalPipeline(both, reranker=reranker, **kw),
    }


# ----------------------------------------------------------------------------- metrics
def doc_ranks(doc_ids_by_chunk: Sequence[str]) -> dict[str, int]:
    """First 1-based chunk position at which each doc appears."""
    ranks: dict[str, int] = {}
    for i, d in enumerate(doc_ids_by_chunk, start=1):
        ranks.setdefault(d, i)
    return ranks


def score_question(q: GoldQuestion, chunk_docs: Sequence[str], ks: Sequence[int],
                   candidate_docs: Sequence[str] | None = None) -> dict[str, float]:
    ranks = doc_ranks(chunk_docs)
    req = q.required_doc_ids
    found_any = [ranks[d] for d in req if d in ranks]
    row: dict[str, float] = {}
    if q.forbidden:  # inverted: the restricted document must not appear at all
        leaked = bool(found_any)
        for k in ks:
            row[f"hit@{k}"] = row[f"recall@{k}"] = 0.0 if leaked else 1.0
        row["mrr"] = 0.0 if leaked else 1.0
        row["leak"] = 1.0 if leaked else 0.0
        if candidate_docs is not None:
            row["cand_recall"] = 0.0 if any(d in set(candidate_docs) for d in req) else 1.0
        return row
    for k in ks:
        hits = [d for d in req if d in ranks and ranks[d] <= k]
        row[f"hit@{k}"] = 1.0 if hits else 0.0
        row[f"recall@{k}"] = len(hits) / len(req)
    row["mrr"] = 1.0 / min(found_any) if found_any else 0.0
    row["leak"] = 0.0
    if candidate_docs is not None:
        cand = set(candidate_docs)
        row["cand_recall"] = sum(d in cand for d in req) / len(req)
    return row


def run_config(name: str, system: Retriever | RetrievalPipeline, gold: Sequence[GoldQuestion], ks: Sequence[int],
               chunk_doc: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    for q in gold:
        query = RetrievalQuery(text=q.question, principal=q.principal, k=max(ks))
        result = system.retrieve(query)
        chunk_docs = [h.chunk.doc_id for h in result.hits]
        cand = None
        if isinstance(system, RetrievalPipeline):
            cand = [chunk_doc[c] for c in stage_candidates(result, "fusion")]
        latency = result.trace.get("latency_ms")
        total = latency.get("total") if isinstance(latency, dict) else latency
        row = {"config": name, "id": q.id, "tags": q.tags, "latency_ms": float(total or 0.0),
               **score_question(q, chunk_docs, ks, cand)}
        rows.append(row)
    return rows


def summarize(rows: Sequence[dict[str, Any]], ks: Sequence[int]) -> dict[str, float]:
    cols = [f"hit@{k}" for k in ks] + [f"recall@{k}" for k in ks if k != 1] + ["mrr"]
    out = {c: round(statistics.fmean(r[c] for r in rows), 3) for c in cols}
    if any("cand_recall" in r for r in rows):
        out["cand_recall"] = round(statistics.fmean(r.get("cand_recall", 0.0) for r in rows), 3)
    out["leaks"] = int(sum(r["leak"] for r in rows))
    out["p50_ms"] = round(statistics.median(r["latency_ms"] for r in rows), 2)
    out["n"] = len(rows)
    return out


def render_table(summaries: dict[str, dict[str, float]]) -> str:
    cols = sorted({c for s in summaries.values() for c in s}, key=_col_order)
    lines = ["| config | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, s in summaries.items():
        lines.append(f"| {name} | " + " | ".join(str(s.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def _col_order(c: str) -> tuple[int, int, str]:
    group = {"hit": 0, "recall": 1, "mrr": 2, "cand_recall": 3, "leaks": 4, "p50_ms": 5, "n": 6}
    head = c.split("@")[0]
    k = int(c.split("@")[1]) if "@" in c else 0
    return (group.get(head, 9), k, c)


def make_reranker(kind: str) -> Reranker:
    if kind == "lexical":
        return LexicalOverlapReranker()
    if kind == "cross-encoder":
        from ..retrieval.settings import RetrievalSettings

        return CrossEncoderReranker(RetrievalSettings().cross_encoder_model)
    if kind == "llm":
        from aie_core.settings import make_llm_client

        return LLMReranker(make_llm_client())
    raise ValueError(f"unknown reranker {kind!r}")


def main(argv: Sequence[str] | None = None, out: Callable[[str], None] = print) -> dict[str, dict[str, float]]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--docs", type=Path, default=SHARED_DATA / "docs")
    ap.add_argument("--gold", type=Path, default=SHARED_DATA / "eval" / "retrieval_gold.jsonl")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--embeddings", choices=["fake-vocab", "settings"], default="fake-vocab")
    ap.add_argument("--reranker", choices=["lexical", "cross-encoder", "llm"], default="lexical")
    ap.add_argument("--candidate-k", type=int, default=30)
    ap.add_argument("--rerank-k", type=int, default=20)
    ap.add_argument("--by-tag", action="store_true", help="also print hit@1 per gold tag")
    ap.add_argument("--json", type=Path, help="write per-question rows to this file")
    args = ap.parse_args(argv)

    embeddings = None
    if args.embeddings == "settings":
        from aie_core.settings import make_embedding_client

        embeddings = make_embedding_client()
    bench = build_bench(embeddings, args.docs)
    gold = load_gold(args.gold)
    ks = sorted(set(args.k))
    chunk_doc = {c.id: c.doc_id for c in bench.chunks}
    systems = configurations(bench, make_reranker(args.reranker), candidate_k=args.candidate_k,
                             rerank_k=args.rerank_k, final_k=max(ks))
    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, float]] = {}
    for name, system in systems.items():
        rows = run_config(name, system, gold, ks, chunk_doc)
        all_rows.extend(rows)
        summaries[name] = summarize(rows, ks)
    out(f"corpus: {len(bench.docs)} docs, {len(bench.chunks)} chunks; gold: {len(gold)} questions "
        f"({sum(q.forbidden for q in gold)} forbidden-doc, scored inverted); reranker: {args.reranker}")
    out(render_table(summaries))
    if args.by_tag:
        tags = sorted({t for q in gold for t in q.tags})
        out("\nhit@1 by tag")
        out("| tag | n | " + " | ".join(summaries) + " |")
        out("|---|---|" + "---|" * len(summaries))
        for t in tags:
            cells = []
            for name in summaries:
                sel = [r for r in all_rows if r["config"] == name and t in r["tags"]]
                cells.append(str(round(statistics.fmean(r["hit@1"] for r in sel), 2)) if sel else "")
            n = sum(t in q.tags for q in gold)
            out(f"| {t} | {n} | " + " | ".join(cells) + " |")
    if args.json:
        args.json.write_text(json.dumps(all_rows, indent=1), encoding="utf-8")
    return summaries


if __name__ == "__main__":  # pragma: no cover
    summaries = main()
    sys.exit(1 if any(s["leaks"] for s in summaries.values()) else 0)
