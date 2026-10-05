# path: book/projects/ragkit/ragkit/eval/chunk_size.py
"""Compare chunking configurations on evidence the answers actually need.

For each configuration the script chunks the corpus and reports:

- chunks, mean and p95 tokens per chunk, and index overhead (indexed tokens / corpus tokens - 1),
  which is what overlap costs you in storage and embedding spend;
- span integrity: the share of questions whose evidence spans all sit inside one chunk,
  independent of retrieval. A configuration that splits a table row from its header scores low
  here no matter how good the retriever is;
- recall@k: the share of questions whose evidence spans are all present in the top-k retrieved
  chunks (for parent-child, after expanding hits to parents);
- MRR of the first retrieved chunk that contains any evidence span;
- context tokens@k: what the generator would have to read.

Retrieval uses a small TF-IDF ranker so the script runs offline with no model. It is a stand-in:
Chapter 12 builds the real lexical, dense, and hybrid retrievers, and the same harness accepts
any `retriever_factory`. Pass `--embeddings` to include the semantic chunker using the embedding
client configured through aie_core settings (or `fake` for the offline hashing fake).

Usage:
    python -m ragkit.eval.chunk_size --docs ../shared-data/docs --gold eval_data/chunk_eval_gold.jsonl --k 3
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np
from pydantic import BaseModel

from ..chunking import (
    BaseChunker,
    FixedTokenChunker,
    MarkdownSectionChunker,
    ParentChildChunker,
    RecursiveChunker,
    SemanticChunker,
    SentenceChunker,
)
from ..documents import Chunk, Document
from ..pipeline import load_documents
from ..tokenizers import RegexTokenizer


class EvidenceQuestion(BaseModel):
    id: str
    doc_id: str
    question: str
    evidence: list[str]  # every span is required to answer


class ConfigResult(BaseModel):
    name: str
    fingerprint: str
    chunks: int
    mean_tokens: float
    p95_tokens: float
    index_overhead: float
    span_integrity: float
    recall_at_k: float
    mrr: float
    context_tokens_at_k: float


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[int]: ...


_MARKUP = re.compile(r"[*`]")
_WS = re.compile(r"\s+")


def canon(text: str) -> str:
    """Comparison form: no emphasis or code markup, lowercase, collapsed whitespace."""
    return _WS.sub(" ", _MARKUP.sub("", text)).strip().lower()


_STOP = frozenset("a an the of to for and or in on at is are be by with what which when how do does i my can it".split())
_TOKEN = re.compile(r"[a-z0-9]+(?:[-/][a-z0-9]+)*")


def _terms(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class TfidfRetriever:
    """Sublinear TF-IDF with cosine similarity. Deterministic; ties break by chunk order."""

    def __init__(self, texts: Sequence[str]) -> None:
        docs = [Counter(_terms(t)) for t in texts]
        df: Counter[str] = Counter()
        for d in docs:
            df.update(d.keys())
        n = len(docs)
        self.vocab = {term: i for i, term in enumerate(sorted(df))}
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1.0 for t in sorted(df)])
        self.matrix = np.zeros((n, len(self.vocab)))
        for row, d in enumerate(docs):
            for term, tf in d.items():
                self.matrix[row, self.vocab[term]] = (1 + math.log(tf)) * self.idf[self.vocab[term]]
        norms = np.linalg.norm(self.matrix, axis=1)
        norms[norms == 0] = 1.0
        self.matrix /= norms[:, None]

    def search(self, query: str, k: int) -> list[int]:
        q = np.zeros(len(self.vocab))
        for term, tf in Counter(_terms(query)).items():
            if term in self.vocab:
                q[self.vocab[term]] = (1 + math.log(tf)) * self.idf[self.vocab[term]]
        if not q.any():
            return []
        scores = self.matrix @ (q / np.linalg.norm(q))
        order = np.argsort(-scores, kind="stable")
        return [int(i) for i in order[:k] if scores[i] > 0]


def _contains_all(text: str, spans: list[str]) -> bool:
    c = canon(text)
    return all(canon(s) in c for s in spans)


def _contains_any(text: str, spans: list[str]) -> bool:
    c = canon(text)
    return any(canon(s) in c for s in spans)


def evaluate_chunker(
    name: str,
    chunker: BaseChunker,
    docs: list[Document],
    questions: list[EvidenceQuestion],
    *,
    k: int = 3,
    retriever_factory: Callable[[Sequence[str]], Retriever] = TfidfRetriever,
    tokenizer: RegexTokenizer | None = None,
) -> ConfigResult:
    tok = tokenizer or RegexTokenizer()
    chunks: list[Chunk] = []
    for d in docs:
        chunks.extend(chunker.chunk(d))
    parents = {c.id: c for c in chunks if c.role == "parent"}
    searchable = [c for c in chunks if c.role != "parent"]
    if not searchable:
        raise ValueError(f"{name}: chunker produced no searchable chunks")
    # what the generator reads for a hit: the parent if there is one, else the chunk itself
    readable = [parents.get(c.parent_id or "", c) for c in searchable]

    corpus_tokens = sum(tok.count(d.text) for d in docs) or 1
    leaf_tokens = [c.token_count for c in searchable]
    retriever = retriever_factory([c.embedding_text() for c in searchable])

    by_doc: dict[str, list[Chunk]] = {}
    for c in readable:
        by_doc.setdefault(c.doc_id, []).append(c)

    integrity = recall = rr_sum = ctx_sum = 0.0
    for q in questions:
        if any(_contains_all(c.text, q.evidence) for c in by_doc.get(q.doc_id, [])):
            integrity += 1
        hits = retriever.search(q.question, k * 4)
        seen: set[str] = set()
        context: list[Chunk] = []
        for i in hits:  # dedupe parents so k means k distinct passages
            if readable[i].id not in seen:
                seen.add(readable[i].id)
                context.append(readable[i])
            if len(context) == k:
                break
        joined = "\n".join(c.text for c in context)
        if _contains_all(joined, q.evidence):
            recall += 1
        for rank, c in enumerate(context, start=1):
            if _contains_any(c.text, q.evidence):
                rr_sum += 1.0 / rank
                break
        ctx_sum += sum(tok.count(c.text) for c in context)

    n = len(questions) or 1
    return ConfigResult(
        name=name,
        fingerprint=chunker.fingerprint(),
        chunks=len(searchable),
        mean_tokens=float(np.mean(leaf_tokens)),
        p95_tokens=float(np.percentile(leaf_tokens, 95)),
        index_overhead=sum(leaf_tokens) / corpus_tokens - 1.0,
        span_integrity=integrity / n,
        recall_at_k=recall / n,
        mrr=rr_sum / n,
        context_tokens_at_k=ctx_sum / n,
    )


def default_grid(embeddings: object | None = None) -> list[tuple[str, BaseChunker]]:
    tok = RegexTokenizer()
    grid: list[tuple[str, BaseChunker]] = [
        ("fixed-64/0", FixedTokenChunker(64, 0, tokenizer=tok)),
        ("fixed-128/16", FixedTokenChunker(128, 16, tokenizer=tok)),
        ("fixed-256/32", FixedTokenChunker(256, 32, tokenizer=tok)),
        ("fixed-512/64", FixedTokenChunker(512, 64, tokenizer=tok)),
        ("recursive-200", RecursiveChunker(200, tokenizer=tok)),
        ("sentence-150/1", SentenceChunker(150, 1, tokenizer=tok)),
        ("section-300", MarkdownSectionChunker(300, tokenizer=tok)),
        ("section-48", MarkdownSectionChunker(48, tokenizer=tok)),
        ("section-48/no-repeat", MarkdownSectionChunker(48, repeat_table_header=False, tokenizer=tok)),
        ("parent-child-800/96", ParentChildChunker(
            MarkdownSectionChunker(800, tokenizer=tok), SentenceChunker(96, 0, tokenizer=tok), tokenizer=tok)),
    ]
    if embeddings is not None:
        grid.append(("semantic-p90", SemanticChunker(embeddings, max_tokens=300, tokenizer=tok)))  # type: ignore[arg-type]
    return grid


def load_questions(path: str | Path) -> list[EvidenceQuestion]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [EvidenceQuestion.model_validate_json(line) for line in lines if line.strip()]


def format_markdown(results: list[ConfigResult], k: int) -> str:
    head = f"| config | chunks | mean tok | p95 tok | overhead | span integrity | recall@{k} | MRR | ctx tok@{k} |"
    lines = [head, "|" + "---|" * 9]
    for r in results:
        overhead = 0.0 if abs(r.index_overhead) < 0.005 else r.index_overhead
        lines.append(
            f"| {r.name} | {r.chunks} | {r.mean_tokens:.0f} | {r.p95_tokens:.0f} | {overhead:+.0%} | "
            f"{r.span_integrity:.2f} | {r.recall_at_k:.2f} | {r.mrr:.2f} | {r.context_tokens_at_k:.0f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--docs", required=True, help="directory of source documents")
    ap.add_argument("--gold", required=True, help="JSONL of EvidenceQuestion")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--format", choices=["md", "json"], default="md")
    ap.add_argument("--embeddings", choices=["none", "fake", "settings"], default="none")
    args = ap.parse_args(argv)

    embeddings = None
    if args.embeddings == "fake":
        from aie_core.embeddings import FakeEmbeddings

        embeddings = FakeEmbeddings(256)
    elif args.embeddings == "settings":
        from aie_core import make_embedding_client

        embeddings = make_embedding_client()

    report = load_documents(args.docs, root=args.docs)
    questions = load_questions(args.gold)
    missing = {q.doc_id for q in questions} - {d.id for d in report.documents}
    if missing:
        print(f"gold references unknown documents: {sorted(missing)}", file=sys.stderr)
        return 2
    results = [evaluate_chunker(n, c, report.documents, questions, k=args.k) for n, c in default_grid(embeddings)]
    if args.format == "json":
        print(json.dumps([r.model_dump() for r in results], indent=2))
    else:
        print(f"corpus: {len(report.documents)} documents, {len(questions)} questions, k={args.k}\n")
        print(format_markdown(results, args.k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
