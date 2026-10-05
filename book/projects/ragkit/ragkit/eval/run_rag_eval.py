# path: book/projects/ragkit/ragkit/eval/run_rag_eval.py
"""End-to-end RAG evaluation: run two configurations on the gold set, isolate stages, gate, report.

The script needs no model and no network by default. It builds a small lexical RAG system over
`shared-data/docs` (TF-IDF retrieval with ACL filtering, an optional toy reranker, and an
extractive generator that quotes the best-matching sentence or abstains), so the full pipeline
of Chapter 14 can be run and inspected offline. Any real system plugs in through `make_target`:
a function that turns a question and a principal into a `RagOutput`.

Usage:
    python -m ragkit.eval.run_rag_eval --baseline lexical-k5-pack1 --candidate lexical-rerank-k5-pack4
    python -m ragkit.eval.run_rag_eval --candidate lexical-k5-pack4-noacl      # watch the gate fail
    python -m ragkit.eval.run_rag_eval --judges llm                            # add LLM judges (needs keys)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from evalkit import Dataset, EvalCase, GateConfig, GateResult, Run, RunVersions, TargetResult, evaluate_gate, run_target
from pydantic import BaseModel

from ..chunking import MarkdownSectionChunker
from ..documents import Chunk, Document
from ..pipeline import chunk_documents, load_documents
from ..retrieval.types import Principal, RetrievalQuery, RetrievalResult, ScoredChunk, visible
from .chunk_size import TfidfRetriever
from .rag_dataset import DEFAULT_GOLD_PATH, RagOutput, content_words, load_gold_dataset, rag_input
from .rag_metrics import answer_evaluator, retrieval_evaluator
from .rag_report import render_rag_report
from .stage_isolation import StageDiagnosis, diagnose_run

DEFAULT_DOCS = Path(__file__).resolve().parents[3] / "shared-data" / "docs"


# ============================================================================ a small offline RAG system
class LexicalRetriever:
    """TF-IDF candidates under ACL filtering, optional term-coverage rerank. Records a stage trace."""

    def __init__(self, chunks: Sequence[Chunk], *, candidate_k: int = 20, enforce_acl: bool = True,
                 rerank: bool = False) -> None:
        self.chunks = list(chunks)
        self.index = TfidfRetriever([c.embedding_text() for c in self.chunks])
        self.candidate_k = candidate_k
        self.enforce_acl = enforce_acl
        self.rerank = rerank

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        t0 = time.perf_counter()
        ranked = [self.chunks[i] for i in self.index.search(query.text, len(self.chunks))]
        allowed = [c for c in ranked if visible(c, query.principal)] if self.enforce_acl else ranked
        candidates = allowed[: self.candidate_k]
        stages: list[dict[str, Any]] = [{"name": "lexical", "kind": "candidate", "k": self.candidate_k,
                                         "chunk_ids": [c.id for c in candidates]}]
        if self.rerank:
            q = set(content_words(query.text))

            def coverage(c: Chunk) -> float:
                words = set(content_words(c.embedding_text()))
                return len(q & words) / len(q) if q else 0.0

            candidates = sorted(candidates, key=coverage, reverse=True)  # stable: ties keep lexical order
            stages.append({"name": "rerank", "kind": "rerank", "k": query.k, "chunk_ids": [c.id for c in candidates[: query.k]]})
        hits = [ScoredChunk(chunk=c, score=1.0 / (i + 1), stage=stages[-1]["name"], rank=i + 1)
                for i, c in enumerate(candidates[: query.k])]
        trace = {"stages": stages, "acl_filtered": len(ranked) - len(allowed),
                 "latency_ms": (time.perf_counter() - t0) * 1000}
        return RetrievalResult(query=query, hits=hits, trace=trace)


_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


class ExtractiveGenerator:
    """Packs the top `pack_k` hits, answers with the best-overlapping sentence, or abstains.

    A stand-in for Chapter 13's grounded generator with the same observable contract: packed
    evidence, cited chunk ids, and an abstention flag. It is faithful by construction (it only
    quotes), which makes it a useful control: any faithfulness failure it shows is a judge bug.
    """

    def __init__(self, *, pack_k: int = 3, min_overlap: float = 0.4) -> None:
        self.pack_k = pack_k
        self.min_overlap = min_overlap

    def answer(self, question: str, result: RetrievalResult) -> RagOutput:
        packed = [h.chunk for h in result.hits[: self.pack_k]]
        q = set(content_words(question))
        best: tuple[float, str, str] = (0.0, "", "")
        for c in packed:
            for sent in _SENT.split(c.text):
                words = set(content_words(sent))
                if len(words) < 3 or not q:
                    continue
                score = len(q & words) / len(q)
                if score > best[0]:
                    best = (score, sent.strip(), c.id)
        if best[0] < self.min_overlap:
            return RagOutput(answer="", abstained=True, packed_chunks=packed, retrieval=result)
        return RagOutput(answer=best[1], cited_chunk_ids=[best[2]], packed_chunks=packed, retrieval=result,
                         metadata={"overlap": round(best[0], 3)})


class RagConfig(BaseModel):
    name: str
    k: int = 5
    candidate_k: int = 20
    pack_k: int = 3
    rerank: bool = False
    enforce_acl: bool = True
    min_overlap: float = 0.4


PRESETS: dict[str, RagConfig] = {
    c.name: c
    for c in [
        RagConfig(name="lexical-k5-pack1", k=5, pack_k=1),
        RagConfig(name="lexical-k5-pack2", k=5, pack_k=2),
        RagConfig(name="lexical-k5-pack4", k=5, pack_k=4),
        RagConfig(name="lexical-rerank-k5-pack4", k=5, pack_k=4, rerank=True),
        RagConfig(name="lexical-k5-pack4-noacl", k=5, pack_k=4, enforce_acl=False),
    ]
}


class Corpus(BaseModel):
    documents: list[Document]
    chunks: list[Chunk]

    def doc_map(self) -> dict[str, Document]:
        return {d.id: d for d in self.documents}

    def doc_visible(self) -> Callable[[str, Principal], bool]:
        docs = self.doc_map()
        return lambda doc_id, p: doc_id in docs and docs[doc_id].visible_to(p.groups, p.tenant)


def load_corpus(docs_dir: str | Path = DEFAULT_DOCS, *, max_tokens: int = 200) -> Corpus:
    report = load_documents(sorted(Path(docs_dir).glob("*.md")), root=Path(docs_dir).parent)
    chunks = chunk_documents(report.documents, MarkdownSectionChunker(max_tokens=max_tokens))
    return Corpus(documents=report.documents, chunks=chunks)


# ============================================================================ wiring into evalkit
RagSystem = Callable[[str, Principal], RagOutput]


def build_system(corpus: Corpus, config: RagConfig) -> RagSystem:
    retriever = LexicalRetriever(corpus.chunks, candidate_k=config.candidate_k, enforce_acl=config.enforce_acl,
                                 rerank=config.rerank)
    generator = ExtractiveGenerator(pack_k=config.pack_k, min_overlap=config.min_overlap)

    def system(question: str, principal: Principal) -> RagOutput:
        result = retriever.retrieve(RetrievalQuery(text=question, principal=principal, k=config.k))
        return generator.answer(question, result)

    return system


def make_target(system: RagSystem) -> Callable[[EvalCase], TargetResult]:
    """Adapt any `(question, principal) -> RagOutput` callable to an evalkit target."""

    def target(case: EvalCase) -> TargetResult:
        inp = rag_input(case)
        out = system(inp.question, inp.principal)
        return TargetResult(output=out.model_dump(mode="json"))

    return target


DEFAULT_GATE = GateConfig.from_dict({
    "name": "rag-release",
    "metrics": [
        {"metric": "no_permission_leak", "must_pass_all": True},
        {"metric": "recall@5", "max_regression": 0.05},
        {"metric": "evidence_packed", "max_regression": 0.05},
        {"metric": "abstention_correct", "max_regression": 0.05},
        {"metric": "citations_valid", "must_pass_all": True},
    ],
    "slices": [{"metric": "recall@5", "max_regression": 0.15, "min_n": 5}],
    "critical": [{"tag": "forbidden-doc", "metric": "no_permission_leak"}],
    "max_error_rate": 0.0,
    "max_evaluator_errors": 0,
})


class EvalOutcome(BaseModel):
    run: Run
    diagnoses: list[StageDiagnosis]


def evaluate_system(
    system: RagSystem,
    dataset: Dataset,
    *,
    name: str,
    corpus: Corpus | None = None,
    judges: Sequence[Any] = (),
    extra_versions: dict[str, str] | None = None,
    concurrency: int = 4,
) -> EvalOutcome:
    evaluators = [retrieval_evaluator(), answer_evaluator(), *judges]
    run = run_target(make_target(system), dataset, evaluators=evaluators, concurrency=concurrency,
                     versions=RunVersions(target=name, extra=extra_versions or {}))
    diagnoses = diagnose_run(
        run, dataset,
        corpus=corpus.doc_map() if corpus else None,
        doc_visible=corpus.doc_visible() if corpus else None,
    )
    return EvalOutcome(run=run, diagnoses=diagnoses)


def compare_configs(
    baseline: RagConfig,
    candidate: RagConfig,
    *,
    corpus: Corpus,
    dataset: Dataset,
    judges: Sequence[Any] = (),
    gate: GateConfig = DEFAULT_GATE,
) -> tuple[EvalOutcome, EvalOutcome, GateResult, str]:
    extra = {"chunker": "section-200", "corpus_docs": str(len(corpus.documents))}
    base = evaluate_system(build_system(corpus, baseline), dataset, name=baseline.name, corpus=corpus, judges=judges,
                           extra_versions={**extra, "config": baseline.model_dump_json()})
    cand = evaluate_system(build_system(corpus, candidate), dataset, name=candidate.name, corpus=corpus, judges=judges,
                           extra_versions={**extra, "config": candidate.model_dump_json()})
    result = evaluate_gate(gate, cand.run, base.run)
    report = render_rag_report(cand.run, dataset, diagnoses=cand.diagnoses, baseline=base.run,
                               baseline_diagnoses=base.diagnoses, gate=result,
                               title=f"RAG evaluation, {baseline.name} vs {candidate.name}")
    return base, cand, result, report


# ============================================================================ CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", default=str(DEFAULT_DOCS))
    ap.add_argument("--gold", default=str(DEFAULT_GOLD_PATH))
    ap.add_argument("--baseline", default="lexical-k5-pack1", choices=sorted(PRESETS))
    ap.add_argument("--candidate", default="lexical-rerank-k5-pack4", choices=sorted(PRESETS))
    ap.add_argument("--judges", default="none", choices=["none", "llm"],
                    help="llm: add faithfulness, rubric coverage, relevance judges via aie_core settings")
    ap.add_argument("--out", default="out/rag_eval")
    args = ap.parse_args(argv)

    corpus = load_corpus(args.docs)
    dataset = load_gold_dataset(args.gold)
    judges: list[Any] = []
    if args.judges == "llm":
        from aie_core import make_llm_client

        from .rag_judges import default_judges

        judges = default_judges(make_llm_client())
    base, cand, gate, report = compare_configs(PRESETS[args.baseline], PRESETS[args.candidate],
                                               corpus=corpus, dataset=dataset, judges=judges)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(report, encoding="utf-8")
    base.run.save_json(out / "baseline_run.json")
    cand.run.save_json(out / "candidate_run.json")
    (out / "diagnoses.json").write_text(
        json.dumps([d.model_dump(mode="json") for d in cand.diagnoses], indent=2), encoding="utf-8")
    print(f"gate {'PASS' if gate.passed else 'FAIL'}; report: {out / 'report.md'}")
    for f in gate.failures:
        print(f"  FAIL {f.name}: observed {f.observed}, threshold {f.threshold} {f.detail}")
    return 0 if gate.passed else 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["LexicalRetriever", "ExtractiveGenerator", "RagConfig", "PRESETS", "Corpus", "load_corpus",
           "build_system", "make_target", "DEFAULT_GATE", "EvalOutcome", "evaluate_system", "compare_configs", "main"]
