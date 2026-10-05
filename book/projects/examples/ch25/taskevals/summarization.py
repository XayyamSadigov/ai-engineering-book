# path: book/projects/examples/ch25/taskevals/summarization.py
"""Summarization evaluators: key-fact coverage, faithfulness, compression ratio.

Case format:
    input    = {"source": "<full text>"}
    expected = {"key_facts": [["root cause", "expired certificate"], ...],   # each fact: any-of phrasings
                "must_keep": ["except contractors"],                           # qualifiers that must survive
                "compression": [0.05, 0.35]}                                   # allowed summary/source token ratio
output   = "<summary text>"

Coverage asks "did the summary keep what matters?", faithfulness asks "did it add or distort
anything?", compression asks "is it actually a summary?". They trade against each other: the
source itself has perfect coverage and faithfulness and a compression ratio of 1.0, which is
why compression must be gated, and a one-line summary is faithful and useless, which is why
coverage must be gated.
"""
from __future__ import annotations

from typing import Any

from aie_core.llm.tokens import count_tokens

from evalkit import EvalCase, Rubric, RubricLevel, Score
from evalkit.metrics import normalize_text

from .text_support import content_words, split_sentences, unsupported_atoms

FAITHFULNESS = Rubric(
    name="faithfulness",
    task=("Judge whether the candidate summary represents the source accurately: no contradictions, "
          "no changed numbers or names, no dropped qualifiers or exceptions, no claims absent from the source."),
    levels=[
        RubricLevel(score=0, description="contradicts the source or changes a number, name, or obligation"),
        RubricLevel(score=1, description="drops a qualifier or exception, or adds a claim the source does not make"),
        RubricLevel(score=2, description="accurate; every statement is supported and no qualifier is lost"),
    ],
    pass_threshold=2,
    flagged_label="distortions",
    version="1",
)


def key_fact_coverage(summary: str, key_facts: list[list[str]]) -> tuple[float, list[int]]:
    """Share of key facts present; a fact is present if any of its phrasings appears (normalized)."""
    norm = normalize_text(summary)
    missing = [i for i, alts in enumerate(key_facts) if not any(normalize_text(a) in norm for a in alts)]
    return (1.0 - len(missing) / len(key_facts) if key_facts else 1.0), missing


def lexical_faithfulness(summary: str, source: str, *, min_overlap: float = 0.5) -> tuple[float, list[str]]:
    """Share of summary sentences with no invented numbers/ids and enough lexical support."""
    sentences = split_sentences(summary)
    if not sentences:
        return 1.0, []
    src_words = content_words(source)
    bad = []
    for s in sentences:
        cw = content_words(s)
        support = len(cw & src_words) / len(cw) if cw else 1.0
        if unsupported_atoms(s, source) or support < min_overlap:
            bad.append(s)
    return 1.0 - len(bad) / len(sentences), bad


def compression_ratio(summary: str, source: str) -> float:
    return count_tokens(summary) / max(count_tokens(source), 1)


SUMMARY_METRICS = ["summary_coverage", "summary_faithfulness", "summary_qualifiers", "summary_compression_ok"]


class SummaryEvaluator:
    name = "summary"
    version = "1"
    metric_names = SUMMARY_METRICS

    def __init__(self, min_coverage: float = 0.8) -> None:
        self.min_coverage = min_coverage

    def __call__(self, case: EvalCase, output: Any) -> list[Score]:
        summary = output if isinstance(output, str) else str(output.get("summary", ""))
        source = case.input["source"]
        exp = case.expected or {}
        cov, missing = key_fact_coverage(summary, exp.get("key_facts", []))
        faith, bad = lexical_faithfulness(summary, source)
        must_keep = exp.get("must_keep", [])
        lost = [q for q in must_keep if normalize_text(q) not in normalize_text(summary)]
        lo, hi = exp.get("compression", [0.0, 0.5])
        ratio = compression_ratio(summary, source)
        return [
            Score(name="summary_coverage", value=cov, passed=cov >= self.min_coverage,
                  detail={"missing_facts": missing} if missing else None),
            Score(name="summary_faithfulness", value=faith, passed=faith >= 1.0,
                  detail={"unsupported": bad} if bad else None),
            Score(name="summary_qualifiers", value=0.0 if lost else 1.0, passed=not lost,
                  detail={"lost": lost} if lost else None),
            Score(name="summary_compression_ok", value=float(lo <= ratio <= hi), passed=lo <= ratio <= hi,
                  detail={"ratio": round(ratio, 3), "band": [lo, hi]}),
        ]


__all__ = ["FAITHFULNESS", "key_fact_coverage", "lexical_faithfulness", "compression_ratio", "SummaryEvaluator",
           "SUMMARY_METRICS"]
