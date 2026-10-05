# path: book/projects/evalkit/evalkit/__init__.py
"""evalkit: the evaluation core of the AI Engineering book (Chapter 24).

Cases and datasets, a runner with lineage, deterministic and classification metrics, LLM
judges with calibration, statistics, Markdown reports, and release gates. Chapters 14 and 25
and the projects import from here.
"""
from . import metrics
from .cases import Dataset, DatasetError, EvalCase, LeakageReport, check_leakage
from .gate import CriticalRule, GateCheck, GateConfig, GateResult, MetricRule, SliceRule, evaluate_gate
from .judges import (
    CORRECTNESS,
    GROUNDEDNESS,
    RELEVANCE,
    JudgeCalibration,
    JudgeEvaluator,
    JudgeResult,
    LLMJudge,
    PairwiseJudge,
    PairwiseResult,
    PairwiseSummary,
    Rubric,
    RubricLevel,
    agreement,
    calibrate_judge,
    cohens_kappa,
    pairwise_summary,
)
from .report import render_report
from .runner import (
    CaseResult,
    Evaluator,
    FunctionEvaluator,
    Run,
    RunVersions,
    Score,
    TargetResult,
    arun_target,
    evaluator,
    run_target,
    score_run,
)
from .stats import (
    CI,
    CaseDelta,
    PairedDelta,
    SliceDelta,
    SliceStat,
    bootstrap_ci,
    compare_runs,
    compare_slices,
    mde_proportion,
    paired_bootstrap,
    per_case_deltas,
    sample_size_for_mde,
    slice_breakdown,
)

__version__ = "0.1.0"

__all__ = [
    "metrics",
    # cases
    "EvalCase", "Dataset", "DatasetError", "LeakageReport", "check_leakage",
    # runner
    "Score", "Evaluator", "FunctionEvaluator", "evaluator", "TargetResult", "CaseResult", "RunVersions", "Run",
    "run_target", "arun_target", "score_run",
    # judges
    "Rubric", "RubricLevel", "GROUNDEDNESS", "CORRECTNESS", "RELEVANCE", "LLMJudge", "JudgeEvaluator", "JudgeResult",
    "PairwiseJudge", "PairwiseResult", "PairwiseSummary", "pairwise_summary",
    "agreement", "cohens_kappa", "calibrate_judge", "JudgeCalibration",
    # stats
    "CI", "bootstrap_ci", "PairedDelta", "paired_bootstrap", "compare_runs", "CaseDelta", "per_case_deltas",
    "SliceStat", "SliceDelta", "slice_breakdown", "compare_slices", "mde_proportion", "sample_size_for_mde",
    # report and gate
    "render_report", "GateConfig", "MetricRule", "SliceRule", "CriticalRule", "GateCheck", "GateResult", "evaluate_gate",
]
