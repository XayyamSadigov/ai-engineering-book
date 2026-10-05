# path: book/projects/examples/ch25/taskevals/__init__.py
"""Task-specific evaluators for Chapter 25, built on evalkit (Chapter 24).

Modules: prompts, rag, trajectory (+ replay over agentkit event logs), extraction, classification, summarization,
tools, synthetic, online, and suites (the CI fast suites with deterministic stand-in models).
"""
from .classification import ClassificationAggregates, LabelEvaluator, classification_aggregates, evaluate_cascade
from .extraction import CRITICAL_FIELDS, FIELD_WEIGHTS, ExtractionEvaluator, evidence_status, weighted_field_prf
from .online import (
    ArmCounts,
    CanaryDecision,
    CanaryMonitor,
    FeedbackEvent,
    TraceRecord,
    corrections_to_cases,
    join_feedback,
    outcome_metrics,
    simulate_false_alarms,
)
from .prompts import PromptContractEvaluator, reply_judge
from .rag import RagAnswerEvaluator, judge_evaluators
from .replay import (
    agentkit_replay_target,
    load_event_logs,
    northwind_projection,
    recorded_run_target,
    replay_fidelity_evaluator,
    trajectory_from_events,
)
from .summarization import FAITHFULNESS, SummaryEvaluator
from .synthetic import SyntheticGenerator, bias_report, validate_candidates
from .tools import ToolUseEvaluator, tool_confusion
from .trajectory import (
    NORTHWIND_TOOLS,
    StatePredicate,
    Step,
    ToolInfo,
    Trajectory,
    TrajectoryEvaluator,
    TrajectorySpec,
    check_trajectory,
)

__all__ = [
    "LabelEvaluator", "ClassificationAggregates", "classification_aggregates", "evaluate_cascade",
    "ExtractionEvaluator", "FIELD_WEIGHTS", "CRITICAL_FIELDS", "evidence_status", "weighted_field_prf",
    "TraceRecord", "FeedbackEvent", "join_feedback", "outcome_metrics", "corrections_to_cases",
    "ArmCounts", "CanaryDecision", "CanaryMonitor", "simulate_false_alarms",
    "PromptContractEvaluator", "reply_judge", "RagAnswerEvaluator", "judge_evaluators",
    "trajectory_from_events", "northwind_projection", "load_event_logs", "recorded_run_target",
    "agentkit_replay_target", "replay_fidelity_evaluator",
    "SummaryEvaluator", "FAITHFULNESS", "SyntheticGenerator", "validate_candidates", "bias_report",
    "ToolUseEvaluator", "tool_confusion",
    "Trajectory", "Step", "ToolInfo", "NORTHWIND_TOOLS", "StatePredicate", "TrajectorySpec", "TrajectoryEvaluator",
    "check_trajectory",
]
