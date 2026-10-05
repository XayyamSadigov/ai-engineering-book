# path: book/projects/p6-research-team/research_team/__init__.py
"""Project 6: a Northwind policy-research team (supervisor, parallel researchers, verifier)
built on agentkit's AgentRuntime, with a single-agent baseline and a benchmark that compares them."""
from .baseline import SingleAgent
from .contracts import (
    AgentUsage, AnswerReport, BudgetSlice, Claim, ClaimVerdict, Conflict, ErrorInfo, EvidenceRef, Plan, RejectedClaim,
    ResearchFindings, ResultEnvelope, Role, SubQuestion, TaskEnvelope, TaskStatus, VerificationReport,
)
from .corpus import Corpus, Passage
from .ledger import Admission, BudgetLedger, TeamBudget, TeamEvent, TeamLog
from .team import ResearchTeam, TeamConfig
from .tracing import PropagatingTracer, span_tree

__version__ = "0.1.0"
__all__ = [
    "Admission", "AgentUsage", "AnswerReport", "BudgetLedger", "BudgetSlice", "Claim", "ClaimVerdict", "Conflict",
    "Corpus", "ErrorInfo", "EvidenceRef", "Passage", "Plan", "PropagatingTracer", "RejectedClaim", "ResearchFindings",
    "ResearchTeam", "ResultEnvelope", "Role", "SingleAgent", "SubQuestion", "TaskEnvelope", "TaskStatus", "TeamBudget",
    "TeamConfig", "TeamEvent", "TeamLog", "VerificationReport", "span_tree",
]
