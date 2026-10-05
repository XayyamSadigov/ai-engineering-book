# path: book/projects/p5-incident-agent/incident_agent/judge.py
"""The rubric judge: evalkit's LLMJudge (Chapter 24) with one incident-report dimension.

It runs only on reports that already pass the deterministic Definition of Done, so it spends
its attention on what rules cannot check: is the stated cause actually supported by the cited
evidence, and are the next steps specific and safe? Its verdict is advisory for publication
(shown to the approver) and drives revisions inside the evaluator-optimizer loop.
"""
from __future__ import annotations

from aie_core.llm.client import LLMClient
from evalkit.judges import JudgeResult, LLMJudge, Rubric, RubricLevel

from .domain.models import Investigation

INCIDENT_REPORT_RUBRIC = Rubric(
    name="incident_report_quality",
    task=("Judge an incident report written for an on-call engineer. Is the likely cause supported by the cited "
          "evidence (not merely plausible), does the report separate observation from inference, and are the "
          "next steps specific, ordered, and safe to execute during an incident?"),
    levels=[
        RubricLevel(score=1, description="cause contradicts or ignores the evidence, or steps are unsafe"),
        RubricLevel(score=2, description="cause is a guess; evidence cited does not support it"),
        RubricLevel(score=3, description="cause plausible but key evidence missing or steps vague"),
        RubricLevel(score=4, description="cause supported by cited evidence; steps specific; minor gaps"),
        RubricLevel(score=5, description="cause supported by converging evidence; steps specific, ordered, safe"),
    ],
    pass_threshold=4,
    flagged_label="problems",
    version="1",
)


class ReportJudge:
    def __init__(self, llm: LLMClient, *, pass_score: int = 4, model: str | None = None) -> None:
        rubric = INCIDENT_REPORT_RUBRIC.model_copy(update={"pass_threshold": pass_score})
        self.judge = LLMJudge(llm, rubric, model=model)

    def __call__(self, inv: Investigation, report: str) -> JudgeResult:
        evidence = [f"[{e.id}] ({e.kind}) {e.text}" for e in inv.evidence.values()]
        return self.judge.judge(input=f"Alert {inv.alert.id} ({inv.alert.rule}): {inv.alert.summary}",
                                answer=report, evidence=evidence)


__all__ = ["INCIDENT_REPORT_RUBRIC", "ReportJudge"]
