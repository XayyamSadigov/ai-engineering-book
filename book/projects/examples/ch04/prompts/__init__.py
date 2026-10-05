# path: book/projects/examples/ch04/prompts/__init__.py
"""Chapter 4: prompts as versioned, tested interface contracts."""
from .examples import FewShotExample, as_variables, select_examples
from .judge import LLMJudge
from .registry import (
    ModelHints,
    PromptNotFoundError,
    PromptRef,
    PromptRegistry,
    PromptSpec,
    PromptVersion,
    RenderedPrompt,
    load_prompt_text,
)
from .regression import (
    Assertion,
    Case,
    Comparison,
    GateDecision,
    JudgeVerdict,
    SuiteResult,
    compare,
    load_cases,
    render_report,
    run_suite,
)
from .rollout import PromptRollout, ReloadResult, bucket
from .template import (
    DATA_TAG,
    PromptDefinitionError,
    PromptRenderError,
    PromptTemplate,
    TemplateSecurityError,
    Untrusted,
    VariableSpec,
    sanitize_untrusted,
)
from .tracing import traced_complete

__all__ = [
    "FewShotExample", "as_variables", "select_examples", "LLMJudge", "ModelHints",
    "PromptNotFoundError", "PromptRef", "PromptRegistry", "PromptSpec", "PromptVersion",
    "RenderedPrompt", "load_prompt_text", "Assertion", "Case", "Comparison", "GateDecision",
    "JudgeVerdict", "SuiteResult", "compare", "load_cases", "render_report", "run_suite",
    "DATA_TAG", "PromptDefinitionError", "PromptRenderError", "PromptTemplate",
    "TemplateSecurityError", "Untrusted", "VariableSpec", "sanitize_untrusted", "traced_complete",
    "PromptRollout", "ReloadResult", "bucket",
]
