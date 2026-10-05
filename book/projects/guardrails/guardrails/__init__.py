# path: book/projects/guardrails/guardrails/__init__.py
"""Layered guardrails for LLM applications (Chapter 27)."""
from .context import (UNTRUSTED_DATA_POLICY, ContextSanitizerCheck, UntrustedDoc, neutralize_untrusted,
                      render_untrusted_context, wrap_untrusted)
from .input import (DenyPatternCheck, InjectionHeuristicCheck, LLMInjectionClassifier, SizeLimitCheck,
                    score_injection)
from .moderation import KeywordModerator, LLMModerator, ModerationCheck, ModerationPolicy, ModerationResult, Moderator
from .output import (ANSWER_PANE_CSP, ActiveContentCheck, CanaryCheck, CitationCheck, SchemaCheck,
                     UrlAllowlistCheck, escape_html, host_allowed)
from .pii import PIIRedactionCheck, PIIVault, detect_pii, iban_valid, luhn_valid, redact_pii
from .pipeline import (Action, BaseCheck, Check, FailMode, Finding, GuardContext, GuardrailPipeline,
                       PipelineResult, Stage, Subject, Verdict)
from .presets import agent_pipeline, rag_pipeline
from .secrets import SecretsCheck, detect_secrets, redact_secrets, shannon_entropy
from .telemetry import RedactingTracer, scrub
from .tenancy import (TenantIsolationError, TenantScopedCache, assert_tenant_scope, in_scope,
                      scoped_cache_key, tenant_guarded)
from .tools import (ToolDecision, ToolPolicyCheck, ToolRule, approval_token, guard_tool_call, recipient_domains,
                    rehydrate_arguments, rehydrate_kinds, token_tolerant_schema)

__version__ = "0.1.0"

__all__ = [
    "guard_tool_call", "rehydrate_arguments", "rehydrate_kinds", "token_tolerant_schema",
    # pipeline
    "Action", "BaseCheck", "Check", "FailMode", "Finding", "GuardContext", "GuardrailPipeline", "PipelineResult",
    "Stage", "Subject", "Verdict",
    # input
    "SizeLimitCheck", "DenyPatternCheck", "InjectionHeuristicCheck", "LLMInjectionClassifier", "score_injection",
    # context
    "UNTRUSTED_DATA_POLICY", "ContextSanitizerCheck", "UntrustedDoc", "neutralize_untrusted",
    "render_untrusted_context", "wrap_untrusted",
    # pii / secrets / telemetry
    "PIIRedactionCheck", "PIIVault", "detect_pii", "redact_pii", "luhn_valid", "iban_valid",
    "SecretsCheck", "detect_secrets", "redact_secrets", "shannon_entropy", "RedactingTracer", "scrub",
    # output
    "ANSWER_PANE_CSP", "ActiveContentCheck", "CanaryCheck", "CitationCheck", "SchemaCheck", "UrlAllowlistCheck",
    "escape_html", "host_allowed",
    # moderation
    "Moderator", "ModerationResult", "KeywordModerator", "LLMModerator", "ModerationPolicy", "ModerationCheck",
    # tenancy
    "TenantIsolationError", "TenantScopedCache", "assert_tenant_scope", "in_scope", "scoped_cache_key",
    "tenant_guarded",
    # tools
    "ToolRule", "ToolDecision", "ToolPolicyCheck", "approval_token", "recipient_domains",
    # presets
    "rag_pipeline", "agent_pipeline",
]
