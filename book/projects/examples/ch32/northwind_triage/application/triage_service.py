# path: book/projects/examples/ch32/northwind_triage/application/triage_service.py
"""The triage use case: select versions via flags, call the classifier through a port,
parse, apply business rules, and record the version manifest on the trace."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..domain import (
    FinalTriage,
    ParseError,
    Ticket,
    apply_business_rules,
    fallback_triage,
    parse_triage,
)
from ..version_manifest import VersionManifest
from .ports import ClassifierPort, ClassifierUnavailable, FlagsPort, PromptStorePort, TracerPort

PROMPT_FLAG = "triage.prompt"
MODEL_FLAG = "triage.model"


@dataclass(frozen=True)
class TriageConfig:
    prompt_id: str = "triage.classify"
    prompt_versions: dict[str, str] = field(default_factory=lambda: {"control": "1.0.0"})
    models: dict[str, str] = field(default_factory=lambda: {"control": "fake-model"})


@dataclass(frozen=True)
class TriageResult:
    triage: FinalTriage
    manifest: VersionManifest
    outcome: str  # ok | parse_error | classifier_unavailable


class TriageService:
    def __init__(
        self,
        classifier: ClassifierPort,
        prompts: PromptStorePort,
        flags: FlagsPort,
        tracer: TracerPort,
        base_manifest: VersionManifest,
        config: TriageConfig | None = None,
    ) -> None:
        self.classifier = classifier
        self.prompts = prompts
        self.flags = flags
        self.tracer = tracer
        self.base_manifest = base_manifest
        self.config = config or TriageConfig()

    def _select(self, unit_id: str) -> tuple[str, str, dict[str, str]]:
        p = self.flags.evaluate(PROMPT_FLAG, unit_id)
        m = self.flags.evaluate(MODEL_FLAG, unit_id)
        # A variant the config does not know falls back to control; never KeyError in prod.
        version = self.config.prompt_versions.get(p.variant, self.config.prompt_versions["control"])
        model = self.config.models.get(m.variant, self.config.models["control"])
        return version, model, {PROMPT_FLAG: p.variant, MODEL_FLAG: m.variant}

    def triage(self, ticket: Ticket, unit_id: str) -> TriageResult:
        version, model, assignments = self._select(unit_id)
        prompt = self.prompts.get(self.config.prompt_id, version)
        manifest = self.base_manifest.with_updates(
            prompts={prompt.id: prompt.label},
            models={"classifier": model},
            flags=assignments,
        )
        with self.tracer.span("triage.request", ticket_id=ticket.id, tenant=ticket.tenant,
                              **manifest.as_span_attributes(), **manifest.as_semconv_attributes()) as span:
            system, user = prompt.render(ticket)
            try:
                out = self.classifier.classify(system=system, user=user, model=model)
            except ClassifierUnavailable as exc:
                result = fallback_triage(ticket, f"classifier unavailable: {exc.reason}")
                span.set_attribute("triage.outcome", "classifier_unavailable")
                span.set_attribute("triage.route", result.route)
                return TriageResult(result, manifest, "classifier_unavailable")

            span.set_attribute("llm.served_model", out.served_model)
            span.set_attribute("llm.input_tokens", out.input_tokens)
            span.set_attribute("llm.output_tokens", out.output_tokens)
            try:
                decision = parse_triage(out.text)
            except ParseError as exc:
                result = fallback_triage(ticket, f"unparseable classifier output: {exc}")
                span.set_attribute("triage.outcome", "parse_error")
                span.set_attribute("triage.route", result.route)
                return TriageResult(result, manifest, "parse_error")

            result = apply_business_rules(ticket, decision)
            span.set_attribute("triage.outcome", "ok")
            span.set_attribute("triage.category", decision.category.value)
            span.set_attribute("triage.priority", result.priority.value)
            span.set_attribute("triage.route", result.route)
            span.set_attribute("triage.confidence", decision.confidence)
            return TriageResult(result, manifest, "ok")
