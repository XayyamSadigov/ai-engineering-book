# path: book/capstone/northwind-assist/northwind_assist/container.py
"""Composition root: the only place that constructs adapters (Chapter 28's `build_app` rule).

Tests and the eval harness call `build_container(settings, llm=..., tracer=...)` with fakes; the
API calls it with nothing and gets the environment's configuration. Everything else receives its
collaborators through constructors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aie_core.embeddings import EmbeddingClient
from aie_core.llm.client import LLMClient
from context import BudgetPolicy, ContextBuilder  # type: ignore[import-not-found]
from northwind_triage.version_manifest import VersionManifest, versioned  # type: ignore[import-not-found]
from prompts import PromptRegistry  # type: ignore[import-not-found]
from ragkit.generation.generator import PROMPT_VERSION
from ragkit.generation.stream import STREAM_PROMPT_ID, STREAM_SYSTEM_PROMPT
from reliability import CircuitBreakerRegistry

from . import _paths
from .config import Settings
from .cost.ledger import CostLedger
from .extraction.service import ExtractionRoute
from .llm.models import ModelLayer
from .memory.service import MemoryService
from .observability.tracing import build_tracer
from .orchestrator import Orchestrator
from .rag.caches import CacheLayer
from .rag.knowledge import Knowledge, build_knowledge
from .rag.service import RagService
from .resilience.wiring import Resilience
from .security.auth import TokenValidator
from .security.guards import Guards
from .tools.service import ToolLayer

PROMPT_DIR = _paths.CAPSTONE_ROOT / "northwind_assist" / "prompt_files"


@dataclass
class Container:
    settings: Settings
    tracer: Any
    models: ModelLayer
    kb: Knowledge
    caches: CacheLayer
    guards: Guards
    rag: RagService
    tools: ToolLayer
    memory: MemoryService
    extraction: ExtractionRoute
    ledger: CostLedger
    resilience: Resilience
    prompts: Any
    context_builder: Any
    validator: TokenValidator
    base_manifest: Any
    orchestrator: Orchestrator = field(init=False)

    def __post_init__(self) -> None:
        self.orchestrator = Orchestrator(self)

    @property
    def agent_prompt_ref(self) -> str:
        return str(self.prompts.get("assist.agent", "prod").ref)

    def manifest_for(self, flags: dict[str, str]) -> Any:
        return self.base_manifest.with_updates(index_version=self.kb.index_version, flags=flags)


def build_container(settings: Settings | None = None, *, llm: LLMClient | None = None,
                    embeddings: EmbeddingClient | None = None, tracer: Any = None,
                    extraction_llm: LLMClient | None = None, docs_dirs: list[str] | None = None,
                    breakers: CircuitBreakerRegistry | None = None, clock: Any = None, backup_llm: LLMClient | None = None,
                    jwks_client: Any = None) -> Container:
    settings = settings or Settings()
    tracer = tracer or build_tracer()
    models = ModelLayer(settings, client=llm, tracer=tracer, breakers=breakers, backup=backup_llm)
    kb = build_knowledge(settings, backend=settings.knowledge_backend, docs_dirs=docs_dirs, embeddings=embeddings,
                         tracer=tracer)
    caches = CacheLayer(retrieval_ttl_s=settings.retrieval_cache_ttl_s, answer_ttl_s=settings.answer_cache_ttl_s)
    guards = Guards(settings, tracer=tracer)
    rag = RagService(kb, caches, guards, tracer=tracer)
    tools = ToolLayer(settings, guards, tracer=tracer, pricing=models.pricing, clock=clock)
    memory = MemoryService(settings)
    extraction = ExtractionRoute(extraction_llm, tracer=tracer)
    ledger = CostLedger(settings.tenant_daily_budget_usd, thresholds=tuple(settings.cost_alert_thresholds),
                        path=settings.cost_ledger_path, **({"clock": clock} if clock is not None else {}))
    resilience = Resilience(settings, clock=clock, model_dependencies=models.model_dependencies)
    prompts = PromptRegistry.from_directory(PROMPT_DIR)
    builder = ContextBuilder(BudgetPolicy(context_window=8000, output_reserve=512), tracer=tracer)
    agent_prompt = prompts.get("assist.agent", "prod")
    manifest = VersionManifest(
        app="northwind-assist", app_version=settings.app_version, git_sha=settings.git_sha,
        environment=settings.environment,
        prompts={STREAM_PROMPT_ID: versioned(PROMPT_VERSION, STREAM_SYSTEM_PROMPT),
                 "assist.agent": f"{agent_prompt.spec.version}#{agent_prompt.content_hash[:12]}"},
        models={alias: p.model_id for alias, p in models.catalog.profiles.items()},
        embedding_model="#".join(x for x in (kb.fingerprint().get("embedding_model"),
                                             kb.fingerprint().get("embedding_space")) if x),
        index_version=kb.index_version,
        tool_schemas=tools.schema_versions(), evaluators={"guards": guards.version, "tool_policy": tools.policy.version},
        flags={"snapshot": resilience.flags.snapshot_hash()}, config=settings.behavior_config())
    return Container(settings=settings, tracer=tracer, models=models, kb=kb, caches=caches, guards=guards, rag=rag,
                     tools=tools, memory=memory, extraction=extraction, ledger=ledger, resilience=resilience,
                     prompts=prompts, context_builder=builder,
                     validator=TokenValidator(settings, jwks_client=jwks_client), base_manifest=manifest)


__all__ = ["Container", "build_container", "PROMPT_DIR"]
