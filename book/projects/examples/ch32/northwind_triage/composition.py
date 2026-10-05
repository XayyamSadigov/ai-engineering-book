# path: book/projects/examples/ch32/northwind_triage/composition.py
"""The composition root: the one place that knows every concrete class.

It reads settings, builds adapters, checks that the configuration is internally consistent
(prompt versions exist and are locked, flag variants map to configured versions), builds the
static version manifest, and wires the TriageService. Tests call it with fakes.
"""
from __future__ import annotations

from aie_core.llm.client import LLMClient
from aie_core.observability import Tracer, get_tracer
from aie_core.settings import make_llm_client

from .adapters.llm_classifier import LLMClassifier
from .adapters.prompt_store import DEFAULT_DIR, FilePromptStore, LockMismatch, PromptNotFound
from .adapters.simulated_model import simulated_llm
from .adapters.tools import CONTRACTS
from .application import MODEL_FLAG, PROMPT_FLAG, TriageConfig, TriageService
from .config import AppSettings, ConfigError
from .flags import FlagEvaluator
from .version_manifest import VersionManifest

APP_NAME = "northwind-triage"


def build_triage_config(settings: AppSettings) -> TriageConfig:
    prompt_versions = {"control": settings.prompt_control_version}
    if settings.prompt_treatment_version:
        prompt_versions["treatment"] = settings.prompt_treatment_version
    models = {"control": settings.llm.llm_model}
    if settings.model_candidate:
        models["candidate"] = settings.model_candidate
    return TriageConfig(prompt_id=settings.prompt_id, prompt_versions=prompt_versions, models=models)


def build_base_manifest(settings: AppSettings, flags: FlagEvaluator) -> VersionManifest:
    config = {
        "classifier_timeout_s": str(settings.classifier_timeout_s),
        "flags_snapshot": flags.snapshot_hash(),
    }
    return VersionManifest(
        app=APP_NAME,
        app_version=settings.app_version,
        git_sha=settings.git_sha,
        environment=settings.environment,
        models={"provider": settings.llm.llm_provider},
        datasets={"triage_golden": settings.dataset_version} if settings.dataset_version else {},
        evaluators={"triage_accuracy": settings.evaluator_version} if settings.evaluator_version else {},
        tool_schemas={c.name: c.label for c in CONTRACTS},
        config=config,
    )


def check_consistency(settings: AppSettings, store: FilePromptStore, flags: FlagEvaluator,
                      triage_config: TriageConfig) -> list[str]:
    problems: list[str] = []
    for variant, version in triage_config.prompt_versions.items():
        try:
            store.get(triage_config.prompt_id, version)
        except PromptNotFound as exc:
            problems.append(f"prompt variant {variant}: {exc}")
    try:
        store.verify_lock()
    except (LockMismatch, FileNotFoundError) as exc:
        problems.append(f"prompts.lock: {exc}")
    for flag_name, known in ((PROMPT_FLAG, triage_config.prompt_versions),
                             (MODEL_FLAG, triage_config.models)):
        cfg = flags.config(flag_name)
        if cfg is None:
            continue
        for alloc in cfg.allocations:
            if alloc.percent > 0 and alloc.variant not in known:
                problems.append(f"flag {flag_name} sends {alloc.percent}% to variant "
                                f"{alloc.variant!r}, which has no configured value")
    return problems


def build_service(settings: AppSettings, *, llm_client: LLMClient | None = None,
                  tracer: Tracer | None = None) -> TriageService:
    store = FilePromptStore(settings.prompt_dir or DEFAULT_DIR)
    flags = (FlagEvaluator.from_file(settings.flags_path, settings.environment)
             if settings.flags_path else FlagEvaluator({}, settings.environment))
    triage_config = build_triage_config(settings)
    problems = check_consistency(settings, store, flags, triage_config)
    if problems:
        raise ConfigError("inconsistent configuration:\n  " + "\n  ".join(problems))

    if llm_client is None:
        llm_client = (simulated_llm(settings.llm.llm_model) if settings.llm.llm_provider == "fake"
                      else make_llm_client(settings.llm))
    classifier = LLMClassifier(llm_client, timeout_s=settings.classifier_timeout_s)
    return TriageService(
        classifier=classifier,
        prompts=store,
        flags=flags,
        tracer=tracer or get_tracer(settings.llm),
        base_manifest=build_base_manifest(settings, flags),
        config=triage_config,
    )


def create_app_from_env():  # pragma: no cover - exercised by `uvicorn --factory`
    from .adapters.http_api import create_app
    from .config import load_settings

    return create_app(build_service(load_settings()))
