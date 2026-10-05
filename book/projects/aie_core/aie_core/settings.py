# path: book/projects/aie_core/aie_core/settings.py
"""Environment-driven configuration and the factories that turn it into clients."""
from __future__ import annotations

from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from .embeddings import EmbeddingClient, FakeEmbeddings, OpenAICompatibleEmbeddings
from .llm.client import LLMClient
from .llm.gateway import InMemoryResponseCache, ModelGateway, RetryPolicy
from .llm.providers import AnthropicClient, FakeLLM, OpenAICompatibleClient
from .observability import Tracer, get_tracer


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: Literal["openai", "anthropic", "fake"] = "fake"
    llm_model: str = "fake-model"
    llm_base_url: str | None = None
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    embedding_provider: Literal["openai", "fake"] = "fake"
    embedding_model: str = "fake-embedding"
    database_url: str | None = None
    redis_url: str | None = None
    trace_sink: Literal["none", "jsonl", "otel", "memory"] = "none"  # "memory": InMemoryTracer, for tests
    trace_path: str = "traces.jsonl"
    request_timeout_s: float = 60.0
    # gateway knobs (all optional; defaults are conservative)
    llm_max_attempts: int = 3
    llm_max_concurrency: int = 16
    llm_requests_per_minute: float | None = None
    llm_tokens_per_minute: float | None = None
    llm_response_cache: bool = False
    llm_fallback_model: str | None = None


def _secret(value: SecretStr | None) -> str | None:
    return value.get_secret_value() if value is not None else None


def make_provider_client(settings: Settings | None = None, model: str | None = None) -> LLMClient:
    """The bare provider adapter for `settings`, with no gateway around it.

    Use it when you compose your own `ModelGateway` (a shared tracer, a custom fallback chain,
    a router that owns several gateways). `model` defaults to `settings.llm_model`.
    """
    settings = settings or Settings()
    return _raw_client(settings, model or settings.llm_model)


def _raw_client(settings: Settings, model: str) -> LLMClient:
    if settings.llm_provider == "openai":
        return OpenAICompatibleClient(
            base_url=settings.llm_base_url or "https://api.openai.com/v1",
            api_key=_secret(settings.openai_api_key),
            default_model=model,
            timeout_s=settings.request_timeout_s,
        )
    if settings.llm_provider == "anthropic":
        kwargs = {"base_url": settings.llm_base_url} if settings.llm_base_url else {}
        return AnthropicClient(
            api_key=_secret(settings.anthropic_api_key),
            default_model=model,
            timeout_s=settings.request_timeout_s,
            **kwargs,
        )
    return FakeLLM(model=model)


def make_llm_client(
    settings: Settings | None = None,
    *,
    tracer: Tracer | None = None,
    wrap: bool | None = None,
) -> LLMClient:
    """The client the application should use.

    By default (`wrap=None`) `fake` returns a bare FakeLLM so tests can script it, and real
    providers come wrapped in a ModelGateway configured from `settings`. `wrap=True` wraps the
    fake too (to test retries, caching, or spans); `wrap=False` returns the bare provider client,
    the same as `make_provider_client`. `tracer` replaces the tracer built from `trace_sink`, so
    an application can pass the one tracer it uses everywhere (Chapter 31).
    """
    settings = settings or Settings()
    if wrap is False:
        return make_provider_client(settings)
    if settings.llm_provider == "fake" and wrap is None:
        return FakeLLM(model=settings.llm_model)
    primary = _raw_client(settings, settings.llm_model)
    fallbacks = [_raw_client(settings, settings.llm_fallback_model)] if settings.llm_fallback_model else []
    from .llm.gateway import RateLimiter

    limiter = (
        RateLimiter(settings.llm_requests_per_minute, settings.llm_tokens_per_minute)
        if settings.llm_requests_per_minute or settings.llm_tokens_per_minute
        else None
    )
    return ModelGateway(
        primary,
        fallbacks=fallbacks,
        retry=RetryPolicy(max_attempts=settings.llm_max_attempts),
        rate_limiter=limiter,
        cache=InMemoryResponseCache() if settings.llm_response_cache else None,
        max_concurrency=settings.llm_max_concurrency,
        tracer=tracer if tracer is not None else get_tracer(settings),
        default_timeout_s=settings.request_timeout_s,
    )


def make_embedding_client(settings: Settings | None = None) -> EmbeddingClient:
    settings = settings or Settings()
    if settings.embedding_provider == "openai":
        return OpenAICompatibleEmbeddings(
            settings.embedding_model,
            base_url=settings.llm_base_url or "https://api.openai.com/v1",
            api_key=_secret(settings.openai_api_key),
            timeout_s=settings.request_timeout_s,
        )
    return FakeEmbeddings(model=settings.embedding_model)


__all__ = ["Settings", "make_llm_client", "make_provider_client", "make_embedding_client"]
