# path: book/projects/aie_core/tests/test_settings.py
from aie_core.llm.gateway import ModelGateway, RateLimiter
from aie_core.llm.providers import AnthropicClient, FakeLLM, OpenAICompatibleClient
from aie_core.embeddings import FakeEmbeddings, OpenAICompatibleEmbeddings
from aie_core.settings import Settings, make_embedding_client, make_llm_client


def test_defaults_are_offline(monkeypatch):
    for var in ("LLM_PROVIDER", "LLM_MODEL", "EMBEDDING_PROVIDER", "TRACE_SINK"):
        monkeypatch.delenv(var, raising=False)
    s = Settings(_env_file=None)
    assert s.llm_provider == "fake" and s.embedding_provider == "fake" and s.trace_sink == "none"
    assert isinstance(make_llm_client(s), FakeLLM)
    assert isinstance(make_embedding_client(s), FakeEmbeddings)


def test_env_drives_settings(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "some-model")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("LLM_REQUESTS_PER_MINUTE", "120")
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "small-model")
    s = Settings(_env_file=None)
    assert s.openai_api_key.get_secret_value() == "sk-secret"
    assert "sk-secret" not in repr(s)
    client = make_llm_client(s)
    assert isinstance(client, ModelGateway)
    assert isinstance(client.primary, OpenAICompatibleClient) and client.primary.default_model == "some-model"
    assert isinstance(client.fallbacks[0], OpenAICompatibleClient) and client.fallbacks[0].default_model == "small-model"
    assert isinstance(client.rate_limiter, RateLimiter)


def test_anthropic_and_openai_embeddings_factories():
    s = Settings(_env_file=None, llm_provider="anthropic", llm_model="m", anthropic_api_key="k", embedding_provider="openai", embedding_model="e")
    client = make_llm_client(s)
    assert isinstance(client, ModelGateway) and isinstance(client.primary, AnthropicClient)
    emb = make_embedding_client(s)
    assert isinstance(emb, OpenAICompatibleEmbeddings) and emb.model == "e"


def test_make_provider_client_is_the_bare_adapter():
    from aie_core import make_provider_client

    s = Settings(_env_file=None, llm_provider="openai", llm_model="m", openai_api_key="k", llm_fallback_model="small")
    raw = make_provider_client(s)
    assert isinstance(raw, OpenAICompatibleClient) and raw.default_model == "m"
    assert make_provider_client(s, model="small").default_model == "small"
    assert isinstance(make_llm_client(s, wrap=False), OpenAICompatibleClient)


def test_injected_tracer_and_wrapped_fake_record_spans():
    from aie_core import CompletionRequest, Message
    from aie_core.observability import InMemoryTracer

    tracer = InMemoryTracer()
    s = Settings(_env_file=None)  # fake provider
    client = make_llm_client(s, tracer=tracer, wrap=True)
    assert isinstance(client, ModelGateway) and isinstance(client.primary, FakeLLM)
    client.complete(CompletionRequest(messages=[Message.user("hi")]))
    assert client.tracer is tracer and tracer.find("llm.complete")
    assert isinstance(make_llm_client(s), FakeLLM)  # default behavior unchanged


def test_memory_trace_sink_from_environment(monkeypatch):
    from aie_core.observability import InMemoryTracer

    monkeypatch.setenv("TRACE_SINK", "memory")
    s = Settings(_env_file=None, llm_provider="openai", llm_model="m", openai_api_key="k")
    assert s.trace_sink == "memory"
    gw = make_llm_client(s)
    assert isinstance(gw.tracer, InMemoryTracer)
