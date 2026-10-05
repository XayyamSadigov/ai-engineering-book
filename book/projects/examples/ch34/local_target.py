# path: book/projects/examples/ch34/local_target.py
"""Describe a serving target (hosted or local) and check a request against its capabilities.

The book's ``aie_core.OpenAICompatibleClient`` already speaks to any OpenAI-compatible server
when given a ``base_url``, so there is nothing to reimplement for transport. What changes when
you route between a hosted endpoint and a self-hosted engine is the *envelope*: whether tools
and JSON-schema output are supported, how long the context is, and which model name the
server expects. This module makes that envelope explicit so the router can refuse or downgrade
a request before the server rejects it with a less helpful error.

Two functions connect it to ``aie_core``: ``envelope_for`` derives the envelope from the same
``CompletionRequest`` the application already built, and ``make_client`` returns an
``OpenAICompatibleClient`` whose ``provider`` names the engine and target (so every span says
which deployment answered) and whose ``supports_response_schema`` matches the target (so
``complete_structured`` falls back to prompt-and-repair instead of sending an unsupported
``response_format``). Capability-changing choices between targets belong to the router
(Chapter 7); a ``ModelGateway`` fallback chain should hold only interchangeable replicas.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ServingTarget(BaseModel):
    name: str
    base_url: str
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    max_context_tokens: int = Field(gt=0)
    supports_tools: bool = True
    supports_response_schema: bool = True
    supports_stream_usage: bool = True
    engine: str | None = None  # e.g. "vllm", "sglang", "llama.cpp", "hosted"; recorded in traces


class RequestEnvelope(BaseModel):
    """The capability-relevant facts about one request, independent of the message contents."""

    input_tokens: int = Field(ge=0)
    max_output_tokens: int = Field(ge=1)
    needs_tools: bool = False
    needs_response_schema: bool = False


def check_fit(target: ServingTarget, env: RequestEnvelope, reserve_tokens: int = 64) -> list[str]:
    """Return a list of human-readable problems; empty means the request can be sent as is."""
    problems: list[str] = []
    needed = env.input_tokens + env.max_output_tokens + reserve_tokens
    if needed > target.max_context_tokens:
        problems.append(
            f"context: needs {needed} tokens (input {env.input_tokens} + output {env.max_output_tokens} "
            f"+ reserve {reserve_tokens}) but {target.name} allows {target.max_context_tokens}"
        )
    if env.needs_tools and not target.supports_tools:
        problems.append(f"tools: {target.name} does not support tool calling")
    if env.needs_response_schema and not target.supports_response_schema:
        problems.append(f"schema: {target.name} does not support JSON-schema constrained output")
    return problems


def choose_target(targets: list[ServingTarget], env: RequestEnvelope) -> ServingTarget | None:
    """First target, in preference order, that fits the request. ``None`` if none does."""
    for t in targets:
        if not check_fit(t, env):
            return t
    return None


def envelope_for(req: Any, *, needs_tools: bool | None = None, needs_response_schema: bool | None = None) -> RequestEnvelope:
    """The envelope of an ``aie_core`` ``CompletionRequest``, counted with ``count_message_tokens``.

    The count is the application's estimate, not the engine's; ``check_fit`` keeps a reserve for
    the difference. Imported lazily so the module stays importable without ``aie_core``.
    """
    from aie_core.llm.tokens import count_message_tokens  # noqa: PLC0415

    return RequestEnvelope(
        input_tokens=count_message_tokens(req.messages, req.model),
        max_output_tokens=req.max_tokens,
        needs_tools=bool(req.tools) if needs_tools is None else needs_tools,
        needs_response_schema=(req.response_schema is not None) if needs_response_schema is None else needs_response_schema,
    )


def make_client(
    target: ServingTarget,
    *,
    timeout_s: float = 60.0,
    transport: Any = None,
    async_transport: Any = None,
) -> Any:
    """Build the book's provider-neutral client for this target.

    ``provider`` becomes ``"<engine>:<target name>"`` so gateway spans and cost reports separate
    a self-hosted replica from the hosted endpoint. Transports are injectable for offline tests.
    Imported lazily so this module stays importable in environments without ``aie_core``.
    """
    import os

    from aie_core.llm.providers import OpenAICompatibleClient  # noqa: PLC0415

    client = OpenAICompatibleClient(
        base_url=target.base_url,
        api_key=os.environ.get(target.api_key_env, "not-needed"),
        default_model=target.model,
        provider=f"{target.engine or 'openai-compatible'}:{target.name}",
        timeout_s=timeout_s,
        transport=transport,
        async_transport=async_transport,
    )
    client.supports_response_schema = target.supports_response_schema
    return client


# Illustrative targets. Capabilities and limits are engine- and version-specific: read them from
# the server's /v1/models response and its documentation, and re-check after every upgrade.
HOSTED = ServingTarget(
    name="hosted", base_url="https://api.example-provider.com/v1", model="hosted-general",
    max_context_tokens=128_000, engine="hosted",
)
LOCAL_VLLM = ServingTarget(
    name="local-gpu", base_url="http://inference.internal:8000/v1", model="open-8b-instruct",
    api_key_env="LOCAL_LLM_API_KEY", max_context_tokens=32_768, engine="vllm",
)
LOCAL_CPU = ServingTarget(
    name="local-cpu", base_url="http://localhost:8080/v1", model="open-8b-instruct-q4",
    api_key_env="LOCAL_LLM_API_KEY", max_context_tokens=8_192, supports_tools=False,
    supports_response_schema=True, engine="llama.cpp",
)
