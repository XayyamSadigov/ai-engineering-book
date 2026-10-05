# path: book/projects/aie_core/aie_core/__init__.py
"""aie_core: the shared LLM client, gateway, embeddings, tracing, and settings for the book."""
from . import embeddings, llm, observability, settings
from .llm import *  # noqa: F401,F403 - re-export the LLM surface at the top level
from .settings import Settings, make_embedding_client, make_llm_client, make_provider_client

__version__ = "0.1.0"
__all__ = ["llm", "embeddings", "observability", "settings", "Settings", "make_llm_client", "make_provider_client", "make_embedding_client", *llm.__all__]
