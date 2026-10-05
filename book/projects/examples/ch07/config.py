# path: book/projects/examples/ch07/config.py
"""Environment-driven settings for the Chapter 7 router."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

from collections.abc import Mapping

from aie_core import LLMClient

from catalog import ModelCatalog, northwind_catalog


class RouterSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_catalog_path: str | None = None        # JSON catalog; built-in illustrative one if unset
    router_min_confidence: float = 0.8           # cascade escalation threshold
    router_long_context_tokens: int = 100_000    # prompt+output size that triggers the long-context route
    router_classifier_min_confidence: float = 0.5

    def load_catalog(self) -> ModelCatalog:
        return ModelCatalog.from_json(self.model_catalog_path) if self.model_catalog_path else northwind_catalog()

    def build_router(self, clients: Mapping[str, LLMClient], **kwargs):
        """The Northwind router with thresholds from the environment. In production each client
        is an aie_core ModelGateway around a real provider; tests pass FakeLLM instances."""
        from router import northwind_router
        return northwind_router(self.load_catalog(), clients, min_confidence=self.router_min_confidence,
                                long_context_threshold=self.router_long_context_tokens,
                                classifier_min_confidence=self.router_classifier_min_confidence, **kwargs)


__all__ = ["RouterSettings"]
