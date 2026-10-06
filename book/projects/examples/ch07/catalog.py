# path: book/projects/examples/ch07/catalog.py
"""Model catalog: what each candidate model can do, what it costs, and where it may run.

Every number in a catalog is illustrative. The catalog is the one place where a model's
capabilities are written down, so the router can check them before it sends a request
and the selection harness can label its results. Profiles are pinned: application code
refers to an alias such as ``nw-small`` and the catalog resolves it to an exact,
versioned model identifier. Changing what an alias points to is a reviewed catalog
change, never a side effect of a provider silently updating a floating name.
"""
from __future__ import annotations

import json
from enum import IntEnum
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field, model_validator

from aie_core import CompletionRequest, count_message_tokens


class Tier(IntEnum):
    """Relative tiers. Lower is cheaper or faster. Only the ordering is meaningful."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    VERY_HIGH = 4


class ModelProfile(BaseModel):
    alias: str                                   # what application code and routes use
    model_id: str                                # pinned, versioned identifier sent to the provider
    provider: str                                # key into the client registry
    context_tokens: int                          # total window: prompt plus output
    max_output_tokens: int
    supports_tools: bool = False
    supports_json_schema: bool = False           # native schema-constrained output
    supports_vision: bool = False
    reasoning_efforts: tuple[str, ...] = ()      # e.g. ("low", "medium", "high"); empty = no knob
    data_zones: tuple[str, ...] = ("any",)       # where inference runs, e.g. ("eu",); "any" = no residency guarantee
    cost_tier: Tier = Tier.MEDIUM
    latency_tier: Tier = Tier.MEDIUM
    input_per_1m: float = 0.0                    # illustrative USD per million input tokens
    output_per_1m: float = 0.0                   # illustrative USD per million output tokens
    notes: str = ""

    @model_validator(mode="after")
    def _check(self) -> "ModelProfile":
        if self.max_output_tokens > self.context_tokens:
            raise ValueError(f"{self.alias}: max_output_tokens exceeds context_tokens")
        return self

    def pricing_entry(self) -> dict[str, float]:
        return {"input_per_1m": self.input_per_1m, "output_per_1m": self.output_per_1m}


class Requirements(BaseModel):
    """What a request needs from whichever model serves it."""

    min_context_tokens: int = 0
    min_output_tokens: int = 0
    needs_tools: bool = False
    needs_json_schema: bool = False
    needs_vision: bool = False
    data_zone: str | None = None
    reasoning_effort: str | None = None

    @classmethod
    def from_request(cls, req: CompletionRequest) -> "Requirements":
        """Derive requirements from the request itself, so callers cannot forget them."""
        has_image = any(
            not isinstance(m.content, str) and any(p.type == "image_url" for p in m.content)
            for m in req.messages
        )
        prompt_tokens = count_message_tokens(req.messages, req.model)
        return cls(
            min_context_tokens=prompt_tokens + req.max_tokens,
            min_output_tokens=req.max_tokens,
            needs_tools=bool(req.tools) and req.tool_choice != "none",
            needs_json_schema=req.response_schema is not None,
            needs_vision=has_image,
            data_zone=req.metadata.get("data_zone"),
            reasoning_effort=req.metadata.get("reasoning_effort"),
        )


class Gap(BaseModel):
    """One way a profile fails a requirement. Hard gaps disqualify; soft gaps degrade."""

    capability: str
    detail: str
    hard: bool = True


def capability_gaps(profile: ModelProfile, need: Requirements) -> list[Gap]:
    gaps: list[Gap] = []
    if need.min_context_tokens > profile.context_tokens:
        gaps.append(Gap(capability="context",
                        detail=f"needs {need.min_context_tokens} tokens, window is {profile.context_tokens}"))
    if need.min_output_tokens > profile.max_output_tokens:
        gaps.append(Gap(capability="output",
                        detail=f"needs {need.min_output_tokens} output tokens, max is {profile.max_output_tokens}"))
    if need.needs_tools and not profile.supports_tools:
        gaps.append(Gap(capability="tools", detail="request carries tools; model has no tool calling"))
    if need.needs_vision and not profile.supports_vision:
        gaps.append(Gap(capability="vision", detail="request carries images; model is text-only"))
    if need.data_zone and need.data_zone not in profile.data_zones:  # "any" never satisfies an explicit zone
        gaps.append(Gap(capability="data_zone",
                        detail=f"request must stay in {need.data_zone}; model runs in {list(profile.data_zones)}"))
    if need.needs_json_schema and not profile.supports_json_schema:
        # Soft: structured output can fall back to prompt-and-parse with validation and repair
        # (Chapter 6). The request still works, but its failure rate changes, so it is flagged.
        gaps.append(Gap(capability="json_schema", hard=False,
                        detail="no native schema mode; falls back to prompt+parse with repair"))
    if need.reasoning_effort and need.reasoning_effort not in profile.reasoning_efforts:
        gaps.append(Gap(capability="reasoning_effort", hard=False,
                        detail=f"effort {need.reasoning_effort!r} unsupported; model offers {list(profile.reasoning_efforts)}"))
    return gaps


def is_compatible(profile: ModelProfile, need: Requirements) -> bool:
    return not any(g.hard for g in capability_gaps(profile, need))


class ModelCatalog(BaseModel):
    profiles: dict[str, ModelProfile] = Field(default_factory=dict)

    @classmethod
    def from_profiles(cls, profiles: Iterable[ModelProfile]) -> "ModelCatalog":
        catalog = cls()
        for p in profiles:
            catalog.add(p)
        return catalog

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelCatalog":
        data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_profiles(ModelProfile.model_validate(p) for p in data["models"])

    def add(self, profile: ModelProfile) -> None:
        if profile.alias in self.profiles:
            raise ValueError(f"duplicate alias {profile.alias!r}")
        self.profiles[profile.alias] = profile

    def get(self, alias: str) -> ModelProfile:
        try:
            return self.profiles[alias]
        except KeyError:
            raise KeyError(f"unknown model alias {alias!r}; known: {sorted(self.profiles)}") from None

    def compatible(self, need: Requirements, among: Iterable[str] | None = None) -> list[ModelProfile]:
        """Compatible profiles, cheapest first, then fastest. Stable for equal tiers."""
        names = list(among) if among is not None else list(self.profiles)
        found = [self.get(n) for n in names if is_compatible(self.get(n), need)]
        return sorted(found, key=lambda p: (p.cost_tier, p.latency_tier))

    def pricing(self) -> dict[str, dict[str, float]]:
        """Prices keyed by pinned model id, in the shape aie_core.PricingTable expects."""
        return {p.model_id: p.pricing_entry() for p in self.profiles.values()}


def northwind_catalog() -> ModelCatalog:
    """The illustrative catalog used throughout Chapter 7. Names are invented; tiers and
    prices are made up to have realistic ratios, not to describe any vendor."""
    return ModelCatalog.from_profiles([
        ModelProfile(alias="nw-small", model_id="small-instruct-2026-03", provider="local",
                     context_tokens=32_000, max_output_tokens=4_000, supports_json_schema=True,
                     data_zones=("onprem",), cost_tier=Tier.LOW, latency_tier=Tier.LOW,
                     input_per_1m=0.10, output_per_1m=0.40,
                     notes="self-hosted; stays inside Northwind's network"),
        ModelProfile(alias="nw-general", model_id="general-2026-02", provider="cloud-a",
                     context_tokens=128_000, max_output_tokens=8_000, supports_tools=True,
                     supports_json_schema=True, supports_vision=True, data_zones=("eu",),
                     cost_tier=Tier.MEDIUM, latency_tier=Tier.MEDIUM,
                     input_per_1m=1.00, output_per_1m=4.00),
        ModelProfile(alias="nw-reasoning", model_id="reasoner-2026-01", provider="cloud-a",
                     context_tokens=200_000, max_output_tokens=32_000, supports_tools=True,
                     supports_json_schema=True, reasoning_efforts=("low", "medium", "high"),
                     data_zones=("eu",), cost_tier=Tier.HIGH, latency_tier=Tier.VERY_HIGH,
                     input_per_1m=5.00, output_per_1m=20.00),
        ModelProfile(alias="nw-longctx", model_id="longctx-2025-12", provider="cloud-b",
                     context_tokens=1_000_000, max_output_tokens=8_000, supports_tools=False,
                     supports_json_schema=False, data_zones=("any",),
                     cost_tier=Tier.HIGH, latency_tier=Tier.HIGH,
                     input_per_1m=2.50, output_per_1m=10.00,
                     notes="large window, no tools, no native schema mode"),
    ])


__all__ = [
    "Tier", "ModelProfile", "Requirements", "Gap", "capability_gaps", "is_compatible",
    "ModelCatalog", "northwind_catalog",
]
