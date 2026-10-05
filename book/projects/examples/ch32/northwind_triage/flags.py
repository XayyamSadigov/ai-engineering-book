# path: book/projects/examples/ch32/northwind_triage/flags.py
"""Feature flags with deterministic percentage rollout and a kill switch.

Assignment is a pure function of (flag salt, unit id): no database lookup, no randomness,
identical on every replica and in every replay. Ramping a variant from 5% to 20% keeps the
first 5% in it, because buckets are compared against a cumulative threshold.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

BUCKETS = 10_000  # basis points: 1 bucket = 0.01% of traffic

Reason = Literal["kill_switch", "override", "allocation", "unknown_flag", "disabled_environment"]


def bucket(unit_id: str, salt: str) -> int:
    """Map a unit (user, tenant, conversation) to [0, BUCKETS). Uniform and stable."""
    digest = hashlib.sha256(f"{salt}:{unit_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % BUCKETS


class Allocation(BaseModel):
    model_config = ConfigDict(frozen=True)
    variant: str
    percent: float = Field(ge=0.0, le=100.0)


class FlagConfig(BaseModel):
    """One flag. Allocations are ordered; the LAST one should be the control so that
    increasing a treatment percentage only moves users out of control, never between
    treatments."""

    model_config = ConfigDict(frozen=True)

    name: str
    allocations: tuple[Allocation, ...]
    salt: str | None = None            # defaults to the flag name; change it to re-shuffle
    kill_switch: bool = False
    safe_variant: str = "control"      # served when killed, unknown, or disabled
    overrides: dict[str, str] = Field(default_factory=dict)  # unit id -> variant (QA, dogfood)
    environments: tuple[str, ...] = ()  # empty = all environments

    @model_validator(mode="after")
    def _check(self) -> "FlagConfig":
        names = [a.variant for a in self.allocations]
        if len(set(names)) != len(names):
            raise ValueError(f"flag {self.name}: duplicate variants {names}")
        total = sum(a.percent for a in self.allocations)
        if abs(total - 100.0) > 1e-9:
            raise ValueError(f"flag {self.name}: allocations sum to {total}, expected 100")
        if self.safe_variant not in names:
            raise ValueError(f"flag {self.name}: safe_variant {self.safe_variant!r} not in {names}")
        unknown = set(self.overrides.values()) - set(names)
        if unknown:
            raise ValueError(f"flag {self.name}: overrides use unknown variants {sorted(unknown)}")
        return self

    @property
    def effective_salt(self) -> str:
        return self.salt or self.name

    def percent_of(self, variant: str) -> float:
        return sum(a.percent for a in self.allocations if a.variant == variant)


class Assignment(BaseModel):
    model_config = ConfigDict(frozen=True)
    flag: str
    variant: str
    reason: Reason
    bucket: int | None = None


class FlagEvaluator:
    """Evaluates flags from an immutable snapshot of configs. Never raises in the request path:
    an unknown flag returns the safe variant, because a typo in a flag name must not take the
    service down."""

    def __init__(self, configs: dict[str, FlagConfig], environment: str = "dev") -> None:
        self._configs = dict(configs)
        self.environment = environment

    @classmethod
    def from_dict(cls, raw: dict[str, Any], environment: str = "dev") -> "FlagEvaluator":
        configs = {name: FlagConfig(name=name, **spec) for name, spec in raw.items()}
        return cls(configs, environment)

    @classmethod
    def from_file(cls, path: str | Path, environment: str = "dev") -> "FlagEvaluator":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")), environment)

    def config(self, name: str) -> FlagConfig | None:
        return self._configs.get(name)

    def evaluate(self, name: str, unit_id: str) -> Assignment:
        cfg = self._configs.get(name)
        if cfg is None:
            return Assignment(flag=name, variant="control", reason="unknown_flag")
        if cfg.kill_switch:
            return Assignment(flag=name, variant=cfg.safe_variant, reason="kill_switch")
        if cfg.environments and self.environment not in cfg.environments:
            return Assignment(flag=name, variant=cfg.safe_variant, reason="disabled_environment")
        if unit_id in cfg.overrides:
            return Assignment(flag=name, variant=cfg.overrides[unit_id], reason="override")
        b = bucket(unit_id, cfg.effective_salt)
        threshold = 0.0
        for alloc in cfg.allocations:
            threshold += alloc.percent * BUCKETS / 100.0
            if b < threshold:
                return Assignment(flag=name, variant=alloc.variant, reason="allocation", bucket=b)
        # Floating-point remainder: fall through to the last allocation.
        return Assignment(flag=name, variant=cfg.allocations[-1].variant, reason="allocation", bucket=b)

    # --------------------------------------------------------------- operations
    def kill(self, name: str) -> "FlagEvaluator":
        """Return a new evaluator with the flag killed. Snapshots are immutable so that one
        request never sees half an update."""
        cfg = self._configs[name]
        return FlagEvaluator({**self._configs, name: cfg.model_copy(update={"kill_switch": True})},
                             self.environment)

    def ramp(self, name: str, variant: str, percent: float, control: str = "control") -> "FlagEvaluator":
        """Set `variant` to `percent`, taking or giving the difference from `control`."""
        cfg = self._configs[name]
        current = cfg.percent_of(variant)
        delta = percent - current
        new_allocs = []
        for a in cfg.allocations:
            if a.variant == variant:
                new_allocs.append(Allocation(variant=variant, percent=percent))
            elif a.variant == control:
                new_allocs.append(Allocation(variant=control, percent=a.percent - delta))
            else:
                new_allocs.append(a)
        new_cfg = FlagConfig(**{**cfg.model_dump(), "allocations": new_allocs})
        return FlagEvaluator({**self._configs, name: new_cfg}, self.environment)

    def snapshot_hash(self) -> str:
        payload = json.dumps({n: c.model_dump(mode="json") for n, c in sorted(self._configs.items())},
                             sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
