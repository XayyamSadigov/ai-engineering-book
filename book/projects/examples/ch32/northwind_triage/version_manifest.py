# path: book/projects/examples/ch32/northwind_triage/version_manifest.py
"""The version manifest: every artifact that can change an AI system's output, in one record.

A manifest is built once at startup (code, static artifacts) and refined per request (the
prompt and model that flags selected). It is attached to every trace span, stored with every
evaluation run, and compared in CI so that a change in behavior can be attributed to a change
in exactly one component.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def content_hash(data: str | bytes, length: int = 12) -> str:
    """Short sha256 of content. Version labels are claims; hashes are evidence."""
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(raw).hexdigest()[:length]


def versioned(version: str, content: str | bytes | None = None) -> str:
    """'1.1.0#3f9a0c2b7d1e': a human label plus the hash of what the label points at."""
    return f"{version}#{content_hash(content)}" if content is not None else version


class VersionManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    app: str
    app_version: str
    git_sha: str
    environment: str
    prompts: dict[str, str] = Field(default_factory=dict)        # prompt id -> version#hash
    models: dict[str, str] = Field(default_factory=dict)         # role -> provider/model
    embedding_model: str | None = None
    index_version: str | None = None
    datasets: dict[str, str] = Field(default_factory=dict)       # dataset -> version#hash
    evaluators: dict[str, str] = Field(default_factory=dict)     # evaluator -> version#hash
    tool_schemas: dict[str, str] = Field(default_factory=dict)   # tool -> version#hash
    flags: dict[str, str] = Field(default_factory=dict)          # flag -> assigned variant
    config: dict[str, str] = Field(default_factory=dict)         # behavior-relevant settings

    # ------------------------------------------------------------------ identity
    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def fingerprint(self) -> str:
        """Stable id of this exact combination. Two requests with the same fingerprint ran
        the same system; group traces and eval results by it."""
        return content_hash(self.canonical_json(), length=16)

    # ------------------------------------------------------------------ derivation
    def with_updates(self, **changes: Any) -> "VersionManifest":
        """Return a copy; dict fields are merged rather than replaced."""
        data = self.model_dump()
        for key, value in changes.items():
            if isinstance(value, dict) and isinstance(data.get(key), dict):
                data[key] = {**data[key], **value}
            else:
                data[key] = value
        return VersionManifest(**data)

    # ------------------------------------------------------------------ export
    def flatten(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for key, value in self.model_dump(mode="json").items():
            if isinstance(value, dict):
                for sub, v in sorted(value.items()):
                    out[f"{key}.{sub}"] = str(v)
            elif value is not None:
                out[key] = str(value)
        return out

    def as_semconv_attributes(self) -> dict[str, str]:
        """The version facts under Chapter 31's span names (`prompt.id`, `prompt.version`,
        `index.version`, `llm.model`, `app.version`), so Chapter 31's trace tooling can group
        this service's traffic by version. Exact only for one prompt and one model; with more,
        rely on the `version.*` attributes."""
        attrs = {"app.version": self.app_version}
        if len(self.prompts) == 1:
            ((prompt_id, label),) = self.prompts.items()
            attrs["prompt.id"], attrs["prompt.version"] = prompt_id, label
        if len(self.models) == 1:
            attrs["llm.model"] = next(iter(self.models.values()))
        if self.index_version:
            attrs["index.version"] = self.index_version
        return attrs

    def as_span_attributes(self, prefix: str = "version") -> dict[str, str]:
        attrs = {f"{prefix}.{k}": v for k, v in self.flatten().items()}
        attrs[f"{prefix}.fingerprint"] = self.fingerprint()
        return attrs

    # ------------------------------------------------------------------ comparison
    def diff(self, other: "VersionManifest", ignore: tuple[str, ...] = ("git_sha", "app_version",
                                                                         "environment")) -> dict[str, tuple[str | None, str | None]]:
        """Keys whose values differ, as (self, other). Used by CI to enforce attribution."""
        a, b = self.flatten(), other.flatten()
        changed: dict[str, tuple[str | None, str | None]] = {}
        for key in sorted(set(a) | set(b)):
            if key.split(".")[0] in ignore:
                continue
            if a.get(key) != b.get(key):
                changed[key] = (a.get(key), b.get(key))
        return changed

    def changed_components(self, other: "VersionManifest") -> set[str]:
        """Top-level components that differ: {'prompts', 'models'} means two things changed."""
        return {key.split(".")[0] for key in self.diff(other)}
