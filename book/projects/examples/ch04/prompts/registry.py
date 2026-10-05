# path: book/projects/examples/ch04/prompts/registry.py
"""Versioned prompt files: parse, validate, hash, look up, and pin.

A prompt file is Markdown with front matter (TOML between `+++` lines, or YAML between
`---` lines when PyYAML is installed) followed by message sections:

    +++
    id = "ticket.classify"
    version = "1.2.0"
    temperature = 0.0
    output_schema = "schemas/ticket_classification.json"
    [variables.ticket_body]
    max_chars = 4000
    +++
    === system ===
    You classify Northwind support tickets...
    === user ===
    {{ ticket_body }}

Files live at `<root>/<id>/<version>.md`. A published id@version is immutable: its content
hash is recorded in `prompts.lock`, and `verify_lock` fails CI when a file changes without
a version bump.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aie_core.llm.types import CompletionRequest, Message, Role

from .template import PromptDefinitionError, PromptTemplate, VariableSpec

_SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
_SECTION = re.compile(r"^===\s*(system|user|assistant)\s*===\s*$", re.MULTILINE)
_ID = r"^[a-z0-9][a-z0-9_.\-]*$"


class ModelHints(BaseModel):
    """Advice to the router (Chapter 7), never a hard binding to a vendor model name."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tier: Literal["small", "medium", "large"] | None = None
    needs_response_schema: bool = False
    min_context_tokens: int | None = None
    notes: str = ""


class PromptSpec(BaseModel):
    """Front matter. Everything that changes model behavior lives here or in the body, so
    the content hash covers the whole decoding policy, not just the wording."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=_ID)
    version: str = Field(pattern=_SEMVER.pattern)
    description: str = ""
    owner: str = ""
    status: Literal["draft", "active", "deprecated"] = "active"
    model_hints: ModelHints = ModelHints()
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, gt=0)
    stop: list[str] | None = None
    output_schema: str | None = None  # path relative to the registry root
    variables: dict[str, VariableSpec] = Field(default_factory=dict)


@dataclass(frozen=True)
class PromptRef:
    """What a trace, a log line, or an eval result records about the prompt it used."""

    id: str
    version: str
    content_hash: str

    def span_attributes(self) -> dict[str, str]:
        return {"prompt.id": self.id, "prompt.version": self.version, "prompt.hash": self.content_hash[:16]}

    def __str__(self) -> str:
        return f"{self.id}@{self.version}"


@dataclass(frozen=True)
class RenderedPrompt:
    ref: PromptRef
    spec: PromptSpec
    messages: list[Message]
    output_schema: dict[str, Any] | None
    render_ms: float

    def to_request(self, **overrides: Any) -> CompletionRequest:
        """Build the provider-neutral request. The prompt's decoding policy is the default;
        callers may override (an experiment, a router choosing `model`), and the override is
        visible in metadata so traces never claim a policy that was not used."""
        params: dict[str, Any] = {
            "messages": self.messages,
            "temperature": self.spec.temperature,
            "max_tokens": self.spec.max_tokens,
            "stop": self.spec.stop,
            "response_schema": self.output_schema,
            "metadata": {**self.ref.span_attributes(), "prompt.overrides": sorted(overrides)},
        }
        params.update(overrides)
        return CompletionRequest(**params)


@dataclass(frozen=True)
class PromptVersion:
    spec: PromptSpec
    template: PromptTemplate = field(repr=False)
    content_hash: str
    output_schema: dict[str, Any] | None = field(default=None, repr=False)
    source: str = ""

    @property
    def ref(self) -> PromptRef:
        return PromptRef(self.spec.id, self.spec.version, self.content_hash)

    def render(self, variables: Mapping[str, Any]) -> RenderedPrompt:
        started = time.perf_counter()
        messages = self.template.render(variables)
        return RenderedPrompt(
            ref=self.ref,
            spec=self.spec,
            messages=messages,
            output_schema=self.output_schema,
            render_ms=(time.perf_counter() - started) * 1000,
        )


# ---------------------------------------------------------------- parsing
def split_front_matter(text: str, where: str = "<string>") -> tuple[dict[str, Any], str]:
    text = text.replace("\r\n", "\n").lstrip("﻿")
    for fence, loader in (("+++", "toml"), ("---", "yaml")):
        if text.startswith(fence + "\n"):
            end = text.find("\n" + fence + "\n", len(fence))
            if end == -1:
                raise PromptDefinitionError(f"{where}: unterminated {fence} front matter")
            raw, body = text[len(fence) + 1 : end], text[end + len(fence) + 2 :]
            return _load_front_matter(raw, loader, where), body
    raise PromptDefinitionError(f"{where}: missing front matter (+++ TOML or --- YAML)")


def _load_front_matter(raw: str, loader: str, where: str) -> dict[str, Any]:
    try:
        if loader == "toml":
            return tomllib.loads(raw)
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise PromptDefinitionError(f"{where}: YAML front matter needs PyYAML (pip install pyyaml)") from exc
        data = yaml.safe_load(raw)
        if not isinstance(data, dict):
            raise PromptDefinitionError(f"{where}: front matter must be a mapping")
        return data
    except PromptDefinitionError:
        raise
    except Exception as exc:  # tomllib.TOMLDecodeError, yaml.YAMLError
        raise PromptDefinitionError(f"{where}: invalid {loader} front matter: {exc}") from exc


def split_sections(body: str, where: str) -> list[tuple[Role, str]]:
    matches = list(_SECTION.finditer(body))
    if not matches:
        raise PromptDefinitionError(f"{where}: no '=== system|user|assistant ===' sections")
    if body[: matches[0].start()].strip():
        raise PromptDefinitionError(f"{where}: text before the first section marker")
    sections = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections.append((Role(m.group(1)), body[m.end() : end].strip("\n")))
    return sections


def content_hash(file_text: str, schema_text: str | None) -> str:
    """sha256 over the normalized file and its output schema: a schema edit changes behavior
    as surely as a wording edit, so it must change the hash too."""
    h = hashlib.sha256(file_text.replace("\r\n", "\n").encode("utf-8"))
    if schema_text is not None:
        h.update(b"\x00schema\x00")
        h.update(json.dumps(json.loads(schema_text), sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return h.hexdigest()


def load_prompt_text(text: str, *, root: Path | None = None, where: str = "<string>") -> PromptVersion:
    meta, body = split_front_matter(text, where)
    try:
        spec = PromptSpec.model_validate(meta)
    except ValidationError as exc:
        raise PromptDefinitionError(f"{where}: invalid front matter:\n{exc}") from exc
    schema, schema_text = None, None
    if spec.output_schema:
        if root is None:
            raise PromptDefinitionError(f"{where}: output_schema needs a registry root")
        schema_path = (root / spec.output_schema).resolve()
        if root.resolve() not in schema_path.parents:
            raise PromptDefinitionError(f"{where}: output_schema escapes the registry root")
        try:
            schema_text = schema_path.read_text(encoding="utf-8")
            schema = json.loads(schema_text)
        except (OSError, json.JSONDecodeError) as exc:
            raise PromptDefinitionError(f"{where}: cannot load output schema {spec.output_schema}: {exc}") from exc
    template = PromptTemplate(split_sections(body, where), spec.variables, name=f"{spec.id}@{spec.version}")
    return PromptVersion(spec, template, content_hash(text, schema_text), schema, where)


def semver_key(version: str) -> tuple[int, int, int]:
    m = _SEMVER.match(version)
    if not m:
        raise ValueError(f"not a MAJOR.MINOR.PATCH version: {version!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


# ---------------------------------------------------------------- the registry
class PromptNotFoundError(KeyError):
    pass


class PromptRegistry:
    """In-memory index of every prompt version, plus named aliases such as `prod`/`canary`."""

    def __init__(self, versions: Iterable[PromptVersion], aliases: Mapping[str, Mapping[str, str]] | None = None) -> None:
        self._by_id: dict[str, dict[str, PromptVersion]] = {}
        for pv in versions:
            slot = self._by_id.setdefault(pv.spec.id, {})
            if pv.spec.version in slot:
                raise PromptDefinitionError(f"duplicate prompt {pv.ref} in {slot[pv.spec.version].source} and {pv.source}")
            slot[pv.spec.version] = pv
        self.aliases: dict[str, dict[str, str]] = {k: dict(v) for k, v in (aliases or {}).items()}
        for pid, table in self.aliases.items():
            for alias, version in table.items():
                target = self._by_id.get(pid, {}).get(version)
                if target is None:
                    raise PromptDefinitionError(f"alias {pid}:{alias} points at missing version {version}")
                if target.spec.status == "draft":
                    raise PromptDefinitionError(f"alias {pid}:{alias} points at draft {version}; drafts are not servable")

    @classmethod
    def from_directory(cls, root: str | Path) -> "PromptRegistry":
        root = Path(root)
        versions = []
        for path in sorted(root.glob("*/*.md")):
            where = str(path.relative_to(root))
            pv = load_prompt_text(path.read_text(encoding="utf-8"), root=root, where=where)
            if path.parent.name != pv.spec.id or path.stem != pv.spec.version:
                raise PromptDefinitionError(f"{where}: path must be {pv.spec.id}/{pv.spec.version}.md")
            versions.append(pv)
        aliases_path = root / "aliases.toml"
        aliases = tomllib.loads(aliases_path.read_text(encoding="utf-8")) if aliases_path.exists() else {}
        return cls(versions, aliases)

    def ids(self) -> list[str]:
        return sorted(self._by_id)

    def versions(self, prompt_id: str) -> list[str]:
        return sorted(self._require(prompt_id), key=semver_key)

    def get(self, prompt_id: str, version: str = "latest") -> PromptVersion:
        """`version` is an exact MAJOR.MINOR.PATCH, an alias (`prod`), or `latest`, which is
        the highest version whose status is `active`. Drafts are reachable only by exact version."""
        table = self._require(prompt_id)
        if version == "latest":
            active = [v for v in table.values() if v.spec.status == "active"]
            if not active:
                raise PromptNotFoundError(f"{prompt_id}: no active version")
            return max(active, key=lambda v: semver_key(v.spec.version))
        resolved = self.aliases.get(prompt_id, {}).get(version, version)
        if resolved not in table:
            raise PromptNotFoundError(f"{prompt_id}@{version} not found; have {self.versions(prompt_id)}")
        return table[resolved]

    def _require(self, prompt_id: str) -> dict[str, PromptVersion]:
        if prompt_id not in self._by_id:
            raise PromptNotFoundError(f"unknown prompt id {prompt_id!r}")
        return self._by_id[prompt_id]

    def lock(self) -> dict[str, str]:
        """Hashes of every servable version. Drafts are excluded so they can be edited freely;
        flipping a draft to `active` is itself an edit, and from then on the hash is pinned."""
        return {
            str(pv.ref): pv.content_hash
            for table in self._by_id.values()
            for pv in table.values()
            if pv.spec.status != "draft"
        }

    def verify_lock(self, lock: Mapping[str, str]) -> list[str]:
        """Problems that must fail CI: a published version whose content changed, or one
        that disappeared. New versions not yet in the lock are fine (the lock is then updated)."""
        current = self.lock()
        problems = []
        for key, expected in sorted(lock.items()):
            if key not in current:
                problems.append(f"{key}: published version was deleted")
            elif current[key] != expected:
                problems.append(f"{key}: content changed without a version bump")
        return problems


__all__ = [
    "ModelHints",
    "PromptSpec",
    "PromptRef",
    "RenderedPrompt",
    "PromptVersion",
    "PromptRegistry",
    "PromptNotFoundError",
    "load_prompt_text",
    "split_front_matter",
    "split_sections",
    "content_hash",
    "semver_key",
]
