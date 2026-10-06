# path: book/projects/evalkit/evalkit/cases.py
"""Evaluation cases and versioned datasets.

A case is data, not test code: an input, what good looks like (`expected` and/or `rubric`),
tags that define slices, and free-form metadata such as the source of the case or the entity
it belongs to. A dataset is an ordered, uniquely keyed list of cases with a name, a
human-assigned version, and a content hash computed from the cases themselves. The hash is
what makes a score reproducible evidence: two runs with the same hash saw the same cases.

File format (JSONL): an optional first line `{"_dataset": {"name": ..., "version": ...,
"description": ...}}` followed by one case per line.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DatasetError(ValueError):
    """Raised for malformed datasets: duplicate ids, bad header, unexpected hash."""


class EvalCase(BaseModel):
    """One evaluation case. `input` and `expected` are any JSON-serializable values. Unknown
    fields are rejected: a typo such as `expeced` or `tag` would otherwise silently drop data."""

    model_config = ConfigDict(extra="forbid")

    id: str
    input: Any
    expected: Any = None
    rubric: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _id_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("case id must not be blank")
        return v

    def group_key(self, group_by: str | None) -> str:
        """The unit that must not straddle a split: an entity id from metadata, else the case id."""
        if group_by is None:
            return self.id
        value = self.metadata.get(group_by)
        return str(value) if value is not None else self.id

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _normalize_for_dup(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


class Dataset:
    """An ordered, versioned collection of `EvalCase` with unique ids."""

    def __init__(
        self,
        cases: Iterable[EvalCase],
        *,
        name: str,
        version: str = "0",
        description: str = "",
    ) -> None:
        self.cases: list[EvalCase] = list(cases)
        self.name = name
        self.version = version
        self.description = description
        seen: set[str] = set()
        dupes = [c.id for c in self.cases if c.id in seen or seen.add(c.id)]  # type: ignore[func-returns-value]
        if dupes:
            raise DatasetError(f"duplicate case ids in {name}: {sorted(set(dupes))[:5]}")
        self._by_id = {c.id: c for c in self.cases}

    # ------------------------------------------------------------------ container protocol
    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[EvalCase]:
        return iter(self.cases)

    def __contains__(self, case_id: object) -> bool:
        return case_id in self._by_id

    def get(self, case_id: str) -> EvalCase:
        return self._by_id[case_id]

    @property
    def ids(self) -> list[str]:
        return [c.id for c in self.cases]

    # ------------------------------------------------------------------ identity
    @property
    def content_hash(self) -> str:
        """SHA-256 over the canonical JSON of every case, order-independent (sorted by id)."""
        h = hashlib.sha256()
        for case in sorted(self.cases, key=lambda c: c.id):
            h.update(case.canonical_json().encode("utf-8"))
            h.update(b"\n")
        return h.hexdigest()

    @property
    def fingerprint(self) -> str:
        """Short identity used in run records and reports: name@version#hash12."""
        return f"{self.name}@{self.version}#{self.content_hash[:12]}"

    def verify_hash(self, expected: str) -> None:
        """Fail loudly when a frozen dataset was edited. Accepts a full hash or a prefix."""
        if not self.content_hash.startswith(expected):
            raise DatasetError(
                f"dataset {self.name}@{self.version} changed: expected hash {expected[:12]}, "
                f"got {self.content_hash[:12]}"
            )

    # ------------------------------------------------------------------ persistence
    @classmethod
    def load_jsonl(cls, path: str | Path, *, name: str | None = None, version: str | None = None,
                   verify: bool = True) -> "Dataset":
        """Load a dataset; when the header records a `content_hash`, check it (`verify=False` skips).
        The hash covers each case's canonical JSON, so adding a field to `EvalCase` changes every
        stored hash: re-freeze datasets deliberately after such a change."""
        path = Path(path)
        header: dict[str, Any] = {}
        cases: list[EvalCase] = []
        with path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
                if lineno == 1 and isinstance(obj, dict) and "_dataset" in obj:
                    header = obj["_dataset"]
                    continue
                try:
                    cases.append(EvalCase.model_validate(obj))
                except ValueError as exc:
                    raise DatasetError(f"{path}:{lineno}: invalid case: {exc}") from exc
        ds = cls(
            cases,
            name=name or header.get("name") or path.stem,
            version=version or str(header.get("version", "0")),
            description=header.get("description", ""),
        )
        if verify and header.get("content_hash"):
            ds.verify_hash(str(header["content_hash"]))
        return ds

    def save_jsonl(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "_dataset": {
                "name": self.name,
                "version": self.version,
                "description": self.description,
                "content_hash": self.content_hash,
            }
        }
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(header, ensure_ascii=False) + "\n")
            for case in self.cases:
                f.write(json.dumps(case.model_dump(mode="json"), ensure_ascii=False) + "\n")
        return path

    # ------------------------------------------------------------------ views
    def subset(self, ids: Iterable[str], *, name: str | None = None) -> "Dataset":
        wanted = set(ids)
        return Dataset(
            [c for c in self.cases if c.id in wanted],
            name=name or self.name,
            version=self.version,
            description=self.description,
        )

    def filter(
        self,
        predicate: Callable[[EvalCase], bool] | None = None,
        *,
        tags: Iterable[str] | None = None,
        name: str | None = None,
    ) -> "Dataset":
        """Cases matching the predicate and carrying every tag in `tags`."""
        required = set(tags or [])
        keep = [c for c in self.cases if required.issubset(c.tags) and (predicate is None or predicate(c))]
        return Dataset(keep, name=name or self.name, version=self.version, description=self.description)

    def tag_counts(self) -> dict[str, int]:
        return dict(Counter(t for c in self.cases for t in c.tags).most_common())

    def slices(self, prefix: str | None = None) -> dict[str, list[str]]:
        """Map each tag (optionally only tags starting with `prefix`) to the case ids carrying it."""
        out: dict[str, list[str]] = {}
        for c in self.cases:
            for t in c.tags:
                if prefix is None or t.startswith(prefix):
                    out.setdefault(t, []).append(c.id)
        return out

    # ------------------------------------------------------------------ splitting
    def split(
        self,
        holdout_fraction: float = 0.3,
        *,
        group_by: str | None = None,
        seed: str | int = 0,
    ) -> tuple["Dataset", "Dataset"]:
        """Deterministic dev/holdout split by group.

        Each group key is hashed with the seed and mapped to [0, 1); groups below
        `holdout_fraction` go to the holdout. Assignment depends only on the group key, so
        adding new cases never moves existing cases between splits, and all cases sharing a
        group (same customer, document, conversation, or paraphrase family) land together.
        """
        if not 0.0 < holdout_fraction < 1.0:
            raise ValueError("holdout_fraction must be in (0, 1)")
        dev: list[EvalCase] = []
        holdout: list[EvalCase] = []
        for c in self.cases:
            digest = hashlib.sha256(f"{seed}:{c.group_key(group_by)}".encode()).hexdigest()
            (holdout if int(digest[:15], 16) / 16**15 < holdout_fraction else dev).append(c)
        return (
            Dataset(dev, name=f"{self.name}-dev", version=self.version, description=self.description),
            Dataset(holdout, name=f"{self.name}-holdout", version=self.version, description=self.description),
        )


class LeakageReport(BaseModel):
    shared_ids: list[str] = Field(default_factory=list)
    shared_groups: list[str] = Field(default_factory=list)
    duplicate_inputs: list[tuple[str, str]] = Field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not (self.shared_ids or self.shared_groups or self.duplicate_inputs)


def check_leakage(a: Dataset, b: Dataset, *, group_by: str | None = None) -> LeakageReport:
    """Find cases that couple two splits: same id, same group, or the same normalized input.

    Normalization lowercases and strips punctuation and whitespace, so trivially reworded
    copies are caught; true paraphrases need a semantic check (embeddings) on top.
    """
    shared_ids = sorted(set(a.ids) & set(b.ids))
    groups_a = {c.group_key(group_by) for c in a} if group_by else set()
    groups_b = {c.group_key(group_by) for c in b} if group_by else set()
    norm_a: dict[str, str] = {}
    for c in a:
        norm_a.setdefault(_normalize_for_dup(c.input), c.id)
    dups = [(norm_a[n], c.id) for c in b if (n := _normalize_for_dup(c.input)) in norm_a and norm_a[n] != c.id]
    return LeakageReport(shared_ids=shared_ids, shared_groups=sorted(groups_a & groups_b), duplicate_inputs=dups)


__all__ = ["EvalCase", "Dataset", "DatasetError", "LeakageReport", "check_leakage"]
