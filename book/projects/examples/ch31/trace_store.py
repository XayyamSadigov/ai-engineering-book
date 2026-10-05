# path: book/projects/examples/ch31/trace_store.py
"""Load span records (JSONL or in-memory) into trace trees you can filter and query.

This is the smallest thing that answers debugging questions offline: rebuild each trace's
tree from parent ids, flag orphans (spans whose parent never arrived, a telemetry gap),
attach evaluation results and user feedback by trace id, and filter by tenant, version,
error class, route, traffic source, or time. A tracing backend does the same with an index;
the queries in `analysis.py` are written against this interface so they port directly.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from aie_core.observability import Span

from semconv import Attr, ErrorClass, SpanName, VERSION_KEYS, normalize

REQUIRED_ROOT_KEYS: tuple[str, ...] = (Attr.TENANT, Attr.ROUTE, Attr.PROMPT_VERSION, Attr.INDEX_VERSION)


@dataclass
class SpanRecord:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    start: float
    end: float | None
    duration_ms: float
    status: str = "ok"
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    resource: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SpanRecord":
        name = d["name"]
        span_id = d.get("span_id") or ""
        return cls(
            # spans from older aie_core versions carry no trace id: keep them, but alone
            trace_id=d.get("trace_id") or f"unlinked-{span_id}",
            span_id=span_id,
            parent_span_id=d.get("parent_span_id"),
            name=name,
            start=float(d.get("start") or 0.0),
            end=d.get("end"),
            duration_ms=float(d.get("duration_ms") or 0.0),
            status=d.get("status", "ok"),
            attributes=normalize(name, d.get("attributes") or {}),
            events=list(d.get("events") or []),
            resource=dict(d.get("resource") or {}),
        )

    @classmethod
    def from_span(cls, span: Span) -> "SpanRecord":
        return cls.from_dict(span.to_dict())

    def get(self, key: str, default: Any = None) -> Any:
        if key in self.attributes:
            return self.attributes[key]
        return self.resource.get(key, default)

    @property
    def is_error(self) -> bool:
        return self.status == "error"


class TraceTree:
    """One request: its spans, their parent/child structure, and joined labels."""

    def __init__(self, trace_id: str, spans: list[SpanRecord]) -> None:
        self.trace_id = trace_id
        self.spans = sorted(spans, key=lambda s: (s.start, s.span_id))
        self.by_id = {s.span_id: s for s in self.spans}
        self.children: dict[str | None, list[SpanRecord]] = defaultdict(list)
        self.orphans: list[SpanRecord] = []
        for s in self.spans:
            if s.parent_span_id is not None and s.parent_span_id not in self.by_id:
                self.orphans.append(s)
                self.children[None].append(s)
            else:
                self.children[s.parent_span_id].append(s)
        roots = self.children.get(None, [])
        real_roots = [s for s in roots if s.parent_span_id is None]
        self.root: SpanRecord | None = (real_roots or roots or [None])[0]
        self.evals: list[dict[str, Any]] = []
        self.feedback: list[dict[str, Any]] = []

    # ---------------------------------------------------------------- lookups
    def get(self, key: str, default: Any = None) -> Any:
        """Root attribute, then resource, then the first span that carries the key."""
        if self.root is not None:
            value = self.root.get(key)
            if value is not None:
                return value
        for s in self.spans:
            value = s.get(key)
            if value is not None:
                return value
        return default

    def find(self, name: str) -> list[SpanRecord]:
        return [s for s in self.spans if s.name == name]

    def first(self, name: str) -> SpanRecord | None:
        found = self.find(name)
        return found[0] if found else None

    def walk(self) -> Iterator[tuple[int, SpanRecord]]:
        def visit(span: SpanRecord, depth: int) -> Iterator[tuple[int, SpanRecord]]:
            yield depth, span
            for child in self.children.get(span.span_id, []):
                yield from visit(child, depth + 1)

        for top in self.children.get(None, []):
            yield from visit(top, 0)

    def total(self, key: str) -> float:
        return float(sum(s.attributes.get(key) or 0 for s in self.spans if isinstance(s.attributes.get(key), (int, float))))

    # ---------------------------------------------------------------- derived properties
    @property
    def start(self) -> float:
        return self.root.start if self.root else (self.spans[0].start if self.spans else 0.0)

    @property
    def duration_ms(self) -> float:
        return self.root.duration_ms if self.root else 0.0

    @property
    def tenant(self) -> str | None:
        return self.get(Attr.TENANT)

    @property
    def versions(self) -> dict[str, Any]:
        return {k: v for k in VERSION_KEYS if (v := self.get(k)) is not None}

    def _recovered(self, s: SpanRecord) -> bool:
        # a failed provider attempt whose parent generation succeeded was retried or fell back
        parent = self.by_id.get(s.parent_span_id or "")
        return s.name == SpanName.LLM_ATTEMPT and parent is not None and not parent.is_error

    @property
    def error_classes(self) -> set[str]:
        """Failure classes that affected the outcome. Recovered attempts are excluded."""
        return {s.attributes[Attr.ERROR_CLASS] for s in self.spans
                if s.attributes.get(Attr.ERROR_CLASS) and not self._recovered(s)}

    @property
    def recovered_errors(self) -> set[str]:
        """Transient failures absorbed by retries or fallbacks: a reliability signal, not quality."""
        return {s.attributes[Attr.ERROR_CLASS] for s in self.spans
                if s.attributes.get(Attr.ERROR_CLASS) and self._recovered(s)}

    @property
    def has_error(self) -> bool:
        return bool(self.error_classes) or (self.root is not None and self.root.is_error)

    @property
    def cost_usd(self) -> float:
        # sum provider attempts only; llm.generate repeats the total and would double count
        attempts = self.find(SpanName.LLM_ATTEMPT)
        if attempts:
            return float(sum(s.attributes.get(Attr.LLM_COST) or 0.0 for s in attempts))
        return float(sum(s.attributes.get(Attr.LLM_COST) or 0.0 for s in self.find(SpanName.GENERATE)))

    @property
    def avoided_cost_usd(self) -> float:
        """Spend a response-cache hit saved (Chapter 30 reports it next to chargeback)."""
        return float(sum(s.attributes.get(Attr.LLM_AVOIDED_COST) or 0.0 for s in self.find(SpanName.LLM_ATTEMPT)))

    @property
    def gold_ids(self) -> list[str] | None:
        for e in self.evals:
            if e.get("gold_ids"):
                return list(e["gold_ids"])
        return None

    @property
    def eval_passed(self) -> bool | None:
        verdicts = [e["passed"] for e in self.evals if e.get("passed") is not None]
        return all(verdicts) if verdicts else None

    @property
    def feedback_value(self) -> int | None:
        return self.feedback[-1]["value"] if self.feedback else None

    @property
    def complete(self) -> bool:
        """True when the trace can answer the lineage questions: a root, no orphans, required keys."""
        return self.root is not None and not self.orphans and all(self.get(k) is not None for k in REQUIRED_ROOT_KEYS)

    # ---------------------------------------------------------------- display
    def render(self, keys: Iterable[str] = ()) -> str:
        lines = []
        for depth, s in self.walk():
            extras = " ".join(f"{k}={s.attributes[k]}" for k in keys if k in s.attributes)
            flag = " !ERR" if s.is_error else ""
            lines.append(f"{'  ' * depth}{s.name:<{max(4, 22 - 2 * depth)}} {s.duration_ms:8.1f} ms{flag} {extras}".rstrip())
        return "\n".join(lines)


class TraceStore:
    """An in-memory collection of traces with joins and filters."""

    def __init__(self, records: Iterable[SpanRecord] = ()) -> None:
        self._spans: dict[str, list[SpanRecord]] = defaultdict(list)
        self._trees: dict[str, TraceTree] | None = None
        self._evals: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._feedback: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.unmatched_feedback: list[dict[str, Any]] = []
        for r in records:
            self.add(r)

    # ---------------------------------------------------------------- loading
    def add(self, record: SpanRecord) -> None:
        self._spans[record.trace_id].append(record)
        self._trees = None

    @classmethod
    def from_jsonl(cls, *paths: str | Path) -> "TraceStore":
        store = cls()
        for path in paths:
            with Path(path).open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        store.add(SpanRecord.from_dict(json.loads(line)))
        return store

    @classmethod
    def from_spans(cls, spans: Iterable[Span]) -> "TraceStore":
        return cls(SpanRecord.from_span(s) for s in spans)

    @classmethod
    def _from_trees(cls, trees: Iterable[TraceTree]) -> "TraceStore":
        store = cls()
        store._trees = {t.trace_id: t for t in trees}
        for t in store._trees.values():
            store._spans[t.trace_id] = list(t.spans)
            store._evals[t.trace_id] = list(t.evals)
            store._feedback[t.trace_id] = list(t.feedback)
        return store

    def _build(self) -> dict[str, TraceTree]:
        if self._trees is None:
            self._trees = {}
            for trace_id, spans in self._spans.items():
                tree = TraceTree(trace_id, spans)
                tree.evals = list(self._evals.get(trace_id, []))
                tree.feedback = list(self._feedback.get(trace_id, []))
                self._trees[trace_id] = tree
        return self._trees

    # ---------------------------------------------------------------- joins
    def attach_evals(self, records: Iterable[dict[str, Any]]) -> int:
        """Eval records carry `trace_id`; returns how many matched a known trace."""
        trees, matched = self._build(), 0
        for r in records:
            tid = r.get("trace_id")
            if tid in trees:
                self._evals[tid].append(r)
                trees[tid].evals.append(r)
                matched += 1
        return matched

    def attach_feedback(self, records: Iterable[dict[str, Any]]) -> int:
        """Feedback references either `trace_id` or the client-visible `response_id`."""
        trees = self._build()
        by_response = {t.get(Attr.RESPONSE_ID): t.trace_id for t in trees.values() if t.get(Attr.RESPONSE_ID)}
        matched = 0
        for r in records:
            tid = r.get("trace_id") or by_response.get(r.get("response_id"))
            if tid in trees:
                self._feedback[tid].append(r)
                trees[tid].feedback.append(r)
                matched += 1
            else:
                self.unmatched_feedback.append(r)
        return matched

    # ---------------------------------------------------------------- access
    def traces(self) -> list[TraceTree]:
        return sorted(self._build().values(), key=lambda t: t.start)

    def get(self, trace_id: str) -> TraceTree:
        return self._build()[trace_id]

    def __len__(self) -> int:
        return len(self._build())

    def __iter__(self) -> Iterator[TraceTree]:
        return iter(self.traces())

    def filter(
        self,
        *,
        tenant: str | None = None,
        route: str | None = None,
        traffic: str | None = None,
        error_class: ErrorClass | str | None = None,
        versions: dict[str, Any] | None = None,
        since: float | None = None,
        until: float | None = None,
        labeled: bool | None = None,
        predicate: Callable[[TraceTree], bool] | None = None,
    ) -> "TraceStore":
        wanted_error = ErrorClass(error_class).value if error_class is not None else None

        def keep(t: TraceTree) -> bool:
            if t.root is None or t.root.name != SpanName.REQUEST:
                return False
            if tenant is not None and t.tenant != tenant:
                return False
            if route is not None and t.get(Attr.ROUTE) != route:
                return False
            if traffic is not None and t.get(Attr.TRAFFIC) != traffic:
                return False
            if wanted_error is not None and wanted_error not in t.error_classes:
                return False
            if versions and any(t.get(k) != v for k, v in versions.items()):
                return False
            if since is not None and t.start < since:
                return False
            if until is not None and t.start >= until:
                return False
            if labeled is not None and (t.gold_ids is not None) != labeled:
                return False
            return predicate(t) if predicate is not None else True

        return TraceStore._from_trees(t for t in self.traces() if keep(t))

    def completeness(self, required: Iterable[str] = REQUIRED_ROOT_KEYS) -> dict[str, float]:
        """Fraction of request traces missing each required key, plus orphan rate."""
        traces = [t for t in self.traces() if t.root is not None and t.root.name == SpanName.REQUEST]
        n = max(1, len(traces))
        out = {k: sum(1 for t in traces if t.get(k) is None) / n for k in required}
        out["orphaned"] = sum(1 for t in self.traces() if t.orphans) / max(1, len(self))
        return out


__all__ = ["SpanRecord", "TraceTree", "TraceStore", "REQUIRED_ROOT_KEYS"]
