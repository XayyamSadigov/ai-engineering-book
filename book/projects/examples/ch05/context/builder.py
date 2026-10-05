# path: book/projects/examples/ch05/context/builder.py
"""ContextBuilder: the eight-stage context pipeline under a token budget.

    items -> permission filter -> relevance filter -> dedupe -> measure
          -> allocate (pinned, floors, caps, priority) -> order -> label/render -> manifest

The builder is pure: no I/O except an optional tracer. Fetching candidates (retrieval,
tool calls, memory lookups) happens before it; calling the model happens after it.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

from aie_core.llm.tokens import count_tokens
from aie_core.llm.types import Message, Role
from aie_core.observability import NoopTracer, Tracer

from .filters import Filter, RequestScope, acl_filter
from .items import ContextItem, Section
from .labels import UNTRUSTED_NOTICE, render_item

TokenCounter = Callable[[str], int]
Placement = Literal["ranked", "best_last", "edges"]

MESSAGE_OVERHEAD = 4  # framing tokens per chat message (approximation; see aie_core.llm.tokens)
ITEM_SEPARATOR = 2  # the blank line between items inside one message
DEDUPE_KINDS = frozenset({"evidence", "tool_result", "memory"})


class ContextOverflowError(Exception):
    """Pinned content alone does not fit. Never truncate it: compact upstream or fail the request."""


class SectionLimits(BaseModel):
    floor: int = 0  # tokens reserved for this section if it has candidates
    cap: int | None = None  # hard ceiling for non-pinned items


class BudgetPolicy(BaseModel):
    context_window: int
    output_reserve: int  # max_tokens you will request; reserved first, never spent on input
    safety_margin: float = 0.05  # our token estimate differs from the provider's tokenizer
    sections: dict[Section, SectionLimits] = Field(default_factory=dict)

    @property
    def input_budget(self) -> int:
        return math.floor((self.context_window - self.output_reserve) * (1.0 - self.safety_margin))

    def limits(self, section: Section) -> SectionLimits:
        return self.sections.get(section, SectionLimits())


class ManifestEntry(BaseModel):
    item_id: str
    source_id: str
    kind: str
    section: Section
    tokens: int
    priority: float
    pinned: bool
    trust: str
    included: bool
    reason: str  # "pinned", "admitted", or why it was dropped
    position: int | None = None  # render order among included items


class BuildResult(BaseModel):
    messages: list[Message]
    manifest: list[ManifestEntry]
    input_budget: int
    used_by_section: dict[str, int]
    estimated_prompt_tokens: int
    prefix_hash: str  # hash of the stable system prefix; changes mean cache misses

    @property
    def included(self) -> list[ManifestEntry]:
        return sorted((e for e in self.manifest if e.included), key=lambda e: e.position or 0)

    @property
    def dropped(self) -> list[ManifestEntry]:
        return [e for e in self.manifest if not e.included]

    def drop_reasons(self) -> dict[str, int]:
        return dict(Counter(e.reason.split(":")[0] for e in self.dropped))

    def explain(self) -> str:
        lines = [f"input budget {self.input_budget}, estimated prompt {self.estimated_prompt_tokens}"]
        for e in sorted(self.manifest, key=lambda e: (not e.included, e.position or 0)):
            mark = "+" if e.included else "-"
            lines.append(f"{mark} {e.section.value:<12} {e.tokens:>6}  {e.source_id:<36} {e.reason}")
        return "\n".join(lines)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _shingles(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def edge_order(ranked: Sequence[Any]) -> list[Any]:
    """Best item first, second-best last, third second, fourth second-to-last, ...

    The weakest items end up in the middle, where long-context models attend least.
    """
    front: list[Any] = []
    back: list[Any] = []
    for i, x in enumerate(ranked):
        (front if i % 2 == 0 else back).append(x)
    return front + back[::-1]


class ContextBuilder:
    def __init__(
        self,
        policy: BudgetPolicy,
        *,
        permission_filters: Sequence[Filter] = (acl_filter,),
        relevance_filters: Sequence[Filter] = (),
        placement: Placement = "edges",
        near_duplicate_threshold: float = 0.9,
        counter: TokenCounter | None = None,
        model: str | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.policy = policy
        self.permission_filters = list(permission_filters)
        self.relevance_filters = list(relevance_filters)
        self.placement = placement
        self.near_duplicate_threshold = near_duplicate_threshold
        self.count: TokenCounter = counter or (lambda text: count_tokens(text, model))
        self.tracer = tracer or NoopTracer()

    # ------------------------------------------------------------------ public
    def build(self, items: Sequence[ContextItem], scope: RequestScope) -> BuildResult:
        with self.tracer.span("context.build", user_id=scope.user_id, tenant=scope.tenant) as span:
            result = self._build(list(items), scope)
            span.set_attribute("context.input_budget", result.input_budget)
            span.set_attribute("context.estimated_prompt_tokens", result.estimated_prompt_tokens)
            span.set_attribute("context.included", len(result.included))
            span.set_attribute("context.dropped", len(result.dropped))
            span.set_attribute("context.drop_reasons", result.drop_reasons())
            span.set_attribute("context.used_by_section", result.used_by_section)
            span.set_attribute("context.prefix_hash", result.prefix_hash)
            span.set_attribute("context.manifest", [e.model_dump(mode="json") for e in result.manifest])
            return result

    # --------------------------------------------------------------- pipeline
    def _build(self, items: list[ContextItem], scope: RequestScope) -> BuildResult:
        decisions: dict[str, str] = {}
        # Queries are always pinned: a prompt without the question is not a smaller prompt, it is a wrong one.
        items = [i.model_copy(update={"pinned": True}) if i.kind == "query" else i for i in items]
        order_index = {item.id: n for n, item in enumerate(items)}

        # 1. Permission. Runs on everything; a pinned item the user may not see is an upstream bug.
        allowed: list[ContextItem] = []
        for item in items:
            reason = next((r for f in self.permission_filters if (r := f(item, scope))), None)
            if reason and item.pinned:
                raise PermissionError(f"pinned item {item.source_id} failed permission check: {reason}")
            if reason:
                decisions[item.id] = f"permission:{reason}"
            else:
                allowed.append(item)

        # 2. Relevance. Pinned items skip it.
        relevant: list[ContextItem] = []
        for item in allowed:
            reason = None if item.pinned else next(
                (r for f in self.relevance_filters if (r := f(item, scope))), None
            )
            if reason:
                decisions[item.id] = f"irrelevant:{reason}"
            else:
                relevant.append(item)

        # 3. Dedupe, keeping the higher-priority copy.
        unique = self._dedupe(relevant, decisions)

        # 4. Measure the rendered size, labels and framing included.
        measured = [i.model_copy(update={"tokens": self._measure(i)}) for i in unique]

        # 5. Allocate.
        admitted = self._allocate(measured, decisions, order_index)

        # 6-7. Order and render.
        ordered = self._order(admitted, order_index)
        messages, stable_prefix = self._render(ordered)

        # 8. Attribute.
        position = {item.id: n for n, item in enumerate(ordered)}
        by_id = {i.id: i for i in measured}
        manifest: list[ManifestEntry] = []
        for item in items:
            final = by_id.get(item.id, item)
            manifest.append(
                ManifestEntry(
                    item_id=item.id,
                    source_id=item.source_id,
                    kind=item.kind,
                    section=item.section,
                    tokens=final.tokens if final.tokens is not None else self._measure(item),
                    priority=item.priority,
                    pinned=item.pinned,
                    trust=item.trust.value,
                    included=item.id in position,
                    reason=decisions.get(item.id, "admitted"),
                    position=position.get(item.id),
                )
            )
        used: dict[str, int] = {s.value: 0 for s in Section}
        for item in admitted:
            used[item.section.value] += item.tokens or 0
        estimate = sum(self.count(m.text) + MESSAGE_OVERHEAD for m in messages) + 3
        return BuildResult(
            messages=messages,
            manifest=manifest,
            input_budget=self.policy.input_budget,
            used_by_section=used,
            estimated_prompt_tokens=estimate,
            prefix_hash=hashlib.sha256(stable_prefix.encode()).hexdigest()[:16],
        )

    def _dedupe(self, items: list[ContextItem], decisions: dict[str, str]) -> list[ContextItem]:
        kept: list[tuple[ContextItem, str, set[str]]] = []
        ranked = sorted(items, key=lambda i: (not i.pinned, -i.priority))
        survivors: set[str] = set()
        for item in ranked:
            if item.kind not in DEDUPE_KINDS:
                survivors.add(item.id)
                continue
            norm, sh = _normalize(item.content), _shingles(item.content)
            dup = next(
                (k for k, kn, ks in kept if kn == norm or _jaccard(sh, ks) >= self.near_duplicate_threshold),
                None,
            )
            if dup is not None and not item.pinned:
                decisions[item.id] = f"duplicate_of:{dup.id}"
                continue
            kept.append((item, norm, sh))
            survivors.add(item.id)
        return [i for i in items if i.id in survivors]

    def _measure(self, item: ContextItem) -> int:
        if item.kind == "turn":
            return self.count(item.content) + MESSAGE_OVERHEAD
        return self.count(render_item(item)) + ITEM_SEPARATOR

    def _allocate(
        self, items: list[ContextItem], decisions: dict[str, str], order_index: dict[str, int]
    ) -> list[ContextItem]:
        budget = self.policy.input_budget
        # Fixed rendering cost that is not an item: the untrusted-data notice and message framing.
        total = self.count(UNTRUSTED_NOTICE) + 3 * MESSAGE_OVERHEAD
        used = {s: 0 for s in Section}
        demand = {s: 0 for s in Section}
        for item in items:
            if not item.pinned:
                demand[item.section] += item.tokens or 0
        floors = {s: min(self.policy.limits(s).floor, demand[s]) for s in Section}

        admitted: list[ContextItem] = []
        pinned = [i for i in items if i.pinned]
        for item in pinned:
            total += item.tokens or 0
            used[item.section] += item.tokens or 0
            decisions[item.id] = "pinned"
            admitted.append(item)
        if total > budget:
            raise ContextOverflowError(
                f"pinned content needs {total} tokens, input budget is {budget}; compact state or shrink the system prompt"
            )

        history_closed = False
        rest = sorted((i for i in items if not i.pinned), key=lambda i: (-i.priority, order_index[i.id]))
        for item in rest:
            t, s = item.tokens or 0, item.section
            cap = self.policy.limits(s).cap
            reserved_for_others = sum(max(0, floors[o] - used[o]) for o in Section if o is not s)
            if s is Section.HISTORY and history_closed:
                decisions[item.id] = "history_gap"  # never keep an older turn after dropping a newer one
                continue
            if cap is not None and used[s] + t > cap:
                reason = "section_cap"
            elif total + t > budget - reserved_for_others:
                reason = "reserved_for_other_sections" if total + t <= budget else "budget"
            else:
                total += t
                used[s] += t
                admitted.append(item)
                continue
            decisions[item.id] = reason
            if s is Section.HISTORY:
                history_closed = True
        return admitted

    def _order(self, items: list[ContextItem], order_index: dict[str, int]) -> list[ContextItem]:
        by_section: dict[Section, list[ContextItem]] = {s: [] for s in Section}
        for item in items:
            by_section[item.section].append(item)
        out: list[ContextItem] = []
        for section in Section:
            group = by_section[section]
            if section is Section.EVIDENCE:
                ranked = sorted(group, key=lambda i: (-i.priority, order_index[i.id]))
                if self.placement == "edges":
                    group = edge_order(ranked)
                elif self.placement == "best_last":
                    group = ranked[::-1]
                else:
                    group = ranked
            elif section is Section.HISTORY:
                group = sorted(group, key=lambda i: (i.metadata.get("turn", 0), order_index[i.id]))
            elif section is Section.STATE:
                # Narrative summary first, exact facts after it: facts are the current truth.
                rank = {"summary": 0, "memory": 1, "fact": 2}
                group = sorted(group, key=lambda i: (rank[i.kind], order_index[i.id]))
            else:
                group = sorted(group, key=lambda i: order_index[i.id])
            out.extend(group)
        return out

    def _render(self, ordered: list[ContextItem]) -> tuple[list[Message], str]:
        def block(section: Section) -> list[str]:
            return [render_item(i) for i in ordered if i.section is section]

        stable_prefix = "\n\n".join([*block(Section.SYSTEM), UNTRUSTED_NOTICE])
        system_text = stable_prefix
        state = block(Section.STATE)
        if state:
            system_text += "\n\n## Conversation state\n" + "\n\n".join(state)
        messages = [Message.system(system_text)]
        for item in ordered:
            if item.section is Section.HISTORY:
                role = Role.USER if item.role == "user" else Role.ASSISTANT
                messages.append(Message(role=role, content=item.content))
        tail = block(Section.TOOL_RESULTS) + block(Section.EVIDENCE)
        query = [i.content for i in ordered if i.section is Section.QUERY]
        if tail or query:
            parts = list(tail)
            if query:
                parts.append("Request:\n" + "\n".join(query))
            messages.append(Message.user("\n\n".join(parts)))
        return messages, stable_prefix


__all__ = [
    "ContextBuilder", "BudgetPolicy", "SectionLimits", "BuildResult", "ManifestEntry",
    "ContextOverflowError", "edge_order", "Placement", "TokenCounter",
]
