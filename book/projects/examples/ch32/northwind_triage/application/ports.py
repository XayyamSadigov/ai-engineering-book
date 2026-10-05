# path: book/projects/examples/ch32/northwind_triage/application/ports.py
"""Ports: the interfaces the application needs, written in the application's own terms.

These types are owned by the application, not by any provider SDK. Adapters translate to
and from them. That translation is the anti-corruption layer: a provider's field names,
error classes, and quirks stop at the adapter and never leak into use-case code.
"""
from __future__ import annotations

import hashlib
import re
from contextlib import AbstractContextManager
from dataclasses import dataclass
from string import Template
from typing import Any, Protocol

from ..domain import Ticket
from ..flags import Assignment


@dataclass(frozen=True)
class ClassifierOutput:
    text: str
    served_model: str          # what the provider says it actually ran
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class ClassifierUnavailable(Exception):
    """The classifier could not produce output. Raised by adapters; the application decides
    what to do (here: route to a human)."""

    def __init__(self, reason: str, retryable: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.retryable = retryable


class ClassifierPort(Protocol):
    def classify(self, *, system: str, user: str, model: str) -> ClassifierOutput: ...


_TICKET_TAG = re.compile(r"<\s*/?\s*ticket\b[^>]*>", re.IGNORECASE)


def strip_delimiters(text: str) -> str:
    """Remove every opening or closing ticket tag from untrusted text, until none is left.

    One pass is not enough: removing the inner tag of '</tic</ticket>ket>' leaves '</ticket>'.
    Repeating until the text stops changing closes that gap, and the pattern also catches case
    and whitespace variants ('</TICKET >', '< /ticket>') that a literal replace would miss.
    """
    while True:
        cleaned = _TICKET_TAG.sub("", text)
        if cleaned == text:
            return cleaned
        text = cleaned


@dataclass(frozen=True)
class PromptVersion:
    id: str
    version: str
    system: str
    user_template: str  # string.Template syntax: $subject, $body, $tenant

    @property
    def sha(self) -> str:
        return hashlib.sha256(f"{self.system}\n---\n{self.user_template}".encode()).hexdigest()[:12]

    @property
    def label(self) -> str:
        return f"{self.version}#{self.sha}"

    def render(self, ticket: Ticket) -> tuple[str, str]:
        # Untrusted ticket text goes only into the user message, inside delimiters, and
        # safe_substitute never evaluates anything (see Chapter 26 on injection).
        user = Template(self.user_template).safe_substitute(
            subject=strip_delimiters(ticket.subject),
            body=strip_delimiters(ticket.body),
            tenant=ticket.tenant,
        )
        return self.system, user


class PromptStorePort(Protocol):
    def get(self, prompt_id: str, version: str) -> PromptVersion: ...


class FlagsPort(Protocol):
    def evaluate(self, name: str, unit_id: str) -> Assignment: ...


class SpanPort(Protocol):
    def set_attribute(self, key: str, value: Any) -> None: ...


class TracerPort(Protocol):
    """Structurally satisfied by aie_core.observability.Tracer; the application does not
    import aie_core."""

    def span(self, name: str, **attributes: Any) -> AbstractContextManager[Any]: ...
