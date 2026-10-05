# path: book/capstone/northwind-assist/northwind_assist/domain/intents.py
"""Deterministic intent routing: rules in code before any model sees the request.

The first routing decision is cheap, testable and explainable: a few ordered rules over the
user's text and the request fields. It picks the *workflow* (grounded answer, agent with tools,
structured extraction, memory command). Only then does the Chapter 7 router pick the *model*
for that workflow. A model-based intent classifier is a reasonable extension once traffic shows
rules missing cases (Chapter 17's decision table: workflow first, agent only where the path is
not known in advance); the rules below are the baseline it must beat.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Intent(str, Enum):
    QUESTION = "rag.answer"          # grounded answer with citations or abstention
    ACTION = "agent.action"          # tool use through the agent loop
    EXTRACT = "extract.document"     # Project 1 structured extraction
    MEMORY = "memory.command"        # explicit profile statements and commands
    SMALLTALK = "assist.smalltalk"   # greetings and help; no retrieval, no tools


@dataclass(frozen=True)
class IntentDecision:
    intent: Intent
    rule: str
    task: str                         # agent task id for the Definition of Done and evals


_RULES: list[tuple[str, Intent, str, re.Pattern[str]]] = [
    ("extract_prefix", Intent.EXTRACT, "extract", re.compile(r"^\s*extract\s*:", re.I)),
    ("remember", Intent.MEMORY, "memory", re.compile(r"^\s*(remember|forget)\b", re.I)),
    ("preference", Intent.MEMORY, "memory",
     re.compile(r"^\s*i (prefer|like) \w+ (answers|replies)\W*$|^\s*(answer|reply) (me )?in \w+\W*$", re.I)),
    ("send_or_draft", Intent.ACTION, "reply", re.compile(r"^\s*(send|draft)\b.*\bTCK-\d{4}-\d{4}\b", re.I)),
    ("create_ticket", Intent.ACTION, "create_ticket",
     re.compile(r"\b(create|open|raise|file) (a )?ticket\b", re.I)),
    ("service_status", Intent.ACTION, "status", re.compile(r"\b(status of|is the .* down|outage)\b", re.I)),
    ("lookup", Intent.ACTION, "lookup", re.compile(r"^\s*(who is|look ?up|find employee)\b", re.I)),
    ("ticket_search", Intent.ACTION, "search", re.compile(r"\b(search|find|show)\b.*\btickets?\b", re.I)),
    ("investigate", Intent.ACTION, "investigate", re.compile(r"^\s*(investigate|research)\b", re.I)),
    ("smalltalk", Intent.SMALLTALK, "smalltalk", re.compile(r"^\s*(hi|hello|hey|thanks|thank you|help)\W*$", re.I)),
]


def route_intent(text: str, *, has_document: bool = False) -> IntentDecision:
    if has_document:
        return IntentDecision(Intent.EXTRACT, "document_field", "extract")
    for name, intent, task, rx in _RULES:
        if rx.search(text):
            return IntentDecision(intent, name, task)
    return IntentDecision(Intent.QUESTION, "default_question", "rag")


__all__ = ["Intent", "IntentDecision", "route_intent"]
