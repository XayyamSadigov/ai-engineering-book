# path: book/projects/ragkit/tests/retrieval_fixtures.py
"""Shared, cached fixtures for the Chapter 12 retrieval tests (imported, not a conftest)."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

from aie_core.llm.types import CompletionRequest

from ragkit.documents import Chunk, short_hash
from ragkit.eval.compare_retrievers import Bench, build_bench, load_corpus, load_gold
from ragkit.retrieval.types import Principal

EMPLOYEE = Principal(user_id="emp-1", tenant="shared", groups=["all"])
RETAIL_EMPLOYEE = Principal(user_id="emp-2", tenant="retail", groups=["all", "retail"])
RETAIL_MANAGER = Principal(user_id="mgr-1", tenant="retail", groups=["all", "managers", "retail"])
LOGISTICS_MANAGER = Principal(user_id="mgr-2", tenant="logistics", groups=["all", "managers", "logistics"])
ONCALL = Principal(user_id="oncall-1", tenant="shared", groups=["all", "it-oncall"])


@lru_cache(maxsize=1)
def corpus() -> tuple[list, list[Chunk]]:
    return load_corpus()


@lru_cache(maxsize=1)
def bench() -> Bench:
    return build_bench()


@lru_cache(maxsize=1)
def gold() -> list:
    return load_gold()


def make_chunk(text: str, *, doc_id: str = "doc-a", tenant: str | None = "shared", acl: list[str] | None = None,
               title: str = "", section: list[str] | None = None, position: int = 0, tags: list[str] | None = None,
               updated_at: str | None = None, parent_id: str | None = None, role: str = "leaf",
               version: str = "1") -> Chunk:
    meta: dict[str, Any] = {"title": title, "tags": tags or []}
    if updated_at:
        meta["updated_at"] = updated_at
    return Chunk(
        id=f"{doc_id}:{short_hash(text, position, role)}", doc_id=doc_id, version=version, text=text,
        content_hash=short_hash(text, length=64), tenant=tenant, acl_groups=acl if acl is not None else ["all"],
        metadata=meta, section_path=section or [], parent_id=parent_id, position=position, char_start=0,
        char_end=len(text), token_count=len(text.split()), role=role, chunker="test",
    )


_PASSAGE_RE = re.compile(r'<passage id="(\d+)"[^>]*title="([^"]*)">\s*<untrusted_data>\s*(.*?)\s*</untrusted_data>', re.S)


def keyword_grader(required: list[str], partial: list[str]):
    """A FakeLLM handler for LLMReranker: grade 3 when a passage (title + body) contains every
    `required` word, 1 when it contains any `partial` word, else 0. It stands in for a model
    and lets tests check the reranker's plumbing (batching, id mapping, ordering)."""

    def handler(req: CompletionRequest) -> dict:
        user = req.messages[-1].text
        grades = []
        for pid, title, body in _PASSAGE_RE.findall(user):
            text = f"{title} {body}".lower()
            if all(w in text for w in required):
                score = 3
            elif any(w in text for w in partial):
                score = 1
            else:
                score = 0
            grades.append({"id": int(pid), "score": score})
        return {"grades": grades}

    return handler


def passages_in(req: CompletionRequest) -> int:
    return len(_PASSAGE_RE.findall(req.messages[-1].text))


def dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


from aie_core.llm.types import Message  # noqa: E402

HISTORY_PTO = [
    Message.user("How many PTO days can I carry over into next year?"),
    Message.assistant("Up to 10 days under PTO Policy 3.0 [hr-pto-policy]."),
]
