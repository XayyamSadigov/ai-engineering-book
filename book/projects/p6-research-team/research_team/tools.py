# path: book/projects/p6-research-team/research_team/tools.py
"""Read-only research tools. ACL and tenant come from the trusted principal in ToolContext,
never from model arguments; an invisible passage is reported as unknown, not as forbidden."""
from __future__ import annotations

from typing import Any

from agentkit import ErrorClass, FunctionTool, SideEffect, ToolContext, ToolOutput

from .corpus import Corpus

SEARCH_HEADER = "SEARCH RESULTS"
PASSAGE_HEADER = "PASSAGE"
SNIPPET_CHARS = 320
PASSAGE_CHARS = 2400


def _obj(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


def make_research_tools(corpus: Corpus, *, k: int = 4) -> list[FunctionTool]:
    def search_docs(ctx: ToolContext, query: str) -> ToolOutput:
        hits = corpus.search(query, ctx.principal, k=k)
        if not hits:
            return ToolOutput(content=f'{SEARCH_HEADER} for "{query}": no results', data={"hits": []})
        lines = [f'{SEARCH_HEADER} for "{query}":']
        for p, score in hits:
            snippet = " ".join(p.text.split())[:SNIPPET_CHARS]
            lines.append(f"[{p.passage_id}] {p.label}: {snippet}")
        return ToolOutput(content="\n".join(lines), data={"hits": [p.passage_id for p, _ in hits]})

    def read_passage(ctx: ToolContext, passage_id: str) -> ToolOutput:
        p = corpus.get(passage_id, ctx.principal)
        if p is None:
            return ToolOutput.failure(f"unknown passage {passage_id!r}; use an id from search results",
                                      ErrorClass.IMPOSSIBLE)
        return ToolOutput(content=f"{PASSAGE_HEADER} [{p.passage_id}] {p.label}\n{p.text[:PASSAGE_CHARS]}",
                          data={"passage_id": p.passage_id, "doc_id": p.doc_id})

    return [
        FunctionTool("search_docs", "Keyword search over Northwind policies and runbooks. Returns passage ids "
                     "with short snippets.", _obj({"query": {"type": "string", "minLength": 3}}, ["query"]),
                     search_docs, side_effect=SideEffect.READ, pass_context=True),
        FunctionTool("read_passage", "Read the full text of one passage by its id from search results.",
                     _obj({"passage_id": {"type": "string", "minLength": 3}}, ["passage_id"]),
                     read_passage, side_effect=SideEffect.READ, pass_context=True),
    ]


__all__ = ["make_research_tools", "SEARCH_HEADER", "PASSAGE_HEADER"]
