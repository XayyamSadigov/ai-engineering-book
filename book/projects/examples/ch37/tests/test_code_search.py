# path: book/projects/examples/ch37/tests/test_code_search.py
from __future__ import annotations

from pathlib import Path

import pytest

from code_search import CodeIndex, split_identifier

PACKAGE = Path(__file__).resolve().parents[3] / "aie_core" / "aie_core"


@pytest.fixture(scope="module")
def idx() -> CodeIndex:
    return CodeIndex.build(PACKAGE)


def test_split_identifier_handles_snake_camel_and_acronyms():
    assert split_identifier("complete_structured") == ["complete", "structured"]
    assert split_identifier("RetryPolicy") == ["retry", "policy"]
    assert split_identifier("HTTPError") == ["http", "error"]
    assert split_identifier("OpenAICompatibleClient") == ["open", "ai", "compatible", "client"]


def test_find_definition_by_name_and_qualified_suffix(idx: CodeIndex):
    [sym] = idx.find_definition("complete_structured")
    assert sym.module == "aie_core.llm.structured" and sym.kind == "function"
    methods = idx.find_definition("FakeLLM.complete")
    assert [m.qualname for m in methods] == ["aie_core.llm.providers.fake.FakeLLM.complete"]


def test_lexical_search_ranks_by_identifier_words(idx: CodeIndex):
    top = [s.qualname for s, _ in idx.search("retry policy backoff", k=5)]
    assert any(q.endswith("RetryPolicy") for q in top)
    top_structured = [s.name for s, _ in idx.search("structured output", k=3)]
    assert "complete_structured" in top_structured


def test_callers_by_name_include_a_same_named_false_positive(idx: CodeIndex):
    callers = {s.qualname for s in idx.callers("_prepare")}
    assert {"aie_core.llm.structured.complete_structured", "aie_core.llm.structured.acomplete_structured"} <= callers
    # CachedEmbeddings has its own _prepare: name-based resolution cannot tell them apart
    assert "aie_core.embeddings.CachedEmbeddings.embed" in callers


def test_import_graph_answers_blast_radius_questions(idx: CodeIndex):
    assert "aie_core.llm.structured" in idx.dependents("aie_core.llm.errors")
    assert idx.dependencies("aie_core.llm.structured") == ["aie_core.llm.client", "aie_core.llm.errors", "aie_core.llm.types"]


def test_ast_chunk_is_a_whole_symbol_with_its_context(idx: CodeIndex):
    [method] = idx.find_definition("FakeLLM.complete")
    chunk = idx.chunk(method)
    assert "class FakeLLM" in chunk  # enclosing class header
    assert "from ..types import" in chunk  # module imports
    assert "def complete(self, req: CompletionRequest) -> Completion:" in chunk
    [fn] = idx.find_definition("complete_structured")
    body = idx.chunk(fn)
    assert body.rstrip().endswith(")")  # ends at the function's last line, not mid-statement
    assert body.count("def complete_structured") == 1


def test_grep_baseline_finds_text_but_not_structure(idx: CodeIndex):
    hits = idx.grep(r"def complete_structured")
    assert hits and hits[0].startswith("aie_core.llm.structured:")
