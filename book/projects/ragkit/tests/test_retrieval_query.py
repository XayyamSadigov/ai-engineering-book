# path: book/projects/ragkit/tests/test_retrieval_query.py
from __future__ import annotations

from retrieval_fixtures import EMPLOYEE

from aie_core.llm.errors import RateLimitError
from aie_core.llm.providers import FakeLLM
from aie_core.llm.types import Message
from ragkit.retrieval.query import (
    ChainedTransformer,
    HyDEGenerator,
    IdentityTransformer,
    MultiQueryExpander,
    QueryDecomposer,
    QueryRewriter,
)
from ragkit.retrieval.types import RetrievalQuery

HISTORY = [
    Message.user("How many PTO days can I carry over into next year?"),
    Message.assistant("Up to 10 days under PTO Policy 3.0 [hr-pto-policy]."),
]


def q(text: str) -> RetrievalQuery:
    return RetrievalQuery(text=text, principal=EMPLOYEE)


def test_identity_keeps_text():
    plan = IdentityTransformer().transform(q("vpn error 412"))
    assert plan.queries == ["vpn error 412"] and plan.original_text == "vpn error 412" and plan.primary == "vpn error 412"


def test_rewriter_resolves_references_from_history_and_keeps_original():
    llm = FakeLLM(responses=[{"query": "PTO carryover deadline for unused days"}])
    plan = QueryRewriter(llm).transform(q("and by when do I have to use them?"), HISTORY)
    assert plan.primary == "PTO carryover deadline for unused days"
    assert plan.original_text == "and by when do I have to use them?"
    sent = llm.last_request.messages[-1].text
    assert "carry over into next year" in sent and "Latest message: and by when" in sent


def test_rewriter_limits_history_and_can_keep_original_query():
    llm = FakeLLM(responses=[{"query": "pto carryover"}])
    long_history = [Message.user(f"turn {i}") for i in range(20)]
    plan = QueryRewriter(llm, max_history_turns=4, keep_original=True).transform(q("and carryover?"), long_history)
    assert plan.queries == ["pto carryover", "and carryover?"] and plan.notes["history_turns"] == 4
    assert "turn 15" not in llm.last_request.messages[-1].text and "turn 19" in llm.last_request.messages[-1].text


def test_rewriter_falls_back_on_drift_or_failure():
    essay = {"query": "x " * 400}
    plan = QueryRewriter(FakeLLM(responses=[essay])).transform(q("vpn 412"), HISTORY)
    assert plan.fallback and plan.queries == ["vpn 412"] and plan.notes["fallback_reason"] == "implausible rewrite"
    plan = QueryRewriter(FakeLLM(responses=[RateLimitError("slow down", provider="fake")])).transform(q("vpn 412"))
    assert plan.fallback and plan.primary == "vpn 412"
    plan = QueryRewriter(FakeLLM(responses=["not json", "still not json"])).transform(q("vpn 412"))
    assert plan.fallback  # malformed structured output after repair is a fallback, not an exception


def test_multi_query_puts_original_first_and_dedupes():
    llm = FakeLLM(responses=[{"queries": ["PTO carry-over limit", "pto carry-over limit", "unused vacation days rollover", "x" * 900]}])
    plan = MultiQueryExpander(llm, n=3).transform(q("How many PTO days carry over?"))
    assert plan.queries[0] == "How many PTO days carry over?"
    assert plan.queries[1:] == ["PTO carry-over limit", "unused vacation days rollover"]
    assert plan.original_text == "How many PTO days carry over?" and plan.strategy == "multi_query"


def test_decomposer_splits_and_keeps_original_for_reranking():
    llm = FakeLLM(responses=[{"sub_questions": ["What is the Trackline tracking ID format?",
                                                "What is the Trackline rate limit per minute?"]}])
    text = "What is the format of a Trackline tracking ID and how many requests per minute can I make?"
    plan = QueryDecomposer(llm).transform(q(text))
    assert plan.primary == text and len(plan.queries) == 3 and plan.notes["sub_questions"][1].endswith("minute?")
    empty = QueryDecomposer(FakeLLM(responses=[{"sub_questions": []}])).transform(q(text))
    assert empty.fallback and empty.queries == [text]


def test_hyde_preserves_original_and_only_adds_passages():
    passage = "Employees may carry over up to X unused PTO days into the next calendar year."
    llm = FakeLLM(responses=[{"passage": passage}])
    plan = HyDEGenerator(llm).transform(q("pto rollover?"))
    assert plan.queries == ["pto rollover?"]  # lexical search never uses invented text
    assert plan.hyde_passages == [passage] and plan.original_text == "pto rollover?"
    assert "placeholders" in llm.last_request.messages[0].text


def test_original_text_survives_chained_transformations():
    llm = FakeLLM(responses=[{"query": "PTO carryover deadline"}, {"queries": ["unused PTO use-by date"]}])
    chain = ChainedTransformer(QueryRewriter(llm), MultiQueryExpander(llm, n=1))
    plan = chain.transform(q("and by when?"), HISTORY)
    assert plan.original_text == "and by when?"
    assert plan.queries == ["PTO carryover deadline", "unused PTO use-by date"]
    assert chain.name == "rewrite+multi_query"
    # the expander saw the rewritten query, not the raw follow-up
    assert "PTO carryover deadline" in llm.requests[1].messages[-1].text
