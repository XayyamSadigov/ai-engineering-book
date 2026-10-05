# path: book/projects/p3-rag-assistant/tests/test_p3_eval_backends.py
"""The CI gate passes with zero leaks and fails on one; SQL registry and Redis backends work."""
from __future__ import annotations

import fakeredis
import pytest
from aie_core.llm.providers import FakeLLM
from aie_core.observability import InMemoryTracer
from conftest import PTO_Q, CountingEmbeddings
from evalkit import evaluate_gate
from ragkit.eval.rag_dataset import load_gold_dataset
from ragkit.eval.run_rag_eval import evaluate_system
from reliability import RedisJobQueue

from rag_assistant.adapters.llm import extractive_handler
from rag_assistant.caching.caches import RedisBytesMap
from rag_assistant.config import DEFAULT_GOLD_PATH, AssistantSettings
from rag_assistant.eval.run_eval import P3_GATE, main, run, service_target
from rag_assistant.ingestion.registry import InMemoryRegistry, SqlRegistry
from rag_assistant.wiring import build_container


@pytest.fixture
def eval_container(make_container):  # type: ignore[no-untyped-def]
    return make_container(answer_cache=False, retrieval_cache=False)


def test_eval_gate_passes_with_zero_leaks(eval_container, tmp_path):
    passed, summary = run(tmp_path / "eval", container=eval_container)
    assert passed, summary["failures"]
    assert summary["metrics"]["no_permission_leak"] == 1.0
    assert summary["metrics"]["recall@5"] >= 0.85
    assert (tmp_path / "eval" / "report.md").read_text().startswith("# Project 3 release evaluation")


def test_eval_gate_fails_on_a_single_leak(eval_container):
    """Simulate an ACL regression: one forbidden-doc case gets the forbidden chunk packed."""
    honest = service_target(eval_container)
    forbidden_chunk = next(h.chunk for ix in eval_container.index_set.existing()
                           for h in ix.bm25.search("SEV1 response target", _superuser(), k=1))

    def leaky(question, principal):  # type: ignore[no-untyped-def]
        out = honest(question, principal)
        if "SEV1" in question:
            out.packed_chunks.append(forbidden_chunk)
        return out

    dataset = load_gold_dataset(DEFAULT_GOLD_PATH)
    outcome = evaluate_system(leaky, dataset, name="leaky", concurrency=1)
    gate = evaluate_gate(P3_GATE, outcome.run)
    assert not gate.passed
    assert any("no_permission_leak" in f.name for f in gate.failures)


def _superuser():  # type: ignore[no-untyped-def]
    from ragkit.retrieval import Principal

    return Principal(user_id="audit", tenant="shared", groups=["all", "it-oncall", "security", "managers", "hr"])


def test_cli_exit_code(monkeypatch, tmp_path, docs_dir):
    monkeypatch.setenv("RAG_DOCS_DIR", str(docs_dir))
    assert main(["--out", str(tmp_path / "cli")]) == 0


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_registry_contract(backend, tmp_path):
    reg = InMemoryRegistry() if backend == "memory" else SqlRegistry(f"sqlite:///{tmp_path / 'reg.db'}")
    assert reg.next_seq() == 1 and reg.next_seq() == 2
    assert reg.bump(["retail", "shared"]) == {"retail": 1, "shared": 1}
    assert reg.generations(["retail", "logistics"]) == {"retail": 1, "logistics": 0}
    reg.set_state("active_version", "v1")
    assert reg.get_state("active_version") == "v1"
    reg.set_state("active_version", None)
    assert reg.get_state("active_version") is None


def test_full_stack_on_sql_registry_redis_queue_and_redis_cache(docs_dir, vocabulary, tmp_path, employee):
    server = fakeredis.FakeServer()
    r = fakeredis.FakeRedis(server=server)
    settings = AssistantSettings(docs_dir=docs_dir, registry_backend="sql",
                                 registry_url=f"sqlite:///{tmp_path / 'reg.db'}", snapshot_dir=tmp_path / "snap")
    worker_side = build_container(settings, llm=FakeLLM(handler=extractive_handler()),
                                  embeddings=CountingEmbeddings(vocabulary=vocabulary), tracer=InMemoryTracer(),
                                  queue=RedisJobQueue(r, namespace="t:jobs"), cache_store=RedisBytesMap(r))
    worker_side.ingestion.sync("folder")
    assert worker_side.drain() == 24
    assert worker_side.embeddings.has_doc("hr-pto-policy")  # vectors cached in Redis, tagged by document
    # an "API replica": separate process state, same registry, snapshots and vector store
    api_side = build_container(settings, llm=FakeLLM(handler=extractive_handler()),
                               embeddings=CountingEmbeddings(vocabulary=vocabulary), tracer=InMemoryTracer(),
                               queue=RedisJobQueue(r, namespace="t:jobs"), cache_store=RedisBytesMap(r),
                               vector_store=worker_side.vector_store)
    api_side.startup()
    assert api_side.answers.ask(PTO_Q, employee).retrieval.doc_ids[0] == "hr-pto-policy"
    assert api_side.answers.ask(PTO_Q, employee).response.cache == "hit"
    # a delete issued by the API side is purged by the worker side and seen by the API side
    api_side.ingestion.delete("hr-pto-policy")
    assert worker_side.drain() == 1
    out = api_side.answers.ask(PTO_Q, employee)
    assert "hr-pto-policy" not in out.retrieval.doc_ids and out.response.cache == "miss"
    assert not worker_side.embeddings.has_doc("hr-pto-policy")
