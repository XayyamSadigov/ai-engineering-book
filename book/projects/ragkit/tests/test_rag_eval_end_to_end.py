# path: book/projects/ragkit/tests/test_rag_eval_end_to_end.py
from __future__ import annotations

import json

from aie_core.llm.providers import FakeLLM
from evalkit import Dataset, Run, compare_runs, evaluate_gate
from rag_eval_fixtures import EMPLOYEE, FakeRetriever, case, chunk

from ragkit.eval.rag_dataset import RagOutput
from ragkit.eval.rag_judges import FaithfulnessJudge
from ragkit.eval.rag_report import render_rag_report
from ragkit.eval.run_rag_eval import DEFAULT_GATE, PRESETS, compare_configs, evaluate_system, load_corpus, main
from ragkit.eval.stage_isolation import FailureStage, stage_counts
from ragkit.retrieval.types import Principal, RetrievalQuery

CHUNKS = [chunk("pto", text="Up to 10 PTO days carry over."), chunk("faq", text="Five days carry over (outdated)."),
          chunk("noise"), chunk("runbook", text="SEV1 response target is 15 minutes.", groups=["it-oncall"])]
RANKINGS = {
    "carryover": ["faq:c0", "pto:c0", "noise:c0"],
    "sev1": ["runbook:c0", "noise:c0"],
    "stipend": ["noise:c0", "faq:c0"],
}
DATASET = Dataset(
    [
        case("Q1", question="carryover", required=["pto"], acceptable=["faq"], tags=["conflicting-versions"]),
        case("Q2", question="sev1", forbidden=["runbook"], abstain=True, tags=["forbidden-doc", "abstain"]),
        case("Q3", question="stipend", required=["remote"], tags=["exact-fact"]),
    ],
    name="tiny",
)


def system_factory(*, pack: int, enforce_acl: bool = True):
    retriever = FakeRetriever(CHUNKS, RANKINGS, enforce_acl=enforce_acl)

    def system(question: str, principal: Principal) -> RagOutput:
        res = retriever.retrieve(RetrievalQuery(text=question, principal=principal, k=3))
        packed = [h.chunk for h in res.hits[:pack]]
        relevant = [c for c in packed if c.doc_id in {"pto", "runbook"}]
        if not relevant:
            return RagOutput(abstained=True, packed_chunks=packed, retrieval=res)
        return RagOutput(answer=relevant[0].text, cited_chunk_ids=[relevant[0].id], packed_chunks=packed, retrieval=res)

    return system


def test_two_configurations_stage_shift_and_paired_comparison():
    corpus = {"pto": 1, "faq": 1, "noise": 1, "runbook": 1}
    base = evaluate_system(system_factory(pack=1), DATASET, name="pack1", concurrency=1)
    cand = evaluate_system(system_factory(pack=2), DATASET, name="pack2", concurrency=1)
    from ragkit.eval.stage_isolation import diagnose_run

    bd = {d.case_id: d.stage for d in diagnose_run(base.run, DATASET, corpus=corpus)}
    cd = {d.case_id: d.stage for d in diagnose_run(cand.run, DATASET, corpus=corpus)}
    assert bd == {"Q1": FailureStage.TRUNCATED_IN_PACKING, "Q2": FailureStage.OK, "Q3": FailureStage.NOT_IN_CORPUS}
    assert cd["Q1"] == FailureStage.OK
    delta = compare_runs(base.run, cand.run, "evidence_packed")
    assert delta.n == 2 and delta.wins == 1 and delta.ties == 1  # answerable cases only; Q2 is inverted
    assert evaluate_gate(DEFAULT_GATE, cand.run, base.run).passed


def test_leaky_configuration_fails_the_gate_even_with_better_recall():
    base = evaluate_system(system_factory(pack=2), DATASET, name="acl", concurrency=1)
    leaky = evaluate_system(system_factory(pack=2, enforce_acl=False), DATASET, name="noacl", concurrency=1)
    assert leaky.run.case_scores("no_permission_leak")["Q2"] == 0.0
    gate = evaluate_gate(DEFAULT_GATE, leaky.run, base.run)
    assert not gate.passed
    assert any("forbidden-doc" in f.name for f in gate.failures)
    report = render_rag_report(leaky.run, DATASET, diagnoses=leaky.diagnoses, baseline=base.run,
                               baseline_diagnoses=base.diagnoses, gate=gate)
    assert "Gate `rag-release`: FAIL" in report
    assert "| Q2 | forbidden-doc, abstain | runbook |" in report
    assert stage_counts(leaky.diagnoses)["permission"] == 1


def test_judge_scores_feed_stage_isolation():
    def handler(req):
        if req.metadata["purpose"] == "eval.judge.claims":
            return json.dumps({"claims": ["claim"]})
        return json.dumps({"verdicts": [{"claim": "claim", "verdict": "unsupported"}]})

    out = evaluate_system(system_factory(pack=2), DATASET, name="judged", concurrency=1,
                          judges=[FaithfulnessJudge(FakeLLM(handler=handler))])
    assert out.run.case_scores("faithfulness") == {"Q1": 0.0}  # only the answered case is judged
    stages = {d.case_id: d.stage for d in out.diagnoses}
    assert stages["Q1"] == FailureStage.GENERATION_IGNORED_EVIDENCE


def test_run_survives_json_roundtrip_for_rescoring(tmp_path):
    out = evaluate_system(system_factory(pack=2), DATASET, name="pack2", concurrency=1)
    path = out.run.save_json(tmp_path / "run.json")
    again = Run.load_json(path)
    assert RagOutput.coerce(again.by_case()["Q1"][0].output).cited_doc_ids == ["pto"]


def test_real_corpus_comparison_offline():
    corpus = load_corpus()
    from ragkit.eval.rag_dataset import load_gold_dataset

    ds = load_gold_dataset()
    base, cand, gate, report = compare_configs(PRESETS["lexical-k5-pack1"], PRESETS["lexical-rerank-k5-pack4"],
                                               corpus=corpus, dataset=ds)
    assert gate.passed
    assert cand.run.mean("evidence_packed") > base.run.mean("evidence_packed")
    assert "truncated-in-packing" in stage_counts(base.diagnoses)
    assert "## Stage isolation" in report and "## Permission leaks" in report


def test_cli_writes_report_and_fails_on_leaky_candidate(tmp_path):
    assert main(["--candidate", "lexical-k5-pack4-noacl", "--out", str(tmp_path)]) == 1
    text = (tmp_path / "report.md").read_text()
    assert "leaked restricted documents" in text
    assert (tmp_path / "candidate_run.json").exists() and (tmp_path / "diagnoses.json").exists()


def test_report_warns_on_acl_dropped_without_failing():
    def system(question, principal):
        out = system_factory(pack=2)(question, principal)
        out.retrieval.trace["acl_dropped"] = 2
        return out

    run = evaluate_system(system, DATASET, name="dropped", concurrency=1).run
    report = render_rag_report(run, DATASET)
    assert "Warning, not a blocker" in report and "Q1 (2)" in report
    assert evaluate_gate(DEFAULT_GATE, run).passed


def test_report_lists_target_errors_as_unchecked_not_as_no_leak():
    def flaky(question, principal):
        if question == "sev1":
            raise TimeoutError("RAG service did not answer")
        return system_factory(pack=2)(question, principal)

    run = evaluate_system(flaky, DATASET, name="flaky", concurrency=1).run
    errored = [r for r in run.results if r.case_id == "Q2"][0]
    assert errored.details["no_permission_leak"] == "target_error"  # the shape that used to crash the report
    report = render_rag_report(run, DATASET)  # must not raise
    assert "could not be checked for leaks" in report and "| Q2 |" in report and "TimeoutError" in report
    assert "None across" not in report  # an unchecked case is never reported as "no leak"
    assert not evaluate_gate(DEFAULT_GATE, run).passed  # and the gate still blocks
    from ragkit.eval.stage_isolation import diagnose_run
    labels = {d.case_id: d.stage for d in diagnose_run(run, DATASET)}
    assert labels["Q2"] == FailureStage.UNCHECKED  # a crashed case is never counted as ok


def test_report_handles_target_errors_scored_as_none():
    from evalkit import run_target
    from ragkit.eval.rag_metrics import retrieval_evaluator
    from ragkit.eval.run_rag_eval import make_target

    def broken(question, principal):
        raise RuntimeError("down")

    run = run_target(make_target(broken), DATASET, evaluators=[retrieval_evaluator()], error_score=None,
                     concurrency=1)
    report = render_rag_report(run, DATASET)
    assert "None across" not in report and report.count("RuntimeError") == 3
