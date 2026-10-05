# path: book/projects/p1-extraction-api/tests/test_eval.py
import pytest
from aie_core.observability import NoopTracer
from conftest import FIXTURE, TODAY

from extraction_api.adapters import InMemoryReviewQueue, ReplayLLM
from extraction_api.application import ExtractionService
from extraction_api.eval.calibration import choose_threshold, expected_calibration_error, reliability
from extraction_api.eval.dataset import load_invoice_gold
from extraction_api.eval.metrics import FieldScore, score_field, score_invoices
from extraction_api.eval.run_eval import evaluate, format_run, main


def test_field_counts_follow_the_definitions():
    fs = FieldScore(field="total")
    score_field(fs, "10.00", "10.00")    # TP
    score_field(fs, "11.00", "10.00")    # wrong value: FP and FN
    score_field(fs, None, "10.00")       # miss: FN
    score_field(fs, "10.00", None)       # hallucinated: FP
    score_field(fs, None, None)          # correct abstention
    assert (fs.tp, fs.fp, fs.fn, fs.correct, fs.n) == (1, 2, 2, 2, 5)
    assert fs.precision == pytest.approx(1 / 3) and fs.recall == pytest.approx(1 / 3)
    assert fs.exact_match == pytest.approx(0.4)


def test_score_invoices_reports_wrong_fields_and_line_items():
    gold = {"vendor": "Acme", "invoice_number": "A-1", "invoice_date": "2026-01-01", "currency": "USD",
            "total": 110.0, "subtotal": 100.0, "tax_amount": 10.0, "tax_rate": 0.1, "due_date": None,
            "po_number": "PO-1",
            "line_items": [{"amount": 60.0, "quantity": 1}, {"amount": 40.0, "quantity": 2}]}
    pred = {**gold, "vendor": "ACME ", "total": "120.00", "line_items": [{"amount": "60.00", "quantity": "1"}]}
    report = score_invoices([("x", pred, gold)])
    doc = report.docs[0]
    assert doc.wrong_fields == ["total"] and doc.critical_correct is False
    assert report.fields["vendor"].tp == 1                 # case and whitespace normalized
    li = report.line_items
    assert (li.tp, li.fp, li.fn) == (1, 0, 1)


def test_choose_threshold_trades_coverage_for_precision():
    scores = [0.95, 0.92, 0.90, 0.85, 0.80, 0.60]
    correct = [True, True, True, False, True, False]
    pick = choose_threshold(scores, correct, target_precision=0.99)
    assert pick.threshold == 0.90 and pick.coverage == pytest.approx(0.5) and pick.precision == 1.0
    loose = choose_threshold(scores, correct, target_precision=0.75)
    assert loose.threshold == 0.80 and loose.accepted == 5
    assert choose_threshold([0.9], [False], 0.9) is None


def test_reliability_and_ece():
    scores = [0.9] * 10
    correct = [True] * 6 + [False] * 4      # claims 90%, delivers 60%: overconfident
    bins = reliability(scores, correct)
    assert len(bins) == 1 and bins[0].accuracy == pytest.approx(0.6)
    assert expected_calibration_error(scores, correct) == pytest.approx(0.3)


async def test_end_to_end_eval_on_fixture_with_replay_model():
    gold = load_invoice_gold(FIXTURE)
    llm = ReplayLLM(invoices=[g.model_dump() for g in gold], tickets=[])
    svc = ExtractionService(llm, InMemoryReviewQueue(), tracer=NoopTracer(), today=lambda: TODAY)
    run = await evaluate(svc, gold, target_precision=0.98, gold_path=str(FIXTURE))

    assert run.report.critical_accuracy == 1.0
    assert all(fs.exact_match == 1.0 for fs in run.report.fields.values())
    # the three documents that are inconsistent in the gold data are exactly the reviewed ones
    reviewed = {k for k, v in run.routes.items() if v == "human_review"}
    assert reviewed == {g.id for g in gold if not g.document_is_consistent} == {"INV-007", "INV-009", "INV-015"}
    assert run.routing.false_accepts == 0 and run.routing.unnecessary_reviews == 0
    assert run.threshold is not None and run.threshold.precision == 1.0
    assert "false accepts" in format_run(run)


def test_cli_runs_offline(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("TRACE_SINK", "none")
    out = tmp_path / "report.json"
    assert main(["--gold", str(FIXTURE), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "line_items" in printed and out.exists()
