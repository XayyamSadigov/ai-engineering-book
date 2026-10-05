# path: book/projects/evalkit/tests/test_example.py
"""The Chapter 24 walkthrough as a test: v2 improves the golden slice, the gate still blocks it."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

import ticket_triage_eval as ex  # noqa: E402

from evalkit import Dataset, GateConfig, check_leakage, compare_slices, evaluate_gate  # noqa: E402


def test_frozen_dataset_is_intact_and_splits_cleanly():
    ds = Dataset.load_jsonl(ex.DATA)
    assert len(ds) == 64 and ds.version == "1"
    header_hash = ex.DATA.read_text().splitlines()[0].split('"content_hash": "')[1][:64]
    ds.verify_hash(header_hash)
    dev, hold = ds.split(0.3, seed="northwind-v1")
    assert check_leakage(dev, hold).clean and len(dev) + len(hold) == 64


def test_candidate_wins_on_golden_but_gate_blocks_on_critical_slice():
    ds = Dataset.load_jsonl(ex.DATA)
    base = ex.evaluate(ds, ex.RULES_V1, "triage-v1")
    cand = ex.evaluate(ds, ex.RULES_V2, "triage-v2")
    slices = {s.slice: s for s in compare_slices(base, cand, "category_correct")}
    assert slices["golden"].delta > 0
    assert slices["adversarial"].delta == -1.0
    gate = evaluate_gate(GateConfig.from_toml(ex.GATE), cand, base)
    assert not gate.passed
    assert "critical [critical] category_correct" in {c.name for c in gate.failures}
    assert cand.versions.prompt == "triage-v2" and cand.versions.dataset == ds.fingerprint


def test_main_writes_report(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "OUT", tmp_path)
    assert ex.main(["x", "run"]) == 1  # gate fails -> non-zero exit, as CI expects
    assert "Gate `ticket-triage-release`: FAIL" in (tmp_path / "report.md").read_text()
