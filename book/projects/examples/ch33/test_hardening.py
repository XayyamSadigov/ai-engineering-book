# path: book/projects/examples/ch33/test_hardening.py
"""Edge cases behind the chapter's guarantees: coverage, test-set uploads, resumes, hyperparameters."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from dataset_builder import build_dataset
from eval_protocol import ShipRule, SystemRun, decide_ship, evaluate
from finetune_job import FakeProvider, Hyperparameters, run_fine_tune
from test_ch33 import SYSTEM, holdout_and_runs, synthetic_corpus


def test_a_candidate_that_skips_rows_does_not_ship():
    labels, holdout, strong, candidate = holdout_and_runs()
    partial = SystemRun(name="ft", predictions=candidate.predictions[:10])
    decision = decide_ship(evaluate(holdout, partial, labels), evaluate(holdout, strong, labels), ShipRule())
    assert not decision.ship and "answered" in decision.reasons[0]


def test_the_test_set_is_never_uploaded_for_training(tmp_path: Path):
    build_dataset(synthetic_corpus(), out_dir=tmp_path, system_prompt=SYSTEM)
    with pytest.raises(ValueError, match="does not match"):
        run_fine_tune(FakeProvider(), tmp_path / "test.jsonl", "small-base",
                      data_card_path=tmp_path / "data_card.json", sleep=lambda _: None)
    with pytest.raises(ValueError, match="frozen test set"):
        run_fine_tune(FakeProvider(), tmp_path / "train.jsonl", "small-base", val_path=tmp_path / "test.jsonl",
                      data_card_path=tmp_path / "data_card.json", sleep=lambda _: None)


def test_a_resume_with_another_base_model_is_refused(tmp_path: Path):
    build_dataset(synthetic_corpus(), out_dir=tmp_path, system_prompt=SYSTEM)
    provider = FakeProvider()
    first = run_fine_tune(provider, tmp_path / "train.jsonl", "base-A", sleep=lambda _: None)
    with pytest.raises(ValueError, match="trains"):
        run_fine_tune(provider, tmp_path / "train.jsonl", "base-B", resume_job_id=first.job_id, sleep=lambda _: None)


def test_nonsense_hyperparameters_are_rejected():
    for bad in ({"n_epochs": -2}, {"learning_rate_multiplier": -1.0}, {"batch_size": 0}):
        with pytest.raises(ValidationError):
            Hyperparameters(**bad)
