"""Tests for Correction generated-token ratio scheduling."""

from types import SimpleNamespace

import pytest

from speculators.train.trainer import Trainer


def _scheduled_trainer(global_step: int) -> Trainer:
    trainer = Trainer.__new__(Trainer)
    trainer.global_step = global_step
    trainer._curriculum_total_steps = 100
    trainer._correction_generated_token_target_ratio = 0.25
    trainer._correction_generated_token_warmup = 0.2
    trainer._correction_generated_token_ramp = 0.4
    return trainer


def test_generated_token_ratio_warmup_ramp_and_plateau():
    assert _scheduled_trainer(0)._current_correction_generated_token_ratio() == 0.0
    assert _scheduled_trainer(20)._current_correction_generated_token_ratio() == 0.0
    assert (
        _scheduled_trainer(40)._current_correction_generated_token_ratio()
        == pytest.approx(0.125)
    )
    assert _scheduled_trainer(60)._current_correction_generated_token_ratio() == pytest.approx(0.25)
    assert _scheduled_trainer(90)._current_correction_generated_token_ratio() == pytest.approx(0.25)


def test_zero_ramp_switches_to_target_after_warmup():
    trainer = _scheduled_trainer(19)
    trainer._correction_generated_token_ramp = 0.0
    assert trainer._current_correction_generated_token_ratio() == 0.0

    trainer.global_step = 20
    assert trainer._current_correction_generated_token_ratio() == pytest.approx(0.25)


def test_generated_batch_choice_is_deterministic_per_global_step():
    first = _scheduled_trainer(42)
    second = _scheduled_trainer(42)
    assert first._sample_correction_generated_token_batch(0.25) == (
        second._sample_correction_generated_token_batch(0.25)
    )
    assert not first._sample_correction_generated_token_batch(0.0)
    assert first._sample_correction_generated_token_batch(1.0)


def test_curriculum_metadata_is_removed_from_model_call_kwargs():
    trainer = Trainer.__new__(Trainer)
    loss_config = object()
    trainer.config = SimpleNamespace(
        train_call_kwargs={
            "loss_config": loss_config,
            "correction_generated_token_curriculum": True,
            "correction_generated_token_target_ratio": 0.25,
            "correction_generated_token_warmup": 0.2,
            "correction_generated_token_ramp": 0.4,
        }
    )
    trainer._init_loss_curricula()

    assert trainer._correction_generated_token_curriculum
    assert trainer._correction_generated_token_target_ratio == 0.25
    assert trainer._correction_generated_token_warmup == 0.2
    assert trainer._correction_generated_token_ramp == 0.4
    assert trainer.config.train_call_kwargs == {"loss_config": loss_config}
