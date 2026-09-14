from pathlib import Path

import pytest
import torch
from transformers import get_linear_schedule_with_warmup

from speculators.train.checkpointer import SingleGPUCheckpointer
from speculators.train.config import TrainConfig
from speculators.train.schedulers import get_wsd_schedule_with_warmup
from speculators.train.trainer import (
    TrainerConfig,
    _resolve_scheduler_steps,
)


def make_config(**overrides) -> TrainerConfig:
    return TrainerConfig(
        lr=1e-4,
        num_epochs=5,
        save_path="checkpoint",
        **overrides,
    )


def test_scheduler_steps_default_to_one_percent_of_training_steps():
    warmup_steps, total_steps = _resolve_scheduler_steps(make_config(), 20)

    assert total_steps == 100
    assert warmup_steps == 1


def test_scheduler_total_steps_only_defaults_warmup_to_one_percent_of_total():
    # default_total_steps is num_epochs * loader_len = 100, but the explicit
    # scheduler_total_steps override must drive the 1% warmup fallback (10, not 1).
    warmup_steps, total_steps = _resolve_scheduler_steps(
        make_config(scheduler_total_steps=1000),
        20,
    )

    assert total_steps == 1000
    assert warmup_steps == 10


def test_max_steps_sets_scheduler_total_steps():
    # max_steps stops the training loop early; the scheduler must decay over the
    # same horizon (30), not num_epochs * loader_len (100).
    warmup_steps, total_steps = _resolve_scheduler_steps(
        make_config(max_steps=30),
        20,
    )

    assert total_steps == 30
    assert warmup_steps == 0  # 1% of 30


def test_scheduler_total_steps_override_wins_over_max_steps():
    warmup_steps, total_steps = _resolve_scheduler_steps(
        make_config(max_steps=30, scheduler_total_steps=250),
        20,
    )

    assert total_steps == 250
    assert warmup_steps == 2  # 1% of 250


def test_scheduler_warmup_ratio_uses_scheduler_total_steps():
    warmup_steps, total_steps = _resolve_scheduler_steps(
        make_config(scheduler_total_steps=200, scheduler_warmup_ratio=0.1),
        20,
    )

    assert total_steps == 200
    assert warmup_steps == 20


def test_scheduler_warmup_steps_take_precedence_over_ratio():
    with pytest.warns(UserWarning, match="using scheduler_warmup_steps"):
        warmup_steps, total_steps = _resolve_scheduler_steps(
            make_config(scheduler_warmup_steps=0, scheduler_warmup_ratio=0.1),
            20,
        )

    assert total_steps == 100
    assert warmup_steps == 0


def test_scheduler_warmup_ratio_must_be_between_zero_and_one():
    with pytest.raises(ValueError, match="scheduler_warmup_ratio"):
        _resolve_scheduler_steps(make_config(scheduler_warmup_ratio=1.1), 20)


def test_scheduler_type_rejects_unsupported_values():
    # --verifier-name-or-path is supplied so the only parse failure is the rejected
    # --scheduler-type choice (not the missing required verifier arg).
    with pytest.raises(SystemExit):
        TrainConfig.resolve(
            ["--verifier-name-or-path", "x", "--scheduler-type", "constant"]
        )


def test_scheduler_resume_restores_optimizer_learning_rate(tmp_path: Path):
    checkpoint_dir = tmp_path / "0"
    checkpoint_dir.mkdir()
    checkpointer = SingleGPUCheckpointer(tmp_path)

    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=10,
        num_training_steps=100,
    )
    for _ in range(25):
        optimizer.step()
        scheduler.step()

    expected_lr = scheduler.get_last_lr()[0]
    checkpointer.save_scheduler_state_dict(scheduler, epoch=0)

    resumed_parameter = torch.nn.Parameter(torch.zeros(()))
    resumed_optimizer = torch.optim.AdamW([resumed_parameter], lr=1e-3)
    resumed_optimizer.load_state_dict(optimizer.state_dict())
    resumed_scheduler = get_linear_schedule_with_warmup(
        resumed_optimizer,
        num_warmup_steps=10,
        num_training_steps=100,
        last_epoch=0,
    )
    assert resumed_optimizer.param_groups[0]["lr"] != pytest.approx(expected_lr)

    checkpointer.load_scheduler_state_dict(resumed_scheduler)

    assert resumed_scheduler.get_last_lr()[0] == pytest.approx(expected_lr)
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)


def _wsd_lrs(total, warmup, decay_ratio=0.1, min_lr_ratio=0.0, base_lr=1.0, steps=None):
    """Drive the schedule step-by-step and return the LR seen at each step."""
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=base_lr)
    sched = get_wsd_schedule_with_warmup(
        opt,
        num_warmup_steps=warmup,
        num_training_steps=total,
        decay_ratio=decay_ratio,
        min_lr_ratio=min_lr_ratio,
    )
    out = []
    for _ in range(steps if steps is not None else total):
        out.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return out


def test_wsd_warms_up_then_holds_peak():
    lrs = _wsd_lrs(total=100, warmup=10)

    assert lrs[0] == pytest.approx(0.0)
    assert lrs[5] == pytest.approx(0.5)
    # Everything between the end of warmup and the start of decay is exactly the peak —
    # this flat middle is the whole point of WSD over cosine.
    assert all(lr == pytest.approx(1.0) for lr in lrs[10:90])


def test_wsd_decays_only_over_the_last_decay_ratio():
    lrs = _wsd_lrs(total=100, warmup=10, decay_ratio=0.1)

    assert lrs[89] == pytest.approx(1.0)
    assert lrs[90] == pytest.approx(1.0)  # p=0 at the first decay step
    assert lrs[95] < 0.35  # 1 - sqrt(0.5)
    assert lrs[99] < 0.06


def test_wsd_zero_decay_ratio_never_leaves_the_plateau():
    # The setting for a run whose budget is unknown: hold peak forever and branch a
    # decay from a checkpoint later. A cosine cannot express this.
    lrs = _wsd_lrs(total=100, warmup=10, decay_ratio=0.0)

    assert all(lr == pytest.approx(1.0) for lr in lrs[10:])


def test_wsd_min_lr_ratio_is_the_floor_and_holds_past_the_budget():
    lrs = _wsd_lrs(total=100, warmup=10, min_lr_ratio=0.2, steps=130)

    # The floor applies to the DECAY, not to warmup — warmup still ramps from 0, as
    # in the transformers schedules. So look past the ramp.
    assert min(lrs[10:]) >= pytest.approx(0.2)
    # Overrunning the budget must clamp, not go negative.
    assert all(lr == pytest.approx(0.2) for lr in lrs[100:])


def test_wsd_warmup_survives_a_decay_window_that_would_swallow_it():
    # decay_ratio=0.9 puts the decay start before the end of warmup; the ramp must still
    # happen, otherwise the first steps run at ~0 LR for no reason.
    lrs = _wsd_lrs(total=100, warmup=50, decay_ratio=0.9)

    assert lrs[0] == pytest.approx(0.0)
    assert lrs[25] == pytest.approx(0.5)
    assert lrs[49] == pytest.approx(0.98)


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_wsd_rejects_out_of_range_ratios(bad):
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    with pytest.raises(ValueError, match="must be in"):
        get_wsd_schedule_with_warmup(opt, 10, 100, decay_ratio=bad)
    with pytest.raises(ValueError, match="must be in"):
        get_wsd_schedule_with_warmup(opt, 10, 100, min_lr_ratio=bad)
