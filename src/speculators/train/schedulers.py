"""Learning-rate schedules that are not provided by ``transformers``."""

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

__all__ = ["get_wsd_schedule_with_warmup"]


def get_wsd_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    decay_ratio: float = 0.1,
    min_lr_ratio: float = 0.0,
    last_epoch: int = -1,
) -> LambdaLR:
    """Warmup-Stable-Decay: ramp up, hold, then decay only at the very end.

    Cosine spends a large share of the budget at a small LR, which is wasted when the
    run is still improving. Its other cost is structural: the decay is a function of
    ``num_training_steps``, so the total budget must be committed before the first step,
    and a run extended past it trains at ~0 LR.

    WSD replaces the long taper with a constant "stable" phase followed by a short decay
    (``decay_ratio`` of the budget). The stable phase can be extended at will, and a
    deliverable checkpoint is produced by branching a short decay from wherever the run
    happens to be -- which is what you want when the step budget is not known up front.

    Shape, as a multiplier on each param group's base LR:

    * ``step < num_warmup_steps``            -- linear ramp ``0 -> 1``
    * up to ``num_training_steps*(1-decay_ratio)`` -- constant ``1.0``
    * the remainder                          -- ``1 -> min_lr_ratio`` as ``1 - sqrt(p)``

    The ``1 - sqrt(p)`` decay is the shape from the WSD paper (arXiv:2404.06395); it
    drops faster than linear early in the window, where most of the benefit lands.

    Args:
        optimizer: Optimizer whose LR is scheduled.
        num_warmup_steps: Steps of linear warmup.
        num_training_steps: Total steps the schedule is defined over. Unlike cosine this
            only sets where the decay *starts*; the stable phase before it is flat, so
            overshooting the estimate costs nothing but a later decay.
        decay_ratio: Fraction of ``num_training_steps`` spent decaying. 0 disables the
            decay entirely (warmup + constant forever), which is the right setting for a
            run whose checkpoints will be decayed separately.
        min_lr_ratio: Floor of the DECAY, as a fraction of base LR. Warmup still ramps
            from 0 (as in the ``transformers`` schedules), so this is not a floor on
            the whole schedule. Steps beyond ``num_training_steps`` stay at this floor
            rather than going negative.
        last_epoch: Index of the last epoch when resuming.

    Returns:
        A ``LambdaLR`` over the schedule above.
    """
    if not 0.0 <= decay_ratio <= 1.0:
        raise ValueError(f"decay_ratio must be in [0, 1], got {decay_ratio}")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}")

    num_decay_steps = int(num_training_steps * decay_ratio)
    # Warmup wins a fight with decay: a short run whose decay window would swallow the
    # ramp must still ramp, otherwise the first steps get a near-zero LR for no reason.
    stable_end = max(num_warmup_steps, num_training_steps - num_decay_steps)

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        if current_step < stable_end:
            return 1.0
        decay_steps = max(1, num_training_steps - stable_end)
        progress = min(1.0, (current_step - stable_end) / decay_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - math.sqrt(progress))

    return LambdaLR(optimizer, lr_lambda, last_epoch)
