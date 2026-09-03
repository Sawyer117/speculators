"""Optimizer construction for speculator training.

Provides a single entry point, :func:`build_optimizers`, that returns the list of
optimizers the trainer should drive. The default ("adamw") returns a single AdamW
optimizer over all parameters, preserving the historical behavior. The "muon" option
returns two optimizers: :class:`~speculators.train.muon_distributed.DistributedMuon` over
the weight matrices and the expert stacks, and ``torch.optim.AdamW`` over everything else
(norms, biases, and the embedding / LM-head matrices, following standard Muon practice).

⚠ This used to say that "Muon works transparently under FSDP2 because the parameters
become DTensors and the orthogonalization dispatches across ranks automatically". It does
not. ``torch.optim.Muon``'s Newton-Schulz transposes, which flips a ``Shard(0)`` parameter
to ``Shard(1)``, and the closing in-place update then raises "in-place operations that
require placement changes are not supported". See ``muon_distributed`` for the fix and for
which parameters take which route.
"""

import logging

import torch
from torch import Tensor
from torch.nn import Module

from speculators.train.muon_distributed import DistributedMuon, split_named_params

logger = logging.getLogger("speculators")

# The parameter split now lives in `muon_distributed.split_named_params`, which also
# routes the 3D expert stacks. The old 2D-only splitter that used to sit here is gone on
# purpose: leaving two competing splitters around is how the experts quietly end up back
# on AdamW, which is most of the parameters and the entire memory saving.


def build_optimizers(model: Module, config) -> list[torch.optim.Optimizer]:
    """Build the optimizer(s) for a training run based on ``config.optimizer``.

    :param model: The model to optimize.
    :param config: A ``TrainerConfig`` holding the optimizer hyperparameters.
    :return: A list of optimizers for the trainer to step in tandem. The default
        "adamw" returns a single optimizer; "muon" returns ``[Muon, AdamW]``.
    """
    if config.optimizer == "adamw":
        # Under EP the routed experts are Shard(0) DTensors on the same mesh as the
        # FSDP-sharded rest, so a single AdamW over all (uniform DTensor) params is fine.
        return [
            torch.optim.AdamW(
                model.named_parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
        ]

    if config.optimizer == "muon":
        # NOT torch.optim.Muon: it runs Newton-Schulz on the DTensor itself, the
        # iteration's transpose flips the shard dim, and the final in-place write fails
        # with "aten.add_.Tensor: in-place operations that require placement changes are
        # not supported" (seen on hyper.py's [32, 16384] HyperMix.fn under 8-way FSDP2).
        # DistributedMuon drops to local tensors for the math instead. It also routes the
        # 3D expert stacks to Muon, which the old 2D-only split did not -- and since the
        # experts are most of the parameters, that split left the memory saving on the
        # table even when it did not crash.
        muon_params, adamw_params = split_named_params(model)
        logger.info(
            "Muon optimizer: %d params via DistributedMuon, %d via AdamW.",
            len(muon_params),
            len(adamw_params),
        )

        optimizers: list[torch.optim.Optimizer] = []
        if muon_params:
            optimizers.append(
                DistributedMuon(
                    muon_params,
                    lr=config.muon_lr,
                    momentum=config.muon_momentum,
                    weight_decay=config.muon_weight_decay,
                    ns_steps=config.muon_ns_steps,
                    adjust_lr_fn=config.muon_adjust_lr_fn,
                )
            )
        if adamw_params:
            optimizers.append(
                torch.optim.AdamW(
                    adamw_params,
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                )
            )
        if not optimizers:
            raise ValueError("No trainable parameters found to optimize.")
        return optimizers

    raise ValueError(f"Unsupported optimizer: {config.optimizer!r}")
