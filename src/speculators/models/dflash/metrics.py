"""Metrics and loss functions for DFlash draft model."""

from collections.abc import Callable
from functools import partial
from typing import Any

import torch

from speculators.models.metrics import (
    compute_accuracy_multi_step,
    confidence_loss,
    dflash_loss_decay,
    kl_div_loss,
    loss_function,
)

_EPS = 1e-5


def compute_metrics(
    logits: torch.Tensor,  # shape: [1, num_anchors*block_size, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, num_anchors*block_size, draft_vocab_size]
    loss_mask: torch.Tensor,  # shape: [1, num_anchors*block_size]
    block_size: int = 1,
    gamma: float = 4.0,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = kl_div_loss,
    confidence_logits: torch.Tensor | None = None,  # shape: [1, num_anchors*block_size]
    confidence_alpha: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Compute loss and accuracy metrics for draft model predictions.

    Args:
        logits: Model logits [1, T, V]
        targets: Target logits [1, T, V]
        loss_mask: Binary mask [1, T]
        block_size: Block size for per-position metrics
        gamma: Temperature for exponential decay in loss weighting
        loss_fn: Loss function

    Returns:
        Tuple of (loss, metrics_dict) where metrics_dict contains:
            - loss: Scalar loss value
            - full_acc: Overall accuracy
            - position {i} acc: Accuracy at position i within blocks
    """
    if loss_fn is None:
        loss_fn = kl_div_loss
    seq_len = logits.shape[1]
    pos_idx = torch.arange(seq_len, device=logits.device) % block_size
    pos_idx = pos_idx.unsqueeze(0)  # shape: [1, T]

    loss = loss_function(
        logits,
        targets,
        loss_mask,
        pos_idx,
        loss_fn=loss_fn,
        decay_fn=partial(dflash_loss_decay, gamma=gamma),
    )

    # DSpark accept-rate (confidence) head: masked + position-decayed BCE against the
    # soft acceptance rate, added to the distribution loss. Same mask/decay convention
    # as the main term (anchors at position 0 are excluded by dflash_loss_decay).
    confidence_scalar = None
    if confidence_logits is not None and confidence_alpha > 0:
        conf_elementwise = confidence_loss(confidence_logits, logits, targets)
        conf_mask = loss_mask.to(conf_elementwise.dtype)
        conf_decay = dflash_loss_decay(pos_idx.to(conf_elementwise.dtype), gamma)
        conf_elementwise = conf_elementwise * conf_mask * conf_decay
        conf_denom = conf_mask.sum(dim=1) + _EPS
        confidence_scalar = (conf_elementwise.sum(dim=1) / conf_denom).mean()
        loss = loss + confidence_alpha * confidence_scalar

    pred_ids = torch.argmax(logits, dim=-1)
    target_ids = torch.argmax(targets, dim=-1)

    correct_per_pos, total_per_pos = compute_accuracy_multi_step(
        pred_ids, target_ids, loss_mask, pos_idx, block_size
    )

    metrics: dict[str, Any] = {}
    metrics["loss_sum"] = loss.detach().clone()
    metrics["loss_total"] = torch.tensor(1.0, device=logits.device)
    if confidence_scalar is not None:
        metrics["confidence_loss_sum"] = confidence_scalar.detach().clone()
        metrics["confidence_loss_total"] = torch.tensor(1.0, device=logits.device)
    # Position 0 is the anchor — intentionally excluded from accuracy
    metrics["full_acc_sum"] = correct_per_pos[1:].sum()
    metrics["full_acc_total"] = total_per_pos[1:].sum()

    for pos in range(1, block_size):
        metrics[f"position_{pos}_acc_sum"] = correct_per_pos[pos]
        metrics[f"position_{pos}_acc_total"] = total_per_pos[pos]
    return loss, metrics
