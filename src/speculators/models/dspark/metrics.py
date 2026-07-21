"""Loss and metrics for the DSpark draft model.

loss = compound_loss(logits, targets)
     + conf_alpha * BCE(confidence, accept_rate)
     + length_alpha * SmoothL1(pred_len, target_len)
     + focal_alpha * CE(first greedy error)

Optional adaptive position weights (CAT / SSAL) replace fixed decay.
"""

from functools import partial
from typing import Any, Literal

import torch
from torch.nn.functional import (
    binary_cross_entropy_with_logits,
    cross_entropy,
    smooth_l1_loss,
    softmax,
)

from speculators.models.metrics import (
    LossConfig,
    compound_loss,
    compute_accuracy_multi_step,
    dpace_loss_decay,
    position_weights,
)

__all__ = [
    "compute_metrics",
]

_EPS = 1e-8

AdaptiveLoss = Literal["none", "cat", "ssal"]
ConfidenceLossWeighting = Literal["uniform", "match-draft"]


def _masked_weighted_mean(
    elementwise: torch.Tensor,  # [1, T]
    loss_mask: torch.Tensor,  # [1, T]
    weights: torch.Tensor | None = None,  # [1, T]
) -> torch.Tensor:
    """Masked mean; optional weights scale the numerator only (same as draft decay)."""
    loss_mask = loss_mask.to(elementwise.dtype)
    weighted = elementwise * loss_mask
    if weights is not None:
        weighted = weighted * weights.to(elementwise.dtype)
    return (weighted.sum(dim=1) / (loss_mask.sum(dim=1) + _EPS)).mean()


def _first_error_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    block_size: int,
    weights: torch.Tensor,
    start_pos: int,
) -> torch.Tensor:
    """CE at each block's first greedy mismatch (chain breaker)."""
    pred_ids = torch.argmax(logits, dim=-1)
    target_ids = torch.argmax(targets, dim=-1)
    vocab = logits.shape[-1]
    ce = cross_entropy(
        logits.reshape(-1, vocab),
        target_ids.reshape(-1),
        reduction="none",
    ).reshape_as(target_ids)

    num_blocks = logits.shape[1] // block_size
    wrong = ((pred_ids != target_ids) & loss_mask.bool()).view(num_blocks, block_size)
    if start_pos > 0:
        wrong = wrong.clone()
        wrong[:, :start_pos] = False

    first_idx = wrong.to(torch.int64).argmax(dim=-1)  # first True; 0 if none
    fe_mask = torch.zeros_like(wrong, dtype=ce.dtype)
    rows = torch.arange(num_blocks, device=logits.device)
    has_err = wrong.any(dim=-1)
    fe_mask[rows[has_err], first_idx[has_err]] = 1.0
    fe_mask = fe_mask.reshape_as(ce)
    return _masked_weighted_mean(ce, fe_mask, weights)


def _accept_length(
    probs: torch.Tensor,  # [num_blocks, draft_slots]
    draft_mask: torch.Tensor,  # [num_blocks, draft_slots]
) -> torch.Tensor:
    prefix = (probs * draft_mask).cumprod(dim=-1)
    return prefix.sum(dim=-1) + 1.0


def compute_metrics(
    logits: torch.Tensor,  # [1, T, draft_vocab_size]
    targets: torch.Tensor,  # [1, T, draft_vocab_size]
    confidence_logits: torch.Tensor | None,  # [1, T] or None
    loss_mask: torch.Tensor,  # [1, T]
    block_size: int,
    loss_config: LossConfig,
    gamma: float = 4.0,
    confidence_head_alpha: float = 1.0,
    confidence_length_alpha: float = 0.0,
    confidence_loss_weighting: ConfidenceLossWeighting = "uniform",
    first_error_focal_alpha: float = 0.0,
    adaptive_loss: AdaptiveLoss = "none",
    ssal_decay_weight: float = 0.0,
    per_position_loss_weight: str = "fixed-exp-decay",
    dpace_alpha: float = 0.5,
    sample_from_anchor: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Compute the DSpark loss and a metrics dict (``*_sum``/``*_total`` pairs)."""

    device = logits.device
    seq_len = logits.shape[1]
    pos_idx = (torch.arange(seq_len, device=device) % block_size).unsqueeze(0)
    start_pos = 0 if sample_from_anchor else 1

    # Analytical overlap (also SSAL score); needed for confidence / accept metrics.
    with torch.no_grad():
        draft_p = softmax(logits.float(), dim=-1)
        target_p = softmax(targets.float(), dim=-1)
        accept_rate = torch.minimum(draft_p, target_p).sum(dim=-1)  # [1, T]

    if per_position_loss_weight == "dpace":
        decay_fn = partial(
            dpace_loss_decay,
            loss_mask=loss_mask,
            block_size=block_size,
            dpace_alpha=dpace_alpha,
        )
        draft_weights = None
    else:
        adaptive_scores = None
        if adaptive_loss == "ssal":
            adaptive_scores = accept_rate
        elif adaptive_loss == "cat":
            with torch.no_grad():
                target_ids = torch.argmax(targets, dim=-1, keepdim=True)
                adaptive_scores = target_p.gather(-1, target_ids).squeeze(-1)
        draft_weights = position_weights(
            pos_idx.to(logits.dtype),
            block_size=block_size,
            gamma=gamma,
            sample_from_anchor=sample_from_anchor,
            adaptive_scores=(
                None if adaptive_scores is None else adaptive_scores.to(logits.dtype)
            ),
            decay_mix=ssal_decay_weight if adaptive_loss == "ssal" else 0.0,
        )
        decay_fn = lambda pos, **_kw: draft_weights  # noqa: E731

    loss, term_losses = compound_loss(
        logits, targets, loss_mask, pos_idx, loss_config=loss_config, decay_fn=decay_fn
    )

    if first_error_focal_alpha > 0.0:
        fe_weights = (
            draft_weights
            if draft_weights is not None
            else position_weights(
                pos_idx.to(logits.dtype),
                block_size=block_size,
                gamma=gamma,
                sample_from_anchor=sample_from_anchor,
            )
        )
        fe_loss = _first_error_focal_loss(
            logits, targets, loss_mask, block_size, fe_weights, start_pos
        )
        loss = loss + first_error_focal_alpha * fe_loss

    num_blocks = seq_len // block_size
    with torch.no_grad():
        accept_blocks = accept_rate.view(num_blocks, block_size)
        draft_mask = loss_mask.to(accept_rate.dtype).view(num_blocks, block_size)[
            :, start_pos:
        ]
        accept_prefix = (accept_blocks[:, start_pos:] * draft_mask).cumprod(dim=-1)

    metrics: dict[str, Any] = {}
    if confidence_logits is not None:
        c_star = accept_rate.detach().to(confidence_logits.dtype)
        bce = binary_cross_entropy_with_logits(
            confidence_logits, c_star, reduction="none"
        )
        conf_weights = (
            draft_weights if confidence_loss_weighting == "match-draft" else None
        )
        conf_loss = _masked_weighted_mean(bce, loss_mask, conf_weights)
        loss = loss + confidence_head_alpha * conf_loss

        conf_prob = confidence_logits.float().sigmoid()
        with torch.no_grad():
            mask_f = loss_mask.to(accept_rate.dtype)
            mask_total = mask_f.sum().clamp_min(1.0)
            metrics["confidence_loss_sum"] = conf_loss.detach().clone()
            metrics["confidence_loss_total"] = torch.ones((), device=device)
            metrics["confidence_abs_error_sum"] = (
                (conf_prob - accept_rate).abs() * mask_f
            ).sum()
            metrics["confidence_abs_error_total"] = mask_total
            metrics["confidence_pred_mean_sum"] = (conf_prob * mask_f).sum()
            metrics["confidence_pred_mean_total"] = mask_total.clone()
            conf_prefix = (
                conf_prob.view(num_blocks, block_size)[:, start_pos:] * draft_mask
            ).cumprod(dim=-1)
            metrics["confidence_cumprod_bias_sum"] = (
                (conf_prefix - accept_prefix) * draft_mask
            ).sum()
            metrics["confidence_cumprod_bias_total"] = draft_mask.sum().clamp_min(1.0)

        if confidence_length_alpha > 0.0:
            pred_len = _accept_length(
                conf_prob.view(num_blocks, block_size)[:, start_pos:], draft_mask
            )
            target_len = _accept_length(accept_blocks[:, start_pos:], draft_mask)
            block_valid = (draft_mask.sum(dim=-1) > 0).to(pred_len.dtype)
            length_loss = smooth_l1_loss(pred_len, target_len, reduction="none")
            length_loss = (length_loss * block_valid).sum() / (
                block_valid.sum() + _EPS
            )
            loss = loss + confidence_length_alpha * length_loss
            with torch.no_grad():
                metrics["confidence_length_loss_sum"] = length_loss.detach().clone()
                metrics["confidence_length_loss_total"] = torch.ones((), device=device)
                metrics["confidence_accept_len_pred_sum"] = (pred_len * block_valid).sum()
                metrics["confidence_accept_len_pred_total"] = block_valid.sum().clamp_min(
                    1.0
                )

    ones = torch.ones((), device=device)
    metrics["loss_sum"] = loss.detach().clone()
    metrics["loss_total"] = ones
    for term_name, term_val in term_losses.items():
        metrics[f"{term_name}_sum"] = term_val
        metrics[f"{term_name}_total"] = ones.clone()

    with torch.no_grad():
        mask_f = loss_mask.to(accept_rate.dtype)
        metrics["accept_rate_sum"] = (accept_rate * mask_f).sum()
        metrics["accept_rate_total"] = mask_f.sum().clamp_min(1.0)
        per_block_len = accept_prefix.sum(dim=-1) + 1.0
        block_valid = (draft_mask.sum(dim=-1) > 0).to(accept_rate.dtype)
        metrics["accept_len_sum"] = (per_block_len * block_valid).sum()
        metrics["accept_len_total"] = block_valid.sum().clamp_min(1.0)

    pred_ids = torch.argmax(logits, dim=-1)
    target_ids = torch.argmax(targets, dim=-1)
    correct_per_pos, total_per_pos = compute_accuracy_multi_step(
        pred_ids, target_ids, loss_mask, pos_idx, block_size
    )
    metrics["full_acc_sum"] = correct_per_pos[start_pos:].sum()
    metrics["full_acc_total"] = total_per_pos[start_pos:].sum()
    for pos in range(start_pos, block_size):
        metrics[f"position_{pos}_acc_sum"] = correct_per_pos[pos]
        metrics[f"position_{pos}_acc_total"] = total_per_pos[pos]

    return loss, metrics
