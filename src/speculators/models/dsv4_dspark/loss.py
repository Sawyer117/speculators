"""DSpark draft training loss (teacher-forced, block-gamma).

Per block position k = 1..gamma with decay ``w_k = exp(-(k-1)/gamma)``::

    L = Σ_k w_k · [ ce_alpha·CE(p_k, q_k) + l1_alpha·L1(p_k, q_k) + conf_alpha·BCE(conf_k, 1 - d_TV) ]

where ``q_k`` is the frozen target's next-token distribution at position k (from
``softmax(lm_head(target_last_hidden))``, supplied as ``target_dist``), CE/L1 are
the DSpark distribution terms, and the confidence head is trained (BCE) against
the detached soft acceptance rate ``1 - d_TV(p_k, q_k)``.

Reuses the repo's metric primitives (``combo_ce_l1_loss`` = ce·CE + l1·L1,
``confidence_loss`` = the soft-accept BCE, ``loss_function`` = masked +
position-decayed reduction). The one DSpark-specific piece is the decay: our
forward emits ``gamma`` logits that are ALL predictions (``logits[:, i]`` predicts
target token ``i+1``), so position 0 must keep the largest weight — we use
``exp(-i/gamma)`` rather than DFlash's ``dflash_loss_decay`` (which zeroes
position 0 as an anchor). The final anchor/target-shift alignment against the
reused data pipeline is settled at trainer-integration time.
"""
from __future__ import annotations

from functools import partial

import torch

from speculators.models.metrics import (
    combo_ce_l1_loss,
    confidence_loss,
    loss_function,
)

from .config import DSparkDraftConfig

_EPS = 1e-5


def dspark_block_decay(pos_idx: torch.Tensor, gamma: float) -> torch.Tensor:
    """``w_k = exp(-(k-1)/gamma)`` for k = 1..gamma, with our 0-based position
    ``i = k-1`` (all gamma positions are predictions -> no position-0 drop)."""
    return torch.exp(-pos_idx.to(torch.float32) / gamma)


def compute_dspark_loss(
    draft_logits: torch.Tensor,   # [N, gamma, V]
    target_dist: torch.Tensor,    # [N, gamma, V]  frozen target next-token distribution
    confidence: torch.Tensor,     # [N, gamma]
    loss_mask: torch.Tensor,      # [N, gamma]
    cfg: DSparkDraftConfig,
) -> tuple[torch.Tensor, dict]:
    """Return ``(loss, metrics)`` for one teacher-forced block batch."""
    n, g, v = draft_logits.shape
    logits = draft_logits.reshape(1, n * g, v)
    targets = target_dist.reshape(1, n * g, v)
    mask = loss_mask.reshape(1, n * g)
    conf = confidence.reshape(1, n * g)
    pos_idx = (torch.arange(n * g, device=logits.device) % g).unsqueeze(0)

    loss_fn = combo_ce_l1_loss(cfg.ce_loss_alpha, cfg.l1_loss_alpha)
    decay_fn = partial(dspark_block_decay, gamma=cfg.decay_gamma)
    loss = loss_function(logits, targets, mask, pos_idx, loss_fn=loss_fn, decay_fn=decay_fn)

    metrics: dict = {
        "loss_sum": loss.detach().clone(),
        "loss_total": torch.tensor(1.0, device=logits.device),
    }

    if cfg.confidence_alpha > 0:
        conf_elem = confidence_loss(conf, logits, targets)  # [1, T] soft-accept BCE
        decay = dspark_block_decay(pos_idx, cfg.decay_gamma).to(conf_elem.dtype)
        m = mask.to(conf_elem.dtype)
        conf_scalar = ((conf_elem * m * decay).sum(dim=1) / (m.sum(dim=1) + _EPS)).mean()
        loss = loss + cfg.confidence_alpha * conf_scalar
        metrics["confidence_loss_sum"] = conf_scalar.detach().clone()
        metrics["confidence_loss_total"] = torch.tensor(1.0, device=logits.device)

    return loss, metrics
