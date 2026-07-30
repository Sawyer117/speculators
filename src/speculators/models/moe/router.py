"""Top-k router for the MoE draft layer.

Score-based selection with an optional aux-loss-free load-balancing bias
(DeepSeek's ``noaux_tc``): a non-gradient per-expert ``bias`` shifts the top-k
*selection* only (never the combine weights), nudged toward uniform expert load
by an explicit rule each training step rather than by an auxiliary loss.
Balancing is OFF by default (the bias stays zero), which reproduces a plain
score router.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from .config import MoEConfig


class Router(nn.Module):
    """Score-based top-k router.

    ``bias`` shifts scores for *selection* (top-k) only; the combine ``weights``
    are gathered from the pre-bias scores, sum-normalized (non-softmax path), then
    scaled by ``route_scale``.
    """

    def __init__(
        self,
        config: MoEConfig,
        *,
        load_balance: bool = False,
        load_balance_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        self.topk = config.n_activated_experts
        self.route_scale = config.route_scale
        self.score_func = config.score_func
        self.n_routed_experts = config.n_routed_experts
        self.weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.hidden_size)
        )
        # Initialize like nn.Linear (kaiming_uniform_, a=sqrt(5)); a from-scratch
        # draft has no checkpoint to overwrite the router, so uninitialized memory
        # would feed garbage into the score function.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Aux-loss-free load-balancing bias (noaux_tc): shifts the top-k SELECTION
        # only, nudged by a rule at train time — NOT by backprop. A persistent
        # BUFFER (not a Parameter) so FSDP leaves it REPLICATED (not Shard(0)): it
        # stays identical across ranks and is updated in-place with the all-reduced
        # global load.
        self.register_buffer("bias", torch.zeros(config.n_routed_experts), persistent=True)
        self.load_balance = load_balance
        self.load_balance_rate = load_balance_rate
        # Set in forward when balancing is on; consumed by update_load_balance_bias.
        self._step_load: torch.Tensor | None = None

    def _score(self, scores: torch.Tensor) -> torch.Tensor:
        if self.score_func == "softmax":
            return scores.softmax(dim=-1)
        if self.score_func == "sigmoid":
            return scores.sigmoid()
        return F.softplus(scores).sqrt()  # sqrtsoftplus

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self._score(F.linear(x.float(), self.weight.float()))
        selection = scores + self.bias.float()
        indices = selection.topk(self.topk, dim=-1)[1]
        weights = scores.gather(1, indices)
        if self.score_func != "softmax":
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.route_scale
        if self.load_balance and self.training:
            # Stash this rank's per-expert selection count (SET, not accumulate, so a
            # grad-checkpoint recompute re-sets the same value -> no double count).
            self._step_load = torch.bincount(
                indices.reshape(-1), minlength=self.n_routed_experts
            ).detach().float()
        return weights, indices

    @torch.no_grad()
    def update_load_balance_bias(self) -> None:
        """Nudge the (non-grad) selection ``bias`` toward uniform expert load without
        an aux loss: ``b_i += rate * sign(avg_load - load_i)`` (under-loaded up,
        over-loaded down), zero-meaned so the bias stays centered over a long run.

        Call ONCE per optimizer step. Under expert-parallelism the router runs on
        each rank's shard of tokens, so the counts are all-reduced to the GLOBAL
        load — the bias is REPLICATED and must update identically on every rank.
        No-op when balancing is off or no load was stashed.
        """
        if not self.load_balance or self._step_load is None:
            return
        load = self._step_load
        if dist.is_initialized():
            dist.all_reduce(load, op=dist.ReduceOp.SUM)  # -> global per-expert load
        delta = self.load_balance_rate * torch.sign(load.mean() - load)
        delta = delta - delta.mean()  # zero-mean the step: keep the bias centered
        self.bias.add_(delta.to(self.bias.dtype))
        self._step_load = None
