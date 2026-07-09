"""Mixture-of-experts FFN for the DSV4 DSpark draft backbone.

``256`` routed experts (SwiGLU with a magnitude clamp) + ``1`` shared expert,
top-``k`` routing with ``sqrtsoftplus`` scoring. The draft layers are all
score-routed (hash routing only applies to the target's first few layers, which
the draft does not include), so no token-id / hash path is needed here.

The per-expert compute (the 256-way grouped matmul) is the natural NPU
insertion point: :class:`MoE` dispatches it through :mod:`.kernels` under the op
``moe_dispatch``. The torch reference loops over the experts that actually
received tokens — correct and fine for CPU parity on the small config; the
accelerator bridge registers a fused grouped-GEMM for real training throughput.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .kernels import get_kernel, torch_kernel

_MOE_OP = "moe_dispatch"


class Router(nn.Module):
    """Score-based top-k router (``sqrtsoftplus``).

    ``bias`` shifts scores for *selection* (top-k) only; the combine
    ``weights`` are gathered from the pre-bias scores, sum-normalized, then
    scaled by ``route_scale`` (this is the non-softmax path).
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.topk = cfg.n_activated_experts
        self.route_scale = cfg.route_scale
        self.score_func = cfg.score_func
        self.weight = nn.Parameter(torch.empty(cfg.n_routed_experts, cfg.hidden_size))
        self.bias = nn.Parameter(torch.zeros(cfg.n_routed_experts))

    def _score(self, scores: torch.Tensor) -> torch.Tensor:
        if self.score_func == "softmax":
            return scores.softmax(dim=-1)
        if self.score_func == "sigmoid":
            return scores.sigmoid()
        return F.softplus(scores).sqrt()  # sqrtsoftplus (DSV4 default)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self._score(F.linear(x.float(), self.weight.float()))
        selection = scores + self.bias.float()
        indices = selection.topk(self.topk, dim=-1)[1]
        weights = scores.gather(1, indices)
        if self.score_func != "softmax":
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.route_scale
        return weights, indices


class Expert(nn.Module):
    """SwiGLU expert with an optional magnitude clamp (fp32 activations)."""

    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)  # gate
        self.w3 = nn.Linear(dim, inter_dim, bias=False)  # up
        self.w2 = nn.Linear(inter_dim, dim, bias=False)  # down
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        h = F.silu(gate) * up
        if weight is not None:
            h = weight * h
        return self.w2(h.to(dtype))


@torch_kernel(_MOE_OP)
def _moe_dispatch_torch(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    experts: nn.ModuleList,
    n_routed_experts: int,
) -> torch.Tensor:
    """Route ``x [tokens, dim]`` to its top-k experts and combine (fp32 accum).

    Loops over the experts that received at least one token — the standard
    reference. The accelerator bridge registers a fused grouped-GEMM under the
    same op for throughput.
    """
    y = torch.zeros_like(x, dtype=torch.float32)
    counts = torch.bincount(indices.flatten(), minlength=n_routed_experts)
    for e in range(n_routed_experts):
        if counts[e] == 0:
            continue
        tok, slot = torch.where(indices == e)
        y[tok] += experts[e](x[tok], weights[tok, slot, None])
    return y


class MoE(nn.Module):
    """Routed + shared mixture of experts."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.dim = cfg.hidden_size
        self.n_routed_experts = cfg.n_routed_experts
        self.router = Router(cfg)
        self.experts = nn.ModuleList(
            Expert(cfg.hidden_size, cfg.moe_inter_dim, cfg.swiglu_limit)
            for _ in range(cfg.n_routed_experts)
        )
        # The released config carries exactly one shared expert; the checkpoint
        # key is ``ffn.shared_experts.{w1,w2,w3}`` (a single expert), so this is
        # a single module, not a list.
        if cfg.n_shared_experts != 1:
            raise ValueError("only n_shared_experts == 1 is supported (matches the release).")
        self.shared_experts = Expert(cfg.hidden_size, cfg.moe_inter_dim, cfg.swiglu_limit)

    def forward(self, x: torch.Tensor, backend: str | None = None) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, self.dim)
        weights, indices = self.router(x)
        y = get_kernel(_MOE_OP, backend)(
            x, weights, indices, self.experts, self.n_routed_experts
        )
        y = y + self.shared_experts(x)
        return y.type_as(x).view(shape)
