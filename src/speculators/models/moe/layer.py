"""The MoE draft FFN: a routed + shared sparse mixture of experts.

:class:`MoE` is the ``nn.Module`` a draft uses in place of a dense FFN. It routes
each token to its top-k experts (:class:`~speculators.models.moe.router.Router`),
combines their SwiGLU outputs, and adds a single always-on shared expert.

The heavy routed-expert compute is dispatched through
:mod:`~speculators.models.moe.backend` under the op ``moe_dispatch``: the
pure-torch reference registered here loops over experts (correct, CPU-parity);
expert-parallel all-to-all (:mod:`~speculators.models.moe.dispatch_ep`) and any
hardware bridge register faster implementations under the same op key.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .backend import get_kernel, torch_kernel
from .config import MoEConfig
from .experts import Expert, GroupedExperts

_MOE_OP = "moe_dispatch"


@torch_kernel(_MOE_OP)
def _moe_dispatch_torch(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    experts: GroupedExperts,
    n_routed_experts: int,
) -> torch.Tensor:
    """Route ``x [tokens, dim]`` to its top-k experts and combine (fp32 accum).

    Pure-torch parity reference (CPU / any accelerator): loops over ALL experts in
    a fixed order, feeding each only its routed tokens. ``experts`` holds the full
    stacked routed-expert weights (non-EP). Faster grouped / expert-parallel
    implementations register under the same op for throughput.
    """
    w1, w3, w2 = experts.local_weights()
    lim = experts.swiglu_limit
    y = torch.zeros_like(x, dtype=torch.float32)
    for e in range(n_routed_experts):
        tok, slot = torch.where(indices == e)
        xe = x[tok].float()
        gate = xe @ w1[e].t().float()
        up = xe @ w3[e].t().float()
        if lim > 0:
            up = torch.clamp(up, min=-lim, max=lim)
            gate = torch.clamp(gate, max=lim)
        h = F.silu(gate) * up
        h = weights[tok, slot, None].float() * h
        y[tok] += h @ w2[e].t().float()
    return y


class MoE(nn.Module):
    """Routed + shared mixture of experts.

    Without expert-parallelism a rank builds all ``n_routed_experts``. Under EP
    (``dispatch_ep`` configured) a rank builds only its disjoint slice of whole
    experts, seeded per-rank; ``ep_expert_offset`` maps a local expert index to
    its global id.
    """

    def __init__(
        self,
        config: MoEConfig,
        *,
        load_balance: bool = False,
        load_balance_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        from .router import Router  # local import keeps router optional at import

        self.dim = config.hidden_size
        self.n_routed_experts = config.n_routed_experts  # GLOBAL count
        self.router = Router(
            config, load_balance=load_balance, load_balance_rate=load_balance_rate
        )

        # Expert-parallel: build only THIS rank's slice of stacked experts. The EP
        # context is read lazily so the layer works with EP never configured.
        from . import dispatch_ep  # noqa: PLC0415

        ep = dispatch_ep.get_context()
        if ep is not None and ep.size > 1:
            n_local = config.n_routed_experts // ep.size
            self.ep_expert_offset = ep.rank * n_local
            seed = 0xE9E9 + ep.rank
        else:
            n_local = config.n_routed_experts
            self.ep_expert_offset = 0
            seed = None
        self.experts = GroupedExperts(
            config.hidden_size, config.moe_inter_dim, n_local, config.swiglu_limit, seed=seed
        )
        self.shared_experts = Expert(
            config.hidden_size, config.moe_inter_dim, config.swiglu_limit
        )

    def forward(self, x: torch.Tensor, backend: str | None = None) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, self.dim)
        weights, indices = self.router(x)
        y = get_kernel(_MOE_OP, backend)(
            x, weights, indices, self.experts, self.n_routed_experts
        )
        y = y + self.shared_experts(x)
        return y.type_as(x).view(shape)

    def update_load_balance_bias(self) -> None:
        """Apply one noaux_tc balancing step to the router (call once per optimizer
        step; no-op when load balancing is off)."""
        self.router.update_load_balance_bias()
