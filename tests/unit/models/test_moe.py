"""Unit tests for the generic MoE draft layer (CPU, no distributed).

Pins the correctness contracts the faster paths must preserve:
* the grouped matmul reference == a naive per-group matmul;
* the expert-parallel dispatch's degenerate (single-rank) path == the per-expert
  loop reference used by :class:`MoE`;
* :class:`MoE` runs end-to-end on CPU and is shape-correct.
"""

from __future__ import annotations

import torch

from speculators.models.moe import GroupedExperts, MoE, MoEConfig
from speculators.models.moe.dispatch_ep import moe_dispatch_ep, reset
from speculators.models.moe.experts import _grouped_matmul_reference, grouped_matmul
from speculators.models.moe.layer import _moe_dispatch_torch


def _config() -> MoEConfig:
    return MoEConfig(
        hidden_size=16,
        moe_inter_dim=32,
        n_routed_experts=4,
        n_activated_experts=2,
        n_shared_experts=1,
        swiglu_limit=0.0,
        score_func="sqrtsoftplus",
    )


def _routing(tokens: int, n_experts: int, topk: int):
    torch.manual_seed(7)
    indices = torch.stack(
        [torch.randperm(n_experts)[:topk] for _ in range(tokens)], dim=0
    )
    weights = torch.rand(tokens, topk)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights, indices


def test_grouped_matmul_matches_reference():
    torch.manual_seed(0)
    counts = torch.tensor([3, 0, 2, 1])
    x = torch.randn(int(counts.sum()), 8)
    weight = torch.randn(4, 8, 5)  # [E, K, M]
    ref = _grouped_matmul_reference(x, weight, counts)
    out = grouped_matmul(x, weight, counts)  # default path == reference on CPU
    assert torch.allclose(out, ref, atol=1e-6)
    assert out.shape == (int(counts.sum()), 5)


def test_ep_degenerate_matches_loop_reference():
    """With no EP configured, moe_dispatch_ep must equal the per-expert loop."""
    reset()  # ensure no EP context leaked from another test
    cfg = _config()
    torch.manual_seed(1)
    experts = GroupedExperts(cfg.hidden_size, cfg.moe_inter_dim, cfg.n_routed_experts, 0.0)
    x = torch.randn(12, cfg.hidden_size)
    weights, indices = _routing(12, cfg.n_routed_experts, cfg.n_activated_experts)

    ref = _moe_dispatch_torch(x, weights, indices, experts, cfg.n_routed_experts)
    ep = moe_dispatch_ep(x, weights, indices, experts, cfg.n_routed_experts)
    assert torch.allclose(ep, ref, atol=1e-4)


def test_moe_forward_cpu():
    reset()
    cfg = _config()
    torch.manual_seed(2)
    moe = MoE(cfg)
    # router weight must be initialized (a from-scratch draft has no checkpoint to
    # overwrite it) — uninitialized memory would feed NaNs into the score function.
    assert torch.isfinite(moe.router.weight).all()
    x = torch.randn(2, 5, cfg.hidden_size)
    y = moe(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_load_balance_bias_updates_only_when_enabled():
    reset()
    cfg = _config()
    moe_off = MoE(cfg)
    moe_off.train()
    moe_off(torch.randn(8, cfg.hidden_size))
    moe_off.update_load_balance_bias()
    assert torch.count_nonzero(moe_off.router.bias) == 0  # off by default -> stays zero

    moe_on = MoE(cfg, load_balance=True, load_balance_rate=1e-2)
    moe_on.train()
    moe_on(torch.randn(64, cfg.hidden_size))
    moe_on.update_load_balance_bias()
    # zero-mean update, but at least some entries move once there is load imbalance
    assert torch.count_nonzero(moe_on.router.bias) > 0
