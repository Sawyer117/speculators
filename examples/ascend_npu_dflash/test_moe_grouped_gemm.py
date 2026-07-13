#!/usr/bin/env python3
"""Parity test: fused grouped-GEMM MoE dispatch vs the eager per-expert loop (CPU).

Validates that `moe_dispatch_grouped` reproduces `_moe_dispatch_torch` (same GroupedExperts,
same routing) in BOTH forward and backward. Run in the austin env (speculators editable):
    python examples/ascend_npu_dflash/test_moe_grouped_gemm.py
CPU (default) uses the pure-torch grouped fallback -> pins the routing / permute / swiglu /
accumulate math. `DEVICE=npu python ...` routes through torch_npu.npu_grouped_matmul (fwd) and
the self-written backward -> validates the REAL op + backward before hooking into training.
"""
import os

import torch

_DEV = os.environ.get("DEVICE", "cpu")
if _DEV == "npu":
    import torch_npu  # noqa: F401

from speculators.models.dsv4_dspark.backbone.moe import GroupedExperts, _moe_dispatch_torch
from speculators.models.dsv4_dspark.backbone.moe_grouped_gemm import moe_dispatch_grouped


def _build(seed=0, T=17, dim=16, inter=32, E=8, topk=3, limit=10.0):
    torch.manual_seed(seed)
    experts = GroupedExperts(dim, inter, E, limit).to(_DEV)  # stacked routed-expert weights
    x = torch.randn(T, dim, dtype=torch.float32, device=_DEV)
    # synthetic routing: topk distinct experts per token + normalized positive weights
    indices = torch.stack([torch.randperm(E)[:topk] for _ in range(T)]).to(_DEV)
    w = torch.rand(T, topk, dtype=torch.float32, device=_DEV) + 0.1
    weights = w / w.sum(-1, keepdim=True)
    return experts, x, weights, indices, E


def _run(fn, experts, x, weights, indices, E):
    for p in experts.parameters():
        p.grad = None
    xg = x.clone().requires_grad_(True)
    y = fn(xg, weights, indices, experts, E)
    y.sum().backward()
    wgrads = [experts.w1.grad.clone(), experts.w2.grad.clone(), experts.w3.grad.clone()]
    return y.detach(), xg.grad.detach(), wgrads


def main():
    print(f">>> device={_DEV}  ({'real npu_grouped_matmul + backward' if _DEV == 'npu' else 'torch grouped fallback'})")
    experts, x, weights, indices, E = _build()
    y_e, gx_e, gw_e = _run(_moe_dispatch_torch, experts, x, weights, indices, E)
    y_g, gx_g, gw_g = _run(moe_dispatch_grouped, experts, x, weights, indices, E)

    fy = (y_e - y_g).abs().max().item()
    fx = (gx_e - gx_g).abs().max().item()
    fw = max((a - b).abs().max().item() for a, b in zip(gw_e, gw_g))
    print(f"[fwd ] max|y_eager - y_grouped|        = {fy:.2e}")
    print(f"[bwd ] max|grad_x  eager - grouped|    = {fx:.2e}")
    print(f"[bwd ] max|grad_W  eager - grouped|    = {fw:.2e}")
    tol = 1e-4
    ok = fy < tol and fx < tol and fw < tol
    print(f"\n{'OK' if ok else 'FAIL'}: grouped-GEMM == eager loop (fwd+bwd, tol={tol}).")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
