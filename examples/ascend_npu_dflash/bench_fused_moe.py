#!/usr/bin/env python
"""Graft A validation — fused Ascend MoE routing vs the eager reference.

Run ON THE BOX (NPU env, e.g. dspark-dsv4-austin):

    python examples/ascend_npu_dflash/bench_fused_moe.py
    # bigger / faithful-ish shape:
    T=8192 DIM=4096 INTER=352 E=256 TOPK=6 python examples/ascend_npu_dflash/bench_fused_moe.py

Compares three implementations of the SAME routed-MoE dispatch math:
  (0) eager per-expert loop        -- moe._moe_dispatch_torch                (ground truth)
  (1) argsort + grouped-GEMM       -- the pre-Graft-A path (int64 argsort -> AiCPU fallback)
  (2) fused permute + grouped-GEMM -- Graft A (npu_moe_token_permute/unpermute)

Checks:
  * FORWARD parity   (2) vs (0)
  * BACKWARD parity  (2) vs (0)  -- grads on x AND on expert w1/w2/w3  (proves the fused
                                    permute/unpermute ops are differentiable)
  * SPEED            (0) vs (1) vs (2), forward-only, ms/call

A green run = (2) is numerically close to (0) AND faster than (1); then it is safe to let
training take the NPU branch in moe_grouped_gemm / moe_ep.
"""
from __future__ import annotations

import os
import time

import torch
import torch_npu  # noqa: F401
from torch_npu.contrib import transfer_to_npu  # noqa: F401

from speculators.models.dsv4_dspark.backbone.moe import (
    GroupedExperts,
    _moe_dispatch_torch,
    swiglu_grouped,
)
from speculators.models.dsv4_dspark.backbone.moe_grouped_gemm import (
    _grouped_matmul,
    moe_dispatch_grouped,
)

torch.manual_seed(0)

DEV = "npu"
T = int(os.environ.get("T", 2000))     # non-multiple of the bucket -> exercises bucket padding
DIM = int(os.environ.get("DIM", 1024))
INTER = int(os.environ.get("INTER", 704))
E = int(os.environ.get("E", 64))
TOPK = int(os.environ.get("TOPK", 6))
SWIGLU_LIMIT = float(os.environ.get("SWIGLU_LIMIT", 0.0))


def make_inputs(requires_grad: bool):
    experts = GroupedExperts(DIM, INTER, n_local=E, swiglu_limit=SWIGLU_LIMIT, seed=0).to(DEV)
    x = torch.randn(T, DIM, device=DEV, requires_grad=requires_grad)
    scores = torch.rand(T, E, device=DEV)
    indices = scores.topk(TOPK, dim=-1).indices              # [T, topk], distinct per row
    weights = torch.rand(T, TOPK, device=DEV)                # router combine weights
    return experts, x, weights, indices


def argsort_grouped(x, weights, indices, experts, n):
    """The pre-Graft-A path (argsort + grouped-GEMM), forced explicitly so we can time it on NPU."""
    w1, w3, w2 = experts.local_weights()
    Tt, dim = x.shape
    topk = indices.shape[1]
    tok = torch.arange(Tt, device=x.device).repeat_interleave(topk)
    exp = indices.reshape(-1)
    wf = weights.reshape(-1).float()
    order = torch.argsort(exp, stable=True)
    tok, wf = tok[order], wf[order]
    counts = torch.bincount(exp[order], minlength=n)
    out = swiglu_grouped(x[tok].float(), w1.float(), w3.float(), w2.float(), counts, wf,
                         experts.swiglu_limit, _grouped_matmul)
    y = torch.zeros(Tt, dim, dtype=torch.float32, device=x.device)
    return y.index_add(0, tok, out)


def parity():
    from speculators.models.dsv4_dspark.backbone.moe_grouped_gemm import _MOE_BUCKET, _bucket_count
    print(f"\n=== PARITY (shape T={T} DIM={DIM} INTER={INTER} E={E} TOPK={TOPK}) ===")
    print(f"    DSPARK_MOE_BUCKET={_MOE_BUCKET}  ->  routed-token count {T} bucketed to {_bucket_count(T)}"
          f"  (pad {_bucket_count(T) - T}); parity below must still hold with padding.")
    experts, x, weights, indices = make_inputs(requires_grad=True)
    leaves = [x, experts.w1, experts.w2, experts.w3]

    y_ref = _moe_dispatch_torch(x, weights, indices, experts, E)     # (0) ground truth
    y_fused = moe_dispatch_grouped(x, weights, indices, experts, E)  # (2) Graft A (NPU branch)

    denom = y_ref.abs().max().clamp_min(1e-9)
    fwd = (y_ref - y_fused).abs().max().item()
    print(f"[fwd ] max|Δ|={fwd:.3e}  rel={(fwd / denom).item():.3e}  ref|max|={denom.item():.3e}")

    g = torch.randn_like(y_ref)
    gref = torch.autograd.grad(y_ref, leaves, g, retain_graph=False)
    gfused = torch.autograd.grad(y_fused, leaves, g, retain_graph=False)
    for name, a, b in zip(["dx ", "dw1", "dw2", "dw3"], gref, gfused):
        d = (a - b).abs().max().item()
        rel = d / a.abs().max().clamp_min(1e-9).item()
        print(f"[bwd ] {name}: max|Δ|={d:.3e}  rel={rel:.3e}")


def bench():
    print("\n=== SPEED (forward-only, ms/call) ===")
    experts, x, weights, indices = make_inputs(requires_grad=False)

    def timed(fn, iters=50, warmup=10):
        with torch.no_grad():
            for _ in range(warmup):
                fn()
            torch.npu.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.npu.synchronize()
        return (time.perf_counter() - t0) / iters * 1000

    t0 = timed(lambda: _moe_dispatch_torch(x, weights, indices, experts, E))
    t1 = timed(lambda: argsort_grouped(x, weights, indices, experts, E))
    t2 = timed(lambda: moe_dispatch_grouped(x, weights, indices, experts, E))
    print(f"(0) eager per-expert loop : {t0:8.3f} ms")
    print(f"(1) argsort + grouped     : {t1:8.3f} ms   ({t0 / t1:.2f}x vs eager)")
    print(f"(2) fused  + grouped (A)  : {t2:8.3f} ms   ({t1 / t2:.2f}x vs argsort, {t0 / t2:.2f}x vs eager)")


if __name__ == "__main__":
    parity()
    bench()
    print("\nDONE. Green = [fwd]/[bwd] rel small (~1e-2 bf16-grouped ok) AND (2) faster than (1).")
