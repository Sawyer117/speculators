#!/usr/bin/env python3
"""Parity test: expert-parallel (EP) MoE dispatch vs the eager per-expert loop.

Two modes, both check that EP dispatch reproduces the eager reference:

  * 1 process (default) -- DEGENERATE EP (size=1): the all-to-all short-circuits, so
    this pins the flatten / local-grouped / combine math == eager loop (fwd + grad_x +
    grad_W). Runs on CPU, no dist:
        python examples/ascend_npu_dflash/test_moe_ep.py

  * N processes (gloo/hccl) -- REAL EP: experts partitioned across ranks, tokens
    all-to-all'd to owners, results all-to-all'd back. Each rank compares its EP output
    to the eager output over the SAME (full) expert set for its OWN batch -> validates the
    all-to-all dispatch AND its autograd backward (grad_x flows back through _AllToAll).
    CPU/gloo (any box, no NPU needed):
        torchrun --nproc_per_node 2 examples/ascend_npu_dflash/test_moe_ep.py
    NPU/hccl (real path):
        DEVICE=npu torchrun --nproc_per_node 8 examples/ascend_npu_dflash/test_moe_ep.py

E (total experts) must be divisible by the process count.
"""
import os

import torch

_DEV = os.environ.get("DEVICE", "cpu")
if _DEV == "npu":
    import torch_npu  # noqa: F401

from speculators.models.dsv4_dspark.backbone import moe_ep
from speculators.models.dsv4_dspark.backbone.moe import GroupedExperts, _moe_dispatch_torch


def _build_experts(seed, E, dim, inter, limit):
    torch.manual_seed(seed)  # SAME seed on every rank -> identical full expert set
    return GroupedExperts(dim, inter, E, limit).to(_DEV)


def _slice_experts(full, rank, L, dim, inter, limit):
    """A GroupedExperts holding full's [rank*L:(rank+1)*L] slice (this rank's EP shard)."""
    local = GroupedExperts(dim, inter, L, limit).to(_DEV)
    with torch.no_grad():
        local.w1.copy_(full.w1[rank * L:(rank + 1) * L])
        local.w3.copy_(full.w3[rank * L:(rank + 1) * L])
        local.w2.copy_(full.w2[rank * L:(rank + 1) * L])
    return local


def _batch(seed, T, dim, E, topk):
    torch.manual_seed(seed)  # per-rank distinct batch
    x = torch.randn(T, dim, dtype=torch.float32, device=_DEV)
    indices = torch.stack([torch.randperm(E)[:topk] for _ in range(T)]).to(_DEV)
    w = torch.rand(T, topk, dtype=torch.float32, device=_DEV) + 0.1
    return x, (w / w.sum(-1, keepdim=True)), indices


def _degenerate():
    """1 process: EP(size=1) == eager, fwd + grad_x + grad_W."""
    E, dim, inter, topk, limit = 8, 16, 32, 3, 10.0
    experts = _build_experts(0, E, dim, inter, limit)
    x, weights, indices = _batch(1, 17, dim, E, topk)
    moe_ep.configure(group=None, rank=0, size=1, experts_per_rank=E)

    def run(fn, exps):
        for p in exps.parameters():
            p.grad = None
        xg = x.clone().requires_grad_(True)
        y = fn(xg, weights, indices, exps, E)
        y.sum().backward()
        gw = [exps.w1.grad.clone(), exps.w2.grad.clone(), exps.w3.grad.clone()]
        return y.detach(), xg.grad.detach(), gw

    y_e, gx_e, gw_e = run(_moe_dispatch_torch, experts)
    y_g, gx_g, gw_g = run(moe_ep.moe_dispatch_ep, experts)
    fy = (y_e - y_g).abs().max().item()
    fx = (gx_e - gx_g).abs().max().item()
    fw = max((a - b).abs().max().item() for a, b in zip(gw_e, gw_g))
    print(f"[degenerate EP=1]  fwd={fy:.2e}  grad_x={fx:.2e}  grad_W={fw:.2e}")
    ok = fy < 1e-4 and fx < 1e-4 and fw < 1e-4
    print(f"{'OK' if ok else 'FAIL'}: EP(size=1) == eager (fwd+bwd).")
    raise SystemExit(0 if ok else 1)


def _distributed():
    """N processes: partition experts, EP dispatch, compare per-rank to eager over full set."""
    import torch.distributed as dist
    backend = "hccl" if _DEV == "npu" else "gloo"
    dist.init_process_group(backend=backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    if _DEV == "npu":
        torch.npu.set_device(rank)

    E, dim, inter, topk, limit = 8, 16, 32, 3, 10.0
    assert E % world == 0, f"E={E} not divisible by world={world}"
    L = E // world
    full = _build_experts(0, E, dim, inter, limit)              # identical on all ranks
    local = _slice_experts(full, rank, L, dim, inter, limit)    # this rank's EP shard
    x, weights, indices = _batch(100 + rank, 17, dim, E, topk)  # per-rank distinct batch

    moe_ep.configure(group=dist.group.WORLD, rank=rank, size=world, experts_per_rank=L)

    xg = x.clone().requires_grad_(True)
    y_ep = moe_ep.moe_dispatch_ep(xg, weights, indices, local, E)
    y_ep.sum().backward()
    gx_ep = xg.grad.detach().clone()

    xr = x.clone().requires_grad_(True)                         # eager over the full expert set
    y_ref = _moe_dispatch_torch(xr, weights, indices, full, E)
    y_ref.sum().backward()

    fy = (y_ep.detach() - y_ref.detach()).abs().max()
    fx = (gx_ep - xr.grad.detach()).abs().max()
    err = torch.stack([fy, fx]).to(_DEV)
    dist.all_reduce(err, op=dist.ReduceOp.MAX)                  # worst rank
    if rank == 0:
        print(f"[EP world={world}]  fwd={err[0].item():.2e}  grad_x={err[1].item():.2e}")
        ok = err[0].item() < 1e-4 and err[1].item() < 1e-4
        print(f"{'OK' if ok else 'FAIL'}: EP all-to-all dispatch == eager (fwd + grad_x).")
    dist.barrier()
    dist.destroy_process_group()


def main():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        _distributed()
    else:
        _degenerate()


if __name__ == "__main__":
    main()
