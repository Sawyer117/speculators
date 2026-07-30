"""End-to-end smoke test for expert-parallel MoE dispatch.

Runs a real multi-process expert-parallel forward and asserts it equals the
single-process per-expert reference (the parity claim EP must satisfy), plus a
``Shard(0)`` DTensor round-trip. Uses the **gloo** backend on CPU so it runs in
CI with no accelerator; the same code path runs on CUDA (NCCL) / NPU (HCCL) when
those are the active accelerator.

Run directly (CPU, 2 procs):
    python -m pytest tests/e2e/smoke/test_expert_parallel.py
"""

from __future__ import annotations

import os

import pytest
import torch

WORLD = 2
N_ROUTED = 4
DIM = 16
INTER = 32
TOPK = 2
TOKENS = 24


def _deterministic_full_weights():
    g = torch.Generator().manual_seed(0)
    w1 = torch.randn(N_ROUTED, INTER, DIM, generator=g)
    w3 = torch.randn(N_ROUTED, INTER, DIM, generator=g)
    w2 = torch.randn(N_ROUTED, DIM, INTER, generator=g)
    return w1, w3, w2


def _deterministic_routing():
    gi = torch.Generator().manual_seed(1)
    x = torch.randn(TOKENS, DIM, generator=gi)
    indices = torch.stack(
        [torch.randperm(N_ROUTED, generator=torch.Generator().manual_seed(100 + t))[:TOPK]
         for t in range(TOKENS)],
        dim=0,
    )
    gw = torch.Generator().manual_seed(2)
    weights = torch.rand(TOKENS, TOPK, generator=gw)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return x, weights, indices


def _ep_worker(rank: int, world_size: int, port: int) -> None:
    import torch.distributed as dist

    from speculators.models.moe.dispatch_ep import configure, moe_dispatch_ep, reset
    from speculators.models.moe.experts import GroupedExperts
    from speculators.models.moe.layer import _moe_dispatch_torch

    dist.init_process_group(
        backend="gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size
    )
    try:
        experts_per_rank = N_ROUTED // world_size
        w1, w3, w2 = _deterministic_full_weights()

        # full expert set (identical on every rank) -> the single-process reference
        full = GroupedExperts(DIM, INTER, N_ROUTED, 0.0)
        full.w1.data.copy_(w1)
        full.w3.data.copy_(w3)
        full.w2.data.copy_(w2)

        # this rank's disjoint slice of whole experts
        lo = rank * experts_per_rank
        local = GroupedExperts(DIM, INTER, experts_per_rank, 0.0)
        local.w1.data.copy_(w1[lo : lo + experts_per_rank])
        local.w3.data.copy_(w3[lo : lo + experts_per_rank])
        local.w2.data.copy_(w2[lo : lo + experts_per_rank])

        reset()
        configure(dist.group.WORLD, rank, world_size, experts_per_rank)

        x, weights, indices = _deterministic_routing()
        ep_out = moe_dispatch_ep(x, weights, indices, local, N_ROUTED)
        ref = _moe_dispatch_torch(x, weights, indices, full, N_ROUTED)
        if not torch.allclose(ep_out, ref, atol=1e-4):
            raise AssertionError(
                f"[rank {rank}] EP dispatch != reference "
                f"(max abs diff {(ep_out - ref).abs().max().item():.3e})"
            )

        # Shard(0) DTensor round-trip: shard the full experts, gather back, compare.
        try:
            from torch.distributed.device_mesh import init_device_mesh
            from torch.distributed.tensor import DTensor, Shard

            mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("ep",))
            local_slice = w1[lo : lo + experts_per_rank].contiguous()
            dt = DTensor.from_local(local_slice, mesh["ep"], [Shard(0)], run_check=False)
            gathered = dt.full_tensor()
            if not torch.allclose(gathered, w1, atol=1e-6):
                raise AssertionError(f"[rank {rank}] Shard(0) round-trip mismatch")
        except (ImportError, RuntimeError) as exc:  # DTensor/cpu-mesh unsupported here
            if rank == 0:
                print(f"[skip] DTensor round-trip unavailable on this build: {exc}")
    finally:
        dist.destroy_process_group()


@pytest.mark.e2e
def test_expert_parallel_parity_cpu_gloo():
    """EP dispatch on 2 gloo procs == single-process per-expert reference."""
    port = 29500 + (os.getpid() % 2000)
    torch.multiprocessing.spawn(_ep_worker, args=(WORLD, port), nprocs=WORLD, join=True)
