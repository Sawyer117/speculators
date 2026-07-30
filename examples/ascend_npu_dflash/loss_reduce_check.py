"""Self-check for DSPARK_GLOBAL_LOSS_REDUCE: verifies the grad math on 2 gloo ranks.

Replicates the global_masked_mean scheme (local_num/global_den*world) + the current
rank-local path + an analytic global reference, then compares the DDP-mean-averaged
gradient of a shared param under each. Proves:
  (1) correctness — global-fix grad == the true token-weighted global grad;
  (2) LR-safety   — when ranks are token-BALANCED, global-fix grad == rank-local grad
                    (so folding the flag into a resumed run does NOT shift the LR);
  (3) it does something — when IMBALANCED, global-fix != rank-local, and global-fix is
                    the correct one.
"""
import os

import torch
import torch.distributed as dist


def _ddp_mean(t):
    r = t.clone()
    dist.all_reduce(r, op=dist.ReduceOp.SUM)
    return r / dist.get_world_size()


def worker(rank, world, port, balanced):
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    try:
        theta = torch.tensor(1.5, requires_grad=True)
        n = 8 if balanced else (4 if rank == 0 else 12)  # supervised-token count per rank
        x = torch.randn(n, generator=torch.Generator().manual_seed(100 + rank))
        num = (theta * x).sum()          # Σ weighted-loss (grad wrt theta = Σx)
        den = torch.tensor(float(n))     # Σ mask

        # --- global-fix path (the code): local_num / global_den * world ---
        gden = den.clone()
        dist.all_reduce(gden, op=dist.ReduceOp.SUM)
        loss_g = num / (gden + 1e-5) * world
        gg = _ddp_mean(torch.autograd.grad(loss_g, theta, retain_graph=True)[0])

        # --- current rank-local path: local_num / local_den ---
        loss_l = num / (den + 1e-5)
        gl = _ddp_mean(torch.autograd.grad(loss_l, theta, retain_graph=True)[0])

        # --- analytic reference: d[(Σ_r Σx_r) / Σ_r n_r]/dtheta = ΣΣx / Σn ---
        sumx = x.sum().detach().clone()
        dist.all_reduce(sumx, op=dist.ReduceOp.SUM)
        ref = sumx / gden

        if rank == 0:
            tag = "balanced  " if balanced else "imbalanced"
            print(f"[{tag}] global-fix={gg.item():.6f}  rank-local={gl.item():.6f}  ref={ref.item():.6f}")
            assert torch.allclose(gg, ref, atol=1e-5), f"[{tag}] global-fix != reference"
            if balanced:
                assert torch.allclose(gg, gl, atol=1e-5), "balanced: global != rank-local (LR would shift!)"
                print("   ✓ correctness (global==ref)   ✓ LR-safe (global==rank-local)")
            else:
                assert not torch.allclose(gg, gl, atol=1e-4), "imbalanced: expected global != rank-local"
                assert not torch.allclose(gl, ref, atol=1e-4), "imbalanced: rank-local should be the wrong one"
                print("   ✓ correctness (global==ref)   ✓ global != rank-local (rank-local is WRONG here)")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    for bal in (True, False):
        port = 29700 + (os.getpid() % 900) + (0 if bal else 1)
        torch.multiprocessing.spawn(worker, args=(2, port, bal), nprocs=2, join=True)
    print("\nALL LOSS-REDUCE CHECKS PASSED (CPU + gloo, 2 ranks)")
