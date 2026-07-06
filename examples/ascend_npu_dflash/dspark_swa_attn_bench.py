#!/usr/bin/env python3
"""Fallback bench for the DSpark draft's NON-CAUSAL sliding-window attention on Ascend NPU.

Training the draft needs FORWARD + BACKWARD. If SDPA can't do the non-causal (bidirectional)
sliding-window attention on NPU, we need a fallback that still has autograd. This benches the
candidates against an EAGER reference (the ground-truth math), reporting forward precision,
forward time, and forward+backward total time:

  - eager   : manual softmax(QK^T*scale + mask) @ V, fp32 internals -> the reference. PURE torch,
              ALWAYS has autograd on NPU. This is the guaranteed fallback (worst case = slower).
  - sdpa    : F.scaled_dot_product_attention(..., attn_mask=<bidir window bool mask>)
  - npu_fa  : torch_npu.npu_fusion_attention (Ascend native fused), best-effort.

vLLM-Ascend's SAS op (the INFERENCE path, PR #11196) computes the same attention math but via a
serving-only fused kernel; the eager reference here is that same math, so "diff vs eager" is
"diff vs the correct attention". A bf16 diff ~1e-2 vs the fp32 eager reference is EXPECTED (bf16
precision), not a bug.

Run in dspark-dsv4-base on ONE NPU:  python dspark_swa_attn_bench.py
"""
import os
import time

import torch
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401
    DEV = "npu:0"
except Exception as e:  # noqa: BLE001
    print(f"!! torch_npu import failed: {e}")
    raise SystemExit(1)

torch.manual_seed(0)
# DTYPE=float32 to prove the bf16 diffs are just dtype rounding (they drop to ~1e-6), not a math bug.
DT = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
    os.environ.get("DTYPE", "bfloat16"), torch.bfloat16)
B, H, L, D = 2, 8, 512, 64          # batch, heads, seq len, head dim (draft-block scale)
WIN = 16                            # bidirectional sliding-window half-width
SCALE = D ** -0.5
NITER = 30

# bidirectional (non-causal) sliding-window mask: keep |qi - ki| <= WIN
qi = torch.arange(L, device=DEV).view(L, 1)
ki = torch.arange(L, device=DEV).view(1, L)
KEEP = ((ki >= qi - WIN) & (ki <= qi + WIN)).view(1, 1, L, L)     # bool [1,1,L,L], True=attend

# fixed q/k/v so every method sees identical inputs (fair precision compare)
Q0 = torch.randn(B, H, L, D, device=DEV, dtype=DT)
K0 = torch.randn(B, H, L, D, device=DEV, dtype=DT)
V0 = torch.randn(B, H, L, D, device=DEV, dtype=DT)


def fresh():
    return (Q0.clone().requires_grad_(True),
            K0.clone().requires_grad_(True),
            V0.clone().requires_grad_(True))


def attn_eager(q, k, v):
    s = (q.float() @ k.float().transpose(-2, -1)) * SCALE
    s = s.masked_fill(~KEEP, float("-inf"))
    return (s.softmax(dim=-1) @ v.float()).to(DT)


def attn_sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=KEEP, scale=SCALE)


def attn_npu(q, k, v):
    am = (~KEEP).squeeze(0).squeeze(0).contiguous()              # [L,L] bool, True = masked out
    return torch_npu.npu_fusion_attention(
        q, k, v, H, "BNSD", atten_mask=am, scale=SCALE, keep_prob=1.0)[0]


REF_OUT = None
REF_GQ = None


def bench(name, fn):
    global REF_OUT, REF_GQ
    # ---- correctness + grad (single run) ----
    try:
        q, k, v = fresh()
        out = fn(q, k, v)
        out.float().sum().backward()
        torch.npu.synchronize()
    except Exception as e:  # noqa: BLE001
        print(f"  {name:<8} FAILED (fwd or bwd): {type(e).__name__}: {str(e)[:70]}")
        return
    if REF_OUT is None:
        REF_OUT, REF_GQ = out.detach().float(), q.grad.detach().float()
        od = gd = 0.0
    else:
        od = (out.detach().float() - REF_OUT).abs().max().item()
        gd = (q.grad.detach().float() - REF_GQ).abs().max().item()
    # ---- forward-only time ----
    for _ in range(3):
        with torch.no_grad():
            fn(*fresh())
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(NITER):
        with torch.no_grad():
            fn(*fresh())
    torch.npu.synchronize()
    fwd = (time.time() - t0) / NITER * 1e3
    # ---- forward + backward time ----
    for _ in range(3):
        q, k, v = fresh()
        fn(q, k, v).float().sum().backward()
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(NITER):
        q, k, v = fresh()
        fn(q, k, v).float().sum().backward()
    torch.npu.synchronize()
    fb = (time.time() - t0) / NITER * 1e3
    print(f"  {name:<8} fwd={fwd:7.3f}ms  fwd+bwd={fb:7.3f}ms  outDiff={od:.2e}  gradDiff={gd:.2e}")


print(f">>> non-causal sliding-window attn  B={B} H={H} L={L} D={D} win=±{WIN}  dtype={DT}  iters={NITER}")
print(">>> (outDiff/gradDiff are vs the eager fp32 reference; ~1e-2 for bf16 is expected)\n")
bench("eager", attn_eager)      # reference; the guaranteed autograd fallback
bench("sdpa", attn_sdpa)        # preferred fast path IF it works on NPU
bench("npu_fa", attn_npu)       # Ascend native fused, best-effort

print("\n>>> read: if sdpa PASSES with small outDiff/gradDiff -> use it (fast, has backward).")
print(">>>       if sdpa FAILS -> eager always works (slower); npu_fa if it passes is the fast fallback.")
print(">>>       compare fwd+bwd ms to see the training-step cost of each.")
