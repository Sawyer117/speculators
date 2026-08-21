#!/usr/bin/env python3
"""Is `torch._grouped_mm` usable on Ascend, and how much does the fused kernel buy?

    python3 examples/ascend_npu_dflash/grouped_mm_parity_bench.py          # one card, ~2 min

WHY THIS EXISTS
---------------
Our MoE backbone calls ``torch_npu.npu_grouped_matmul`` and wraps it in a hand-written
``torch.autograd.Function`` with a hand-derived backward. None of that is necessary:

* ``aten::_grouped_mm`` exists in torch 2.12 with CUDA, Meta, Autograd and a
  CompositeExplicitAutograd fallback, so it already RUNS on any backend -- just unfused.
* Its backward (``GroupedMmBackward0``) is written in ``derivatives.yaml`` IN TERMS OF OTHER
  ATEN OPS. A traced backward is ``transpose`` + ``_grouped_mm`` twice, computing exactly the
  ``dx = grad @ w^T`` and ``dw = x^T @ grad`` we hand-wrote. So registering only the FORWARD
  for the NPU dispatch key accelerates the backward too, and the model needs no
  ``autograd.Function`` at all -- the autograd key sits ABOVE the backend key in the dispatch
  chain, so graph construction is already handled device-agnostically.
* ``Ascend/op-plugin`` master registers ``_scaled_grouped_mm`` (quantised, MoE inference) but
  not ``_grouped_mm`` (bf16, MoE training). Both existing adapters call the identical
  ``aclnnGroupedMatmul/V4/V5/WeightNz`` set, so no new kernel is needed -- only a third
  signature adapter.

This script produces the two things needed to act on that: proof the adapter is numerically
correct, and the speed ratio that is the entire argument for asking op-plugin to absorb it.

★ THE dw CASE IS THE ONE THAT BITES. A MoE grouped matmul has three shapes, and they do NOT
all reduce along the same axis:

    forward  x[T,K] @ w[E,K,M] -> [T,M]      grouped along experts      group_type=0
    dx    grad[T,M] @ w^T[E,M,K] -> [T,K]    grouped along experts      group_type=0
    dw     x^T[K,T] @ grad[T,M] -> [E,K,M]   reduced along T (tokens)   group_type=2  <-- !

Hardcoding group_type=0 computes dw silently WRONG. The adapter below is transcribed from
torchtitan-npu (Huawei's own, ``torchtitan_npu/ops/_grouped_mm.py``), which carries that
distinction; the end-to-end autograd check here is what proves it, since dw only appears in
the backward.

WHAT IT PRINTS
    [1] parity   fwd / dx / dw, adapter vs direct vendor op vs pure-torch oracle
    [2] autograd end-to-end: grads through torch._grouped_mm vs through the oracle
    [3] ★ speed: generic fallback vs adapter vs direct vendor call -> the N for the issue

Safe: read-only, allocates a few hundred MB, touches no checkpoint, needs ONE card.
Runs on CPU/CUDA too (skips the NPU-specific rows) so the logic can be checked anywhere.
"""

from __future__ import annotations

import os
import sys
import time

import torch

# Our real training shapes: hidden 4096, moe_inter 2048, MAX_ANCHORS 512 x block 5 = 2560
# draft tokens, 256 routed experts over EP8 = 32 local. Token counts per expert are uneven in
# reality, so the benchmark uses a skewed split rather than a uniform one.
T_TOKENS = int(os.environ.get("BENCH_TOKENS", "2560"))
K_DIM = int(os.environ.get("BENCH_K", "4096"))
M_DIM = int(os.environ.get("BENCH_M", "2048"))
E_LOCAL = int(os.environ.get("BENCH_EXPERTS", "32"))
ITERS = int(os.environ.get("BENCH_ITERS", "20"))
DTYPE = torch.bfloat16


def _dev() -> torch.device:
    acc = torch.accelerator.current_accelerator()
    return torch.device(acc.type, 0) if acc is not None else torch.device("cpu")


def _is_npu(dev: torch.device) -> bool:
    return dev.type == "npu"


def _skewed_counts(total: int, groups: int, device) -> torch.Tensor:
    """Uneven token counts, as a real router produces. Sums to `total` exactly."""
    g = torch.arange(1, groups + 1, dtype=torch.float64)
    w = (g % 5 + 1) / (g % 5 + 1).sum()
    c = (w * total).floor().to(torch.int64)
    c[-1] += total - int(c.sum())
    return c.to(device)


def _oracle(x: torch.Tensor, w: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Pure-torch grouped matmul. Always available, and the thing everything is judged against."""
    outs, off = [], 0
    for e in range(w.shape[0]):
        n = int(counts[e])
        outs.append(x[off:off + n].float() @ w[e].float())
        off += n
    return torch.cat(outs, dim=0) if outs else x.new_zeros((0, w.shape[-1]))


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    d = (a.float() - b.float()).abs().max()
    s = b.float().abs().max().clamp_min(1e-9)
    return float(d / s)


def _sync() -> None:
    """Device-agnostic barrier; a no-op (and not an error) on CPU, where there is nothing async."""
    try:
        if torch.accelerator.current_accelerator() is not None:
            torch.accelerator.synchronize()
    except Exception:  # noqa: BLE001 - timing must not fail on an exotic backend
        pass


def _timed(fn, warmup: int = 3, iters: int = ITERS) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def register_npu_adapter() -> bool:
    """Register torch_npu.npu_grouped_matmul as aten::_grouped_mm for the NPU dispatch key.

    Transcribed from torchtitan-npu ``torchtitan_npu/ops/_grouped_mm.py``
    (Copyright (c) 2026 Huawei Technologies, BSD-style) -- reproduced rather than invented so
    the group_type distinction comes from the people who own the kernel.
    """
    try:
        import torch_npu  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        print(f"    (torch_npu unavailable: {exc}); skipping adapter")
        return False

    @torch.library.impl("aten::_grouped_mm", "PrivateUse1")
    def _(self, mat2, offs, bias=None, out_dtype=None):  # noqa: ANN001
        # dw reduces along the token axis, not the expert axis -> group_type=2. Both operands
        # being 2-D is what identifies that case.
        split_along_k = self.ndim == 2 and mat2.ndim == 2
        return torch_npu.npu_grouped_matmul(
            [self], [mat2],
            group_list=offs.to(dtype=torch.int64), group_list_type=0,
            split_item=2, group_type=(2 if split_along_k else 0),
            bias=[bias] if bias is not None else None, output_dtype=out_dtype,
        )[0]

    return True


def _direct_vendor(x, w, counts):
    """What our code does TODAY: the vendor op with a cumulative group list."""
    import torch_npu  # noqa: PLC0415

    return torch_npu.npu_grouped_matmul(
        [x], [w], bias=None, group_list=torch.cumsum(counts, 0).to(torch.int64),
        split_item=3, group_type=0, group_list_type=0,
    )[0]


def main() -> None:  # noqa: PLR0915
    dev = _dev()
    print("=" * 78)
    print(f" grouped_mm parity + bench   device={dev}  torch={torch.__version__}")
    print(f" shapes: x[{T_TOKENS},{K_DIM}] @ w[{E_LOCAL},{K_DIM},{M_DIM}]  {DTYPE}")
    print("=" * 78)

    counts = _skewed_counts(T_TOKENS, E_LOCAL, dev)
    offs = torch.cumsum(counts, 0).to(torch.int32)
    x = torch.randn(T_TOKENS, K_DIM, dtype=DTYPE, device=dev)
    w = torch.randn(E_LOCAL, K_DIM, M_DIM, dtype=DTYPE, device=dev)
    print(f" token split per expert: min={int(counts.min())} max={int(counts.max())} "
          f"(skewed on purpose -- a real router is not uniform)")

    # ---- [3a] generic fallback FIRST: registration is global and cannot be undone -------
    # On NPU this is the CompositeExplicitAutograd decomposition; on CUDA it is already the
    # native kernel, so the ratio at the end is only meaningful on NPU. Labelled accordingly
    # rather than printing a "1.03x speedup" that means nothing.
    pre_label = "generic fallback (未注册)" if _is_npu(dev) else f"native {dev.type} kernel"
    print(f"\n[3a] BEFORE registering -- torch._grouped_mm ({pre_label})")
    fallback_ms = None
    try:
        out_fb = torch._grouped_mm(x, w, offs=offs)
        fallback_ms = _timed(lambda: torch._grouped_mm(x, w, offs=offs))
        print(f"     ✓ runs: {tuple(out_fb.shape)}   {fallback_ms:8.3f} ms")
    except Exception as exc:  # noqa: BLE001
        print(f"     ✗ {type(exc).__name__}: {str(exc)[:150]}")

    # ---- register ------------------------------------------------------------------
    print("\n[0] registering the NPU adapter (aten::_grouped_mm -> npu_grouped_matmul)")
    have_npu = _is_npu(dev) and register_npu_adapter()
    print(f"    adapter active: {have_npu}")

    # ---- [1] parity on the three shapes ---------------------------------------------
    print("\n[1] PARITY vs the pure-torch oracle (bf16; ~1e-2 relative is normal)")
    ref = _oracle(x, w, counts)
    try:
        got = torch._grouped_mm(x, w, offs=offs)
        print(f"    fwd  torch._grouped_mm      rel={_rel(got, ref):.2e}")
    except Exception as exc:  # noqa: BLE001
        print(f"    fwd  torch._grouped_mm      ✗ {str(exc)[:110]}")
    if have_npu:
        try:
            print(f"    fwd  vendor op (today)      rel={_rel(_direct_vendor(x, w, counts), ref):.2e}")
        except Exception as exc:  # noqa: BLE001
            print(f"    fwd  vendor op (today)      ✗ {str(exc)[:110]}")

    # ---- [2] autograd end to end: this is what exercises the dw path ----------------
    print("\n[2] AUTOGRAD end-to-end  ★ the only check that exercises dw (group_type=2)")
    xg = x.clone().requires_grad_(True)
    wg = w.clone().requires_grad_(True)
    xr = x.clone().requires_grad_(True)
    wr = w.clone().requires_grad_(True)
    try:
        g = torch.randn(T_TOKENS, M_DIM, dtype=DTYPE, device=dev)
        out = torch._grouped_mm(xg, wg, offs=offs)
        print(f"    grad_fn = {type(out.grad_fn).__name__}  (no autograd.Function of ours)")
        dx, dw = torch.autograd.grad(out, [xg, wg], grad_outputs=g)
        rout = _oracle(xr, wr, counts)
        rdx, rdw = torch.autograd.grad(rout, [xr, wr], grad_outputs=g.float())
        print(f"    dx  rel={_rel(dx, rdx):.2e}")
        print(f"    dw  rel={_rel(dw, rdw):.2e}   ← 若这一行明显差于 dx,就是 group_type 用错了")
    except Exception as exc:  # noqa: BLE001
        print(f"    ✗ {type(exc).__name__}: {str(exc)[:170]}")

    # ---- [3b] the number the issue needs --------------------------------------------
    print("\n[3b] SPEED  ★ this ratio is the whole argument for op-plugin absorbing it")
    rows = []
    try:
        rows.append(("torch._grouped_mm (adapter)" if have_npu else "torch._grouped_mm",
                     _timed(lambda: torch._grouped_mm(x, w, offs=offs))))
    except Exception as exc:  # noqa: BLE001
        print(f"     adapter path failed: {str(exc)[:110]}")
    if have_npu:
        try:
            rows.append(("vendor op直调 (today)", _timed(lambda: _direct_vendor(x, w, counts))))
        except Exception as exc:  # noqa: BLE001
            print(f"     vendor path failed: {str(exc)[:110]}")
    if fallback_ms is not None:
        rows.insert(0, (pre_label, fallback_ms))
    for name, ms in rows:
        print(f"     {name:<32} {ms:8.3f} ms")
    if fallback_ms is not None and len(rows) > 1 and _is_npu(dev):
        best = min(ms for n, ms in rows if "fallback" not in n)
        print(f"\n     ★ 融合 vs 通用回落 = {fallback_ms / best:.2f}×   ← 提 issue 用这个数")
    elif not _is_npu(dev):
        print(f"\n     (非 NPU:上面两行走的是同一条 {dev.type} 原生 kernel,比值无意义)")
    print("\n" + "=" * 78)


if __name__ == "__main__":
    sys.exit(main())
