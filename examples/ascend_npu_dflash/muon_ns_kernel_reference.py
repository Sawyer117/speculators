"""Newton-Schulz orthogonalization: reference, oracle and benchmark for a fused rewrite.

WHY THIS FILE EXISTS. This iteration is the single hot spot of ``DistributedMuon`` on
the DSV4-DSpark draft. Measured on 8x Ascend 910B1 (faithful 3L x 256E, EP8, bf16
experts):

    opt_ms   992     of a step_ms of 2940   -> 34% of every training step
    AdamW's  99                             -> Muon costs 10x the optimizer time

and the arithmetic says that 992 ms is FLOP-bound, not a code smell:

    per expert stack x = [E=32, 2048, 4096] bf16, per iteration:
      bmm(x, x^T)       2*32*2048*2048*4096 = 1.10e12 FLOPs
      bmm(gram, gram)   2*32*2048^3         = 0.55e12
      bmm(poly, x)      2*32*2048*2048*4096 = 1.10e12
                                              --------
                                              2.75e12
    x 5 iterations x 9 stacks (3 layers x w1/w2/w3) = 1.24e14 FLOPs / step / rank
    1.24e14 / 0.992 s = 125 TFLOPS ~= 33% of the 910B1's ~376 TFLOPS bf16 peak.

So this is not a latency or launch-overhead problem. It is a dense-GEMM problem running
at a third of peak, with structure that ``torch.bmm`` cannot exploit.

WHERE THE WINS ARE (in the order I would try them):

1. ``gram = x @ x^T`` IS SYMMETRIC, and so is ``gram @ gram``. A general bmm computes
   all n^2 output entries; only the triangle is needed. That is a **1.43x FLOP
   reduction on the whole iteration** (2.75e12 -> 1.93e12), pure arithmetic that no
   amount of tiling recovers, because torch has no batched SYRK:

       bmm(x, x^T)      1.10e12 -> 0.55e12   (SYRK)
       bmm(gram, gram)  0.55e12 -> 0.275e12  (symmetric squared)
       bmm(poly, x)     1.10e12 -> 1.10e12   (SYMM: same FLOPs, half the reads)

2. EPILOGUE FUSION. Per iteration per stack the current code moves roughly 4.8 GB:
   gram is 268 MB in bf16, x is 537 MB, and ``poly = b*gram + c*gram2`` plus
   ``x = a*x + (poly@x)`` are two extra full round trips over them. Over 5 iterations
   and 9 stacks that is ~216 GB/step, ~135 ms at 1.6 TB/s -- about 14% of the 992 ms,
   nearly all of which folds into the two GEMM epilogues.

3. ACCUMULATE IN FP32, STORE BF16. The reference already runs bf16 in and bf16 out;
   the accumulator dtype is the kernel's choice, and fp32 accumulation is what keeps
   the 5-step schedule where the coefficients expect it.

NOT a win, already checked: reassociating ``poly @ x`` as
``b*(gram@x) + c*(gram@(gram@x))`` costs 3.30e12 instead of 2.75e12, because n=2048 is
smaller than m=4096, so forming gram^2 is cheaper than applying gram to x twice. The
current form is already FLOP-optimal.

WHAT A REPLACEMENT MUST PRESERVE (the coefficients are load-bearing):

* ``COEFF_PRIMARY`` deliberately does NOT converge to 1. As a map on singular values,
  p(x) = a x + b x^3 + c x^5 has p(1) = 0.7010, so 1 is not a fixed point -- by design.
  Do not "fix" that, and do not reorder the polynomial.
* The tall/wide transpose is what keeps the gram matrix at ``min(m, n)`` on a side.
  Every expert stack lands at [32, 2048, 4096] after it; w2 is stored [32, 4096, 2048]
  and is transposed in. The 37 matrix-route parameters pass through here too but are
  small after the transpose (markov_w1/w2 are [129280, 256], so their gram is 256x256).
  **The expert stacks are the whole cost -- optimize for [E, 2048, 4096].**
* Output must be the orthogonalized tensor in the INPUT dtype and layout.

CORRECTNESS BAR. ``check()`` compares a candidate against this reference. The meaningful
metric is not elementwise closeness but distance from orthogonality, ``mean|sigma - 1|``
of the result, because that is what the iteration exists to minimise and what the
training consumes. A kernel may differ from the reference in the last bf16 bits and be
correct; one that matches elementwise while moving that metric has changed the run.

Run:  python muon_ns_kernel_reference.py            # correctness + bench, small shapes
      python muon_ns_kernel_reference.py --full     # the real [32, 2048, 4096] x 9
"""

# A standalone CLI reference for a kernel rewrite, not library code: printing (T201),
# asserting (S101) and comparing against literal tensor ranks (PLR2004) are the point.
# ruff: noqa: T201, S101, PLR2004
from __future__ import annotations

import argparse
import time

import torch

# Quintic Newton-Schulz coefficients, chosen to maximize the slope at zero.
COEFF_PRIMARY = (3.4445, -4.7750, 2.0315)
# DeepSeek-V4's hybrid schedule: the last two steps swap to a gentler polynomial that
# settles the singular values at 1 instead of overshooting them.
COEFF_SECONDARY = (2.0, -1.5, 0.5)

# The nine expert stacks are the entire cost. E is the per-rank expert count
# (256 experts / EP8), inter=2048, dim=4096; w2 arrives transposed and lands here too.
PRODUCTION_STACK = (32, 2048, 4096)
PRODUCTION_CALLS_PER_STEP = 9  # 3 layers x {w1, w2, w3}


def newton_schulz(
    grad: torch.Tensor, steps: int = 5, eps: float = 1e-7, hybrid_ns: bool = False
) -> torch.Tensor:
    """THE REFERENCE. Verbatim from ``speculators/train/muon_distributed.py``."""
    if grad.ndim not in (2, 3):
        raise ValueError(
            f"Newton-Schulz expects a 2D or 3D tensor, got {tuple(grad.shape)}"
        )
    is_2d = grad.ndim == 2
    x = grad.unsqueeze(0) if is_2d else grad

    original_dtype = x.dtype
    x = x.bfloat16()

    # The iteration wants wide matrices; transpose tall ones and undo it at the end.
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.transpose(1, 2)

    x = x / (torch.linalg.norm(x, dim=(-2, -1), keepdim=True) + eps)
    a, b, c = COEFF_PRIMARY
    for i in range(steps):
        if hybrid_ns and i >= steps - 2:
            a, b, c = COEFF_SECONDARY
        gram = torch.bmm(x, x.transpose(1, 2))              # SYMMETRIC -> SYRK
        poly = b * gram + c * torch.bmm(gram, gram)     # SYMMETRIC -> fuse the axpby
        x = a * x + torch.bmm(poly, x)                      # SYMM -> fuse the axpby
    if transposed:
        x = x.transpose(1, 2)
    if is_2d:
        x = x.squeeze(0)
    return x.to(original_dtype)


def verify_structure(shape: tuple[int, int, int] = (4, 512, 1024)) -> None:
    """Check the symmetry claim lead #1 rests on, rather than asserting it in prose.

    ``gram = x @ x^T`` is symmetric by construction, and the square of a symmetric
    matrix is symmetric too -- so both of the first two bmms compute a full n^2 output
    where only the triangle carries information. Printed as a ratio against the tensor's
    own scale so bf16 rounding is visible for what it is.
    """
    dev = _device()
    x = torch.randn(*shape, device=dev, dtype=torch.bfloat16)
    gram = torch.bmm(x, x.transpose(1, 2))
    gram2 = torch.bmm(gram, gram)
    for name, t in (("x @ x^T", gram), ("gram @ gram", gram2)):
        asym = (t - t.transpose(1, 2)).abs().max().item()
        scale = t.abs().max().item()
        ratio = asym / (scale + 1e-30)
        verdict = "SYMMETRIC" if ratio < 1e-2 else "NOT symmetric"
        print(f"  {name:<12} max|A - A^T| / max|A| = {ratio:.2e}   -> {verdict}")


def orthogonality_error(y: torch.Tensor) -> float:
    """mean|sigma - 1| of the result -- what the iteration exists to minimise.

    This, not elementwise agreement, is the bar a replacement has to clear: the training
    consumes the singular-value spectrum, so a kernel may differ from the reference in
    the
    last bf16 bits and still be correct, and may agree elementwise while having changed
    the spectrum and therefore the run.
    """
    z = y.float()
    if z.ndim == 2:
        z = z.unsqueeze(0)
    if z.shape[-2] > z.shape[-1]:
        z = z.transpose(1, 2)
    sigma = torch.linalg.svdvals(z)
    return (sigma - 1.0).abs().mean().item()


def flops_per_call(shape: tuple[int, int, int], steps: int) -> int:
    """Dense-bmm FLOPs for one call, matching the accounting in the module docstring."""
    e, m, n = shape
    if m > n:
        m, n = n, m
    return steps * 2 * e * (m * m * n + m * m * m + m * m * n)


def _device() -> torch.device:
    has_acc = hasattr(torch, "accelerator")
    acc = torch.accelerator.current_accelerator() if has_acc else None
    if acc is not None and getattr(torch, acc.type, None) is not None:
        try:
            if getattr(torch, acc.type).is_available():
                return torch.device(acc.type)
        except Exception:  # noqa: BLE001, S110 - probing for a device, never fatal
            pass
    return torch.device("cpu")


def _sync(dev: torch.device) -> None:
    mod = getattr(torch, dev.type, None)
    if mod is not None and hasattr(mod, "synchronize"):
        mod.synchronize()


def check(
    impl, shape: tuple[int, int, int], steps: int = 5, hybrid: bool = False
) -> None:
    """Compare a candidate against the reference on both bars."""
    dev = _device()
    g = torch.randn(*shape, device=dev, dtype=torch.bfloat16)
    ref = newton_schulz(g, steps, hybrid_ns=hybrid)
    got = impl(g, steps, hybrid_ns=hybrid)
    assert got.shape == ref.shape, f"shape {tuple(got.shape)} != {tuple(ref.shape)}"
    assert got.dtype == ref.dtype, f"dtype {got.dtype} != {ref.dtype}"
    rel = ((got.float() - ref.float()).norm() / (ref.float().norm() + 1e-12)).item()
    e_ref, e_got = orthogonality_error(ref), orthogonality_error(got)
    print(f"  rel_fro_diff       {rel:.3e}")
    print(f"  mean|sigma-1| ref  {e_ref:.4f}")
    print(f"  mean|sigma-1| impl {e_got:.4f}   <- THIS is the bar; must not get worse")


def bench(impl, shape: tuple[int, int, int], steps: int = 5, iters: int = 10) -> float:
    dev = _device()
    g = torch.randn(*shape, device=dev, dtype=torch.bfloat16)
    for _ in range(3):
        impl(g, steps)
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        impl(g, steps)
    _sync(dev)
    ms = (time.perf_counter() - t0) * 1000 / iters
    tf = flops_per_call(shape, steps) / (ms / 1000) / 1e12
    per_step = ms * PRODUCTION_CALLS_PER_STEP
    print(
        f"  {ms:8.2f} ms/call   {tf:7.1f} TFLOPS   "
        f"({per_step:.0f} ms/step at {PRODUCTION_CALLS_PER_STEP} calls)"
    )
    return ms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true",
                    help=f"use the production stack {PRODUCTION_STACK} (needs ~2 GB)")
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args()

    shape = PRODUCTION_STACK if args.full else (4, 512, 1024)
    dev = _device()
    print(f"device={dev}  shape={shape}  steps={args.steps}")
    print(f"dense-bmm FLOPs/call = {flops_per_call(shape, args.steps):.3e}")
    print("\nstructure the kernel can exploit:")
    verify_structure(shape)
    print("\nreference vs itself (sanity):")
    check(newton_schulz, shape, args.steps)
    print("\nreference throughput:")
    bench(newton_schulz, shape, args.steps)
    if not args.full:
        print(f"\n(reduced shape; rerun with --full for the real {PRODUCTION_STACK})")


if __name__ == "__main__":
    main()
