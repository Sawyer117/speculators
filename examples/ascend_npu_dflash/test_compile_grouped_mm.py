#!/usr/bin/env python
"""GO/NO-GO for the compile path (Graft C precondition) — torch.compile + inductor_npu_ext on NPU.

Run in the CLONED env AFTER installing inductor_npu_ext:
    python examples/ascend_npu_dflash/test_compile_grouped_mm.py
    # to SEE recompile events explicitly:
    TORCH_LOGS=recompiles python examples/ascend_npu_dflash/test_compile_grouped_mm.py

Answers three questions that decide whether compile is viable for us:
  1. Does torch.compile(backend="inductor") produce a WORKING NPU kernel for the grouped-swiglu
     experts forward (torch._grouped_mm + npu_swiglu)?           -> parity vs eager.
  2. Is it SHAPE-GENERIC? maybe_mark_dynamic on the token dim should give ONE compiled kernel that
     handles ALL routed-token counts -> NO per-shape recompile (our eager path recompiles 20-60s
     per new shape; that's the 42%-of-wall-clock we want to kill).
  3. Is it fast (no recompile stall on a new token count)?

GREEN  = parity rel small (~1e-2 bf16 ok) AND only the FIRST token-count compiles; every later
         (different) count runs fast.  -> compile path viable, I write Graft B+C.
RED    = every new token count is slow (recompiles per shape) OR inductor_npu_ext codegen errors.
         -> compile not viable on this stack; stay on bucketing / rethink.
"""
from __future__ import annotations

import time

import torch
import torch_npu  # noqa: F401

# ★ engage inductor_npu_ext's AutoFuse/AscendC codegen (torchtitan-npu entry.py:92 does exactly this).
# Without this import, torch.compile falls through to torch_npu's BUILT-IN _inductor (Triton backend),
# which needs an active Ascend-Triton driver and dies with "0 active drivers" if only CUDA-triton is
# installed. Importing inductor_npu_ext registers the AscendC codegen path instead.
import inductor_npu_ext  # noqa: F401

# NB: do NOT `from torch_npu.contrib import transfer_to_npu` — torchtitan-npu does not use it, and on
# torch 2.12 + torch_npu 2.12.0rc1 its _patch_cuda() crashes (_apply_patches signature mismatch). We
# only need `import torch_npu`, which registers the "npu" device + torch.npu.* (no CUDA remap needed).


# --- NPU compile shim: torch 2.10's Triton/TMA capability probes call get_device_capability() /
# get_device_properties().major, which are None on NPU -> "'>=' NoneType vs tuple" crash DURING
# torch.compile (before it ever reaches the grouped_mm). inductor_npu_ext codegens AscendC (NOT
# Triton), so forcing benign no-Triton values is correct. torchtitan-npu also sets capture_scalar
# _outputs for the data-dependent offs path. ---
def _install_npu_compile_shim():
    def _safe_cap(_f):
        def inner(*a, **k):
            try:
                c = _f(*a, **k)
            except Exception:  # noqa: BLE001
                c = None
            return c if c is not None else (0, 0)
        return inner

    for _name in ("cuda", "npu"):
        mod = getattr(torch, _name, None)
        gc = getattr(mod, "get_device_capability", None) if mod is not None else None
        if gc is not None:
            mod.get_device_capability = _safe_cap(gc)
    torch._dynamo.config.capture_scalar_outputs = True


_install_npu_compile_shim()


# --- register aten::_grouped_mm -> npu_grouped_matmul (verbatim from torchtitan-npu ops/_grouped_mm.py) ---
@torch.library.impl("aten::_grouped_mm", "PrivateUse1")
def _grouped_mm_npu(self, mat2, offs, bias=None, out_dtype=None):
    split_along_k = self.ndim == 2 and mat2.ndim == 2
    return torch_npu.npu_grouped_matmul(
        [self], [mat2], group_list=offs.to(dtype=torch.int64),
        group_list_type=0, split_item=2,
        group_type=(2 if split_along_k else 0),
        bias=[bias] if bias is not None else None, output_dtype=out_dtype,
    )[0]


DEV = "npu"
DIM, INTER, E = 4096, 768, 256          # DSV4-ish draft MoE dims (bump if you want)
torch.manual_seed(0)

# stacked expert weights: w13 = fused [gate; up], w2 = down
w13 = (torch.randn(E, 2 * INTER, DIM, device=DEV, dtype=torch.bfloat16) * 0.02)
w2 = (torch.randn(E, DIM, INTER, device=DEV, dtype=torch.bfloat16) * 0.02)


def experts_fn(x, counts):
    """torchtitan-npu _run_experts_grouped_mm: fused w13 grouped-GEMM -> npu_swiglu -> w2 grouped-GEMM."""
    offs = torch.cumsum(counts, dim=0, dtype=torch.int32)
    h = torch._grouped_mm(x.bfloat16(), w13.transpose(-2, -1), offs=offs)   # [M, 2*INTER]
    h = torch_npu.npu_swiglu(h, dim=-1)                                     # [M, INTER]
    return torch._grouped_mm(h, w2.transpose(-2, -1), offs=offs)           # [M, DIM]


def make_inputs(m):
    x = torch.randn(m, DIM, device=DEV, dtype=torch.bfloat16)
    idx = torch.randint(0, E, (m,), device=DEV)
    counts = torch.zeros(E, dtype=torch.int64, device=DEV)
    counts.scatter_add_(0, idx, torch.ones(m, dtype=torch.int64, device=DEV))
    return x, counts


# arbitrary, NON-round token counts (what routing actually produces) -> the recompile trigger
M_LIST = [500, 1024, 1500, 2048, 3000, 777]

print("=== EAGER baseline (reference outputs) ===")
# generate inputs ONCE and reuse for eager AND compiled — else parity compares different random data.
INPUTS = {m: make_inputs(m) for m in M_LIST}
ref = {}
for m in M_LIST:
    ref[m] = experts_fn(*INPUTS[m]).float()
    print(f"  M={m:5}: out {tuple(ref[m].shape)}")

print("\n=== COMPILED (backend='inductor' + maybe_mark_dynamic on token dim) ===")
print("  (first call compiles — may take 20-60s with AutoFuse codegen; LATER calls must be FAST)")
torch.compiler.reset()
compiled = torch.compile(experts_fn, backend="inductor")

for i, m in enumerate(M_LIST):
    x, counts = INPUTS[m]                       # SAME inputs the eager reference used (real parity)
    torch._dynamo.maybe_mark_dynamic(x, 0)     # token dim symbolic -> one kernel for every M
    torch.npu.synchronize()
    t0 = time.perf_counter()
    out = compiled(x, counts).float()
    torch.npu.synchronize()
    dt = time.perf_counter() - t0
    d = (out - ref[m]).abs().max().item()
    rel = d / ref[m].abs().max().clamp_min(1e-6).item()
    if i == 0:
        tag = "(1st compile)"
    elif dt > 3.0:
        tag = "⚠ RECOMPILE"
    else:
        tag = "fast ✅"
    print(f"  M={m:5}: {dt * 1000:9.1f} ms  {tag:14}  parity max|Δ|={d:.3e} rel={rel:.3e}")

try:
    from torch._dynamo.utils import counters
    print("\ndynamo stats:", dict(counters.get("stats", {})),
          "\n  (unique_graphs / autograd_captures ~1-2 = shape-generic ✅ ; ~len(M_LIST) = recompiled per shape ❌)")
except Exception as e:  # noqa: BLE001
    print("\n(dynamo stats unavailable:", e, ")")

print("\nVERDICT: GREEN if parity rel small AND only the 1st M compiles (rest 'fast'). Then I write Graft B+C.")
print("         RED  if every new M is slow (recompiles per shape) or a codegen error above.")


# =====================================================================================================
# BACKWARD parity + SPEED + MEMORY (self-contained — no speculators import; same math as moe_compile.py,
# which is validated by construction). Training needs gradients (torch.compile compiles the backward via
# aot_autograd), so this proves the compiled BACKWARD matches eager, and quantifies the speed/mem.
# =====================================================================================================
import time  # noqa: E402


def experts_g(x, counts, w13, w2):  # weights as args -> grad-enabled (globals are grad-free constants)
    offs = torch.cumsum(counts, dim=0, dtype=torch.int32)
    h = torch._grouped_mm(x.bfloat16(), w13.bfloat16().transpose(-2, -1), offs=offs)
    h = torch_npu.npu_swiglu(h, dim=-1)
    return torch._grouped_mm(h, w2.bfloat16().transpose(-2, -1), offs=offs)


print("\n=== BACKWARD parity (compiled bwd via aot_autograd vs eager) ===")
w13g = torch.cat([w13, w13], dim=1).detach().clone().requires_grad_()   # [E, 2*INTER, DIM]
w2g = w2.detach().clone().requires_grad_()
xb, cb = INPUTS[1500]
xb = xb.detach().clone().requires_grad_()
compiled_g = torch.compile(experts_g, backend="inductor")
y_e = experts_g(xb, cb, w13g, w2g).float()
torch._dynamo.maybe_mark_dynamic(xb, 0)
y_c = compiled_g(xb, cb, w13g, w2g).float()
print(f"[fwd] rel={((y_e - y_c).abs().max() / y_e.abs().max().clamp_min(1e-6)).item():.3e}")
gg = torch.randn_like(y_e)
ge = torch.autograd.grad(y_e, [xb, w13g, w2g], gg)
gc = torch.autograd.grad(y_c, [xb, w13g, w2g], gg)
for nm, a, b in zip(["dx  ", "dw13", "dw2 "], ge, gc):
    print(f"[bwd] {nm}: rel={((a - b).abs().max() / a.abs().max().clamp_min(1e-6)).item():.3e}")


def timed(fn, iters=30, warmup=5):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


print("\n=== SPEED (steady per-call, fwd-only; M=2048) ===")
xe, ce = INPUTS[2048]
te = timed(lambda: experts_fn(xe, ce))     # eager npu_grouped_matmul (ACL kernel already warm)
tc = timed(lambda: compiled(xe, ce))       # compiled AscendC (one kernel)
print(f"eager {te:.2f} ms   compiled {tc:.2f} ms   ({te / tc:.2f}x per-call)")
print("  (the BIG win is recompile-avoidance across VARYING M above — eager recompiles 20-60s per new")
print("   shape in training; compiled = one kernel. Steady per-call is the smaller, secondary gain.)")

print("\n=== MEMORY (peak reserved, compiled fwd, M=3000) ===")
try:
    torch.npu.reset_peak_memory_stats()
    with torch.no_grad():
        compiled(*INPUTS[3000])
    torch.npu.synchronize()
    print(f"peak reserved: {torch.npu.max_memory_reserved() / 1e9:.2f} GB  (no bucket padding — compile is")
    print("  dynamic-shape, so it AVOIDS the bucketing padding memory the 2.10 path pays)")
except Exception as e:  # noqa: BLE001
    print(f"(mem stats unavailable: {e})")

print("\nGREEN if [bwd] rel small — the compiled backward matches eager, so Graft B+C is training-ready.")
