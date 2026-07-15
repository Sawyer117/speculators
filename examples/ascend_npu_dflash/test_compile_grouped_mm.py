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
ref = {}
for m in M_LIST:
    x, counts = make_inputs(m)
    ref[m] = experts_fn(x, counts).float()
    print(f"  M={m:5}: out {tuple(ref[m].shape)}")

print("\n=== COMPILED (backend='inductor' + maybe_mark_dynamic on token dim) ===")
print("  (first call compiles — may take 20-60s with AutoFuse codegen; LATER calls must be FAST)")
torch.compiler.reset()
compiled = torch.compile(experts_fn, backend="inductor")

for i, m in enumerate(M_LIST):
    x, counts = make_inputs(m)
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
