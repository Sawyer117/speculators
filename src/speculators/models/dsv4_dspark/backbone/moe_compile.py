"""Graft B+C: torch.compile'd grouped-GEMM MoE experts (opt-in ``DSPARK_COMPILE=1``) — **SEED TECH**.

Kills the per-shape grouped-GEMM recompile — measured **42% of wall-clock** once ``MAX_ANCHORS`` solved
the HS-bound (see ``docs/deployment/ascend-npu-dsv4-dspark-compile-recompile.md``) — by compiling the
experts forward ONCE with a symbolic token dim (``maybe_mark_dynamic``) → **one AscendC kernel for every
routed-token count** (no recompile). Validated bit-exact (fwd+bwd, ``unique_graphs=1``) → ~**1.74×**.

★ **DEFAULT OFF and MUST STAY OFF on the main (torch 2.10) stack.** It requires a matched stack our
training/serve env does NOT have:
    torch 2.12.0+cpu · torch_npu 2.12.0rc1 · inductor_npu_ext (gitcode Ascend/torchair @3c9418c2) · triton-ascend.
Enabling it now would desync training vs the 2.10 serve (invalidating rolled data). This is a seed
capability for a future FULL-stack (train+serve) upgrade. Everything here is lazy-imported so importing
this module on the 2.10 stack is a no-op until :func:`enable` is called.

Graft B = torchtitan-npu ``_run_experts_grouped_mm`` (``torch._grouped_mm`` + fused ``w13`` + ``npu_swiglu``).
Graft C = ``torch.compile(backend="inductor")`` + ``maybe_mark_dynamic(x, 0)`` + the NPU compile shims.
"""
from __future__ import annotations

import torch

_ENABLED = False
_COMPILED = None


def _install_compile_shims() -> None:
    """torch 2.12's Triton/TMA probes call get_device_capability()/...properties().major → None on NPU →
    '>=' TypeError during compile. inductor_npu_ext codegens AscendC (not Triton), so benign no-Triton
    values are safe. Also set capture_scalar_outputs for the data-dependent ``offs`` (torchtitan-npu does)."""
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


def _register_grouped_mm_npu() -> None:
    """aten::_grouped_mm → npu_grouped_matmul (verbatim from torchtitan-npu ops/_grouped_mm.py)."""
    import torch_npu  # noqa: PLC0415

    try:
        @torch.library.impl("aten::_grouped_mm", "PrivateUse1")
        def _(self, mat2, offs, bias=None, out_dtype=None):
            split_along_k = self.ndim == 2 and mat2.ndim == 2
            return torch_npu.npu_grouped_matmul(
                [self], [mat2], group_list=offs.to(dtype=torch.int64),
                group_list_type=0, split_item=2,
                group_type=(2 if split_along_k else 0),
                bias=[bias] if bias is not None else None, output_dtype=out_dtype,
            )[0]
    except RuntimeError:
        pass  # already registered (idempotent)


def _experts_grouped_mm(w13, w2, x, counts, swiglu_limit, scores):
    """Fused-w13 grouped-GEMM SwiGLU (torchtitan-npu ``_run_experts_grouped_mm``). This is what Graft C
    compiles. ``w13 = [gate; up]`` stacked ``[E, 2*inter, dim]``; ``w2`` ``[E, dim, inter]``;
    ``x[N, dim]`` routed tokens (sorted by expert); ``counts[E]``; ``scores[N]`` router weights."""
    import torch_npu  # noqa: PLC0415

    offs = torch.cumsum(counts, dim=0, dtype=torch.int32)
    h = torch._grouped_mm(x.bfloat16(), w13.bfloat16().transpose(-2, -1), offs=offs)   # [N, 2*inter]
    if swiglu_limit > 0:
        gate, up = h.chunk(2, -1)
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        gate = torch.clamp(gate, max=swiglu_limit)
        h = torch.cat([gate, up], dim=-1)
    h = torch_npu.npu_swiglu(h, dim=-1)                                                # [N, inter]
    h = h * scores.reshape(-1, 1).to(h.dtype)
    return torch._grouped_mm(h, w2.bfloat16().transpose(-2, -1), offs=offs).float()    # [N, dim]


def enable() -> None:
    """Install + compile the experts path. Call ONCE at startup, ONLY on the torch-2.12 stack with
    ``DSPARK_COMPILE=1``. No-op / unsafe on the 2.10 main stack."""
    global _ENABLED, _COMPILED
    if _ENABLED:
        return
    _install_compile_shims()
    _register_grouped_mm_npu()
    import inductor_npu_ext  # noqa: F401, PLC0415  (engages AscendC codegen; bypasses torch_npu's Triton inductor)

    _COMPILED = torch.compile(_experts_grouped_mm, backend="inductor")
    _ENABLED = True


def run(w1, w3, w2, x, counts, swiglu_limit, scores):
    """Compiled experts forward used by ``moe_grouped_gemm._fused_permute_dispatch_npu`` when enabled.
    Fuses ``w13 = cat([w1, w3])`` and marks the token dim dynamic (one kernel for every token count)."""
    w13 = torch.cat([w1, w3], dim=1)          # [E, 2*inter, dim]
    torch._dynamo.maybe_mark_dynamic(x, 0)
    # ★ scores[N] shares x's token dim N. If ONLY x is marked, dynamo/inductor still specializes on
    # scores' concrete shape -> a fresh per-shape recompile every time the routed-token count changes
    # (each DSPARK_MOE_BUCKET step) = the recompile spikes that survived compilation. Mark it too so
    # ALL token-dependent inputs are dynamic -> one shape-generic kernel, no recompile. (The isolated
    # test_compile_grouped_mm.py used a single token count, so it never exposed this.)
    torch._dynamo.maybe_mark_dynamic(scores, 0)
    return _COMPILED(w13, w2, x, counts, swiglu_limit, scores)
