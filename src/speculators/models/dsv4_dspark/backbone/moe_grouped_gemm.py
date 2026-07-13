"""Fused grouped-GEMM implementation of the DSV4 DSpark MoE dispatch (opt-in NPU).

Replaces the 256-way per-expert eager loop (`moe._moe_dispatch_torch`) with ONE grouped
matmul per projection (gate / up / down), sorted by expert. This is the same math but:
  * variable per-expert token counts go into a single `npu_grouped_matmul` (group_list) that
    handles them WITHOUT a per-shape kernel recompile -> kills the forward recompile spikes;
  * 256 small GEMMs collapse to 3 grouped GEMMs -> big throughput win.

Registers itself under `("moe_dispatch", "npu")` (import this module from the NPU bridge and
`set_active_backend("npu")`). CPU-runnable via a pure-torch grouped fallback, so the routing /
permute / swiglu logic unit-tests without an NPU (test_moe_grouped_gemm.py).

⚠️ Distributed note: grouped-GEMM needs each expert's FULL weight locally (it stacks them).
That conflicts with per-expert FSDP2 sharding (weights are gathered only inside each expert's
forward hook). So this is a clean win for single-card / expert-replicated / **EP** (each rank
owns whole experts) layouts; the per-expert-FSDP faithful run needs EP (Track B) or a
gather-before-group. The MATH here is layout-independent and is what the unit test pins down.
"""
from __future__ import annotations

import torch

from .kernels import register_kernel
from .moe import GroupedExperts, swiglu_grouped

_MOE_OP = "moe_dispatch"


def _grouped_matmul_torch(x: torch.Tensor, weight: torch.Tensor,
                          counts: torch.Tensor) -> torch.Tensor:
    """Reference grouped matmul: x[T, K] blocks (by `counts`) @ weight[E, K, M] -> [T, M].

    Autograd-native (split + matmul + cat); used on CPU and as the parity oracle.
    """
    outs = []
    off = 0
    for e in range(weight.shape[0]):
        n = int(counts[e])
        outs.append(x[off:off + n] @ weight[e])
        off += n
    return torch.cat(outs, dim=0) if outs else x.new_zeros((0, weight.shape[-1]))


class _NpuGroupedMatmul(torch.autograd.Function):
    """torch_npu.npu_grouped_matmul with an explicit backward (mirrors MindSpeed GMMFunction).

    x[T, K], weight[E, K, M], counts[E] (raw per-group token counts) -> [T, M].
    """

    @staticmethod
    def forward(ctx, x, weight, counts):
        import torch_npu
        group_list = torch.cumsum(counts, dim=0).to(torch.int64)
        out = torch_npu.npu_grouped_matmul(
            [x], [weight], bias=None, group_list=group_list,
            split_item=3, group_type=0, group_list_type=0,
        )[0]
        ctx.save_for_backward(x, weight, group_list)
        return out

    @staticmethod
    def backward(ctx, grad):
        import torch_npu
        x, weight, group_list = ctx.saved_tensors
        # dx = grad[T, M] @ weight[E, K, M]^T  (grouped) ; dw = x^T @ grad (grouped)
        dx = torch_npu.npu_grouped_matmul(
            [grad], [weight.transpose(1, 2)], bias=None, group_list=group_list,
            split_item=3, group_type=0, group_list_type=0,
        )[0]
        dw = torch_npu.npu_grouped_matmul(
            [x.transpose(0, 1)], [grad], bias=None, group_list=group_list,
            split_item=2, group_type=2, group_list_type=0,
        )[0]
        return dx, dw.reshape(weight.shape), None


def _grouped_matmul(x, weight, counts):
    if x.device.type == "npu":
        return _NpuGroupedMatmul.apply(x, weight, counts)
    return _grouped_matmul_torch(x, weight, counts)


def moe_dispatch_grouped(x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor,
                         experts: GroupedExperts, n_routed_experts: int) -> torch.Tensor:
    """Grouped-GEMM equivalent of moe._moe_dispatch_torch. Returns y[T, dim] (fp32).

    ``experts`` holds stacked weights (``w1/w3 [E, inter, dim]``, ``w2 [E, dim, inter]``);
    no per-forward restacking. Reads ``.to_local()`` when they are Shard(0) DTensors.
    """
    T, dim = x.shape
    device = x.device
    w1, w3, w2 = experts.local_weights()

    # flatten routing: each token -> topk (token, expert, weight); sort by expert
    topk = indices.shape[1]
    tok_ids = torch.arange(T, device=device).repeat_interleave(topk)      # [T*topk]
    exp_ids = indices.reshape(-1)                                          # [T*topk]
    w_flat = weights.reshape(-1).float()                                   # [T*topk]
    order = torch.argsort(exp_ids, stable=True)
    tok_ids, exp_ids, w_flat = tok_ids[order], exp_ids[order], w_flat[order]
    counts = torch.bincount(exp_ids, minlength=n_routed_experts)           # [E]

    xg = x[tok_ids].float()                                                # [T*topk, dim]
    out = swiglu_grouped(xg, w1.float(), w3.float(), w2.float(), counts, w_flat,
                         experts.swiglu_limit, _grouped_matmul)            # [T*topk, dim]

    y = torch.zeros(T, dim, dtype=torch.float32, device=device)
    y.index_add_(0, tok_ids, out)                                         # sum a token's topk contributions
    return y


register_kernel(_MOE_OP, "npu", moe_dispatch_grouped)


def enable() -> None:
    """Activate the fused grouped-GEMM MoE process-wide (call once at NPU training startup).

    Importing this module already registers the impl under ("moe_dispatch", "npu"); this just
    flips the active backend to "npu" (ops without an npu impl fall back to torch, so it's
    safe to switch globally). Usage in the trainer/model init on NPU:
        from speculators.models.dsv4_dspark.backbone import moe_grouped_gemm
        moe_grouped_gemm.enable()
    """
    from .kernels import set_active_backend
    set_active_backend("npu")
