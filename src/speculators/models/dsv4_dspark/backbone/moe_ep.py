"""Expert-parallel (EP) MoE dispatch for the DSV4 DSpark draft backbone (opt-in NPU).

Where per-expert FSDP shards every expert across all ranks, EP instead gives each
rank a DISJOINT slice of WHOLE experts (256 experts / EP=8 -> 32 whole experts per
rank). Tokens are routed to their expert's owner rank via all-to-all, the owner runs
the fused grouped-GEMM over its LOCAL experts (weights are whole & local -> grouped
works directly, no per-expert all-gather), then results are all-to-all'd back and the
top-k contributions are summed. This is what lets the faithful 256-expert draft use
the grouped-GEMM throughput win (see ``moe_grouped_gemm``) that per-expert FSDP blocks.

Dropless (no capacity / no token drop): every routed (token, slot) is delivered.
EP=8 on a single node -> intra-node HCCL all-to-all (the cross-node EP deadlock is a
shm_broadcast issue, not triggered here).

Design mined from MindSpeed's ``legacy_a2a_token_dispatcher`` + ``comm_utils`` (the
input/output-splits handshake) but written standalone -- NO megatron ``parallel_state``
dependency; the EP group is a plain ``dist.new_group``. The token permute is a plain
``argsort`` (correct on CPU+NPU, gloo-testable); swapping in the fused
``npu_moe_init_routing`` is a v2 perf follow-up (also fixes the int64-argsort AiCPU
fallback).

⚠️ Gradients cross the dispatch: the token all-to-all is wrapped in an autograd
Function (``_AllToAll``) whose backward is the reverse all-to-all, so grads flow back
to the router / upstream. The router weight ``w`` rides the same tensor (extra column)
so its gradient flows too; only the integer local-expert-id ride is non-differentiable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .kernels import register_kernel
from .moe import GroupedExperts, swiglu_grouped
from .moe_grouped_gemm import _fused_permute_dispatch_npu, _grouped_matmul

_MOE_OP = "moe_dispatch"

# DSPARK_EP_CHECK=1: validate the counts handshake below before the payload rides on it.
_EP_CHECK = os.environ.get("DSPARK_EP_CHECK") == "1"


def _verify_splits(input_splits, output_splits, group, size, rank, where) -> None:
    """Prove the ranks agree on the SIZE of the token all-to-all, before it is issued.

    WHY THIS EXISTS. Every collective in a step is issued by every rank, in the same
    order -- the sequence counter in the trace says so (all eight ranks sat at
    ``seq=179`` through the whole optimizer). The counts match. Yet ranks {0,1,2,3}
    then block forever in a plain device ``synchronize()`` while {4,5,6,7} drain in
    1.3 s and run on. A device sync waits on work already queued, so one of those 179
    collectives completed on half the mesh and never completed on the other half.

    A collective can do that ONLY if the ranks disagree on how many bytes are moving:
    the ranks that were sent everything they expected complete, the ranks still short
    of their expected receive block forever. Count agreement cannot catch it.

    This model has exactly one variable-size collective -- the EP token all-to-all --
    and its sizes come from the handshake above. That handshake is
    ``all_to_all_single`` on a ``torch.bincount`` result, i.e. **int64 over HCCL**,
    which is the same datatype class that already handed this project uninitialised
    memory out of an int64 all-gather.

    So: gather the whole [size, size] table and check it. Two assertions, in order:

      1. **my own row must round-trip.** I know exactly what I put in, so if the copy
         that comes back through the collective differs, the CHECK channel is itself
         corrupt and its verdict on anything else is worthless. Reported separately.
      2. **``output_splits`` must equal column ``rank`` of the table** -- what every
         other rank says it is sending me. This is the actual claim under test.

    The check channel is int32 on purpose: a check must not ride the datatype it is
    checking. One 256-byte all-gather per dispatch, opt-in, off by default.
    """
    mine = input_splits.to(torch.int32).contiguous()
    table = torch.empty(size * size, dtype=torch.int32, device=mine.device)
    dist.all_gather_into_tensor(table, mine, group=group)
    table = table.view(size, size)  # table[r] == rank r's input_splits

    sent, echoed = input_splits.tolist(), table[rank].tolist()
    if sent != echoed:
        raise RuntimeError(
            f"[EP-CHECK {where}] rank {rank}: the check channel corrupted its own "
            f"row -- put in {sent}, got back {echoed}. int32 all-gather is not "
            "trustworthy on "
            "this stack, so nothing below can be concluded; fix the transport first."
        )

    expected, actual = table[:, rank].tolist(), output_splits.tolist()
    if expected != actual:
        raise RuntimeError(
            f"[EP-CHECK {where}] rank {rank}: counts handshake disagrees. The other "
            f"ranks say they are sending me {expected}, the int64 all-to-all "
            f"handshake returned {actual}. The token all-to-all would then post the "
            "wrong receive size and "
            "hang on whichever ranks come up short -- which is the observed failure. "
            f"My own sends: {sent}."
        )


@dataclass
class _EPCtx:
    group: object  # the EP process group (dist.new_group), or None for degenerate
    rank: int
    size: int
    experts_per_rank: int  # = n_routed_experts // size (== len(local experts))


_EP: _EPCtx | None = None  # process-wide EP context, set by configure()


def configure(group, rank: int, size: int, experts_per_rank: int) -> None:
    """Install the process-wide EP context (call once at NPU/EP training startup)."""
    global _EP
    _EP = _EPCtx(group=group, rank=rank, size=size, experts_per_rank=experts_per_rank)


class _AllToAll(torch.autograd.Function):
    """Variable-split all-to-all with an autograd backward (reverse all-to-all).

    forward:  send ``in_splits`` rows to each rank, receive ``out_splits`` -> [sum(out), *].
    backward: the transpose -- send grad with ``out_splits``, receive ``in_splits``.
    """

    @staticmethod
    def forward(ctx, x, out_splits, in_splits, group):
        ctx.out_splits, ctx.in_splits, ctx.group = out_splits, in_splits, group
        y = x.new_empty([sum(out_splits), *x.shape[1:]])
        dist.all_to_all_single(y, x.contiguous(), out_splits, in_splits, group=group)
        return y

    @staticmethod
    def backward(ctx, g):
        gx = g.new_empty([sum(ctx.in_splits), *g.shape[1:]])
        dist.all_to_all_single(gx, g.contiguous(), ctx.in_splits, ctx.out_splits, group=ctx.group)
        return gx, None, None, None


def _a2a_ints(x: torch.Tensor, out_splits, in_splits, group) -> torch.Tensor:
    """Non-differentiable variable-split all-to-all for integer side-channels (expert ids)."""
    y = x.new_empty([sum(out_splits), *x.shape[1:]])
    dist.all_to_all_single(y, x.contiguous(), out_splits, in_splits, group=group)
    return y


def _local_grouped_ffn(x: torch.Tensor, local_eid: torch.Tensor, w: torch.Tensor,
                       experts: GroupedExperts, n_local: int) -> torch.Tensor:
    """Grouped-GEMM SwiGLU over the LOCAL experts, per-unit output (no token combine).

    ``x[N, dim]`` units each tagged with ``local_eid[N]`` (0..n_local-1) and router weight
    ``w[N]``. Sorts by local expert, runs the grouped SwiGLU, returns outputs re-ordered
    back to the input order. Same math as ``moe_grouped_gemm.moe_dispatch_grouped`` minus
    the flatten/scatter (which EP does around the all-to-all). ``experts`` holds stacked
    weights (``.to_local()`` when Shard(0) DTensors).
    """
    if x.device.type == "npu":
        # Fused Ascend routing: units are already flattened (k=1), so permute/unpermute over
        # local_eid replaces the int64 argsort. Returns [N, dim] in input order (== out[inv]).
        return _fused_permute_dispatch_npu(x, local_eid.reshape(-1, 1), w, experts, n_local)

    w1, w3, w2 = experts.local_weights()
    order = torch.argsort(local_eid, stable=True)
    inv = torch.argsort(order, stable=True)
    xs, ws = x[order].float(), w[order].float()
    counts = torch.bincount(local_eid, minlength=n_local)
    out = swiglu_grouped(xs, w1.float(), w3.float(), w2.float(), counts, ws,
                         experts.swiglu_limit, _grouped_matmul)            # [N, dim]
    return out[inv]                                                        # back to input order


def _flatten_route(x, weights, indices):
    """(token, slot) -> flat (tok_ids, global expert ids, router weights, per-unit features)."""
    T = x.shape[0]
    topk = indices.shape[1]
    tok = torch.arange(T, device=x.device).repeat_interleave(topk)  # [T*topk]
    eid = indices.reshape(-1)                                       # global expert id
    w = weights.reshape(-1).float()
    return tok, eid, w


def moe_dispatch_ep(x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor,
                    experts: GroupedExperts, n_routed_experts: int) -> torch.Tensor:
    """EP grouped-GEMM MoE dispatch. ``experts`` is the LOCAL slice; returns y[T, dim] fp32."""
    ep = _EP
    T, dim = x.shape
    device = x.device
    tok, eid, w = _flatten_route(x, weights, indices)

    # ---- degenerate: no EP (single rank / unconfigured) == grouped over all local experts ----
    if ep is None or ep.size == 1 or not dist.is_initialized():
        n_local = ep.experts_per_rank if ep is not None else n_routed_experts
        out = _local_grouped_ffn(x[tok], eid, w, experts, n_local)
        y = torch.zeros(T, dim, dtype=torch.float32, device=device)
        y.index_add_(0, tok, out)
        return y

    # ---- route each unit to its expert's owner rank, sort by owner ----
    owner = torch.div(eid, ep.experts_per_rank, rounding_mode="floor")  # dest rank
    leid = eid - owner * ep.experts_per_rank                            # local expert id on owner
    order = torch.argsort(owner, stable=True)
    tok, owner, leid, w = tok[order], owner[order], leid[order], w[order]
    xf = x[tok].float()

    # ---- counts handshake: my sends -> everyone's receives (MindSpeed pattern) ----
    input_splits = torch.bincount(owner, minlength=ep.size)            # tokens I send to each rank
    output_splits = torch.empty_like(input_splits)
    dist.all_to_all_single(output_splits, input_splits, group=ep.group)  # tokens I receive from each
    if _EP_CHECK:
        _verify_splits(input_splits, output_splits, ep.group, ep.size, ep.rank, _MOE_OP)
    in_s, out_s = input_splits.tolist(), output_splits.tolist()

    # ---- dispatch: token features + router weight ride ONE autograd all-to-all ----
    payload = torch.cat([xf, w[:, None]], dim=1)                       # [N, dim+1]
    recv = _AllToAll.apply(payload, out_s, in_s, ep.group)            # [M, dim+1]
    x_recv, w_recv = recv[:, :dim], recv[:, dim]
    leid_recv = _a2a_ints(leid, out_s, in_s, ep.group)               # [M] (non-diff index)

    # ---- owner runs grouped-GEMM over its LOCAL experts ----
    out_recv = _local_grouped_ffn(x_recv, leid_recv, w_recv, experts, ep.experts_per_rank)

    # ---- combine: results back to origin (splits swapped), then sum top-k per token ----
    out_back = _AllToAll.apply(out_recv, in_s, out_s, ep.group)      # [N, dim], aligned with tok
    y = torch.zeros(T, dim, dtype=torch.float32, device=device)
    y.index_add_(0, tok, out_back)
    return y


register_kernel(_MOE_OP, "npu", moe_dispatch_ep)


def enable() -> None:
    """Activate EP grouped-GEMM MoE dispatch process-wide (overrides plain grouped-GEMM).

    ``configure(...)`` must be called first (with the EP group/rank/size). Registering
    happens at import; this flips the active backend to "npu". The local per-owner compute
    still uses the validated ``_grouped_matmul`` op.
    """
    from .kernels import set_active_backend
    set_active_backend("npu")
