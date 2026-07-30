"""Expert-parallel (EP) MoE dispatch — the all-to-all forward.

Where per-expert FSDP shards every expert across all ranks, EP instead gives each
rank a DISJOINT slice of WHOLE experts (e.g. 256 experts / EP=8 -> 32 whole
experts per rank). Tokens are routed to their expert's owner rank via all-to-all,
the owner runs the grouped SwiGLU over its LOCAL experts (weights are whole and
local -> the grouped path works directly, no per-expert all-gather), then results
are all-to-all'd back and the top-k contributions summed. This is what lets a
large-expert-count draft train with grouped-GEMM throughput that per-expert FSDP
blocks.

Dropless (no capacity / no token drop): every routed (token, slot) is delivered.
The EP group is a plain process group; there is NO dependency on a model-parallel
framework. The token permute is a plain ``argsort`` (correct on CPU + any
accelerator, gloo-testable).

Gradients cross the dispatch: the token all-to-all is wrapped in an autograd
Function (:class:`_AllToAll`) whose backward is the reverse all-to-all, so grads
flow back to the router / upstream. The router weight rides the same tensor (an
extra column) so its gradient flows too; only the integer local-expert-id ride is
non-differentiable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from .backend import register_kernel, set_active_backend
from .experts import GroupedExperts, grouped_matmul, swiglu_grouped

_MOE_OP = "moe_dispatch"
EP_BACKEND = "ep"


@dataclass
class EPContext:
    """Process-wide expert-parallel context, installed by :func:`configure`."""

    group: object  # the EP process group, or None for the degenerate single-rank case
    rank: int
    size: int
    experts_per_rank: int  # = n_routed_experts // size (== len(local experts))


_EP: EPContext | None = None


def configure(group, rank: int, size: int, experts_per_rank: int) -> None:
    """Install the process-wide EP context (call once at EP training startup)."""
    global _EP
    _EP = EPContext(group=group, rank=rank, size=size, experts_per_rank=experts_per_rank)


def get_context() -> EPContext | None:
    """Return the installed EP context (or None if EP was never configured)."""
    return _EP


def reset() -> None:
    """Clear the EP context (mainly for tests)."""
    global _EP
    _EP = None


class _AllToAll(torch.autograd.Function):
    """Variable-split all-to-all with an autograd backward (reverse all-to-all).

    forward:  send ``in_splits`` rows to each rank, receive ``out_splits`` -> [sum(out), *].
    backward: the transpose — send grad with ``out_splits``, receive ``in_splits``.
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
        dist.all_to_all_single(
            gx, g.contiguous(), ctx.in_splits, ctx.out_splits, group=ctx.group
        )
        return gx, None, None, None


def _a2a_ints(x: torch.Tensor, out_splits, in_splits, group) -> torch.Tensor:
    """Non-differentiable variable-split all-to-all for integer side-channels."""
    y = x.new_empty([sum(out_splits), *x.shape[1:]])
    dist.all_to_all_single(y, x.contiguous(), out_splits, in_splits, group=group)
    return y


def _local_grouped_ffn(
    x: torch.Tensor,
    local_eid: torch.Tensor,
    w: torch.Tensor,
    experts: GroupedExperts,
    n_local: int,
) -> torch.Tensor:
    """Grouped SwiGLU over the LOCAL experts, per-unit output (no token combine).

    ``x[N, dim]`` units each tagged with ``local_eid[N]`` (0..n_local-1) and router
    weight ``w[N]``. Sorts by local expert, runs the grouped SwiGLU, returns outputs
    re-ordered back to the input order. ``experts`` holds stacked weights
    (``.to_local()`` when Shard(0) DTensors).
    """
    w1, w3, w2 = experts.local_weights()
    order = torch.argsort(local_eid, stable=True)
    inv = torch.argsort(order, stable=True)
    xs, ws = x[order].float(), w[order].float()
    counts = torch.bincount(local_eid, minlength=n_local)
    out = swiglu_grouped(
        xs, w1.float(), w3.float(), w2.float(), counts, ws, experts.swiglu_limit, grouped_matmul
    )  # [N, dim]
    return out[inv]  # back to input order


def _flatten_route(x, weights, indices):
    """(token, slot) -> flat (token ids, global expert ids, router weights)."""
    tokens = x.shape[0]
    topk = indices.shape[1]
    tok = torch.arange(tokens, device=x.device).repeat_interleave(topk)  # [T*topk]
    eid = indices.reshape(-1)  # global expert id
    w = weights.reshape(-1).float()
    return tok, eid, w


def moe_dispatch_ep(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    experts: GroupedExperts,
    n_routed_experts: int,
) -> torch.Tensor:
    """EP MoE dispatch. ``experts`` is the LOCAL slice; returns ``y[T, dim]`` (fp32)."""
    ep = _EP
    tokens, dim = x.shape
    device = x.device
    tok, eid, w = _flatten_route(x, weights, indices)

    # ---- degenerate: no EP (single rank / unconfigured) == grouped over all local experts ----
    if ep is None or ep.size == 1 or not dist.is_initialized():
        n_local = ep.experts_per_rank if ep is not None else n_routed_experts
        out = _local_grouped_ffn(x[tok], eid, w, experts, n_local)
        y = torch.zeros(tokens, dim, dtype=torch.float32, device=device)
        y.index_add_(0, tok, out)
        return y

    # ---- route each unit to its expert's owner rank, sort by owner ----
    owner = torch.div(eid, ep.experts_per_rank, rounding_mode="floor")  # dest rank
    leid = eid - owner * ep.experts_per_rank  # local expert id on owner
    order = torch.argsort(owner, stable=True)
    tok, owner, leid, w = tok[order], owner[order], leid[order], w[order]
    xf = x[tok].float()

    # ---- counts handshake: my sends -> everyone's receives ----
    input_splits = torch.bincount(owner, minlength=ep.size)  # tokens I send to each rank
    output_splits = torch.empty_like(input_splits)
    dist.all_to_all_single(output_splits, input_splits, group=ep.group)  # tokens I receive
    in_s, out_s = input_splits.tolist(), output_splits.tolist()

    # ---- dispatch: token features + router weight ride ONE autograd all-to-all ----
    payload = torch.cat([xf, w[:, None]], dim=1)  # [N, dim+1]
    recv = _AllToAll.apply(payload, out_s, in_s, ep.group)  # [M, dim+1]
    x_recv, w_recv = recv[:, :dim], recv[:, dim]
    leid_recv = _a2a_ints(leid, out_s, in_s, ep.group)  # [M] (non-diff index)

    # ---- owner runs grouped SwiGLU over its LOCAL experts ----
    out_recv = _local_grouped_ffn(x_recv, leid_recv, w_recv, experts, ep.experts_per_rank)

    # ---- combine: results back to origin (splits swapped), then sum top-k per token ----
    out_back = _AllToAll.apply(out_recv, in_s, out_s, ep.group)  # [N, dim], aligned with tok
    y = torch.zeros(tokens, dim, dtype=torch.float32, device=device)
    y.index_add_(0, tok, out_back)
    return y


register_kernel(_MOE_OP, EP_BACKEND, moe_dispatch_ep)


def enable() -> None:
    """Activate EP MoE dispatch process-wide (overrides the per-expert loop).

    :func:`configure` must be called first (with the EP group/rank/size). Importing
    this module already registers the impl under ``("moe_dispatch", "ep")``; this
    flips the active backend so :class:`~speculators.models.moe.layer.MoE` uses it.
    """
    set_active_backend(EP_BACKEND)
