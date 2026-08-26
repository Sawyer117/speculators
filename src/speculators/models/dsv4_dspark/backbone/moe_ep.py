"""Expert-parallel MoE dispatch.

Where FSDP shards every expert across every rank and gathers them back for each
forward, expert parallelism gives each rank a disjoint slice of *whole* experts and
moves the tokens instead: each routed (token, slot) is sent to the rank that owns its
expert, computed there over local weights, and sent back. With 256 experts over 8 ranks
that is 32 whole experts per rank and no expert all-gather at all.

Dropless: there is no capacity limit and no token is discarded. A rank's share of a step
is whatever the router sends it, which can be nothing.

Gradients cross the dispatch. The token all-to-all is an autograd ``Function`` whose
backward is the reverse all-to-all, so gradients reach the router and the layers below
it. The router weight rides the same tensor as an extra column so that it, too, is
differentiated; only the integer expert ids travel without gradient.

Plain ``torch.distributed`` throughout -- ``all_to_all_single`` and a reference grouped
matmul. Nothing here is device-specific.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from speculators.train import expert_parallel

from .moe import GroupedExperts, swiglu_grouped


def _grouped_matmul(
    x: torch.Tensor, w: torch.Tensor, counts: torch.Tensor
) -> torch.Tensor:
    """``x @ w[e]`` per group, for ``x`` already sorted by group.

    ``counts[e]`` rows belong to group ``e`` and ``w[E, in, out]`` holds one matrix per
    group. A per-group loop: correct on any device, and the oracle a fused grouped GEMM
    would be checked against.
    """
    outs = []
    start = 0
    for e, n in enumerate(counts.tolist()):
        if n:
            outs.append(x[start : start + n] @ w[e])
            start += n
    if not outs:
        return x.new_zeros((0, w.shape[-1]))
    return torch.cat(outs, dim=0)


class _AllToAll(torch.autograd.Function):
    """Variable-split all-to-all whose backward is the same exchange, reversed.

    Forward sends ``in_splits[r]`` rows to rank ``r`` and receives ``out_splits[r]``
    from it; backward swaps the two, which is exactly the transpose of the forward
    permutation.
    """

    @staticmethod
    def forward(ctx, x, out_splits, in_splits, group):
        ctx.out_splits, ctx.in_splits, ctx.group = out_splits, in_splits, group
        y = x.new_empty([sum(out_splits), *x.shape[1:]])
        dist.all_to_all_single(y, x.contiguous(), out_splits, in_splits, group=group)
        return y

    @staticmethod
    def backward(ctx, grad):
        out = grad.new_empty([sum(ctx.in_splits), *grad.shape[1:]])
        dist.all_to_all_single(
            out, grad.contiguous(), ctx.in_splits, ctx.out_splits, group=ctx.group
        )
        return out, None, None, None


def _exchange_ids(
    ids: torch.Tensor, out_splits: list[int], in_splits: list[int], group
) -> torch.Tensor:
    """The same exchange for the integer expert ids, which carry no gradient."""
    out = ids.new_empty([sum(out_splits), *ids.shape[1:]])
    dist.all_to_all_single(out, ids.contiguous(), out_splits, in_splits, group=group)
    return out


def _local_experts_forward(
    x: torch.Tensor,
    local_ids: torch.Tensor,
    router_weights: torch.Tensor,
    experts: GroupedExperts,
) -> torch.Tensor:
    """Run this rank's experts over the rows it owns, one row per (token, slot).

    ``local_ids`` index into the local slice. Rows are sorted by expert so the grouped
    SwiGLU sees contiguous groups, then restored to the caller's order -- the all-to-all
    that follows expects them in the order it sent.
    """
    n_local = experts.num_local_experts
    w1, w3, w2 = experts.local_weights()
    order = torch.argsort(local_ids, stable=True)
    inverse = torch.argsort(order, stable=True)
    counts = torch.bincount(local_ids, minlength=n_local)
    out = swiglu_grouped(
        x[order].float(),
        w1.float(),
        w3.float(),
        w2.float(),
        counts,
        router_weights[order].float(),
        experts.swiglu_limit,
        _grouped_matmul,
    )
    return out[inverse]


def moe_dispatch_ep(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    experts: GroupedExperts,
) -> torch.Tensor:
    """Route ``x [tokens, dim]`` to its top-k experts across ranks and combine.

    ``experts`` holds this rank's slice; ``indices`` are global expert ids. Returns
    ``[tokens, dim]`` in fp32, the same contract as the replicated dispatch.
    """
    ep = expert_parallel.context()
    n_local = experts.num_local_experts
    num_tokens, dim = x.shape

    # One row per routed (token, slot): which token it came from, which expert wants
    # it, and the router weight to apply.
    topk = indices.shape[1]
    token = torch.arange(num_tokens, device=x.device).repeat_interleave(topk)
    expert = indices.reshape(-1)
    weight = weights.reshape(-1).float()

    if ep is None or ep.size == 1 or not dist.is_initialized():
        # Every expert is already local; the exchange would be an identity.
        out = _local_experts_forward(x[token], expert, weight, experts)
        return torch.zeros(
            num_tokens, dim, dtype=torch.float32, device=x.device
        ).index_add_(0, token, out)

    # Sort by owning rank so each rank's rows are contiguous, which is what
    # all_to_all_single's split arguments describe.
    owner = torch.div(expert, n_local, rounding_mode="floor")
    local_id = expert - owner * n_local
    order = torch.argsort(owner, stable=True)
    token, owner, local_id, weight = (
        token[order],
        owner[order],
        local_id[order],
        weight[order],
    )

    # Tell each rank how many rows it is about to receive; its answer is how many this
    # rank receives. Both sides need both numbers before any payload moves.
    send_counts = torch.bincount(owner, minlength=ep.size)
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts, group=ep.group)
    send, recv = send_counts.tolist(), recv_counts.tolist()

    # The router weight rides as an extra column so it crosses the same differentiable
    # exchange as the activations.
    payload = torch.cat([x[token].float(), weight[:, None]], dim=1)
    received = _AllToAll.apply(payload, recv, send, ep.group)
    local_id = _exchange_ids(local_id, recv, send, ep.group)

    computed = _local_experts_forward(
        received[:, :dim], local_id, received[:, dim], experts
    )

    # Back to the rank the rows came from -- splits swapped -- then sum each token's
    # top-k contributions.
    returned = _AllToAll.apply(computed, send, recv, ep.group)
    return torch.zeros(
        num_tokens, dim, dtype=torch.float32, device=x.device
    ).index_add_(0, token, returned)
