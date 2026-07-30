"""Expert weights and the grouped SwiGLU compute for the MoE draft layer.

Routed experts are held as **stacked weights** (:class:`GroupedExperts`:
``w1/w3 [E, inter, dim]``, ``w2 [E, dim, inter]``) rather than a ``ModuleList``
of per-expert Linears. This is the layout expert-parallelism wants — one
``Shard(0)`` DTensor per projection, and a single grouped GEMM instead of an
``E``-way Python loop.

The grouped matmul has a portable reference (:func:`_grouped_matmul_reference`,
a per-group ``split`` + ``matmul``, correct on CPU / any accelerator) and an
opt-in ``torch._grouped_mm`` fast path for CUDA that a caller may switch on with
:func:`enable_cuda_grouped_mm`. The default is the portable reference so the
layer is correct everywhere with nothing enabled.
"""

from __future__ import annotations

import contextlib
import math

import torch
import torch.nn.functional as F
from torch import nn

# Opt-in: use ``torch._grouped_mm`` on CUDA instead of the per-group reference.
# OFF by default — the reference runs everywhere; the fused path is a throughput
# win that must be validated on the target GPU before it is relied on.
_USE_CUDA_GROUPED_MM = False


def enable_cuda_grouped_mm(flag: bool = True) -> bool:
    """Enable the ``torch._grouped_mm`` fast path (CUDA only). Returns whether it
    is now active (requires the running torch to expose ``torch._grouped_mm``)."""
    global _USE_CUDA_GROUPED_MM
    _USE_CUDA_GROUPED_MM = bool(flag) and hasattr(torch, "_grouped_mm")
    return _USE_CUDA_GROUPED_MM


def _grouped_matmul_reference(
    x: torch.Tensor, weight: torch.Tensor, counts: torch.Tensor
) -> torch.Tensor:
    """Portable grouped matmul: ``x[T, K]`` blocks (sized by ``counts``) each times
    ``weight[e] [K, M]`` -> ``[T, M]``. Autograd-native; the parity oracle."""
    outs = []
    off = 0
    for e in range(weight.shape[0]):
        n = int(counts[e])
        outs.append(x[off : off + n] @ weight[e])
        off += n
    return torch.cat(outs, dim=0) if outs else x.new_zeros((0, weight.shape[-1]))


def grouped_matmul(
    x: torch.Tensor, weight: torch.Tensor, counts: torch.Tensor
) -> torch.Tensor:
    """Grouped ``x @ weight`` with ``weight[E, K, M]`` and per-group ``counts[E]``.

    Uses the portable reference by default; the ``torch._grouped_mm`` CUDA fast
    path when :func:`enable_cuda_grouped_mm` has switched it on and ``x`` is on
    CUDA. Both compute the same result (validate the fast path on GPU).
    """
    if _USE_CUDA_GROUPED_MM and x.is_cuda:
        # offs = exclusive/cumulative group boundaries expected by torch._grouped_mm.
        offs = torch.cumsum(counts, dim=0).to(torch.int32)
        return torch._grouped_mm(x, weight, offs=offs)  # noqa: SLF001
    return _grouped_matmul_reference(x, weight, counts)


def swiglu_grouped(
    xg: torch.Tensor,
    w1: torch.Tensor,
    w3: torch.Tensor,
    w2: torch.Tensor,
    counts: torch.Tensor,
    w_flat: torch.Tensor,
    swiglu_limit: float,
    matmul=grouped_matmul,
) -> torch.Tensor:
    """SwiGLU over token groups, one group per (local) expert.

    ``xg[N, dim]`` are routed tokens already sorted by expert; ``counts[E]`` the
    per-expert token count; ``w1/w3 [E, inter, dim]``, ``w2 [E, dim, inter]`` the
    stacked expert weights; ``w_flat[N]`` the per-token router weight (applied
    pre-down, matching :class:`Expert`). ``matmul(x, W, counts)`` does the grouped
    ``x @ W`` with ``W[E, in, out]``.
    """
    gate = matmul(xg, w1.transpose(1, 2), counts)  # [N, inter]
    up = matmul(xg, w3.transpose(1, 2), counts)
    if swiglu_limit > 0:
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        gate = torch.clamp(gate, max=swiglu_limit)
    h = F.silu(gate) * up
    h = w_flat[:, None] * h
    return matmul(h, w2.transpose(1, 2), counts)  # [N, dim]


class Expert(nn.Module):
    """SwiGLU expert with an optional magnitude clamp (fp32 activations).

    Used for the single **shared** expert (a plain dense FFN). The routed experts
    live in :class:`GroupedExperts` as stacked weights.
    """

    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)  # gate
        self.w3 = nn.Linear(dim, inter_dim, bias=False)  # up
        self.w2 = nn.Linear(inter_dim, dim, bias=False)  # down
        self.swiglu_limit = swiglu_limit

    def forward(
        self, x: torch.Tensor, weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        h = F.silu(gate) * up
        if weight is not None:
            h = weight * h
        return self.w2(h.to(dtype))


class GroupedExperts(nn.Module):
    """Routed experts as stacked weights (``w1/w3 [E, inter, dim]``, ``w2 [E, dim, inter]``).

    ``E`` is the number of experts held **locally**: all ``n_routed_experts``
    without EP, or ``n_routed_experts // ep_size`` under expert-parallelism (each
    rank owns a disjoint slice, seeded per-rank so the shards don't init
    identically). Under EP the stacked weights are wrapped as ``Shard(0)``
    DTensors at setup so the optimizer / clip / checkpoint see uniform DTensors;
    the forward reads ``.to_local()``.
    """

    def __init__(
        self,
        dim: int,
        inter_dim: int,
        n_local: int,
        swiglu_limit: float,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.num_local_experts = n_local
        self.swiglu_limit = swiglu_limit
        # Weight layout [out, in] per expert (like nn.Linear.weight), stacked over
        # experts, so a checkpoint round-trips to/from per-expert
        # ``experts.{i}.w{1,2,3}.weight`` by a plain stack/index.
        self.w1 = nn.Parameter(torch.empty(n_local, inter_dim, dim))  # gate
        self.w3 = nn.Parameter(torch.empty(n_local, inter_dim, dim))  # up
        self.w2 = nn.Parameter(torch.empty(n_local, dim, inter_dim))  # down
        # nn.Linear default init is kaiming_uniform_(a=sqrt(5)) -> U(-1/sqrt(fan_in));
        # apply it per-expert (fan_in = dim for w1/w3, inter for w2 — same for every
        # expert, so a single fill matches). Per-rank seed under EP so disjoint
        # shards differ.
        b1, b2 = 1.0 / math.sqrt(dim), 1.0 / math.sqrt(inter_dim)
        ctx = torch.random.fork_rng(devices=[]) if seed is not None else contextlib.nullcontext()
        with ctx, torch.no_grad():
            if seed is not None:
                torch.manual_seed(seed)
            self.w1.uniform_(-b1, b1)
            self.w3.uniform_(-b1, b1)
            self.w2.uniform_(-b2, b2)

    def local_weights(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(w1, w3, w2) as plain local tensors (``.to_local()`` when Shard(0) DTensors)."""

        def loc(p):
            return p.to_local() if hasattr(p, "to_local") else p

        return loc(self.w1), loc(self.w3), loc(self.w2)
