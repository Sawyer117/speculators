"""Muon that survives FSDP2 + expert-parallel DTensors.

WHY THIS EXISTS. ``torch.optim.Muon`` runs its Newton-Schulz iteration directly on the
DTensor and lets sharding propagation work it out. The iteration transposes
(``bmm(x, x.transpose(1, 2))``), which flips the shard dim, and the final in-place write
then dies:

    aten.add_.Tensor: in-place operations that require placement changes are not supported
    Spec(f32[4, 16384](S(0))) += Spec(bf16[4, 16384](S(1)))

Observed on the DSV4-DSpark draft at ``hyper.py`` ``HyperMix.fn`` ([32, 16384], a 1:512
matrix) under 8-way FSDP2.

The fix is taken from torchtitan-npu's ``DistributedMuon``
(``torchtitan_npu/patches/optimizer/muon_optimizer.py``, itself derived from
``rakkit/torchtitan`` dist-scion): **never let DTensor ops carry the iteration.** Drop to
plain local tensors, orthogonalize there, and handle distribution explicitly. This module
keeps that math verbatim and drops the TP / HSDP / swap-to-host / bucket-merge machinery
that this setup does not use (one mesh, FSDP2 + EP).

Two routes, matching how the draft is actually sharded:

* **experts** -- ``w1/w3 [E_local, inter, dim]``, ``w2 [E_local, dim, inter]``: 3D and
  ``Shard(0)`` on the EXPERT axis, so every rank holds WHOLE expert matrices and the
  iteration is exact locally with **zero communication**. This route is also where the
  memory saving lives: the experts are most of the 21B, and one momentum buffer (Muon)
  instead of two (AdamW) is the whole point. Routing only 2D params to Muon -- what the
  previous wiring did -- leaves the experts on AdamW and saves nothing.
* **matrices** -- 2D, row-sharded by FSDP2. A row-shard is not a matrix, so the full
  gradient is gathered, the iteration runs identically on every rank, and each rank keeps
  its own slice. Redundant compute, but these are the small parameters.

COST, MEASURED (8x A2, faithful EP config, steady state). The ~5% figure this
file used to quote was AdamW's, not Muon's:

    AdamW   opt_ms  99 ms / step_ms 2155 ms  =  4.6%
    Muon    opt_ms 1010 ms / step_ms 2960 ms  =  34%   (+37% wall-clock/step)

The 1 s is Newton-Schulz FLOPs on the 3D expert stacks, not communication: 5
iterations x 3 ``bmm`` each, over every local expert. ``ns_steps`` is therefore
the first knob to reach for if step time has to come down, and ``hybrid_ns``
settles the singular values in fewer of them.

⚠ ``validate_expert_shard_dim0`` is not decoration. If experts were ever sharded INSIDE a
matrix instead of across the expert axis, the local route would orthogonalize incomplete
matrices and the run would train on quietly wrong updates -- no error, just worse results.
"""

from __future__ import annotations

import logging
import math

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard
from torch.optim import Optimizer

logger = logging.getLogger("speculators")

# Quintic Newton-Schulz coefficients, chosen to maximize the slope at zero.
COEFF_PRIMARY = (3.4445, -4.7750, 2.0315)
# DeepSeek-V4's hybrid schedule: the last two steps swap to a gentler polynomial that
# settles the singular values at 1 instead of overshooting them.
COEFF_SECONDARY = (2.0, -1.5, 0.5)

_EXPERT_HINTS = ("experts", "expert")
_ADAMW_NAME_HINTS = ("embed_tokens", "lm_head")


def zeropower_via_newtonschulz5(
    grad: Tensor, steps: int = 5, eps: float = 1e-7, hybrid_ns: bool = False
) -> Tensor:
    """Orthogonalize ``grad`` (2D, or 3D as a batch of matrices) by Newton-Schulz."""
    if grad.ndim not in (2, 3):
        raise ValueError(f"Newton-Schulz expects a 2D or 3D tensor, got {tuple(grad.shape)}")
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
        gram = torch.bmm(x, x.transpose(1, 2))
        poly = b * gram + c * torch.bmm(gram, gram)
        x = a * x + torch.bmm(poly, x)

    if transposed:
        x = x.transpose(1, 2)
    if is_2d:
        x = x.squeeze(0)
    return x.to(original_dtype)


def normalise_grad(g: Tensor, adjust_lr_fn: str) -> Tensor:
    """Scale the orthogonalized update so its RMS is comparable across matrix shapes."""
    a, b = g.size(-2), g.size(-1)
    if adjust_lr_fn == "match_rms_adamw":
        return g * 0.18 * math.sqrt(max(a, b))  # Moonshot / DeepSeek-V4 constant
    return g * math.sqrt(max(1.0, a / b))  # Keller Jordan's original


def validate_expert_shard_dim0(param: Tensor, name: str) -> None:
    """Refuse an expert stack sharded inside its matrices rather than across them."""
    if not isinstance(param, DTensor):
        return
    for placement in param.placements:
        if isinstance(placement, Shard) and placement.dim != 0:
            raise RuntimeError(
                f"expert parameter {name!r} has placement {placement}: it is sharded INSIDE "
                "each expert matrix, so a local Newton-Schulz would orthogonalize incomplete "
                "matrices and train on wrong updates. Expert stacks must be Shard(0)."
            )


def classify(name: str, param: Tensor) -> str:
    """Route a parameter to ``"expert"``, ``"matrix"`` or ``"adamw"``.

    3D expert stacks and 2D weight matrices get the orthogonalized update; embeddings, the
    LM head, and everything 1D (norms, biases, the per-head sink) go to AdamW, following
    Muon's own convention that only matrices are orthogonalized.
    """
    if any(hint in name for hint in _ADAMW_NAME_HINTS):
        return "adamw"
    if param.ndim == 3 and any(hint in name for hint in _EXPERT_HINTS):
        return "expert"
    if param.ndim == 2 and min(param.shape) > 1:
        return "matrix"
    return "adamw"


def split_named_params(model) -> tuple[list[tuple[str, Tensor]], list[tuple[str, Tensor]]]:
    """Split trainable parameters into the Muon group and the AdamW group."""
    muon: list[tuple[str, Tensor]] = []
    adamw: list[tuple[str, Tensor]] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (adamw if classify(name, param) == "adamw" else muon).append((name, param))
    return muon, adamw


def _local_shard_of(full: Tensor, like: DTensor) -> Tensor:
    """Take back this rank's slice of ``full``, matching ``like``'s shard split."""
    mesh = like.device_mesh
    shard_dim = next(p.dim for p in like.placements if isinstance(p, Shard))
    chunks = torch.chunk(full, mesh.size(), dim=shard_dim)
    idx = mesh.get_local_rank()
    piece = chunks[idx] if idx < len(chunks) else full.new_zeros((0, *full.shape[1:]))
    local = like.to_local()
    if piece.shape != local.shape:
        raise RuntimeError(
            f"reshard mismatch: rebuilt {tuple(piece.shape)} but the local shard is "
            f"{tuple(local.shape)} — DTensor's split rule is not plain chunk here"
        )
    return piece


class DistributedMuon(Optimizer):
    """Muon for FSDP2 (+ expert-parallel) models living on a single device mesh."""

    def __init__(
        self,
        named_params: list[tuple[str, Tensor]],
        lr: float = 2e-2,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        eps: float = 1e-7,
        adjust_lr_fn: str = "match_rms_adamw",
        hybrid_ns: bool = False,
    ) -> None:
        names = [n for n, _ in named_params]
        params = [p for _, p in named_params]
        super().__init__(
            params,
            {
                "lr": lr,
                "momentum": momentum,
                "nesterov": nesterov,
                "weight_decay": weight_decay,
                "ns_steps": ns_steps,
                "eps": eps,
                "adjust_lr_fn": adjust_lr_fn,
                "hybrid_ns": hybrid_ns,
            },
        )
        self._route: dict[int, str] = {}
        counts = {"expert": 0, "matrix": 0}
        for name, param in zip(names, params, strict=True):
            route = classify(name, param)
            if route == "expert":
                validate_expert_shard_dim0(param, name)
            self._route[id(param)] = route
            counts[route] = counts.get(route, 0) + 1
        logger.info(
            "DistributedMuon: %d expert stacks (local route, no comm) + %d matrices "
            "(gather route)",
            counts.get("expert", 0),
            counts.get("matrix", 0),
        )

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    self._step_param(param, group)
        return loss

    def _orthogonalize(self, effective: Tensor, group: dict) -> Tensor:
        upd = zeropower_via_newtonschulz5(
            effective, group["ns_steps"], group["eps"], group["hybrid_ns"]
        )
        return normalise_grad(upd, group["adjust_lr_fn"])

    def _step_param(self, param: Tensor, group: dict) -> None:
        state = self.state.setdefault(param, {})
        grad, m = param.grad, group["momentum"]

        # The momentum buffer stays SHARDED, always. Keeping it in gathered shape would
        # put a whole copy of every matrix on every rank -- for this draft that is ~1 GB
        # of pure waste per rank, which is a large bite out of the very saving Muon is
        # here for. Only the iteration input is gathered, and only transiently.
        g_local = grad.to_local() if isinstance(grad, DTensor) else grad
        buf = state.get("momentum_buffer")
        if buf is None:
            buf = state["momentum_buffer"] = torch.zeros_like(g_local)
        buf.lerp_(g_local, 1.0 - m)
        effective = g_local.add(buf, alpha=m) if group["nesterov"] else buf

        if self._route[id(param)] == "matrix" and isinstance(grad, DTensor):
            # A row-shard is not a matrix: gather the effective gradient, run the
            # identical iteration on every rank, then keep this rank's slice back.
            eff_full = DTensor.from_local(
                effective, grad.device_mesh, grad.placements
            ).full_tensor()
            update = _local_shard_of(self._orthogonalize(eff_full, group), param)
        else:
            # Experts: whole matrices are already local, so nothing to gather.
            update = self._orthogonalize(effective, group)

        # Write through .to_local(): the in-place add is exactly what torch's Muon dies
        # on, and doing it on plain tensors is what avoids the placement propagation.
        p_local = param.to_local() if isinstance(param, DTensor) else param
        if group["weight_decay"]:
            p_local.mul_(1.0 - group["lr"] * group["weight_decay"])
        p_local.add_(update.to(p_local.dtype), alpha=-group["lr"])
