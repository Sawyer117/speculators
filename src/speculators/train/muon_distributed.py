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

COST AND BENEFIT, MEASURED (8x A2, faithful EP config; medians / means over
~1700 matched steps of two runs on the SAME lr schedule -- verified identical at
step 57 and 2002). The ~5% cost this file used to quote was AdamW's, not Muon's:

    opt_ms      AdamW    99   Muon  1010    10.2x
    step_ms     AdamW  2040   Muon  3030     1.49x

    steps 1900-2004      AdamW     Muon
    accept_len            2.453    2.793   +13.9%
    position_0_acc        0.709    0.750    +5.8%
    position_4_acc        0.246    0.357   +45.1%

The 1 s is Newton-Schulz FLOPs on the 3D expert stacks, not communication: 5
iterations x 3 ``bmm`` each, over every local expert.

⚠ ``ns_steps`` is NOT a usable knob for cutting that. Measured on the real shapes
in bf16 (GPU, mean|sigma-1| of the orthogonalized result), no coefficient schedule
is usable at 3 steps -- the best is 0.21, i.e. barely orthogonalized at all:

    5 steps, expert w1/w3 [2048, 4096]        3 steps
    single triplet repeated   0.1650           0.2811
    hybrid (this file)        0.0104           0.2811
    MindSpeed "quintic"       0.0159           0.2087
    "polar_express"           0.0842           0.6075

⚠ READ THE TABLE FOR WHAT IT IS. It measures distance from orthogonal, which is
NOT the training objective. ``COEFF_PRIMARY`` does not converge to 1 by design:
as a map on singular values, p(x) = a x + b x^3 + c x^5 has p(1) = 0.7010, so 1 is
not even a fixed point, and iterating it oscillates (1.19, 0.90, 0.82, 0.77, ...,
1.13). Muon accepts sigma in roughly [0.7, 1.3]. ``COEFF_SECONDARY`` has p(1) = 1
and p'(1) = 0 -- a genuine quadratically-convergent fixed point -- which is why
hybrid measures better here. Whether that helps accept_len is UNTESTED, so
``hybrid_ns`` defaults OFF, matching upstream.

⚠ AND THIS PORT IS NOT EQUIVALENT TO UPSTREAM AT ``ns_steps=5``. torchtitan-npu
switches at an ABSOLUTE ``i >= 8`` with ``steps=10`` (8 primary + 2 secondary);
this file switches at ``steps - 2``. The two agree at 10 steps, but at our 5 steps
upstream's condition never fires -- their ``hybrid_ns=True`` would be a complete
no-op -- while this gives 3 primary + 2 secondary, a schedule upstream never
validated. Anything claimed for hybrid here rests on our own measurement only.

MindSpeed has no fused NPU kernel here to borrow: its Newton-Schulz accepts a
``use_syrk`` argument and documents that the NPU path falls back to plain matmul.

Muon wins even after paying the 1.49x: at EQUAL WALL CLOCK (AdamW step 2970 vs
Muon step 2000) accept_len is 2.650 vs 2.796, and AdamW needs ~step 4500 to reach
2.796 -- 2.2x the steps, 1.5x the wall clock. The gain also grows monotonically
with position, which is what matters for a longer block.

⚠ Two limits on the above. It is step ~2000 of a 124,480-step run (1.6% in; AdamW
ends at accept_len 3.87), and it is one seed each -- nothing here says the lead
survives to convergence. And Muon is genuinely WORSE early: behind on loss from
step ~25 to ~200 (steps 50-75: ce 29.9 vs 9.8). It passes AdamW on accept_len
around step 100 and on loss around step 250, so a 100-step smoke test reads as a
regression.

⚠ ``validate_expert_shard_dim0`` is not decoration. If experts were ever sharded INSIDE a
matrix instead of across the expert axis, the local route would orthogonalize incomplete
matrices and the run would train on quietly wrong updates -- no error, just worse results.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.distributed as dist
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

        # ⚠ THE COLLECTIVE MUST NOT BE CONDITIONAL. The matrix route calls
        # ``full_tensor()`` -- an all-gather over the whole mesh. Skipping a parameter
        # on the ranks where its grad happens to be None, while the others gather, makes
        # the ranks disagree on how many collectives to issue, and it does not fail --
        # it HANGS until HCCL_EXEC_TIMEOUT (1800 s here), dying far from the cause.
        # So a missing grad is treated as a zero grad, and every rank walks the same
        # parameters in the same order. AdamW never had this exposure -- its step is
        # element-wise on local shards and issues no collectives at all.
        stepped = 0
        for group in self.param_groups:
            for param in group["params"]:
                self._step_param(param, group)
                stepped += param.grad is not None
        self._assert_ranks_agree(stepped)
        return loss

    def _assert_ranks_agree(self, stepped: int) -> None:
        """Turn a rank disagreement into an immediate error instead of a 30-min hang.

        One scalar all-gather per step (negligible next to ~1 s of Newton-Schulz). If
        ranks ever walk different parameter sets, this says so on the spot, naming the
        counts, rather than leaving a collective half-issued for the watchdog to find.
        """
        if not (dist.is_available() and dist.is_initialized()):
            return
        param = next(
            (p for g in self.param_groups for p in g["params"] if p.numel()), None
        )
        if param is None:
            return
        device = (param.to_local() if isinstance(param, DTensor) else param).device
        counts = torch.tensor([stepped], device=device, dtype=torch.int64)
        gathered = [torch.zeros_like(counts) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, counts)
        seen = [int(t.item()) for t in gathered]
        if len(set(seen)) > 1:
            raise RuntimeError(
                "DistributedMuon: ranks disagree on how many parameters carried a "
                f"gradient ({seen}). They would issue different numbers of collectives "
                "and the run would hang until the HCCL timeout. This means the ranks "
                "took different code paths in the backward."
            )

    def _orthogonalize(self, effective: Tensor, group: dict) -> Tensor:
        upd = zeropower_via_newtonschulz5(
            effective, group["ns_steps"], group["eps"], group["hybrid_ns"]
        )
        return normalise_grad(upd, group["adjust_lr_fn"])

    def _step_param(self, param: Tensor, group: dict) -> None:
        state = self.state.setdefault(param, {})
        m = group["momentum"]

        # The mesh and placements come from the PARAMETER, not the gradient, because a
        # rank may arrive here with grad None -- see the note in step(). A missing grad
        # becomes a zero grad so the collective below is still issued in lockstep; the
        # momentum simply decays, which is what a zero gradient should do anyway.
        p_local = param.to_local() if isinstance(param, DTensor) else param
        g_local = param.grad
        if g_local is None:
            g_local = torch.zeros_like(p_local)
        elif isinstance(g_local, DTensor):
            g_local = g_local.to_local()

        # The momentum buffer stays SHARDED, always. Keeping it in gathered shape would
        # put a whole copy of every matrix on every rank -- for this draft that is ~1 GB
        # of pure waste per rank, which is a large bite out of the very saving Muon is
        # here for. Only the iteration input is gathered, and only transiently.
        buf = state.get("momentum_buffer")
        if buf is None:
            buf = state["momentum_buffer"] = torch.zeros_like(g_local)
        buf.lerp_(g_local, 1.0 - m)
        effective = g_local.add(buf, alpha=m) if group["nesterov"] else buf

        if self._route[id(param)] == "matrix" and isinstance(param, DTensor):
            # A row-shard is not a matrix: gather the effective gradient, run the
            # identical iteration on every rank, then keep this rank's slice back.
            eff_full = DTensor.from_local(
                effective, param.device_mesh, param.placements
            ).full_tensor()
            update = _local_shard_of(self._orthogonalize(eff_full, group), param)
        else:
            # Experts: whole matrices are already local, so nothing to gather.
            update = self._orthogonalize(effective, group)

        # Write through .to_local(): the in-place add is exactly what torch's Muon dies
        # on, and doing it on plain tensors is what avoids the placement propagation.
        if group["weight_decay"]:
            p_local.mul_(1.0 - group["lr"] * group["weight_decay"])
        p_local.add_(update.to(p_local.dtype), alpha=-group["lr"])
