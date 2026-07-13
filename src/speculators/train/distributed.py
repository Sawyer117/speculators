"""Single source of truth for distributed training topology.

Stores local_rank, SP/DP sizes and ranks, and process groups.
All other modules should import getters from here rather than
maintaining their own distributed state.
"""

from __future__ import annotations

import contextlib
import logging
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed import ProcessGroup
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

logger = logging.getLogger("speculators")

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_local_rank: int = 0
_rank: int = 0
_world_size: int = 1
_is_distributed: bool = False

_sp_size: int = 1
_sp_rank: int = 0
_dp_size: int = 1
_dp_rank: int = 0

_sp_group: ProcessGroup | None = None
_dp_group: ProcessGroup | None = None


# ---------------------------------------------------------------------------
# Getters
# ---------------------------------------------------------------------------


def get_local_rank() -> int:
    return _local_rank


def get_rank() -> int:
    return _rank


def get_world_size() -> int:
    return _world_size


def is_distributed() -> bool:
    return _is_distributed


def get_sp_group() -> ProcessGroup | None:
    return _sp_group


def get_dp_group() -> ProcessGroup | None:
    return _dp_group


def get_sp_size() -> int:
    return _sp_size


def get_sp_rank() -> int:
    return _sp_rank


def get_dp_size() -> int:
    return _dp_size


def get_dp_rank() -> int:
    return _dp_rank


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def _init_sp_process_groups(rank: int, world_size: int, sp_size: int) -> None:
    """Initialize sequence-parallel and data-parallel process groups.

    SP groups use contiguous ranks (e.g. sp_size=2, world_size=4: {0,1}, {2,3}).
    DP groups use strided ranks (e.g. sp_size=2, world_size=4: {0,2}, {1,3}).
    """
    global _sp_group, _dp_group, _sp_size, _sp_rank, _dp_size, _dp_rank  # noqa: PLW0603

    if sp_size <= 0:
        raise ValueError(f"sp_size must be positive, got {sp_size}")

    if world_size % sp_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by sp_size ({sp_size})"
        )

    dp_size = world_size // sp_size

    sp_group = None
    for i in range(dp_size):
        sp_ranks = list(range(i * sp_size, (i + 1) * sp_size))
        pg = dist.new_group(sp_ranks)
        if rank in sp_ranks:
            sp_group = pg

    dp_group = None
    for i in range(sp_size):
        dp_ranks = list(range(i, world_size, sp_size))
        pg = dist.new_group(dp_ranks)
        if rank in dp_ranks:
            dp_group = pg

    if sp_group is None or dp_group is None:
        raise RuntimeError("Failed to initialize SP/DP process groups")

    _sp_group = sp_group
    _dp_group = dp_group
    _sp_size = sp_size
    _sp_rank = rank % sp_size
    _dp_size = dp_size
    _dp_rank = rank // sp_size


def maybe_setup_distributed(sp_size: int = 1) -> None:
    """Set up distributed training if launched with ``torchrun``.

    Always populates the module-level topology state so that callers
    can use the getter functions regardless of whether SP is enabled.
    Process groups are always created when distributed — with
    ``sp_size == 1`` the DP group spans all ranks and each SP group
    contains a single rank.
    """
    global _local_rank, _rank, _is_distributed, _world_size  # noqa: PLW0603

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = "LOCAL_RANK" in os.environ

    _local_rank = local_rank
    _is_distributed = distributed

    if not distributed:
        return

    torch.accelerator.set_device_index(local_rank)
    acc = torch.accelerator.current_accelerator()
    if acc is None:
        raise ValueError("No accelerator found")
    backend = torch.distributed.get_default_backend_for_device(acc)
    # device_id must be a torch.device (torch>=2.x reads device_id.type). The bare
    # int works on CUDA, but transfer_to_npu rewrites it to a str on Ascend, so
    # build the device explicitly from the active accelerator.
    dist.init_process_group(backend, device_id=torch.device(acc.type, local_rank))

    _rank = dist.get_rank()
    _world_size = dist.get_world_size()

    _init_sp_process_groups(_rank, _world_size, sp_size)

    logger.info(
        f"Started distributed with local_rank={local_rank}, "
        f"dp_size={_dp_size}, sp_size={_sp_size}",
        extra={"override_rank0_filter": True},
    )


def maybe_destroy_distributed() -> None:
    """Destroy the distributed process group if using distributed training."""
    global _is_distributed, _local_rank, _rank, _world_size  # noqa: PLW0603
    global _sp_size, _sp_rank, _dp_size, _dp_rank  # noqa: PLW0603
    global _sp_group, _dp_group  # noqa: PLW0603

    if not _is_distributed:
        return

    dist.destroy_process_group()
    logger.info(
        "Destroyed distributed process group",
        extra={"override_rank0_filter": True},
    )

    _is_distributed = False
    _local_rank = 0
    _rank = 0
    _world_size = 1
    _sp_size = 1
    _sp_rank = 0
    _dp_size = 1
    _dp_rank = 0
    _sp_group = None
    _dp_group = None


@contextlib.contextmanager
def build_on_meta():
    """Construct module PARAMETERS on the meta device (no real storage).

    Overrides ``nn.Module.register_parameter`` to move each parameter to ``meta``
    immediately after it is created, so a large model is built with ~zero resident
    memory (only one parameter is briefly real before being replaced). Buffers
    (rope caches, vocab maps) are left real -- they are small and some are
    non-persistent, so they must keep their computed values.

    Intended for the non-rank0 ranks: their real weights arrive via
    ``set_model_state_dict(..., broadcast_from_rank0=True)`` in the trainer's FSDP
    setup, which materializes the sharded meta params in place. This is the
    torchtitan/torchtune large-model init pattern. Forcing ``.to("meta")``
    explicitly (rather than a default-device context) is robust to
    ``transfer_to_npu`` remapping factory ``device=`` args to npu.

    Model init code that touches parameter DATA (weight loading, random init)
    must no-op when the params are meta -- see the ``is_meta`` guards in
    ``DSV4DSparkDraftModel.load_verifier_weights`` / ``_init_backbone_params``.
    """
    orig = nn.Module.register_parameter

    def _register_on_meta(self, name, param):
        if param is not None and not param.is_meta:
            param = nn.Parameter(param.data.to("meta"), requires_grad=param.requires_grad)
        orig(self, name, param)

    nn.Module.register_parameter = _register_on_meta
    try:
        yield
    finally:
        nn.Module.register_parameter = orig


def shard_experts_as_dtensor(model: torch.nn.Module, mesh) -> None:
    """Convert each GroupedExperts' stacked weights to ``Shard(0)`` DTensors on ``mesh``.

    Under EP each rank already holds only its local ``[n_local, ...]`` slice; wrap it as
    ``DTensor.from_local(..., [Shard(0)])`` so the global param is ``[E, ...]`` sharded on
    the expert dim. The experts are then uniform DTensors (like the FSDP-sharded rest, on
    the SAME mesh) -> the optimizer / clip / DCP checkpoint need no plain-vs-DTensor special
    casing. FSDP is told to ignore them (see fsdp_ignored_params); the MoE reads ``.to_local()``
    and moves tokens with all-to-all instead of an FSDP all-gather.
    """
    from torch.distributed.tensor import DTensor, Shard  # noqa: PLC0415

    for module in model.modules():
        if type(module).__name__ != "GroupedExperts":
            continue
        for name in ("w1", "w2", "w3"):
            p = getattr(module, name)
            if isinstance(p.data, DTensor):
                continue
            dt = DTensor.from_local(p.data, mesh, [Shard(0)], run_check=False)
            setattr(module, name, torch.nn.Parameter(dt, requires_grad=p.requires_grad))


def apply_fully_sharded(model: torch.nn.Module, mesh=None):
    """Applies torch FSDP fully_shard to the model, wrapping layers in FSDPModule.

    A model may expose ``fsdp_wrap_plan() -> list[nn.Module]`` to declare the FSDP unit
    granularity (children before parents), and ``fsdp_ignored_params()`` to keep some
    params OUT of FSDP entirely (expert-parallel routed experts: rank-local Shard(0)
    DTensors moved by the MoE all-to-all, not FSDP all-gather). ``mesh`` (when given) is
    the DeviceMesh every fully_shard uses -- pass the SAME mesh the experts were sharded on
    (:func:`shard_experts_as_dtensor`) so expert and non-expert DTensors live on one mesh.

    Model should be validated with SpeculatorModel.verify_training_compatible()
    before calling this function.
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    plan = getattr(model, "fsdp_wrap_plan", None)
    modules = plan() if callable(plan) else list(model.layers)  # type: ignore[union-attr]

    ig = getattr(model, "fsdp_ignored_params", None)
    ignored_params = (ig() if callable(ig) else None) or None
    extra = {"mp_policy": mp_policy}
    if ignored_params:
        extra["ignored_params"] = ignored_params  # torch FSDP2 >=2.5
    if mesh is not None:
        extra["mesh"] = mesh

    for module in modules:
        fully_shard(module, **extra)

    fully_shard(model, **extra)

    return model
