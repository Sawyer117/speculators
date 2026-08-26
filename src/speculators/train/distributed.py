"""Single source of truth for distributed training topology.

Stores local_rank, SP/DP sizes and ranks, and process groups.
All other modules should import getters from here rather than
maintaining their own distributed state.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist
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
    dist.init_process_group(backend, device_id=local_rank)

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


def rank_local_param_keys(model: torch.nn.Module) -> set[str]:
    """``state_dict`` names of the parameters this model keeps rank-local.

    A model declares them by defining ``ep_local_param_keys() -> set[str]``. Under
    expert parallelism the routed experts are partitioned, not replicated: each rank
    builds and owns a disjoint slice, so those names must be skipped by the rank-0
    broadcast, sharded explicitly rather than by FSDP, and kept out of FSDP entirely.
    A model that does not define the hook has no rank-local parameters.
    """
    hook = getattr(model, "ep_local_param_keys", None)
    return set(hook()) if callable(hook) else set()


def shard_rank_local_params(model: torch.nn.Module, mesh, param_keys: set[str]) -> None:
    """Wrap each rank-local parameter as a ``Shard(0)`` DTensor on ``mesh``.

    The rank already holds only its slice, so this states the global shape rather than
    moving any data. The point is uniformity: with the slice wrapped on the same mesh
    FSDP shards the rest over, every parameter in the model is a DTensor, and the
    optimizer, gradient clipping and distributed checkpointing need no
    plain-tensor-versus-DTensor branch. The model's forward reads ``.to_local()`` and
    moves activations itself instead of relying on an FSDP all-gather.
    """
    from torch.distributed.tensor import DTensor, Shard  # noqa: PLC0415

    for key in sorted(param_keys):
        module_path, _, attr = key.rpartition(".")
        module = model.get_submodule(module_path)
        param = getattr(module, attr)
        if isinstance(param.data, DTensor):
            continue
        local = DTensor.from_local(param.data, mesh, [Shard(0)], run_check=False)
        setattr(
            module, attr, torch.nn.Parameter(local, requires_grad=param.requires_grad)
        )


def apply_fully_sharded(
    model: torch.nn.Module,
    param_dtype: torch.dtype = torch.bfloat16,
    mesh=None,
):
    """Applies torch FSDP fully_shard to the model, wrapping layers in FSDPModule.

    Assumes the model has a `layers` attribute containing the decoder layers, unless it
    defines ``fsdp_wrap_plan() -> list[nn.Module]`` to declare the unit granularity
    itself (children before parents).

    Parameters the model declares rank-local (:func:`rank_local_param_keys`) are kept
    out of FSDP: they are already sharded, on their own axis. ``mesh``, when given, is
    the ``DeviceMesh`` every ``fully_shard`` call uses — pass the same one they were
    sharded on, so expert and non-expert parameters live on a single mesh.

    Model should be validated with SpeculatorModel.verify_training_compatible()
    before calling this function.
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=torch.float32,
    )

    plan = getattr(model, "fsdp_wrap_plan", None)
    modules = plan() if callable(plan) else list(model.layers)  # type: ignore[union-attr,arg-type]

    shard_kwargs: dict = {"mp_policy": mp_policy}
    if mesh is not None:
        shard_kwargs["mesh"] = mesh
    ignored = {model.get_parameter(k) for k in rank_local_param_keys(model)}
    if ignored:
        shard_kwargs["ignored_params"] = ignored

    for module in modules:
        fully_shard(module, **shard_kwargs)

    fully_shard(model, **shard_kwargs)

    return model
