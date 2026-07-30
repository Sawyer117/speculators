"""Training-side setup for expert-parallel (EP) MoE drafts.

Wires the model-side EP dispatch (:mod:`speculators.models.moe.dispatch_ep`) into
a training run:

* :func:`setup_expert_parallel` — build the EP process group (backend derived from
  the current accelerator, never hardcoded) and install the EP context, so that
  each :class:`~speculators.models.moe.layer.MoE` built afterwards holds only its
  disjoint slice of whole experts.
* :func:`shard_experts` — wrap each rank's local stacked expert weights as
  ``Shard(0)`` DTensors over the EP mesh, so the optimizer / grad-clip /
  checkpoint see one uniform global tensor per projection.
* :func:`update_moe_load_balance` — apply the per-step noaux_tc balancing update
  across all MoE layers (call once per optimizer step).
* :func:`full_expert_weights` — gather the sharded experts back to full stacked
  tensors for saving a dense checkpoint.

Device-agnostic: the process-group backend comes from
``torch.distributed.get_default_backend_for_device`` and the device type from
``torch.accelerator`` (via :mod:`speculators.utils.util`), so the same code runs
on CUDA (NCCL), Ascend NPU (HCCL), or CPU (gloo).
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from speculators.models.moe.dispatch_ep import EPContext, configure, enable
from speculators.models.moe.layer import MoE


def _accelerator_device_type() -> str:
    """The current accelerator's device type ('cuda' / 'npu' / ...), or 'cpu'."""
    acc = torch.accelerator.current_accelerator()
    return acc.type if acc is not None else "cpu"


def setup_expert_parallel(
    ep_size: int, n_routed_experts: int, device_type: str | None = None
) -> tuple[EPContext, DeviceMesh]:
    """Create the EP process group + mesh and install the EP context.

    Must be called AFTER ``torch.distributed`` is initialized and BEFORE the MoE
    model is built (the MoE reads the context to size its local experts). For this
    minimal setup EP spans the whole world, so ``ep_size`` must equal the world
    size and divide ``n_routed_experts`` evenly.
    """
    if not dist.is_initialized():
        raise RuntimeError("init the default process group before setup_expert_parallel()")
    world = dist.get_world_size()
    if ep_size != world:
        raise ValueError(f"minimal EP spans the world: ep_size ({ep_size}) must == world ({world})")
    if n_routed_experts % ep_size != 0:
        raise ValueError(f"n_routed_experts ({n_routed_experts}) must be divisible by ep_size ({ep_size})")

    device_type = device_type or _accelerator_device_type()
    mesh = init_device_mesh(device_type, (ep_size,), mesh_dim_names=("ep",))
    group = mesh.get_group("ep")
    rank = dist.get_rank(group)
    experts_per_rank = n_routed_experts // ep_size
    configure(group, rank, ep_size, experts_per_rank)
    enable()  # flip the active MoE backend to expert-parallel dispatch
    return EPContext(group=group, rank=rank, size=ep_size, experts_per_rank=experts_per_rank), mesh


def shard_experts(model: torch.nn.Module, mesh: DeviceMesh) -> None:
    """Wrap every MoE's local stacked experts as ``Shard(0)`` DTensors over ``mesh``.

    Each rank already holds its disjoint local slice (``[n_local, ...]``), so the
    weights are promoted with ``DTensor.from_local`` (no re-scatter): the global
    logical tensor is ``[n_routed_experts, ...]`` sharded on dim 0.
    """
    ep_mesh = mesh["ep"] if "ep" in mesh.mesh_dim_names else mesh
    for module in model.modules():
        if not isinstance(module, MoE):
            continue
        experts = module.experts
        for name in ("w1", "w3", "w2"):
            local = getattr(experts, name)
            if isinstance(local.data, DTensor):
                continue
            dt = DTensor.from_local(local.data, ep_mesh, [Shard(0)], run_check=False)
            setattr(experts, name, torch.nn.Parameter(dt, requires_grad=local.requires_grad))


def update_moe_load_balance(model: torch.nn.Module) -> None:
    """Apply one noaux_tc balancing step across all MoE layers (once per optimizer
    step). No-op for layers with load balancing off."""
    for module in model.modules():
        if isinstance(module, MoE):
            module.update_load_balance_bias()


def full_expert_weights(
    experts: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather sharded experts to full stacked tensors ``(w1, w3, w2)`` for a dense
    save. Works whether or not the experts are DTensors."""

    def full(p: torch.Tensor) -> torch.Tensor:
        return p.full_tensor() if isinstance(p, DTensor) else p

    return full(experts.w1.data), full(experts.w3.data), full(experts.w2.data)
