"""Process-wide expert-parallel context.

Expert parallelism partitions an MoE's routed experts across ranks -- whole experts per
rank, no replication -- and moves tokens to their expert's owner rather than gathering
every expert onto every rank. Which experts a rank owns has to be settled before the
model is built, because the rank only ever allocates its own slice, so the context is
installed once at startup and read by the MoE constructor.

It is deliberately process-wide rather than a model attribute: a run either is
expert-parallel or is not, and every MoE layer in it has to agree.

The context carries the parallelism only. How many experts that works out to per rank
is the model's own number, so the model divides it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.distributed as dist

__all__ = [
    "ExpertParallelContext",
    "configure",
    "configure_from_world",
    "context",
    "is_active",
    "reset",
]


@dataclass(frozen=True)
class ExpertParallelContext:
    """The expert-parallel group this process belongs to."""

    group: object | None
    rank: int
    size: int


_CONTEXT: ExpertParallelContext | None = None


def configure(group: object | None, rank: int, size: int) -> None:
    """Install the expert-parallel context for this process."""
    global _CONTEXT  # noqa: PLW0603 - one context per process, by design
    if size < 1:
        raise ValueError(f"expert-parallel size must be >= 1, got {size}")
    if not 0 <= rank < size:
        raise ValueError(f"expert-parallel rank {rank} is outside [0, {size})")
    _CONTEXT = ExpertParallelContext(group=group, rank=rank, size=size)


def configure_from_world() -> ExpertParallelContext:
    """Install a context spanning the whole distributed world.

    A narrower group -- expert parallelism inside a node, data parallelism across
    nodes -- is a matter of passing a different group to :func:`configure`; nothing
    below this reads the world size again.
    """
    if not dist.is_initialized():
        raise RuntimeError(
            "expert parallelism requires distributed training; launch with torchrun."
        )
    configure(dist.group.WORLD, dist.get_rank(), dist.get_world_size())
    return ExpertParallelContext(
        group=dist.group.WORLD, rank=dist.get_rank(), size=dist.get_world_size()
    )


def context() -> ExpertParallelContext | None:
    """The installed context, or None when this run is not expert-parallel."""
    return _CONTEXT


def is_active() -> bool:
    """True when the routed experts are partitioned across more than one rank."""
    return _CONTEXT is not None and _CONTEXT.size > 1


def reset() -> None:
    """Drop the context. For tests, and for a process that reconfigures."""
    global _CONTEXT  # noqa: PLW0603 - see configure
    _CONTEXT = None
