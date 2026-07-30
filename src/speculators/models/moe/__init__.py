"""Generic sparse Mixture-of-Experts (MoE) draft FFN + expert-parallel dispatch.

A reusable MoE layer for speculator drafts whose verifier is itself an MoE model
(so the draft's distribution should match a sparse FFN). The routed experts are
stacked weights (:class:`GroupedExperts`), which is the layout that lets the
optimizer / checkpoint see uniform ``Shard(0)`` DTensors and lets a large expert
count train expert-parallel (:mod:`dispatch_ep`, with the training-side setup in
:mod:`speculators.train.expert_parallel`).

Correctness is device-agnostic: the layer runs on CPU / CUDA / any accelerator
with the pure-torch reference dispatch; faster kernels register opt-in under the
same op key (:mod:`backend`).
"""

from __future__ import annotations

from .config import MoEConfig
from .experts import Expert, GroupedExperts, enable_cuda_grouped_mm
from .layer import MoE
from .router import Router

__all__ = [
    "Expert",
    "GroupedExperts",
    "MoE",
    "MoEConfig",
    "Router",
    "enable_cuda_grouped_mm",
]
