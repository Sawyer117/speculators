"""Faithful DeepSeek-V4-Flash DSpark draft (Plan 甲, HF-native / Track A).

A self-contained, backend-agnostic reproduction of DeepSeek's DSpark *draft*
for DeepSeek-V4-Flash: a 3-layer semi-autoregressive drafter whose layers mirror
the target decoder shape (multi-head latent attention + attention sink +
256-expert MoE + hyper-connections), plus the DSpark-method parts (``main_proj``
target-hidden conditioning, a low-rank Markov logit-bias head, and a per-position
confidence head). Trained teacher-forced on cached target hidden states.

The port is clean-room: mechanism-descriptive naming, no private-repo or
accelerator imports, GPU/CPU-runnable, with NPU kernels inserted opt-in via
:mod:`.backbone.kernels`. Correctness is parity-anchored to the released
reference; see ``docs/deployment/ascend-npu-dsv4-dspark-landing-plan.md``.
"""
from __future__ import annotations

from .config import DSparkDraftConfig
from .draft import (
    ConfidenceHead,
    DSparkDraftModel,
    MarkovHead,
)

__all__ = [
    "ConfidenceHead",
    "DSparkDraftConfig",
    "DSparkDraftModel",
    "MarkovHead",
]
