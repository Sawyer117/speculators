"""Training-only low-rank prediction heads for intermediate draft layers.

Each head reads mean-pooled mHC streams and predicts the same aligned teacher
argmax as the final head. It does not inject tokens into the next layer.
"""

from __future__ import annotations

import math
import os

import torch
from torch import nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint


def settings_from_env(saved=None):
    """Persist the resolved recipe; an unset environment preserves saved settings."""
    mode = os.environ.get("DSPARK_INTER_HEADS", "").strip().lower()
    if mode == "off":
        return None
    if mode not in ("", "aux"):
        raise ValueError("This branch supports DSPARK_INTER_HEADS=off or aux")
    if not mode and not saved:
        return None
    value = dict(saved or {})
    for key, name, default, cast in (
        ("rank", "DSPARK_INTER_RANK", 256, int),
        ("weight", "DSPARK_INTER_AUX_WEIGHT", 0.2, float),
        ("schedule", "DSPARK_INTER_AUX_SCHEDULE", "const", str),
        ("decay_steps", "DSPARK_INTER_AUX_DECAY_STEPS", 6118, int),
        ("weight_end", "DSPARK_INTER_AUX_WEIGHT_END", 0.08, float),
    ):
        value[key] = cast(os.environ.get(name) or value.get(key, default))
    if value["rank"] < 1 or value["decay_steps"] < 1:
        raise ValueError("Head rank and schedule boundary must be positive")
    if value["schedule"] not in {"const", "step"}:
        raise ValueError("Auxiliary schedule must be const or step")
    if any(
        not math.isfinite(value[k]) or value[k] < 0 for k in ("weight", "weight_end")
    ):
        raise ValueError("Auxiliary weights must be finite and nonnegative")
    return value


def weight_at(settings, step):
    if settings["schedule"] == "step" and step >= settings["decay_steps"]:
        return settings["weight_end"]
    return settings["weight"]


class IntermediateHead(nn.Module):
    """[batch,tokens,mHC,width] -> mean -> RMSNorm -> low-rank code."""

    def __init__(self, hidden_size, rank, vocab_size, eps, index):
        super().__init__()
        self.eps = eps
        self.norm_weight = nn.Parameter(torch.ones(hidden_size))
        self.down = nn.Parameter(torch.empty(rank, hidden_size))
        self.codebook = nn.Parameter(torch.empty(vocab_size, rank))
        generator = torch.Generator().manual_seed(20260928 + index)
        with torch.no_grad():
            if not self.down.is_meta:
                self.down.copy_(
                    torch.randn(rank, hidden_size, generator=generator)
                    * hidden_size**-0.5
                )
                self.codebook.copy_(
                    torch.randn(vocab_size, rank, generator=generator) * rank**-0.5
                )

    def forward(self, streams):
        x = streams.mean(dim=2).flatten(0, 1)
        normalized = x.float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + self.eps
        )
        normalized = normalized.to(x.dtype) * self.norm_weight.to(x.dtype)
        return normalized @ self.down.to(x.dtype).t()


def _chunk_ce(code, codebook, labels):
    logits = (code @ codebook.to(code.dtype).t()).float()
    return functional.cross_entropy(logits, labels, reduction="none")


def chunked_ce(code, codebook, labels, chunk=512):
    """Recompute vocabulary logits during backward instead of retaining all of them."""
    if chunk < 1:
        raise ValueError("CE chunk must be positive")
    if not code.shape[0]:
        return code.sum(-1).float()
    return torch.cat(
        [
            checkpoint(
                _chunk_ce,
                code[i : i + chunk],
                codebook,
                labels[i : i + chunk],
                use_reentrant=False,
            )
            for i in range(0, code.shape[0], chunk)
        ]
    )
