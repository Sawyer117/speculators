"""E11: per-layer low-rank prediction heads, optionally feeding their guess forward.

After each draft layer except the last, a low-rank head reads the layer output and
predicts the slot's target token (auxiliary loss, ``DSPARK_INTER_HEADS=aux``). With
``DSPARK_INTER_HEADS=inject`` the head's argmax for slot k-1 -- its guess of the token
at slot k's own position -- is embedded and added, through a zero-initialized gate, to
slot k's input to the next layer. That gives slot k a committed (discrete) guess of
what came before it at depth, which attention over mask-derived hidden states does not.

Everything stays one parallel forward: the head is 4096 -> r -> vocab (r = 256), about
6% of a full lm_head per layer, and the injected ids are the model's own guesses, so
no target leaks even under the non-causal block mask.

Parameters are initialized from a private generator, seeded per layer, so building
these modules adds no draws to the global RNG: turning the heads on does not move the
anchor sampling or hidden-state noise that follow model construction, and every rank
builds the same weights without a broadcast.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

_INIT_SEED = 20260928


class IntermediateHead(nn.Module):
    """Mean over the mHC streams -> RMSNorm -> down (hidden -> r) -> codebook."""

    def __init__(
        self, hidden_size: int, rank: int, vocab_size: int, eps: float, index: int
    ) -> None:
        super().__init__()
        self.eps = eps
        self.norm_weight = nn.Parameter(torch.ones(hidden_size))
        self.down = nn.Parameter(torch.empty(rank, hidden_size))
        self.codebook = nn.Parameter(torch.empty(vocab_size, rank))
        g = torch.Generator().manual_seed(_INIT_SEED + index)
        with torch.no_grad():
            if not self.down.is_meta:
                down = torch.randn(rank, hidden_size, generator=g) * hidden_size**-0.5
                codebook = torch.randn(vocab_size, rank, generator=g) * rank**-0.5
                self.down.copy_(down)
                self.codebook.copy_(codebook)

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        """``streams [1, T, hc, H]`` -> ``z [T, r]``, the code the codebook scores."""
        x = streams.mean(dim=2)[0]  # [T, H]
        xf = x.float()
        x = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype)
        x = x * self.norm_weight.to(x.dtype)
        return x @ self.down.to(x.dtype).t()


def _chunk_ce(
    z: torch.Tensor, codebook: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    logits = (z @ codebook.to(z.dtype).t()).float()
    return torch.nn.functional.cross_entropy(logits, labels, reduction="none")


def chunked_ce(
    z: torch.Tensor, codebook: torch.Tensor, labels: torch.Tensor, chunk: int = 512
) -> torch.Tensor:
    """Per-token CE of ``z @ codebook.T`` against ``labels``, chunk by chunk.

    Each chunk is checkpointed, so backward recomputes its [chunk, vocab] logits
    instead of keeping them: the full fp32 tensor would be ~1.5 GB per layer at
    2,880 slots (192 anchors x 15).
    """
    out = [
        checkpoint(
            _chunk_ce,
            z[i : i + chunk],
            codebook,
            labels[i : i + chunk],
            use_reentrant=False,
        )
        for i in range(0, z.shape[0], chunk)
    ]
    return torch.cat(out)


@torch.no_grad()
def chunked_argmax(
    z: torch.Tensor, codebook: torch.Tensor, chunk: int = 512
) -> torch.Tensor:
    cb = codebook.to(z.dtype)
    return torch.cat(
        [(z[i : i + chunk] @ cb.t()).argmax(-1) for i in range(0, z.shape[0], chunk)]
    )


def shift_within_blocks(
    ids: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slot k receives slot k-1's id; slot 0 of every block receives nothing.

    Returns ``(shifted [T], valid [T] bool)``. Blocks never leak into one another.
    """
    blocks = ids.view(-1, block_size)
    shifted = torch.zeros_like(blocks)
    shifted[:, 1:] = blocks[:, :-1]
    valid = torch.ones_like(blocks, dtype=torch.bool)
    valid[:, 0] = False
    return shifted.reshape(-1), valid.reshape(-1)
