"""Two-tap dynamic causal convolution over BLOCK POSITIONS, wrapped around each sublayer.

Transcribed from the DFlash2 reference implementation, ``z-lab/dflash``
(``dflash/model.py::_grouped_dynamic_convolve`` / ``GroupedDynamicCausalConv``, Apache-2.0),
rather than from the third-party port, so the semantics come from the authors.

WHAT IT IS FOR. The official model card states the purpose directly: *"two-tap dynamic
convolutions in the backbone keep the draft from decaying toward the end of the block."* A
block drafter emits all K positions from one forward pass, so position t's hidden state was
produced without knowing anything the draft chose at t-1. Acceptance therefore decays along
the block -- our own run reads p1 0.688 down to p5 0.282, and even the released DeepSeek
draft is at 51% by position 7. This mixes each position's hidden state with its predecessor's
*inside the residual flow*, giving the backbone a directional, block-local prior that
non-causal block attention cannot express.

★ WHY THIS AND NOT THE SELECT HEAD'S KIND OF FIX. This runs entirely inside the
block-parallel forward and never looks at a chosen token, so at serve it costs nothing beyond
its own arithmetic. The Markov/select terms are sequential: each step pays a V-wide GEMV
(66 MB of codebook traffic per position at batch 1). Two mechanisms, very different price.

SHAPE. Operates on ``[N, gamma, dim]`` -- N blocks, gamma block positions, hidden -- which is
exactly what a sublayer input already looks like here, so no reshaping and no masking are
needed: blocks are separated by the batch dimension. (The vllm-ascend serve port expresses the
same thing on a FLATTENED ``[T, dim]`` tensor and must mask with ``position % block_size`` to
stop one block reading the previous block's tail. Equivalent; different runtime layout. Any
serve-side implementation of this MUST use the masked form.)

PLACEMENT. Mirrors the reference line for line. The reference wraps each sublayer between its
input norm and the residual add:

    residual = h;  h = input_layernorm(h)
    h, k = conv.prepare(h);  h = sublayer(h);  h = conv.finish(h, k)
    h = residual + h

Ours differs only in that the residual is mHC rather than a plain add, so "produce the
sublayer input" is ``attn_hc`` + ``attn_norm`` and "fold back" is ``place``. The convolution
sits between them, i.e. AFTER the norm:

    residual = streams
    post, comb, x = attn_hc(streams);  x = attn_norm(x)
    x, k = attn_conv.prepare(x);  x = attn(x, ...);  x = attn_conv.finish(x, k)
    streams = place(x, residual, post, comb)

⚠ One consequence of the mHC difference: ``place`` scales the sublayer output by ``post``
before folding it into the streams, where the reference adds it unscaled. Harmless at identity
init, but it means this module's output is rescaled downstream -- worth remembering if its
gradients look small.

INITIALISATION. The reference leaves ``base_kernel`` uninitialised because it loads trained
weights. We train from scratch, so it starts as an EXACT identity: the zero-lag static
coefficient is 1, every other tap is 0, and the dynamic projection is zeroed. Step 0 is then
bit-identical to a run without the convolution, which is what makes a paired A/B readable.
Gradient still reaches both (d out / d coeff = the tap's values, which are nonzero), so they
learn from step 1.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def grouped_dynamic_convolve(
    hidden: torch.Tensor,      # [N, gamma, dim]
    dynamic: torch.Tensor,     # [N, gamma, taps, groups]  -- per-token, per-group
    base: torch.Tensor,        # [taps, dim]               -- static, per-channel
    group_size: int,
) -> torch.Tensor:
    """``sum_tap (base[tap] + dynamic[tap]) * shift(hidden, tap)``, causal within the block.

    Transcribed from ``z-lab/dflash``. The static term is per-channel and the dynamic term is
    per-group (``group_size`` channels share one coefficient), which is what keeps the
    projection that produces it affordable.
    """
    batch, length, hidden_size = hidden.shape
    groups = hidden_size // group_size
    taps = base.shape[0]
    blocks = hidden.view(batch, length, groups, group_size)
    dyn = dynamic.reshape(batch, length, taps, groups, 1)
    out = torch.zeros_like(blocks)
    for offset in range(taps):
        # Shift along the BLOCK-POSITION axis and pad at the front: position t reads t-offset,
        # and the first `offset` positions of every block read zero. Blocks cannot leak into
        # one another because they are separate batch rows.
        values = blocks if offset == 0 else F.pad(blocks[:, :-offset], (0, 0, 0, 0, offset, 0))
        kernel = base[offset].view(1, 1, groups, group_size).to(hidden.dtype)
        out = out + kernel * values
        out = torch.addcmul(out, dyn[:, :, offset], values)
    return out.view_as(hidden)


class GroupedDynamicCausalConv(nn.Module):
    """One sublayer's pair of convolutions: ``prepare`` before it, ``finish`` after it.

    Both coefficient sets come from a SINGLE projection evaluated in ``prepare``; ``finish``
    reuses the half it was handed. That is the reference's design and it matters: the
    projection is the expensive part (``dim x 2*taps*groups``), and evaluating it once per
    sublayer rather than twice halves it.
    """

    def __init__(self, hidden_size: int, kernel_size: int, group_size: int) -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(f"hidden_size {hidden_size} not divisible by group_size {group_size}")
        self.kernel_size = kernel_size
        self.group_size = group_size
        groups = hidden_size // group_size
        # Leading 2 = the two sides: [0] used by prepare, [1] by finish.
        self.base_kernel = nn.Parameter(torch.zeros(2, kernel_size, hidden_size))
        self.kernel_projection = nn.Linear(hidden_size, 2 * kernel_size * groups, bias=False)
        with torch.no_grad():
            self.base_kernel[:, 0, :] = 1.0      # zero-lag coefficient -> identity
            self.kernel_projection.weight.zero_()

    def prepare(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        groups = hidden.shape[-1] // self.group_size
        dynamic = self.kernel_projection(hidden).view(
            *hidden.shape[:-1], 2, self.kernel_size, groups
        )
        return (
            grouped_dynamic_convolve(
                hidden, dynamic[..., 0, :, :], self.base_kernel[0], self.group_size
            ),
            dynamic[..., 1, :, :],
        )

    def finish(self, hidden: torch.Tensor, dynamic: torch.Tensor) -> torch.Tensor:
        return grouped_dynamic_convolve(
            hidden, dynamic, self.base_kernel[1], self.group_size
        )
