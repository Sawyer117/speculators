"""Rotary position embedding for the DSV4 DSpark draft (interleaved, YaRN-capable).

Clean-room reproduction of the reference RoPE:

* :func:`precompute_freqs_cis` builds complex exponentials ``e^{i·t·θ_k}`` with
  optional YaRN frequency interpolation (a smooth linear ramp between the
  ``beta_fast`` / ``beta_slow`` correction dims). The draft's sliding-window
  attention runs YaRN **off** (pass ``original_seq_len=0``) with the base
  ``rope_theta`` — matching the reference, which disables YaRN on the pure
  sliding path.
* :func:`apply_rotary_emb` rotates the trailing ``rope_head_dim`` slice of a
  ``[..., D]`` tensor using the **interleaved** pairing ``(x0,x1),(x2,x3),…``.
  ``inverse=True`` conjugates the rotation to de-rotate the attention output's
  rope slice (needed because DSV4 shares K=V, so V carried the rotation).

The result is applied out-of-place (returns a new tensor) rather than the
reference's in-place ``copy_`` so it composes cleanly with autograd.
"""
from __future__ import annotations

import math
from functools import lru_cache

import torch


@lru_cache(maxsize=4)
def precompute_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: float,
    beta_slow: float,
    device: str = "cpu",
) -> torch.Tensor:
    """Complex rotary frequencies ``[seqlen, dim//2]`` with optional YaRN.

    ``dim`` is the rope slice width (``rope_head_dim``). With
    ``original_seq_len == 0`` YaRN is disabled and plain ``1/base^(2k/dim)``
    frequencies are used (the draft's sliding-window path).
    """

    def correction_dim(num_rotations: float) -> float:
        return dim * math.log(original_seq_len / (num_rotations * 2 * math.pi)) / (
            2 * math.log(base)
        )

    def correction_range(low_rot: float, high_rot: float) -> tuple[int, int]:
        low = math.floor(correction_dim(low_rot))
        high = math.ceil(correction_dim(high_rot))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp(lo: float, hi: float, n: int) -> torch.Tensor:
        if lo == hi:
            hi += 0.001
        ramp = (torch.arange(n, dtype=torch.float32, device=device) - lo) / (hi - lo)
        return torch.clamp(ramp, 0, 1)

    # ★ device=device (default "cpu") on every arange so freqs_cis is built as a REAL tensor even under
    # transformers `from_pretrained`'s ambient `with torch.device("meta")` init context (used by the
    # --from-pretrained warm-start path). Without it the arange inherits the meta default → freqs_cis is
    # a meta tensor → `.to(device)` below fails "Cannot copy out of meta tensor; no data!". No effect on
    # the normal path (torch's default device is already cpu there).
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    if original_seq_len > 0:
        low, high = correction_range(beta_fast, beta_slow)
        smooth = 1 - linear_ramp(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen, dtype=torch.float32, device=device)
    angles = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    return freqs_cis.to(device)


def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Rotate ``x`` (``[B, S, ..., rope_dim]``) by ``freqs_cis`` (``[S, rope_dim//2]``).

    Interleaved pairing; out-of-place. ``inverse`` conjugates (de-rotation).
    """
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    # Broadcast freqs over batch + any head dims: shape [1, S, (1,)*extra, rope//2].
    seq_len = xc.shape[1]
    extra = xc.ndim - 2  # dims between the seq axis and the rope axis
    view_shape = (1, seq_len, *([1] * (extra - 1)), xc.shape[-1]) if extra >= 1 else (1, seq_len, xc.shape[-1])
    freqs_cis = freqs_cis.view(*view_shape)
    rotated = torch.view_as_real(xc * freqs_cis).flatten(-2)
    return rotated.to(x.dtype)
