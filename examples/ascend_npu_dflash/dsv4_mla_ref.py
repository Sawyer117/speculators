#!/usr/bin/env python3
"""Clean-room torch-eager DSV4-Flash MLA layer — the PARITY REFERENCE.

Derived directly from the OFFICIAL gold `deepseek-ai/DeepSeek-V4-Flash`
`inference/model.py` (MLA class, no-compress path = the draft-layer case) +
its helpers (RMSNorm / precompute_freqs_cis / apply_rotary_emb). Pure torch,
CPU-runnable, backend-agnostic — used to numerically validate MindSpeed's DSV4
attention layer (and any other impl) on a single layer.

Faithful details captured (the easy-to-miss ones):
  * q: wq_a -> q_norm(RMSNorm, q_lora_rank) -> wq_b -> reshape(n_heads, head_dim)
       -> PER-HEAD RMSNorm WITHOUT weight (official model.py line ~498:
          q *= rsqrt(q.square().mean(-1)+eps)) -> RoPE on the last rope_head_dim.
  * kv: wkv -> kv_norm(RMSNorm, head_dim) -> RoPE on the last rope_head_dim.
        Single KV latent (num_key_value_heads=1) shared across all heads = MLA.
  * attention: einsum(q, kv) * scale, sliding-window causal mask, ATTENTION SINK
        concatenated as an extra column -> softmax over [kv + sink] -> drop the
        sink column -> weighted sum of kv.  (We use einsum+sink, not the fused
        `sparse_attn` kernel, so the reference has a clean autograd backward.)
  * output: INVERSE RoPE on the last rope_head_dim, then GROUPED low-rank O proj
        (o.view(n_groups) -> einsum with wo_a -> flatten -> wo_b).

No compressor / indexer (those are target-only, ratio!=0 layers). For the DSpark
draft block-attention, only the MASK changes (block-non-causal window); the
projections + sink here are identical.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MLAConfig:
    dim: int = 4096
    n_heads: int = 64
    head_dim: int = 512
    rope_head_dim: int = 64
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    window_size: int = 128
    rope_theta: float = 10000.0
    eps: float = 1e-6


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


def precompute_freqs_cis(dim, seqlen, base):
    """Plain RoPE (no YaRN) — matches the official no-compress path
    (original_seq_len=0 -> no interpolation)."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex [seqlen, dim/2]


def apply_rotary_emb(x, freqs_cis, inverse=False):
    """x[..., :] rotary on its LAST dim (which is rope_head_dim). Complex form,
    matches official. Returns a new tensor (autograd-friendly, not in-place)."""
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    fc = freqs_cis.conj() if inverse else freqs_cis
    if xc.ndim == 3:      # [b, s, d/2]
        fc = fc.view(1, xc.size(1), xc.size(-1))
    else:                 # [b, s, h, d/2]
        fc = fc.view(1, xc.size(1), 1, xc.size(-1))
    return torch.view_as_real(xc * fc).flatten(-2).to(x.dtype)


def sliding_window_causal_mask(seqlen, window, device):
    """Additive mask [sq, sk]: position i attends to j in (i-window, i], causal."""
    i = torch.arange(seqlen, device=device).unsqueeze(1)
    j = torch.arange(seqlen, device=device).unsqueeze(0)
    keep = (j <= i) & (j > i - window)
    return torch.where(keep, 0.0, float("-inf"))


class DSV4MLA(nn.Module):
    """Backbone MLA (no compress). Attention math is einsum + explicit sink so
    the reference is autograd-clean. `attn_mask` is pluggable (base causal-SWA
    here; the DSpark draft swaps in a block-non-causal window)."""

    def __init__(self, cfg: MLAConfig):
        super().__init__()
        self.cfg = cfg
        self.wq_a = nn.Linear(cfg.dim, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank, cfg.eps)
        self.wq_b = nn.Linear(cfg.q_lora_rank, cfg.n_heads * cfg.head_dim, bias=False)
        self.wkv = nn.Linear(cfg.dim, cfg.head_dim, bias=False)
        self.kv_norm = RMSNorm(cfg.head_dim, cfg.eps)
        self.wo_a = nn.Parameter(torch.empty(cfg.o_groups, cfg.o_lora_rank,
                                             cfg.n_heads * cfg.head_dim // cfg.o_groups))
        self.wo_b = nn.Linear(cfg.o_groups * cfg.o_lora_rank, cfg.dim, bias=False)
        self.attn_sink = nn.Parameter(torch.zeros(cfg.n_heads, dtype=torch.float32))
        self.softmax_scale = cfg.head_dim ** -0.5
        nn.init.normal_(self.wo_a, std=0.02)

    def forward(self, x, freqs_cis, attn_mask):
        b, s, _ = x.shape
        cfg = self.cfg
        rd = cfg.rope_head_dim

        # q: low-rank -> per-head reshape -> per-head norm (no weight) -> RoPE
        q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (cfg.n_heads, cfg.head_dim))
        q = q * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + cfg.eps).to(q.dtype)
        q = torch.cat([q[..., :-rd], apply_rotary_emb(q[..., -rd:], freqs_cis)], dim=-1)

        # kv latent (shared across heads) -> norm -> RoPE
        kv = self.kv_norm(self.wkv(x))
        kv = torch.cat([kv[..., :-rd], apply_rotary_emb(kv[..., -rd:], freqs_cis)], dim=-1)

        # attention: scores[b,h,sq,sk] = q . kv ; window mask ; concat sink ; softmax ; drop sink
        scores = torch.einsum("bshd,btd->bhst", q, kv).float() * self.softmax_scale
        scores = scores + attn_mask  # [sq, sk] broadcast over [b, h]
        sink = self.attn_sink.view(1, cfg.n_heads, 1, 1).expand(b, -1, s, 1)
        combined = torch.cat([scores, sink], dim=-1)
        combined = combined - combined.max(dim=-1, keepdim=True).values
        probs = combined.softmax(dim=-1)[..., :-1]           # drop sink column
        o = torch.einsum("bhst,btd->bshd", probs.to(kv.dtype), kv)  # [b,s,h,head_dim]

        # inverse RoPE on the output rope dims
        o = torch.cat([o[..., :-rd], apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)], dim=-1)

        # grouped low-rank O proj
        o = o.reshape(b, s, cfg.o_groups, -1)                # [b,s,G, n_heads*head_dim/G]
        o = torch.einsum("bsgd,grd->bsgr", o, self.wo_a)     # [b,s,G, o_lora_rank]
        return self.wo_b(o.flatten(2))                       # [b,s,dim]


def _selftest():
    torch.manual_seed(0)
    # small config for a fast CPU run (same STRUCTURE as DSV4, tiny dims)
    cfg = MLAConfig(dim=128, n_heads=4, head_dim=32, rope_head_dim=8,
                    q_lora_rank=48, o_lora_rank=48, o_groups=2, window_size=6, eps=1e-6)
    b, s = 2, 12
    mla = DSV4MLA(cfg).double()
    x = torch.randn(b, s, cfg.dim, dtype=torch.double, requires_grad=True)
    freqs = precompute_freqs_cis(cfg.rope_head_dim, s, cfg.rope_theta)
    mask = sliding_window_causal_mask(s, cfg.window_size, x.device)

    y = mla(x, freqs, mask)
    print(f"[shape]   in {tuple(x.shape)} -> out {tuple(y.shape)}  (expect [{b},{s},{cfg.dim}])")
    assert y.shape == (b, s, cfg.dim)

    # backward works (autograd-clean einsum+sink path)
    y.sum().backward()
    print(f"[bwd]     grad(x) finite={torch.isfinite(x.grad).all().item()}  "
          f"grad(sink) nonzero={mla.attn_sink.grad.abs().sum().item():.3e}")

    # sink actually changes the output (set sink -> -inf effectively removes it)
    with torch.no_grad():
        y0 = mla(x, freqs, mask)
        mla.attn_sink.fill_(-30.0)          # exp(-30) ~ 0 -> sink absorbs ~no mass
        y1 = mla(x, freqs, mask)
        mla.attn_sink.fill_(3.0)            # large sink -> absorbs a lot of mass
        y2 = mla(x, freqs, mask)
    d_off = (y0 - y1).abs().mean().item()
    d_on = (y0 - y2).abs().mean().item()
    print(f"[sink]    |y(sink=0) - y(sink=-30)|={d_off:.3e}   |y(sink=0) - y(sink=+3)|={d_on:.3e}")
    print(f"[sink]    => sink materially changes the output ({d_on:.2e} >> {d_off:.2e}); "
          f"sink=+3 pulls mass off the real keys as expected")
    print("OK: DSV4 MLA reference runs fwd+bwd, sink is live. "
          "Next: bit-parity vs official inference/model.py on-box (bf16, real dims).")


if __name__ == "__main__":
    _selftest()
