#!/usr/bin/env python3
"""Eager-fp32 parity + fwd/bwd smoke for the faithful DSV4 DSpark draft.

Stage-1 correctness gate for ``speculators/models/dsv4_dspark`` (run BEFORE any
NPU kernel is inserted). Three parts:

  A. mHC Sinkhorn parity — our torch reference vs a transcription of the OFFICIAL
     tilelang kernel (`DeepSeek-V4-Flash-DSpark/inference/kernel.py::hc_split_sinkhorn`).
     Registered as the ``official`` backend and compared to ``torch`` through the
     same dispatch the NPU bridge uses, so this also exercises kernel insertion.
  B. Full-draft fwd/bwd smoke — build the small config, run the teacher-forced
     block-gamma forward, check shapes / finiteness, and that ``loss.backward()``
     populates grads on every trainable param while the shared embed + lm_head
     stay frozen.
  C. (next) Full-draft numerical parity vs a de-kernelized official ``forward_spec``.

Runs on CPU (no NPU needed) in an env with torch. fp32 throughout — this gates
the MATH, not precision. Usage::

    python dsv4_dspark_parity.py
"""
from __future__ import annotations

import sys

import torch

from speculators.models.dsv4_dspark import DSparkDraftConfig, DSparkDraftModel
from speculators.models.dsv4_dspark.backbone import kernels
from speculators.models.dsv4_dspark.backbone.hyper import HyperConnection

torch.manual_seed(0)
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


# --------------------------------------------------------------------------- #
# A. mHC Sinkhorn: torch reference vs a transcription of the official kernel.
# --------------------------------------------------------------------------- #
def official_hc_split_sinkhorn(module: HyperConnection, streams: torch.Tensor):
    """Transcribed 1:1 from the official tilelang ``hc_split_sinkhorn_kernel_``
    (kernel.py:378-425). Shares the mixes computation with our reference; only
    the split + Sinkhorn is the object of comparison.
    """
    hc, eps = module.hc_mult, module.hc_eps
    flat = module.input_norm(streams.flatten(start_dim=2).float())
    mixes = torch.nn.functional.linear(flat, module.fn.float())  # [..., (2+hc)*hc]
    scale, base = module.scale.float(), module.base.float()

    pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = mixes[..., 2 * hc :].view(*mixes.shape[:-1], hc, hc) * scale[2] + base[2 * hc :].view(hc, hc)

    # comb = softmax(-1) + eps
    comb = comb - comb.max(dim=-1, keepdim=True).values
    comb = torch.exp(comb)
    comb = comb / comb.sum(dim=-1, keepdim=True) + eps
    # comb = comb / (comb.sum(-2) + eps)
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(module.hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    collapsed = (pre.unsqueeze(-1) * streams).sum(dim=2).to(streams.dtype)
    return post, comb, collapsed


def part_a() -> None:
    print("A. mHC Sinkhorn parity (our torch vs official kernel transcription)")
    kernels.register_kernel("mhc_hyper_connection", "official", official_hc_split_sinkhorn)

    cfg = DSparkDraftConfig().small()
    hc = HyperConnection(cfg).float()
    torch.nn.init.normal_(hc.fn, std=0.02)
    torch.nn.init.normal_(hc.base, std=0.02)
    torch.nn.init.normal_(hc.scale, std=0.02)
    streams = torch.randn(2, 7, cfg.hc_mult, cfg.hidden_size)

    post_t, comb_t, coll_t = hc(streams, backend="torch")
    post_o, comb_o, coll_o = hc(streams, backend="official")

    # Both paths run the identical fp32 math (ours uses torch.softmax, the
    # official transcription writes exp/sum by hand) — agreement to the fp32
    # softmax noise floor confirms our sinkhorn == the official kernel.
    for label, a, b in [("post", post_t, post_o), ("comb", comb_t, comb_o), ("collapsed", coll_t, coll_o)]:
        d = (a - b).abs().max().item()
        check(f"sinkhorn {label} == official", d < 1e-5, f"max_abs={d:.2e}")
    # comb must be (approximately) doubly stochastic after Sinkhorn.
    row = comb_t.sum(-1)
    col = comb_t.sum(-2)
    check("comb rows ~1", (row - 1).abs().max().item() < 1e-3, f"max|row-1|={(row - 1).abs().max().item():.2e}")
    check("comb cols ~1", (col - 1).abs().max().item() < 1e-3, f"max|col-1|={(col - 1).abs().max().item():.2e}")


# --------------------------------------------------------------------------- #
# B. Full-draft fwd/bwd smoke.
# --------------------------------------------------------------------------- #
def part_b() -> None:
    print("B. Full-draft fwd/bwd smoke (small config, fp32)")
    cfg = DSparkDraftConfig().small()
    model = DSparkDraftModel(cfg).float()
    # sane inits so random empty-() params don't NaN
    for p in model.parameters():
        if p.dim() >= 2:
            torch.nn.init.normal_(p, std=0.02)
    model.freeze_target_weights()

    n, w, g = 3, cfg.window_size, cfg.block_size
    ctx = torch.randn(n, w, cfg.hidden_size * cfg.num_target_layers)
    block_ids = torch.randint(0, cfg.vocab_size, (n, g))
    markov_ids = torch.randint(0, cfg.vocab_size, (n, g))

    logits, conf, hidden = model(ctx, block_ids, markov_ids)
    check("logits shape", tuple(logits.shape) == (n, g, cfg.vocab_size), str(tuple(logits.shape)))
    check("confidence shape", tuple(conf.shape) == (n, g), str(tuple(conf.shape)))
    check("outputs finite", torch.isfinite(logits).all().item() and torch.isfinite(conf).all().item())

    # A stand-in loss (real loss lives in loss.py); just exercise autograd.
    loss = logits.float().log_softmax(-1).mean() + conf.float().mean()
    loss.backward()

    trained = [n_ for n_, p in model.named_parameters() if p.requires_grad]
    no_grad = [n_ for n_, p in model.named_parameters() if p.requires_grad and p.grad is None]
    check("all trainable params got grad", not no_grad, f"missing: {no_grad[:4]}")
    check("embed frozen", not model.embed_tokens.weight.requires_grad)
    check("lm_head frozen", not model.lm_head.weight.requires_grad)
    print(f"      trainable tensors: {len(trained)}  |  params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


def main() -> int:
    print(f"torch {torch.__version__}\n")
    part_a()
    print()
    part_b()
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        return 1
    print("ALL PASS — Stage-1 A+B. Next: C (vs de-kernelized official forward_spec) + NPU precision.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
