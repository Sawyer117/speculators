#!/usr/bin/env python3
"""Is the rank-256 Markov head SATURATED? -- decide before spending a run on `gated`.

    python3 examples/ascend_npu_dflash/markov_spectrum_probe.py <weights_dir_or_file> [...]

WHY THIS EXISTS
---------------
`markov_head_type="gated"` multiplies the predecessor embedding by a hidden-conditioned
sigmoid gate before the same `markov_w2` projection. It therefore adds NO rank: the head
stays a rank-`r` approximation of a V x V transition matrix (V=129,280, r=256 -- a ~250x
compression). What the gate buys is a CONTEXT-DEPENDENT MIXTURE of rank-<=r matrices that
all share one r-dimensional basis.

So the gate can only help if that basis has slack. If the r dimensions are already fully
spent on unconditional bigram statistics, gating cannibalises: it trades bigram fidelity
for context sensitivity and can net out negative. That question is answerable from the
trained weights alone -- no training, no serving, no eval machine.

WHAT IT REPORTS
    participation ratio  (sum s^2)^2 / sum s^4  -- "how many dimensions are really in use",
                         robust to the long tail in a way a hard energy threshold is not.
    energy-coverage rank how many dimensions carry 50/80/90/95/99% of the spectral energy.

For the TRANSITION MATRIX itself (`markov_w2 @ markov_w1.T`, V x V and far too large to
form) the singular values are obtained exactly via thin QR:
    w1 = Q1 R1, w2 = Q2 R2  =>  w2 w1^T = Q2 (R2 R1^T) Q1^T
so svdvals(w2 w1^T) == svdvals(R2 R1^T), an r x r problem. That composed spectrum is the
one that matters -- w1 and w2 individually can look full-rank while their product does not.

READING IT
    effective rank << 256   -> SLACK. The basis is not spent; `gated` has room. Proceed.
    effective rank ~= 256   -> SATURATED. Gating would reallocate, not add. Prefer giving
                              the selection its OWN rank (a separate low-rank term) over
                              gating the existing one.

Run it on the RELEASED draft first: it scores 4.42 and is the reference for what a healthy
spectrum looks like at this vocabulary and rank. Then on ours, to see whether we differ.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import torch


def _load_pair(path: str) -> dict[str, torch.Tensor]:
    """Find markov_w1/markov_w2 in a safetensors file, a sharded dir, or a .pt/.bin."""
    from safetensors.torch import safe_open

    out: dict[str, torch.Tensor] = {}
    files: list[str] = []
    if os.path.isdir(path):
        idx = glob.glob(os.path.join(path, "*.index.json"))
        if idx:
            wmap = json.load(open(idx[0]))["weight_map"]
            files = sorted({
                os.path.join(path, v) for k, v in wmap.items() if "markov_w" in k
            })
        if not files:
            files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    else:
        files = [path]

    for f in files:
        if os.path.getsize(f) < 1024:
            raise SystemExit(
                f"{f} is {os.path.getsize(f)} bytes -- a git-lfs POINTER, not weights.\n"
                "Run this where the real checkpoint lives (the serving box)."
            )
        if f.endswith(".safetensors"):
            with safe_open(f, framework="pt") as h:
                for k in h.keys():
                    if "markov_w1" in k or "markov_w2" in k:
                        out["w1" if "markov_w1" in k else "w2"] = h.get_tensor(k).float()
        else:
            sd = torch.load(f, map_location="cpu", weights_only=True)
            for k, v in sd.items():
                if "markov_w1" in k or "markov_w2" in k:
                    out["w1" if "markov_w1" in k else "w2"] = v.float()
    if "w1" not in out or "w2" not in out:
        raise SystemExit(f"no markov_w1/markov_w2 found under {path}")
    return out


def _report(name: str, s: torch.Tensor, r: int) -> float:
    e = s**2
    e = e / e.sum()
    c = e.cumsum(0)
    pr = float(e.sum() ** 2 / (e**2).sum())  # participation ratio
    cov = "  ".join(f"{int(t*100)}%->{int((c < t).sum()) + 1}" for t in (0.5, 0.8, 0.9, 0.95, 0.99))
    print(f"  {name:<22} 有效秩 {pr:6.1f} / {r}   ({pr/r:5.1%})   能量覆盖: {cov}")
    return pr


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit(__doc__.strip().splitlines()[2].strip())
    for path in paths:
        t = _load_pair(path)
        w1, w2 = t["w1"], t["w2"]  # [V, r] each
        r = w1.shape[1]
        print(f"\n=== {path}")
        print(f"  markov_w1 {tuple(w1.shape)}   markov_w2 {tuple(w2.shape)}")
        _report("w1 (前驱嵌入 A)", torch.linalg.svdvals(w1), r)
        _report("w2 (后继嵌入 B)", torch.linalg.svdvals(w2), r)
        # exact spectrum of the V x V transition matrix, via thin QR -- see module docstring
        r1 = torch.linalg.qr(w1, mode="r").R
        r2 = torch.linalg.qr(w2, mode="r").R
        pr = _report("★ 转移矩阵 w2·w1ᵀ", torch.linalg.svdvals(r2 @ r1.T), r)
        frac = pr / r
        print()
        if frac < 0.45:
            print(f"  ⟹ 有余量({frac:.0%} 在用)。基底没被吃满,gated 有地方施展。")
        elif frac < 0.75:
            print(f"  ⟹ 中间地带({frac:.0%} 在用)。门能施展但会有取舍,值得跑但要看住 bigram 侧。")
        else:
            print(f"  ⟹ 已饱和({frac:.0%} 在用)。加门是重新分配而非新增 —— 优先给 select 单开一份秩。")


if __name__ == "__main__":
    main()
