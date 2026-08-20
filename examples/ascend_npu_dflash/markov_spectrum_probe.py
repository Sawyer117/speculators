#!/usr/bin/env python3
"""Is the rank-256 Markov head SATURATED? -- decide before spending a run on `gated`.

    python3 examples/ascend_npu_dflash/markov_spectrum_probe.py <dir_or_file> [...]
    python3 examples/ascend_npu_dflash/markov_spectrum_probe.py --find [root ...]

Accepts a released/converted HF dir (sharded safetensors), a single .safetensors/.pt, or
one of OUR training checkpoints (a DCP dir: ``.metadata`` + ``*.distcp``) -- markov_w1/w2
are replicated, not expert-sharded, so a single process can read them straight out.
``--find`` walks the usual storage roots and reports every checkpoint it can see, so no
one has to remember which box keeps what.

WHY THIS EXISTS
---------------
`markov_head_type="dflash2"` scales the predecessor embedding by H(h_t) before the same
`markov_w2` projection. It therefore adds NO rank: the head stays a rank-`r` approximation
of a V x V transition matrix (V=129,280, r=256 -- a ~250x compression). What H buys is a
CONTEXT-DEPENDENT MIXTURE of rank-<=r matrices that all share one r-dimensional basis.

So H can only help if that basis has slack. If the r dimensions are already fully
spent on unconditional bigram statistics, modulation cannibalises: it trades bigram fidelity
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
    effective rank << 256   -> SLACK. The basis is not spent; train the `dflash2` head at
                              the current markov_rank.
    effective rank ~= 256   -> SATURATED. H would only reallocate an already-full basis, so
                              raise markov_rank (320/384) before training that head.

Run it on the RELEASED draft first: it scores 4.42 and is the reference for what a healthy
spectrum looks like at this vocabulary and rank. Then on ours, to see whether we differ.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import torch


_ROOTS = (
    "/share", "/home/canada_group_folder", os.path.expanduser("~"),
    "/data", "/mnt", "/workspace",
)


def _is_dcp(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, ".metadata"))


def _load_dcp(path: str) -> dict[str, torch.Tensor]:
    """Read markov_w1/w2 out of a DCP checkpoint without building the model.

    Safe single-process: both tensors are replicated across ranks (only the MoE experts
    are DTensor-sharded), so reading them by key needs no process group and no EP layout.
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict_from_keys

    reader = dcp.FileSystemReader(path)
    keys = [k for k in reader.read_metadata().state_dict_metadata if "markov_w" in k]
    if not keys:
        raise SystemExit(f"{path}: DCP checkpoint has no markov_w* key")
    loaded = _load_state_dict_from_keys(keys, storage_reader=reader)
    out: dict[str, torch.Tensor] = {}
    for k, v in loaded.items():
        out["w1" if "markov_w1" in k else "w2"] = v.float()
    return out


def find_checkpoints(roots: tuple[str, ...] | list[str]) -> None:
    """Walk storage roots and print every checkpoint that carries a Markov head."""
    print("扫描中(HF 分片 / 单文件 / DCP)...")
    found = 0
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root, topdown=True):
            dirnames[:] = [d for d in dirnames if not d.startswith((".git", "__pycache__"))]
            if ".metadata" in filenames:
                dirnames[:] = []
                try:
                    import torch.distributed.checkpoint as dcp

                    md = dcp.FileSystemReader(dirpath).read_metadata()
                    if any("markov_w" in k for k in md.state_dict_metadata):
                        print(f"  [DCP]         {dirpath}")
                        found += 1
                except Exception:  # noqa: BLE001 - a listing must not die on one bad dir
                    pass
                continue
            idx = [f for f in filenames if f.endswith(".index.json")]
            if idx:
                try:
                    wmap = json.load(open(os.path.join(dirpath, idx[0])))["weight_map"]
                    if any("markov_w" in k for k in wmap):
                        print(f"  [HF sharded]  {dirpath}")
                        found += 1
                except Exception:  # noqa: BLE001
                    pass
    print(f"\n共 {found} 个。挑一到两个再跑一次,不带 --find。")


def _load_pair(path: str) -> dict[str, torch.Tensor]:
    """Find markov_w1/markov_w2 in a DCP dir, a safetensors file/dir, or a .pt/.bin."""
    from safetensors.torch import safe_open

    if _is_dcp(path):
        return _load_dcp(path)

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
    argv = sys.argv[1:]
    if argv and argv[0] == "--find":
        find_checkpoints(argv[1:] or _ROOTS)
        return
    paths = argv
    if not paths:
        raise SystemExit(
            "用法: markov_spectrum_probe.py <ckpt 目录或文件> ...\n"
            "      markov_spectrum_probe.py --find [根目录 ...]   # 不知道路径就先跑这个"
        )
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
            print(f"  ⟹ 有余量({frac:.0%} 在用)。基底没被吃满,rank 256 足以同时扛无条件 bigram 与上下文调制。按现有 rank 训 dflash2 头。")
        elif frac < 0.75:
            print(f"  ⟹ 中间地带({frac:.0%} 在用)。可以按现有 rank 训,但要盯住 bigram 侧有没有被换走。")
        else:
            print(f"  ⟹ 已饱和({frac:.0%} 在用)。H 只能在一组已被占满的基底上重新分配 —— 先把 markov_rank 开大(320/384)再训 dflash2 头。")


if __name__ == "__main__":
    main()
