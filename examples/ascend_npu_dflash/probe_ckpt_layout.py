#!/usr/bin/env python3
"""What namespace is a trained checkpoint actually in, and would the released layout load?

Written before changing how checkpoints are saved, because the change is only safe if we
know three things for certain rather than by reading the loader:

  1. the exact key namespace `save_pretrained` emits today,
  2. whether the routed experts are STACKED ([E, out, in], our grouped-GEMM packing) or
     per-expert (what the released checkpoint and every vLLM/vllm-ascend DSV4 loader expect),
  3. what a load of the released layout would hit -- which keys would go unmatched.

It reads safetensors headers only: no model construction, no accelerator, no torch beyond
what safetensors needs. That matters because the question is about key names and shapes, and
building a 256-expert model to ask about key names would be a worse experiment, not a better
one.

USAGE
  python3 examples/ascend_npu_dflash/probe_ckpt_layout.py $RUN/ckpt_faithful_ep_<ts>/<epoch>
  python3 ... <ckpt> --released /share/.../DeepSeek-V4-Flash-0731-w8a8   # compare namespaces
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
from collections import Counter


def st_keys(path: str) -> dict[str, list[int]]:
    """{tensor name: shape} from the safetensors header alone."""
    out: dict[str, list[int]] = {}
    for fn in sorted(os.listdir(path)):
        if not fn.endswith(".safetensors"):
            continue
        with open(os.path.join(path, fn), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k != "__metadata__":
                out[k] = v.get("shape", [])
    return out


def namespace_of(keys) -> Counter:
    """Coarse bucket of the leading component, which is what a loader keys off."""
    c: Counter = Counter()
    for k in keys:
        head = k.split(".")[0]
        c[head if not head.isdigit() else "<int>"] += 1
    return c


def expert_style(keys: dict[str, list[int]]) -> str:
    stacked = [k for k in keys if re.search(r"experts\.w\d$", k)]
    per_ex = [k for k in keys if re.search(r"experts\.\d+\.w\d\.weight$", k)]
    if stacked and not per_ex:
        shp = keys[stacked[0]]
        return f"STACKED — e.g. {stacked[0]} {shp} ({len(stacked)} tensors, E={shp[0] if shp else '?'})"
    if per_ex and not stacked:
        return f"PER-EXPERT — e.g. {per_ex[0]} {keys[per_ex[0]]} ({len(per_ex)} tensors)"
    if per_ex and stacked:
        return f"MIXED — {len(stacked)} stacked and {len(per_ex)} per-expert"
    return "no routed-expert tensors found"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", help="a directory holding model.safetensors (or shards)")
    ap.add_argument("--released", default=None, metavar="DIR",
                    help="a released DeepSeek checkpoint, to diff the two namespaces")
    ap.add_argument("--show", type=int, default=12, help="sample keys to print per group")
    args = ap.parse_args()

    ours = st_keys(args.ckpt)
    if not ours:
        print(f"!! no .safetensors under {args.ckpt}")
        return 2

    print("=" * 74)
    print(f" OURS  {args.ckpt}")
    print("=" * 74)
    print(f"  tensors      : {len(ours)}")
    print(f"  namespaces   : {dict(namespace_of(ours).most_common(8))}")
    print(f"  routed experts: {expert_style(ours)}")
    print(f"  sample keys  :")
    for k in sorted(ours)[: args.show]:
        print(f"    {k}  {ours[k]}")

    if args.released:
        rel = {k: v for k, v in st_keys(args.released).items() if k.startswith("mtp.")}
        print()
        print("=" * 74)
        print(f" RELEASED (mtp.* only)  {args.released}")
        print("=" * 74)
        print(f"  mtp.* tensors : {len(rel)}")
        print(f"  routed experts: {expert_style(rel)}")
        print(f"  sample keys   :")
        for k in sorted(rel)[: args.show]:
            print(f"    {k}  {rel[k]}")

        # The decisive number: how much of each side the other side would fail to match.
        print()
        print("-- if a loader expecting one namespace were handed the other --")
        print(f"  keys only in ours     : {len(set(ours) - set(rel))}")
        print(f"  keys only in released : {len(set(rel) - set(ours))}")
        print(f"  keys in common        : {len(set(ours) & set(rel))}")
        print("  ⟹ zero in common means no loader can read both without a mapping;")
        print("     that mapping is the whole cost of changing what we emit.")

    # Buffers and non-parameter entries complicate any rename, so surface them.
    odd = [k for k in ours if any(t in k for t in ("freqs", "rope", "inv_freq", "d2t", "t2d"))]
    if odd:
        print(f"\n  non-weight entries ({len(odd)}): {odd[: args.show]}")
        print("  these are not module parameters; a rename scheme has to decide about them too.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
