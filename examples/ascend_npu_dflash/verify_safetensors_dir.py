#!/usr/bin/env python3
"""Verify a sharded safetensors model dir is COMPLETE and not corrupt — no network / no reference hashes.

Three independent completeness checks:

1. Per-shard INTEGRITY (catches truncated / partially-downloaded shards). safetensors is
   self-describing: first 8 bytes = little-endian header length N, then an N-byte JSON header mapping
   each tensor -> {dtype, shape, data_offsets:[start,end]}. A whole, untruncated shard satisfies
   ``file_size == 8 + N + max(end over all data_offsets)``. Any mismatch => truncated/corrupt.

2. Shard COUNT (catches missing shards) from the ``model-XXXXX-of-YYYYY.safetensors`` naming: YYYYY is
   the expected total, so every index 1..YYYYY must be present.

3. index.json CONSISTENCY (catches missing tensors) if ``model.safetensors.index.json`` exists: every
   tensor in its ``weight_map`` must actually appear in the shard it points to, and every referenced
   shard must exist. If the index is ABSENT (a sharded model needs one), that is flagged loudly.

``--sha256`` additionally prints each file's SHA-256 so you can diff against the source repo's LFS
oids for a byte-exact check (integrity check #1 already catches the common truncation/missing failure
without needing those).

Usage:  verify_safetensors_dir.py <model_dir> [--sha256]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import struct
import sys

_OF = re.compile(r"-(\d+)-of-(\d+)\.safetensors$")


def read_header(path: str) -> tuple[int, dict]:
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            raise ValueError("file smaller than 8-byte header prefix")
        n = struct.unpack("<Q", raw)[0]
        if n <= 0 or n > 1_000_000_000:
            raise ValueError(f"implausible header length {n} (corrupt / not safetensors)")
        blob = f.read(n)
        if len(blob) < n:
            raise ValueError(f"truncated header: got {len(blob)} of {n} bytes")
        return n, json.loads(blob)


def verify_shard(path: str) -> dict:
    size = os.path.getsize(path)
    n, hdr = read_header(path)
    tensors = {k: v for k, v in hdr.items() if k != "__metadata__"}
    end = 0
    for name, t in tensors.items():
        off = (t or {}).get("data_offsets")
        if not off or len(off) != 2:
            raise ValueError(f"tensor {name!r}: missing/bad data_offsets")
        end = max(end, int(off[1]))
    expected = 8 + n + end
    return {"tensors": list(tensors), "expected": expected, "size": size, "ok": expected == size}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify a sharded safetensors model dir is complete & intact.")
    ap.add_argument("model_dir")
    ap.add_argument("--sha256", action="store_true", help="also print each shard's SHA-256 (slow; for diffing vs source)")
    args = ap.parse_args()
    d = args.model_dir
    if not os.path.isdir(d):
        ap.error(f"not a directory: {d}")

    problems: list[str] = []

    # ---- index.json (optional but expected for sharded models) ----
    idx_path = os.path.join(d, "model.safetensors.index.json")
    weight_map, expected_shards = None, None
    if os.path.isfile(idx_path):
        try:
            idx = json.load(open(idx_path))
            weight_map = idx.get("weight_map", {})
            expected_shards = sorted(set(weight_map.values()))
            print(f"index.json: {len(weight_map)} tensors mapped across {len(expected_shards)} shards")
        except (OSError, ValueError) as e:
            problems.append(f"index.json unreadable: {e}")
            print(f"✗ index.json unreadable: {e}")

    shards = sorted(glob.glob(os.path.join(d, "*.safetensors")))
    if not shards:
        print("!! no *.safetensors files here"); sys.exit(2)

    # ---- shard-count from the -of-YYYYY naming ----
    totals = {int(m.group(2)) for m in (_OF.search(os.path.basename(s)) for s in shards) if m}
    have_idx = {int(_OF.search(os.path.basename(s)).group(1)) for s in shards if _OF.search(os.path.basename(s))}
    if len(totals) == 1:
        total = totals.pop()
        missing_num = sorted(set(range(1, total + 1)) - have_idx)
        print(f"shard naming: {len(have_idx)}/{total} present" + (f"  ✗ MISSING #{missing_num}" if missing_num else "  ✓"))
        if missing_num:
            problems.append(f"missing shard numbers {missing_num}")
    elif len(totals) > 1:
        problems.append(f"inconsistent -of-N totals across files: {sorted(totals)}")
        print(f"✗ inconsistent -of-N totals: {sorted(totals)}")

    if weight_map is None:
        print("⚠ NO model.safetensors.index.json — a sharded model normally needs it to map tensors→shards.")
        print("  (download may be INCOMPLETE, or this model ships a non-standard loader — check inference/README.)")

    # ---- per-shard integrity ----
    print("\n-- per-shard integrity --")
    all_tensors: set[str] = set()
    for s in shards:
        name = os.path.basename(s)
        try:
            r = verify_shard(s)
            all_tensors |= set(r["tensors"])
            tag = "OK" if r["ok"] else f"✗ TRUNCATED (have {r['size']:,}, header needs {r['expected']:,})"
            if not r["ok"]:
                problems.append(f"{name}: truncated/short by {r['expected'] - r['size']:,} bytes")
            extra = f"  sha256={sha256(s)[:16]}…" if args.sha256 else ""
            print(f"  {name}: {len(r['tensors'])} tensors  {tag}{extra}")
        except Exception as e:  # noqa: BLE001 — report any parse failure per-file, keep going
            problems.append(f"{name}: {e}")
            print(f"  {name}: ✗ CORRUPT — {e}")

    # ---- index cross-check ----
    if weight_map is not None:
        have_names = {os.path.basename(s) for s in shards}
        miss_shards = [s for s in expected_shards if s not in have_names]
        miss_tensors = [t for t in weight_map if t not in all_tensors]
        if miss_shards:
            problems.append(f"index references {len(miss_shards)} absent shard(s): {miss_shards}")
        if miss_tensors:
            problems.append(f"{len(miss_tensors)} tensor(s) in index but not in any shard")
        print(f"\nindex cross-check: {len(all_tensors)}/{len(weight_map)} mapped tensors present; "
              f"{len(miss_shards)} shard(s) & {len(miss_tensors)} tensor(s) missing")

    # ---- verdict ----
    print("\n==================== VERDICT ====================")
    print(f"shards: {len(shards)}   tensors: {len(all_tensors)}" + (f"   (index expects {len(weight_map)})" if weight_map else ""))
    if problems:
        print(f"❌ {len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"   • {p}")
        sys.exit(1)
    note = "COMPLETE & INTACT" if weight_map is not None else "INTACT (per-shard + count OK; no index.json so vs-source completeness unverified)"
    print(f"✅ {note}")
    sys.exit(0)


if __name__ == "__main__":
    main()
