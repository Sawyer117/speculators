#!/usr/bin/env python3
"""Verify a downloaded model dir is COMPLETE and not corrupt — the WHOLE dir, not just the shards.

No network and no reference hashes needed for the baseline checks:

1. Recurse EVERY file (incl subdirs like ``encoding/`` ``inference/``): flag ZERO-BYTE files (the
   classic partial-download symptom) and print a full inventory (per subdir count + bytes).
2. Every ``*.json`` (config.json, generation_config.json, tokenizer configs, …) is PARSE-checked — a
   truncated JSON fails to load, so this catches half-written small files.
3. Every ``*.safetensors`` (anywhere) is header-checked: first 8 bytes = LE header length N, then an
   N-byte JSON header of ``{tensor: {dtype, shape, data_offsets:[s,e]}}``. A whole shard satisfies
   ``file_size == 8 + N + max(e)``; a mismatch => truncated/corrupt.
4. Sharded ``model-XXXXX-of-YYYYY.safetensors`` naming → all 1..YYYYY must be present (missing shards);
   ``model.safetensors.index.json`` (if present) → every mapped tensor must appear in its shard.

For an AUTHORITATIVE "nothing missing vs the source repo" check, pass ``--manifest FILE`` where FILE is
the Hugging Face tree listing:
    curl -kL 'https://huggingface.co/api/models/<org>/<repo>/tree/main?recursive=true' -o hf_tree.json
Every remote file must exist locally with the same size; for LFS files the local SHA-256 must equal the
remote ``lfs.oid`` (only computed for files the manifest marks as LFS, so it stays fast).

Usage:  verify_safetensors_dir.py <model_dir> [--manifest hf_tree.json] [--sha256]
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
            raise ValueError("smaller than the 8-byte header prefix")
        n = struct.unpack("<Q", raw)[0]
        if n <= 0 or n > 1_000_000_000:
            raise ValueError(f"implausible header length {n} (corrupt / not safetensors)")
        blob = f.read(n)
        if len(blob) < n:
            raise ValueError(f"truncated header: {len(blob)} of {n} bytes")
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


def human(n: float) -> str:
    for u in ("B", "K", "M", "G", "T"):
        if n < 1024 or u == "T":
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}T"


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify a WHOLE downloaded model dir is complete & intact.")
    ap.add_argument("model_dir")
    ap.add_argument("--manifest", metavar="HF_TREE_JSON",
                    help="HF tree listing (…/api/models/<repo>/tree/main?recursive=true) for authoritative diff")
    ap.add_argument("--sha256", action="store_true", help="also print each shard's SHA-256")
    args = ap.parse_args()
    d = os.path.abspath(args.model_dir)
    if not os.path.isdir(d):
        ap.error(f"not a directory: {d}")

    problems: list[str] = []

    # ---- recurse every file ----
    files = []  # (relpath, abspath, size)
    for dp, _dns, fns in os.walk(d):
        for fn in fns:
            ap_ = os.path.join(dp, fn)
            try:
                files.append((os.path.relpath(ap_, d), ap_, os.path.getsize(ap_)))
            except OSError as e:
                problems.append(f"{os.path.relpath(ap_, d)}: stat failed ({e})")
    if not files:
        print("!! empty directory"); sys.exit(2)

    # ---- inventory (per top-level component) ----
    by_top: dict[str, list[int]] = {}
    for rel, _abs, sz in files:
        top = rel.split(os.sep)[0] if os.sep in rel else "(root)"
        by_top.setdefault(top, []).append(sz)
    print("-- inventory --")
    for top in sorted(by_top):
        szs = by_top[top]
        print(f"  {top}: {len(szs)} file(s), {human(sum(szs))}")

    # ---- zero-byte files ----
    zero = [rel for rel, _abs, sz in files if sz == 0]
    if zero:
        problems.append(f"{len(zero)} ZERO-byte file(s): {zero}")
        print(f"\n✗ {len(zero)} zero-byte file(s): {zero}")

    # ---- JSON parse-check (catches truncated small files) ----
    bad_json = []
    for rel, abs_, sz in files:
        if rel.endswith(".json") and sz > 0:
            try:
                json.load(open(abs_))
            except (OSError, ValueError) as e:
                bad_json.append(rel); problems.append(f"{rel}: invalid JSON ({e})")
    print(f"\njson: {sum(1 for r,_,_ in files if r.endswith('.json'))} file(s) checked, "
          f"{len(bad_json)} invalid" + (f" {bad_json}" if bad_json else " ✓"))

    # ---- safetensors integrity (anywhere) ----
    st = [(rel, abs_) for rel, abs_, _ in files if rel.endswith(".safetensors")]
    print(f"\n-- safetensors integrity ({len(st)} file(s)) --")
    all_tensors: set[str] = set()
    for rel, abs_ in st:
        try:
            r = verify_shard(abs_)
            all_tensors |= set(r["tensors"])
            tag = "OK" if r["ok"] else f"✗ TRUNCATED (have {r['size']:,}, need {r['expected']:,})"
            if not r["ok"]:
                problems.append(f"{rel}: truncated by {r['expected']-r['size']:,} bytes")
            extra = f"  sha256={sha256(abs_)[:16]}…" if args.sha256 else ""
            print(f"  {rel}: {len(r['tensors'])} tensors  {tag}{extra}")
        except Exception as e:  # noqa: BLE001
            problems.append(f"{rel}: {e}"); print(f"  {rel}: ✗ CORRUPT — {e}")

    # ---- shard count + index.json (top-level model-*.safetensors) ----
    top_shards = [os.path.basename(s) for s in glob.glob(os.path.join(d, "*.safetensors"))]
    totals = {int(m.group(2)) for m in (_OF.search(s) for s in top_shards) if m}
    if len(totals) == 1:
        total = totals.pop()
        have = {int(_OF.search(s).group(1)) for s in top_shards if _OF.search(s)}
        miss = sorted(set(range(1, total + 1)) - have)
        print(f"\nshard count: {len(have)}/{total} present" + (f"  ✗ MISSING #{miss}" if miss else "  ✓"))
        if miss:
            problems.append(f"missing shard numbers {miss}")
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.isfile(idx):
        try:
            wm = json.load(open(idx)).get("weight_map", {})
            miss_t = [t for t in wm if t not in all_tensors]
            print(f"index.json: {len(wm)} tensors mapped, {len(miss_t)} missing from shards"
                  + ("  ✓" if not miss_t else ""))
            if miss_t:
                problems.append(f"{len(miss_t)} tensor(s) in index.json absent from shards")
        except (OSError, ValueError) as e:
            problems.append(f"index.json unreadable: {e}")
    elif top_shards:
        print("index.json: ABSENT — normal only if the model ships a custom loader (check inference/README).")

    # ---- authoritative diff vs HF manifest ----
    if args.manifest:
        try:
            man = json.load(open(args.manifest))
        except (OSError, ValueError) as e:
            ap.error(f"--manifest unreadable: {e}")
        entries = man if isinstance(man, list) else man.get("siblings", man.get("tree", []))
        local = {rel: sz for rel, _abs, sz in files}
        remote_files = miss_files = size_bad = sha_bad = 0
        for e in entries:
            if not isinstance(e, dict) or e.get("type") == "directory":
                continue
            path = e.get("path") or e.get("rfilename")
            if not path:
                continue
            remote_files += 1
            if path not in local:
                miss_files += 1; problems.append(f"MISSING vs source: {path}")
                continue
            lfs = e.get("lfs") or {}
            rsize = lfs.get("size") or e.get("size")
            if rsize is not None and int(rsize) != local[path]:
                size_bad += 1; problems.append(f"size mismatch {path}: local {local[path]} vs source {rsize}")
                continue
            oid = lfs.get("oid") or lfs.get("sha256")
            if oid and len(oid) == 64:  # LFS sha256 → verify byte-exact
                if sha256(os.path.join(d, path)) != oid:
                    sha_bad += 1; problems.append(f"SHA256 mismatch {path}")
        print(f"\nmanifest diff: {remote_files} source file(s) — "
              f"{miss_files} missing, {size_bad} size-mismatch, {sha_bad} sha-mismatch"
              + ("  ✓" if not (miss_files or size_bad or sha_bad) else ""))

    # ---- verdict ----
    print("\n==================== VERDICT ====================")
    print(f"files: {len(files)}   safetensors: {len(st)}   tensors: {len(all_tensors)}")
    if problems:
        print(f"❌ {len(problems)} PROBLEM(S):")
        for p in problems[:40]:
            print(f"   • {p}")
        if len(problems) > 40:
            print(f"   … and {len(problems)-40} more")
        sys.exit(1)
    if args.manifest:
        print("✅ COMPLETE vs source manifest & INTACT (every file present, sizes/hashes match)")
    else:
        print("✅ INTACT (all files non-empty, JSON valid, shards whole & counted) — "
              "for vs-source completeness add --manifest")
    sys.exit(0)


if __name__ == "__main__":
    main()
