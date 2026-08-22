#!/usr/bin/env python3
"""Verify a downloaded HF/ModelScope checkpoint is COMPLETE and not truncated.

Counting files proves nothing: a shard cut off mid-download is still a file, and the failure
shows up much later as a load error or, worse, as garbage output. safetensors is
self-describing -- its header records the exact byte range of every tensor -- so truncation is
*provable*, and the index proves no shard is missing. Both checks read only headers, so a
400 GB checkpoint verifies in seconds rather than hours.

WHAT IT CHECKS (all offline)
  1. the expected companion files exist (config.json, tokenizer, the index)
  2. every shard named in model.safetensors.index.json is present
  3. each shard's declared size == 8 + header_len + max(data_offsets)  <- catches truncation
  4. every tensor the index promises actually appears in that shard's header
  5. no shard is present that the index does not know about

OPTIONAL ONLINE CROSS-CHECK (--remote)
  Ask ModelScope for its file list and diff names + sizes against local. Needs network; on the
  Ascend boxes that means an https_proxy with a self-signed cert, so certificate verification
  is disabled deliberately (we are comparing sizes, not trusting content).

OPTIONAL --sha256
  Hash every file and compare with ModelScope's. CORRECT but SLOW -- it reads every byte, so a
  400 GB checkpoint takes hours and saturates the disk. Only reach for it when the header
  checks pass yet the model still misbehaves.

USAGE
  python3 examples/ascend_npu_dflash/verify_model_weights.py /data/ckpt/DeepSeek-V4-Flash-0731-w8a8
  python3 ... /data/ckpt/DeepSeek-V4-Flash-0731-w8a8 \
      --remote https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import sys
import urllib.request

EXPECTED_COMPANIONS = ("config.json",)
NICE_TO_HAVE = ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
                "configuration.json", "quant_model_description.json")


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n} B"


def read_safetensors_header(path: str):
    """(header_dict, data_start, declared_end) or raise ValueError.

    Layout: u64 little-endian header length, then that many bytes of JSON, then the tensor
    blob. Each tensor's ``data_offsets`` are relative to the start of the blob, so the file's
    true length is fully determined by the header -- which is what makes truncation detectable
    without reading the payload.
    """
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) < 8:
            raise ValueError("shorter than the 8-byte header length field")
        n = int.from_bytes(raw, "little")
        if n <= 0 or n > (1 << 32):
            raise ValueError(f"implausible header length {n}")
        blob = fh.read(n)
        if len(blob) < n:
            raise ValueError(f"header truncated: wanted {n} bytes, got {len(blob)}")
        try:
            header = json.loads(blob)
        except json.JSONDecodeError as exc:
            raise ValueError(f"header is not valid JSON: {exc}") from exc
    end = 0
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        offs = (meta or {}).get("data_offsets")
        if not offs or len(offs) != 2:
            raise ValueError(f"tensor {name!r} has no usable data_offsets")
        end = max(end, int(offs[1]))
    return header, 8 + n, 8 + n + end


def modelscope_files(url: str):
    """[(path, size, sha256)] from the ModelScope repo-files API, or None if unreachable."""
    tail = url.rstrip("/").split("/models/")[-1]
    api = f"https://www.modelscope.cn/api/v1/models/{tail}/repo/files?Revision=master&Recursive=True"
    # The boxes sit behind an MITM proxy with a self-signed cert. We compare names and sizes,
    # never execute what comes back, so skipping verification here is a considered choice.
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(api, timeout=60, context=ctx) as resp:
            payload = json.load(resp)
    except Exception as exc:                                    # noqa: BLE001 - report, don't crash
        print(f"  ⚠ could not reach ModelScope ({type(exc).__name__}: {exc})")
        print(f"    the offline checks above still stand on their own.")
        return None
    files = (payload.get("Data") or {}).get("Files") or []
    out = []
    for f in files:
        if f.get("Type") == "tree":
            continue
        out.append((f.get("Path"), int(f.get("Size") or 0), f.get("Sha256")))
    return out


def sha256_of(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="local checkpoint directory")
    ap.add_argument("--remote", metavar="URL", default=None,
                    help="ModelScope model URL to cross-check names/sizes against")
    ap.add_argument("--sha256", action="store_true",
                    help="also hash every file (SLOW: reads every byte) and compare with --remote")
    args = ap.parse_args()

    root = os.path.abspath(args.model)
    if not os.path.isdir(root):
        print(f"!! not a directory: {root}")
        return 2
    print("=" * 72)
    print(f" VERIFY  {root}")
    print("=" * 72)

    problems: list[str] = []

    # --- 1. companions -------------------------------------------------------------------
    for name in EXPECTED_COMPANIONS:
        if not os.path.isfile(os.path.join(root, name)):
            problems.append(f"missing required file: {name}")
    missing_nice = [n for n in NICE_TO_HAVE if not os.path.isfile(os.path.join(root, n))]
    print(f"\n-- companions --")
    print(f"  required present : {[n for n in EXPECTED_COMPANIONS if os.path.isfile(os.path.join(root, n))]}")
    if missing_nice:
        print(f"  absent (may be fine): {missing_nice}")

    # --- 2. index ------------------------------------------------------------------------
    # The index is NOT always HF's model.safetensors.index.json: Ascend quantized checkpoints
    # (msmodelslim output, loaded with --quantization ascend) ship
    # quant_model_weights.safetensors.index.json instead. Looking only for the HF name reports
    # a complete checkpoint as "sharded with no index", which is both wrong and alarming.
    indices = sorted(f for f in os.listdir(root) if f.endswith(".safetensors.index.json"))
    index_path = os.path.join(root, indices[0]) if indices else os.path.join(root, "model.safetensors.index.json")
    if len(indices) > 1:
        print(f"  note: {len(indices)} index files present, using {indices[0]}: {indices}")
    on_disk = sorted(f for f in os.listdir(root) if f.endswith(".safetensors"))
    expected_shards, weight_map = set(), {}
    if os.path.isfile(index_path):
        with open(index_path) as fh:
            index = json.load(fh)
        weight_map = index.get("weight_map") or {}
        expected_shards = set(weight_map.values())
        total_declared = (index.get("metadata") or {}).get("total_size")
        print(f"\n-- index --")
        print(f"  file             : {os.path.basename(index_path)}")
        print(f"  tensors promised : {len(weight_map)}")
        print(f"  shards promised  : {len(expected_shards)}")
        if total_declared:
            print(f"  total_size       : {human(int(total_declared))}")
    else:
        print(f"\n-- index --\n  no *.safetensors.index.json "
              f"({'single-file checkpoint' if len(on_disk) == 1 else 'UNEXPECTED for a sharded model'})")
        if len(on_disk) > 1:
            problems.append("sharded checkpoint with no index — cannot prove completeness")
        expected_shards = set(on_disk)

    missing = sorted(expected_shards - set(on_disk))
    extra = sorted(set(on_disk) - expected_shards)
    for m in missing:
        problems.append(f"shard named in the index is ABSENT: {m}")
    for e in extra:
        problems.append(f"shard on disk that the index does not know about: {e}")

    # --- 3+4. per-shard header vs. real size, and tensor presence --------------------------
    print(f"\n-- shards ({len(on_disk)} on disk) --")
    seen_tensors: set[str] = set()
    total_bytes = 0
    bad = 0
    for i, name in enumerate(on_disk, 1):
        path = os.path.join(root, name)
        actual = os.path.getsize(path)
        total_bytes += actual
        try:
            header, _, declared = read_safetensors_header(path)
        except ValueError as exc:
            problems.append(f"{name}: unreadable header — {exc}")
            print(f"  [{i:>3}/{len(on_disk)}] {name}  !! {exc}")
            bad += 1
            continue
        seen_tensors |= {k for k in header if k != "__metadata__"}
        if actual != declared:
            delta = actual - declared
            problems.append(
                f"{name}: size {human(actual)} but its own header declares {human(declared)} "
                f"({'TRUNCATED by ' + human(-delta) if delta < 0 else 'has ' + human(delta) + ' of trailing junk'})")
            print(f"  [{i:>3}/{len(on_disk)}] {name}  !! size {actual} vs declared {declared}")
            bad += 1
        elif i % 25 == 0 or i == len(on_disk):
            print(f"  [{i:>3}/{len(on_disk)}] … ok so far ({human(total_bytes)} scanned)")

    if weight_map:
        promised = set(weight_map)
        absent = promised - seen_tensors
        if absent:
            problems.append(f"{len(absent)} tensor(s) promised by the index are in NO shard, "
                            f"e.g. {sorted(absent)[:3]}")

    print(f"\n  shards ok        : {len(on_disk) - bad}/{len(on_disk)}")
    print(f"  tensors found    : {len(seen_tensors)}"
          + (f" / {len(weight_map)} promised" if weight_map else ""))
    print(f"  total on disk    : {human(total_bytes)}")

    # --- 5. optional online cross-check ---------------------------------------------------
    if args.remote:
        print(f"\n-- ModelScope cross-check --")
        remote = modelscope_files(args.remote)
        if remote:
            rmap = {p: (s, h) for p, s, h in remote if p}
            local_names = set(os.listdir(root))
            miss = sorted(p for p in rmap if p not in local_names and "/" not in p)
            print(f"  files upstream   : {len(rmap)}")
            for p in miss:
                problems.append(f"present upstream but NOT local: {p}")
            for p, (size, digest) in sorted(rmap.items()):
                lp = os.path.join(root, p)
                if not os.path.isfile(lp) or not size:
                    continue
                actual = os.path.getsize(lp)
                if actual != size:
                    problems.append(f"{p}: local {human(actual)} vs upstream {human(size)}")
                elif args.sha256 and digest:
                    got = sha256_of(lp)
                    if got != digest:
                        problems.append(f"{p}: sha256 mismatch (local {got[:12]}… upstream {digest[:12]}…)")
            if not miss:
                print(f"  every upstream file is present locally")

    # --- verdict --------------------------------------------------------------------------
    print("\n" + "=" * 72)
    if problems:
        print(f" ✗ {len(problems)} PROBLEM(S)")
        for p in problems:
            print(f"   - {p}")
        print("=" * 72)
        return 1
    print(" ✓ COMPLETE — every shard present, every header self-consistent,")
    print("   every tensor the index promises accounted for.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
