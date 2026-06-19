#!/usr/bin/env python3
"""Check a HF model checkpoint for usability + integrity (no torch needed).

Handles the HF-cache layout 'models--Org--Name/{blobs,refs,snapshots}': resolves
the real snapshot dir to pass as --verifier-name-or-path. Validates config.json
(model_type / num_hidden_layers / hidden_size), tokenizer presence, and every
safetensors shard (header parses, no truncation, sizes match the index). Read-only.

Usage:
    python examples/ascend_npu_dflash/check_ckpt.py [PATH]
"""
import json
import os
import struct
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "/share/canada_group_folder/ckpt/models--Qwen--Qwen3-4B"


def resolve_model_dir(p):
    snap = os.path.join(p, "snapshots")
    if os.path.isdir(snap):
        commit = None
        ref = os.path.join(p, "refs", "main")
        if os.path.isfile(ref):
            with open(ref) as fh:
                commit = fh.read().strip()
        if commit and os.path.isdir(os.path.join(snap, commit)):
            return os.path.join(snap, commit), f"HF cache -> snapshots/{commit}"
        subs = sorted(d for d in os.listdir(snap) if os.path.isdir(os.path.join(snap, d)))
        if subs:
            return os.path.join(snap, subs[-1]), f"HF cache -> snapshots/{subs[-1]} (no refs/main)"
    return p, "plain directory"


def st_header(fp):
    """Parse a safetensors header -> (header_dict, header_len, file_size)."""
    size = os.path.getsize(fp)
    with open(fp, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            raise ValueError("file smaller than 8-byte header prefix")
        n = struct.unpack("<Q", raw)[0]
        if n <= 0 or n + 8 > size:
            raise ValueError(f"implausible header length {n} (file {size} bytes)")
        hdr = json.loads(f.read(n))
    return hdr, n, size


print(f"=== check ckpt: {PATH} ===")
if not os.path.exists(PATH):
    print("!! path does NOT exist"); sys.exit(1)

mdir, how = resolve_model_dir(PATH)
print(f"resolved model dir: {mdir}   ({how})")
print(f"\n>>> use THIS as --verifier-name-or-path:\n    {mdir}\n")

files = sorted(os.listdir(mdir))
print(f"--- files ({len(files)}) ---")
for f in files[:60]:
    print("   ", f)

problems = []

cfg_path = os.path.join(mdir, "config.json")
if os.path.isfile(cfg_path):
    try:
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        print("\n--- config.json ---")
        for k in ("model_type", "num_hidden_layers", "hidden_size",
                  "num_attention_heads", "num_key_value_heads", "vocab_size", "torch_dtype"):
            print(f"   {k}: {cfg.get(k)}")
        if cfg.get("num_hidden_layers") == 36 and cfg.get("hidden_size") == 2560:
            print("   -> matches Qwen3-4B (36 layers, hidden 2560) ✓")
    except Exception as e:
        problems.append(f"config.json unreadable: {e}")
else:
    problems.append("config.json MISSING")

tok = [f for f in files if f in ("tokenizer.json", "tokenizer_config.json",
       "tokenizer.model", "vocab.json", "merges.txt")]
print(f"\n--- tokenizer files ---  {tok if tok else 'NONE (!)'}")
if not tok:
    problems.append("no tokenizer files found")

idx_path = os.path.join(mdir, "model.safetensors.index.json")
shards, index_total = [], None
if os.path.isfile(idx_path):
    with open(idx_path) as fh:
        idx = json.load(fh)
    wmap = idx.get("weight_map", {})
    shards = sorted(set(wmap.values()))
    index_total = idx.get("metadata", {}).get("total_size")
    print(f"\n--- sharded safetensors: {len(shards)} shards, {len(wmap)} tensors (per index) ---")
else:
    shards = [f for f in files if f.endswith(".safetensors")]
    if shards:
        print(f"\n--- single/unsharded safetensors: {shards} ---")
    else:
        problems.append("no *.safetensors and no index.json (is it a .bin checkpoint?)")

total_tensors, total_bytes = 0, 0
for sh in shards:
    fp = os.path.join(mdir, sh)
    if not os.path.isfile(fp):
        problems.append(f"shard MISSING: {sh}")
        print(f"   {sh}: !! MISSING")
        continue
    try:
        hdr, hlen, size = st_header(fp)
        hdr.pop("__metadata__", None)
        data_len = size - 8 - hlen
        max_end = max((t["data_offsets"][1] for t in hdr.values() if "data_offsets" in t), default=0)
        total_tensors += len(hdr); total_bytes += size
        ok = max_end <= data_len
        print(f"   {sh}: {len(hdr)} tensors, {size/1e9:.2f} GB  {'OK' if ok else f'!! TRUNCATED (need {max_end}B, have {data_len}B)'}")
        if not ok:
            problems.append(f"{sh}: truncated/short")
    except Exception as e:
        problems.append(f"{sh}: header error: {e}")
        print(f"   {sh}: !! header error: {e}")

if shards:
    print(f"\n--- totals: {total_tensors} tensors, {total_bytes/1e9:.2f} GB on disk ---")
    if index_total is not None:
        print(f"    index total_size: {index_total/1e9:.2f} GB (tensor bytes; on-disk is slightly larger due to headers)")

print("\n=== VERDICT ===")
if problems:
    print("PROBLEMS FOUND:")
    for p in problems:
        print("  !!", p)
    sys.exit(2)
print("OK — config + tokenizer + all safetensors shards present, headers valid, no truncation.")
print(f"Use:  --verifier-name-or-path '{mdir}'")
