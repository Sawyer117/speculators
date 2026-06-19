#!/usr/bin/env python3
"""Inspect a training JSONL for speculators prepare_data.py compatibility.

Usage:
    python examples/ascend_npu_dflash/inspect_jsonl.py [PATH]

Default PATH is the 10k subset. Read-only. Prints a bounded, paste-friendly
summary: record count, top-level key sets, conversation turn structure
(role/content vs from/value), roles, <think> usage, and 2 truncated samples.
prepare_data.py needs a top-level `conversations` (or `messages`) list of turns
with `role`/`content` (or sharegpt `from`/`value`).
"""
import json
import os
import sys
from collections import Counter

PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "/share/canada_group_folder/dataset/perfectblend_train_10ksubset.jsonl"
SAMPLE = 300       # records to parse for aggregates
PREVIEW = 180      # chars per content preview
MAX_TURNS_SHOWN = 6


def trunc(s, n=PREVIEW):
    s = str(s).replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + f"...(+{len(s) - n} chars)"


print(f"=== JSONL inspect: {PATH} ===")
if not os.path.exists(PATH):
    print("!! path does NOT exist"); sys.exit(0)

if os.path.isdir(PATH):
    print("This is a DIRECTORY (maybe an already-prepared Arrow dataset). Contents:")
    for f in sorted(os.listdir(PATH))[:40]:
        print("   ", f)
    print("\n-> if you see *.arrow / dataset_info.json / token_freq.pt, it's PREPARED:")
    print("   set DATA_DIR to this dir and skip prepare.")
    sys.exit(0)

print(f"is_file: True   size: {os.path.getsize(PATH) / 1e6:.2f} MB")

n_records = 0
malformed = 0
keysets = Counter()
conv_key = Counter()
turn_style = Counter()
roles = Counter()
turns_per = []
think_records = 0
samples = []

with open(PATH, "r", encoding="utf-8") as fh:
    for i, line in enumerate(fh):
        line = line.strip()
        if not line:
            continue
        n_records += 1
        if i >= SAMPLE:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            malformed += 1
            continue
        keysets[tuple(sorted(rec.keys()))] += 1
        ck = "conversations" if "conversations" in rec else (
            "messages" if "messages" in rec else None)
        conv_key[ck] += 1
        conv = rec.get(ck) if ck else None
        if isinstance(conv, list):
            turns_per.append(len(conv))
            has_think = False
            for t in conv:
                if not isinstance(t, dict):
                    continue
                if "role" in t and "content" in t:
                    turn_style["role/content"] += 1
                    roles[t.get("role")] += 1
                    if t.get("role") == "assistant" and "<think>" in str(t.get("content", "")):
                        has_think = True
                elif "from" in t and "value" in t:
                    turn_style["from/value"] += 1
                    roles[t.get("from")] += 1
                    if t.get("from") in ("gpt", "assistant") and "<think>" in str(t.get("value", "")):
                        has_think = True
            if has_think:
                think_records += 1
        if len(samples) < 2:
            samples.append(rec)

scanned = min(n_records, SAMPLE)
print(f"records (non-empty lines): {n_records}   (parsed first {scanned}, malformed: {malformed})")
print("\n--- top-level key sets ---")
for ks, c in keysets.most_common(5):
    print(f"   {c:>4}x  {list(ks)}")
print(f"\n--- conversation field ---  {dict(conv_key)}")
print(f"--- turn key style ---       {dict(turn_style)}")
print(f"--- roles seen ---           {dict(roles)}")
if turns_per:
    print(f"--- turns/record ---         min={min(turns_per)} max={max(turns_per)} "
          f"avg={sum(turns_per) / len(turns_per):.1f}")
print(f"--- records w/ <think> ---   {think_records}/{scanned}")

for si, rec in enumerate(samples):
    print(f"\n=== sample record #{si} ===")
    print(f"top-level keys: {list(rec.keys())}")
    conv = rec.get("conversations") or rec.get("messages") or []
    for ti, t in enumerate(conv[:MAX_TURNS_SHOWN]):
        if isinstance(t, dict):
            role = t.get("role") or t.get("from") or "?"
            content = t.get("content") if "content" in t else t.get("value", "")
            print(f"  [{ti}] {role}: {trunc(content)}")
    if len(conv) > MAX_TURNS_SHOWN:
        print(f"  ... (+{len(conv) - MAX_TURNS_SHOWN} more turns)")

print("\n--- verdict hints ---")
ok_conv = conv_key.get("conversations", 0) > 0 or conv_key.get("messages", 0) > 0
ok_turn = turn_style.get("role/content", 0) > 0 or turn_style.get("from/value", 0) > 0
print(f"  has conversations/messages list: {ok_conv}")
print(f"  turns use role/content or from/value: {ok_turn}")
print("  -> if both True, it's a raw JSONL ready for prepare_data.py (run prepare).")
