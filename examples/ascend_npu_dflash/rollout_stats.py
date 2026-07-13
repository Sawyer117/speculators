#!/usr/bin/env python3
"""Rollout length/speed stats + AR/MTP/DSpark content-identity compare.

Reads rollout JSONL produced by scripts/response_regeneration/script.py, whose
lines look like:
    {"id","conversations":[{from:human,value},{from:gpt,value}],
     "metadata":{"idx","finish_reason","latency_s","usage":{prompt_tokens,
                 completion_tokens,total_tokens},...}}

Two modes:
  1) ONE file  -> full distribution: count, completion-token mean/median/p90/p99/
     min/max/SUM, finish_reason breakdown (stop vs length=truncated), mean latency.
     This is the Qwen3-4B baseline you extrapolate from.
  2) 2+ files  -> per-file stats PLUS pairwise content-identity (exact-match rate,
     aligned by id) and per-request throughput compare (tok/s, speedup vs first).
     Use for AR vs MTP vs DSpark at temperature=0 (they should match byte-for-byte).

Usage:
    python rollout_stats.py FILE [FILE2 FILE3 ...] [--labels AR,MTP,DSpark]
    # optional wall-clock for aggregate throughput + full-run extrapolation:
    python rollout_stats.py sample.jsonl --wall-sec 123 --total-rows 1400000 --machines 4
"""
import argparse
import json
import statistics as st
import sys
from collections import Counter


def pct(sorted_vals, p):
    if not sorted_vals:
        return 0
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def load(path):
    """Return list of records: {id, gpt_text, ctoks, ptoks, latency, finish, error}."""
    recs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            md = o.get("metadata", {}) or {}
            usage = md.get("usage") or {}
            gpt = ""
            for turn in o.get("conversations", []):
                if turn.get("from") == "gpt":
                    gpt = turn.get("value", "")
            recs.append({
                "id": o.get("id"),
                "gpt": gpt,
                "ctoks": usage.get("completion_tokens"),
                "ptoks": usage.get("prompt_tokens"),
                "latency": md.get("latency_s"),
                "finish": md.get("finish_reason"),
                "error": md.get("error"),
            })
    return recs


def summarize(label, recs):
    ok = [r for r in recs if r["error"] is None]
    err = [r for r in recs if r["error"] is not None]
    ct = sorted(r["ctoks"] for r in ok if isinstance(r["ctoks"], (int, float)))
    lat = [r["latency"] for r in ok if isinstance(r["latency"], (int, float))]
    fin = Counter(r["finish"] for r in ok)
    print(f"\n===== {label} =====")
    print(f"rows: {len(recs)}  ok: {len(ok)}  errors: {len(err)}")
    if ct:
        print(f"completion_tokens: mean {st.mean(ct):.1f}  median {st.median(ct):.0f}  "
              f"p90 {pct(ct,90):.0f}  p99 {pct(ct,99):.0f}  min {ct[0]}  max {ct[-1]}")
        print(f"completion_tokens TOTAL (this file): {sum(ct):,}")
    else:
        # fallback: no usage in jsonl -> char length of gpt text (rough proxy)
        chars = sorted(len(r["gpt"]) for r in ok)
        if chars:
            print("!! no usage.completion_tokens in file — falling back to CHAR length:")
            print(f"gpt chars: mean {st.mean(chars):.0f}  median {st.median(chars):.0f}  "
                  f"p90 {pct(chars,90):.0f}  min {chars[0]}  max {chars[-1]}")
    if fin:
        tot = sum(fin.values())
        parts = ", ".join(f"{k}={v} ({100*v/tot:.1f}%)" for k, v in fin.most_common())
        print(f"finish_reason: {parts}   (length = hit max_tokens = truncated)")
    if lat and ct:
        # per-request tok/s (NOT aggregate; under concurrency aggregate is higher)
        tps = [r["ctoks"] / r["latency"] for r in ok
               if isinstance(r["ctoks"], (int, float)) and isinstance(r["latency"], (int, float)) and r["latency"] > 0]
        if tps:
            print(f"per-request: mean latency {st.mean(lat):.2f}s  mean tok/s {st.mean(tps):.1f}")
    return {"ok": len(ok), "ct": ct, "recs": ok}


def compare(labels, datasets):
    """Pairwise exact-match identity, aligned by id (falls back to order)."""
    print("\n===== CONTENT IDENTITY (temperature=0 → speculative MTP/DSpark should MATCH AR) =====")
    base_label, base = labels[0], datasets[0]
    base_map = {r["id"]: r["gpt"] for r in base["recs"]}
    for label, ds in zip(labels[1:], datasets[1:]):
        common = matched = 0
        first_diff = None
        for r in ds["recs"]:
            if r["id"] in base_map:
                common += 1
                if r["gpt"] == base_map[r["id"]]:
                    matched += 1
                elif first_diff is None:
                    first_diff = r["id"]
        rate = 100 * matched / common if common else 0
        flag = "✅ identical" if matched == common and common else "⚠️  DIVERGES"
        print(f"{label} vs {base_label}: {matched}/{common} exact match ({rate:.1f}%)  {flag}"
              + (f"  first diff id={first_diff}" if first_diff else ""))
    # aggregate speed compare (mean per-request tok/s)
    print("\n----- per-request tok/s (higher = faster; MTP/DSpark should beat AR) -----")
    base_tps = None
    for label, ds in zip(labels, datasets):
        tps = [r["ctoks"] / r["latency"] for r in ds["recs"]
               if isinstance(r["ctoks"], (int, float)) and isinstance(r["latency"], (int, float)) and r["latency"] > 0]
        m = st.mean(tps) if tps else 0
        if base_tps is None:
            base_tps = m
        sp = f"  speedup {m/base_tps:.2f}x" if base_tps else ""
        print(f"{label}: {m:.1f} tok/s{sp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--labels", help="comma-separated labels, e.g. AR,MTP,DSpark")
    ap.add_argument("--wall-sec", type=float, help="wall-clock of the sample run (for aggregate throughput)")
    ap.add_argument("--total-rows", type=int, help="full dataset row count for extrapolation (e.g. 1400000)")
    ap.add_argument("--machines", type=int, default=1, help="machines for --num-shards scaling")
    args = ap.parse_args()

    labels = (args.labels.split(",") if args.labels
              else [f.split("/")[-1] for f in args.files])
    if len(labels) != len(args.files):
        print("!! --labels count must match number of files"); sys.exit(1)

    datasets = [summarize(lab, load(f)) for lab, f in zip(labels, args.files)]

    if len(args.files) >= 2:
        compare(labels, datasets)

    # aggregate throughput + full-run extrapolation (uses FIRST file as the sample)
    if args.wall_sec and args.wall_sec > 0:
        s = datasets[0]
        rows, toks = s["ok"], sum(s["ct"])
        print(f"\n===== AGGREGATE THROUGHPUT ({labels[0]} sample) =====")
        print(f"wall {args.wall_sec:.0f}s  rows {rows}  → {rows/args.wall_sec:.2f} rows/s  "
              f"{toks/args.wall_sec:,.0f} tok/s (aggregate, at your concurrency)")
        if args.total_rows and rows:
            full_s = args.total_rows / (rows / args.wall_sec)
            print(f"\n===== FULL-RUN EXTRAPOLATION ({args.total_rows:,} rows) =====")
            print(f"1 machine : {full_s/3600:.1f} h  ({full_s/86400:.2f} days)")
            if args.machines > 1:
                print(f"{args.machines} machines: {full_s/3600/args.machines:.1f} h  "
                      f"(linear via --num-shards {args.machines})")


if __name__ == "__main__":
    main()
