#!/usr/bin/env python3
"""Stitch fragmented DSV4-DSpark training logs into ONE continuous metric log.

Training runs in resume SEGMENTS: each kill+resume writes a new ``faithful_ep_<TS>.log`` and
``global_step`` continues (with a small overlap at the resume boundary). To see the FULL curve
you must merge the per-step records of several segment logs into one, keyed by ``global_step``.

Parsing is IDENTICAL to ``analyze_train_run.py.load()`` — scan ``key=value`` pairs and flush a
record at each ``global_step=`` (line-wrap agnostic). Output re-emits each record as ONE line
``k=v … global_step=N`` (``global_step`` LAST, so the re-parser's flush keeps records separate),
which ``analyze_train_run.py`` then reads natively as a single run.

De-dup: segments are ordered by mtime; on a duplicate ``global_step`` the LATER (resumed) segment
wins. Reports each segment's step range, de-duped overlaps, and GAPS (missing ``global_step`` spans
= where the log broke — the whole reason this exists).

Usage
-----
  # explicit chain (you pick the segment logs — order doesn't matter, sorted by mtime):
  python stitch_train_logs.py seg1.log seg2.log seg3.log --out full.log

  # then feed the stitched full curve to the analyzer (e.g. as the baseline of a NEW run):
  python analyze_train_run.py new.log --baseline full.log --out ./cmp

  # convenience: gather a chain by a save-path substring found in each log's checkpoint line
  # (best-effort — copy-resume can break the path link, so prefer the explicit form):
  python stitch_train_logs.py --chain ckpt_faithful_ep_20260729_092941 --run-dir . --out full.log
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

# Same pair regex as analyze_train_run.py: key=value with NO space around '=' (skips env dumps).
_PAIR = re.compile(r"([A-Za-z0-9_/]+)=(-?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|nan|inf))")
# trainer.py checkpoint lines carry the save-path (banner echoes are NOT in the log file):
#   "No previous training checkpoint found in '<save_path>'."  (fresh)
#   "Found checkpoint at <save_path>/<N>."                      (resume)
_CKPT_PATH = re.compile(r"(?:checkpoint found in|Found checkpoint at)\s*'?([^'\s]+)")


def parse_records(path: str) -> list[tuple[int, dict]]:
    """(global_step, {k: value_str}) records — flush on global_step, matching analyze.load()."""
    text = open(path, errors="ignore").read()
    recs: list[tuple[int, dict]] = []
    cur: dict[str, str] = {}
    for k, v in _PAIR.findall(text):
        cur[k] = v
        if k == "global_step":
            try:
                gs = int(v)
            except ValueError:
                cur = {}
                continue
            recs.append((gs, cur))
            cur = {}
    return recs


def log_mentions_chain(path: str, substr: str) -> bool:
    """Best-effort: does this log's checkpoint line reference the chain's save-path substr?

    The checkpoint message is emitted before training, so a 256 KB head read is enough."""
    head = open(path, errors="ignore").read(256 * 1024)
    return any(substr in m.group(1) for m in _CKPT_PATH.finditer(head))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stitch resume-segment training logs into one continuous global_step curve.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage")[-1],
    )
    ap.add_argument("logs", nargs="*", help="explicit segment log files (the chain; any order)")
    ap.add_argument("--chain", metavar="SAVEPATH_SUBSTR",
                    help="auto-gather logs whose checkpoint line references this save-path substr")
    ap.add_argument("--run-dir", default=".", help="dir scanned for --chain (default: cwd)")
    ap.add_argument("--glob", default="faithful_ep_*.log", help="--chain scan glob (default: faithful_ep_*.log)")
    ap.add_argument("--out", metavar="FILE", help="write combined log here (default: stdout)")
    args = ap.parse_args()

    logs = list(args.logs)
    if args.chain:
        for p in glob.glob(os.path.join(args.run_dir, args.glob)):
            if p not in logs and log_mentions_chain(p, args.chain):
                logs.append(p)
    if not logs:
        ap.error("no logs — pass explicit segment files or --chain SUBSTR")

    # chronological order = mtime; later segment overwrites on duplicate global_step (resume wins).
    logs = sorted(set(logs), key=os.path.getmtime)

    merged: dict[int, dict] = {}
    per_seg: list[tuple[str, int, int, int]] = []
    dropped = 0
    for p in logs:
        recs = parse_records(p)
        if not recs:
            print(f"  [skip] {os.path.basename(p)}: no global_step records", file=sys.stderr)
            continue
        steps = [gs for gs, _ in recs]
        per_seg.append((os.path.basename(p), len(recs), min(steps), max(steps)))
        for gs, rec in recs:
            if gs in merged:
                dropped += 1
            merged[gs] = rec  # later segment (later mtime) wins

    if not merged:
        ap.error("no metric records found in any log")

    order = sorted(merged)
    fh = open(args.out, "w") if args.out else sys.stdout
    for gs in order:
        rec = merged[gs]
        pairs = " ".join(f"{k}={v}" for k, v in rec.items() if k != "global_step")
        fh.write(f"{pairs} global_step={gs}\n")  # global_step LAST → re-parser flushes per line
    if args.out:
        fh.close()

    # ---- report to stderr (so stdout stays a clean log when --out is omitted) ----
    gaps: list[tuple[int, int]] = []
    prev = None
    for gs in order:
        if prev is not None and gs - prev > 1:
            gaps.append((prev + 1, gs - 1))
        prev = gs
    print(f"# stitched {len(per_seg)} segment(s) → {len(order)} unique steps "
          f"(global_step {order[0]}..{order[-1]})", file=sys.stderr)
    for name, n, lo, hi in per_seg:
        print(f"   {name}: {n} steps  [{lo}..{hi}]", file=sys.stderr)
    print(f"   overlaps de-duped (later segment won): {dropped}", file=sys.stderr)
    if gaps:
        miss = sum(b - a + 1 for a, b in gaps)
        print(f"   ⚠ {len(gaps)} GAP(s), {miss} steps missing (where the log broke):", file=sys.stderr)
        for a, b in gaps:
            print(f"     {a}..{b}  ({b - a + 1} missing)", file=sys.stderr)
    else:
        print("   ✓ no gaps — continuous global_step coverage", file=sys.stderr)
    if args.out:
        print(f"   → wrote {args.out}   (feed to: analyze_train_run.py {args.out})", file=sys.stderr)


if __name__ == "__main__":
    main()
