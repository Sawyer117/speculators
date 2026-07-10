#!/usr/bin/env python3
"""Summarize a speculators training log: loss / acceptance / timing / NaN.

Parses the rich-logger metric lines trainer.py emits each step
(``train/*``, ``profile/*``, ``mem/*``, ``lr``, ``global_step``) even when the
terminal wrapped them across physical lines, and prints a compact report:
  * the FIRST NaN step and the 10 steps leading into it (to see a divergence ramp
    vs a sudden overflow), plus the lr at that point;
  * loss trajectory (first / min / last-good) and whether it was rising pre-NaN;
  * acceptance progress (accept_len, full_acc);
  * per-stage timing medians with recompile spikes (fwd >> median) called out;
  * peak reserved memory.

Usage:
    python analyze_train_log.py <logfile>
    <cmd> 2>&1 | tee run.log            # capture, then:
    python analyze_train_log.py run.log
    python analyze_train_log.py -       # read stdin
"""
from __future__ import annotations

import math
import re
import sys
from statistics import median

# key=value with NO space around '=' (so env dumps like "LOCAL_RANK = 0" are skipped)
_PAIR = re.compile(r"([A-Za-z0-9_/]+)=(-?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|nan|inf))")


def load(path: str) -> list[dict]:
    text = sys.stdin.read() if path == "-" else open(path, errors="ignore").read()
    recs, cur = [], {}
    for k, v in _PAIR.findall(text):
        cur[k] = v
        if k == "global_step":  # always the last field of a record → flush
            recs.append(cur)
            cur = {}
    return recs


def f(rec: dict, key: str):
    v = rec.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return math.nan


def isnan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


def fmt(x) -> str:
    if x is None:
        return "-"
    if isnan(x):
        return "nan"
    return f"{x:.3g}"


def col(recs, key):
    return [f(r, key) for r in recs if f(r, key) is not None and not isnan(f(r, key))]


def step_of(r) -> int:
    return int(float(r["global_step"]))


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: analyze_train_log.py <logfile|->")
        return
    recs = load(sys.argv[1])
    if not recs:
        print("no metric records found (is this the right log?)")
        return
    print(f"records: {len(recs)}   steps: {step_of(recs[0])}..{step_of(recs[-1])}\n")

    # --- NaN onset ---
    nan_i = next((i for i, r in enumerate(recs) if isnan(f(r, "train/loss"))), None)
    if nan_i is None:
        print("NaN: none in train/loss  ✅")
    else:
        print(f"NaN: first at global_step={step_of(recs[nan_i])}  (record #{nan_i})")
        print("  10 steps leading in — watch for a rising-loss ramp (divergence) vs a jump:")
        for r in recs[max(0, nan_i - 10):nan_i + 1]:
            print(
                f"    {step_of(r):>7}: loss={fmt(f(r,'train/loss'))}"
                f"  ce={fmt(f(r,'train/ce_loss'))}"
                f"  tv={fmt(f(r,'train/tv_loss'))}"
                f"  conf={fmt(f(r,'train/confidence_loss'))}"
                f"  lr={fmt(f(r,'lr'))}"
            )
    print()

    # --- loss trajectory (good steps) ---
    good = [r for r in recs if f(r, "train/loss") is not None and not isnan(f(r, "train/loss"))]
    if good:
        traj = [(step_of(r), f(r, "train/loss")) for r in good]
        lo = min(traj, key=lambda t: t[1])
        hi = max(traj, key=lambda t: t[1])
        print(
            f"loss  first={traj[0][1]:.3f}@{traj[0][0]}"
            f"  min={lo[1]:.3f}@{lo[0]}"
            f"  max={hi[1]:.3f}@{hi[0]}"
            f"  last_good={traj[-1][1]:.3f}@{traj[-1][0]}"
        )

    # --- acceptance ---
    al = [(step_of(r), f(r, "train/accept_len")) for r in good if f(r, "train/accept_len") is not None]
    if al:
        hi = max(al, key=lambda t: t[1])
        print(f"accept_len  last={al[-1][1]:.3f}  max={hi[1]:.3f}@{hi[0]}  (block drafts up to block_size)")
    for k in ("train/full_acc", "train/accept_rate"):
        c = col(good, k)
        if c:
            print(f"{k.split('/')[1]:<12} last={c[-1]:.4g}  max={max(c):.4g}")
    print()

    # --- timing ---
    fwd = col(recs, "profile/fwd_ms")
    if fwd:
        mfwd = median(fwd)
        spikes = [
            (step_of(r), f(r, "profile/fwd_ms"))
            for r in recs
            if f(r, "profile/fwd_ms") and not isnan(f(r, "profile/fwd_ms")) and f(r, "profile/fwd_ms") > 5 * mfwd
        ]
        print("timing (median ms, recompile spikes excluded from medians):")
        print(
            f"  fetch={median(col(recs,'profile/fetch_ms')):.0f}"
            f"  fwd={mfwd:.0f}  bwd={median(col(recs,'profile/bwd_ms')):.0f}"
            f"  opt={median(col(recs,'profile/opt_ms')):.0f}"
            f"  step={median(col(recs,'profile/step_ms')):.0f}"
        )
        tp = col(recs, "profile/tokens_per_s")
        ff = col(recs, "profile/fetch_frac")
        if tp:
            print(f"  tokens_per_s median={median(tp):.0f}")
        if ff:
            print(f"  fetch_frac median={median(ff):.3f}  ->  {'HS/serve-bound' if median(ff) > 0.5 else 'COMPUTE-bound'}")
        print(f"  recompile spikes (fwd>5x median): {len(spikes)}" + (f"  e.g. {spikes[:6]}" if spikes else ""))

    mr = col(recs, "mem/max_reserved_gb")
    if mr:
        print(f"\nmem  max_reserved={max(mr):.1f} GB")


if __name__ == "__main__":
    main()
