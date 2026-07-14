#!/usr/bin/env python3
"""Analyze a speculators (DSV4-DSpark) training run: quality + timing + spikes, console + plots.

Parses the rich-logger metric lines trainer.py emits each step (``train/*``, ``profile/*``,
``mem/*``, ``lr``, ``global_step``), even when the terminal wrapped them across physical lines.
Prints a structured report AND (if matplotlib is present) writes PNGs to an output folder:

  quality   : loss (total/ce/tv/confidence), accept_len, accept_rate, full_acc, per-position
              accuracy (pos1..posK), confidence calibration (pred vs observed, cumprod bias).
  timing    : STEADY-STATE medians (spikes excluded) for fetch/fwd/bwd/opt/step + tokens/s.
  spikes    : detects recompile spikes (a stage >> its steady median), reports which stage,
              count, magnitude, inter-spike interval + periodicity, and total wall-clock overhead.
  HS/fetch  : fetch_ms / fetch_frac — is HS the bottleneck? (usually not once anchors are up).
  memory    : peak reserved / alloc.
  NaN       : first NaN step + the ramp into it.
  notes     : auto-surfaced findings (e.g. "spikes are fwd-only", "bwd dominates steady state").

Console-only works anywhere (no torch/matplotlib needed). Plots need matplotlib.

Usage:
    python analyze_train_run.py <logfile> [--out plots_dir]
    <cmd> 2>&1 | tee run.log ; python analyze_train_run.py run.log --out ./analysis
    python analyze_train_run.py -                     # read stdin, console only
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from statistics import median, mean, pstdev

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


def isnan(x) -> bool:
    return x is None or (isinstance(x, float) and x != x)


def f(rec: dict, key: str):
    v = rec.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def step_of(r) -> int:
    return int(f(r, "global_step") or -1)


def col(recs, key):
    """Finite values of `key` across records (drops None/nan/inf)."""
    out = []
    for r in recs:
        v = f(r, key)
        if v is not None and v == v and abs(v) != float("inf"):
            out.append(v)
    return out


def series(recs, key):
    """(step, value) pairs for a key, finite only — for plotting."""
    return [(step_of(r), f(r, key)) for r in recs
            if f(r, key) is not None and f(r, key) == f(r, key)]


def fmt(x) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and x != x:
        return "nan"
    return f"{x:.4g}"


def detect_positions(recs) -> list[str]:
    """How many per-position accuracy keys exist (pos1..posK) — K = block_size-1 = gamma."""
    ks = set()
    for r in recs:
        for k in r:
            m = re.match(r"train/position_(\d+)_acc", k)
            if m:
                ks.add(int(m.group(1)))
    return [f"train/position_{i}_acc" for i in sorted(ks)]


def spike_report(recs, key, k_thresh=3.0):
    """Steady median (spikes excluded) + list of spike (step, value) for a timing stage."""
    vals = col(recs, key)
    if not vals:
        return None
    med0 = median(vals)
    # iteratively exclude >k*median to get a spike-free steady baseline
    steady = [v for v in vals if v <= k_thresh * med0]
    med = median(steady) if steady else med0
    spikes = [(step_of(r), f(r, key)) for r in recs
              if f(r, key) is not None and f(r, key) == f(r, key) and f(r, key) > k_thresh * med]
    return {"median": med, "steady_med": med, "spikes": spikes, "n": len(vals)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile")
    ap.add_argument("--out", default=None, help="folder for PNG plots (created if missing)")
    ap.add_argument("--spike-k", type=float, default=3.0, help="spike = stage > k*steady-median")
    args = ap.parse_args()

    recs = load(args.logfile)
    if not recs:
        print("!! no metric records parsed — is this a trainer.py rich-logger log?")
        return
    steps = [step_of(r) for r in recs]
    print("=" * 78)
    print(f" TRAINING RUN ANALYSIS   {len(recs)} steps   (global_step {min(steps)}..{max(steps)})")
    print("=" * 78)

    # ---------------- NaN ----------------
    # Only a REAL nan counts — a missing train/loss just means a truncated/partial record
    # (e.g. a spike step whose block wrapped), not a divergence.
    def _is_real_nan(r):
        v = f(r, "train/loss")
        return v is not None and v != v
    nan_i = next((i for i, r in enumerate(recs) if _is_real_nan(r)), None)
    if nan_i is None:
        print("NaN            : none in train/loss  ✅")
    else:
        print(f"NaN            : ⚠️ first at step {step_of(recs[nan_i])} — ramp:")
        for r in recs[max(0, nan_i - 8):nan_i + 1]:
            print(f"                 {step_of(r):>7}: loss={fmt(f(r,'train/loss'))}"
                  f" ce={fmt(f(r,'train/ce_loss'))} grad_norm={fmt(f(r,'profile/grad_norm'))}")

    # ---------------- quality ----------------
    good = [r for r in recs if f(r, "train/loss") is not None and not isnan(f(r, "train/loss"))]
    def last_n_med(key, n=20):
        v = col(good[-n:], key)
        return median(v) if v else None
    print("\n-- QUALITY (last-20-step medians) " + "-" * 44)
    lt = [(step_of(r), f(r, "train/loss")) for r in good if f(r, "train/loss") is not None]
    if lt:
        lo = min(lt, key=lambda t: t[1])
        print(f"loss           : first {fmt(lt[0][1])} → min {fmt(lo[1])}@{lo[0]} → last {fmt(lt[-1][1])}"
              f"   (ce {fmt(last_n_med('train/ce_loss'))} | tv {fmt(last_n_med('train/tv_loss'))})")
    al = [(step_of(r), f(r, "train/accept_len")) for r in good if f(r, "train/accept_len") is not None]
    if al:
        hi = max(al, key=lambda t: t[1])
        # simple trend: last-20 median vs first-20 median
        first_al = median([v for _, v in al[:20]]) if len(al) >= 2 else al[0][1]
        last_al = median([v for _, v in al[-20:]])
        trend = "↑" if last_al > first_al + 1e-3 else ("↓" if last_al < first_al - 1e-3 else "→")
        print(f"accept_len     : first {fmt(first_al)} → last {fmt(last_al)} {trend}  (max {fmt(hi[1])}@{hi[0]})"
              f"   [block ceiling = block_size; target released draft AL 3.94 @ num_spec=5]")
    for k, lab in [("train/accept_rate", "accept_rate"), ("train/full_acc", "full_acc")]:
        v = last_n_med(k)
        if v is not None:
            print(f"{lab:15}: {fmt(v)}")
    pos_keys = detect_positions(recs)
    if pos_keys:
        vals = [last_n_med(k) for k in pos_keys]
        print(f"per-position   : " + "  ".join(f"p{i+1}={fmt(v)}" for i, v in enumerate(vals))
              + f"   (γ={len(pos_keys)} draft positions)")
    # confidence calibration
    cl = last_n_med("train/confidence_loss")
    if cl is not None:
        print(f"confidence     : loss {fmt(cl)} | abs_err {fmt(last_n_med('train/confidence_abs_error'))}"
              f" | pred_mean {fmt(last_n_med('train/confidence_pred_mean'))}"
              f" vs accept_rate {fmt(last_n_med('train/accept_rate'))}"
              f" | cumprod_bias {fmt(last_n_med('train/confidence_cumprod_bias'))}")

    # ---------------- timing (steady-state) ----------------
    print("\n-- TIMING (steady-state medians, spikes excluded) " + "-" * 28)
    stages = ["profile/fetch_ms", "profile/fwd_ms", "profile/bwd_ms", "profile/opt_ms", "profile/step_ms"]
    reps = {s: spike_report(recs, s, args.spike_k) for s in stages}
    for s in stages:
        r = reps[s]
        if r:
            print(f"{s.split('/')[1]:10}: {r['steady_med']:8.1f} ms   (steady)")
    tp = col(recs, "profile/tokens_per_s")
    if tp:
        # steady tokens/s excludes spike steps (very low tps)
        steady_tp = [v for v in tp if v > 0.5 * median(sorted(tp)[len(tp)//2:])]
        print(f"tokens/s   : {median(tp):8.0f} (median incl spikes) | {median(steady_tp):.0f} (steady)")

    # ---------------- spikes ----------------
    print("\n-- SPIKES (recompile / stalls) " + "-" * 47)
    step_r = reps["profile/step_ms"]
    fwd_r, bwd_r = reps["profile/fwd_ms"], reps["profile/bwd_ms"]
    if step_r:
        spikes = step_r["spikes"]
        smed = step_r["steady_med"]
        if not spikes:
            print(f"step_ms spikes : none > {args.spike_k}× steady ({smed:.0f} ms)  ✅")
        else:
            mags = [v for _, v in spikes]
            # attribute each spike to the stage that blew up
            fwd_spk = {s for s, _ in (fwd_r["spikes"] if fwd_r else [])}
            bwd_spk = {s for s, _ in (bwd_r["spikes"] if bwd_r else [])}
            n_fwd = sum(1 for s, _ in spikes if s in fwd_spk)
            n_bwd = sum(1 for s, _ in spikes if s in bwd_spk)
            overhead = sum(v - smed for _, v in spikes)
            total = sum(col(recs, "profile/step_ms"))
            ints = [spikes[i][0] - spikes[i-1][0] for i in range(1, len(spikes))]
            print(f"step_ms spikes : {len(spikes)} steps > {args.spike_k}× steady ({smed:.0f} ms)")
            print(f"  magnitude    : max {max(mags):.0f} ms ({max(mags)/smed:.0f}×) | median spike {median(mags):.0f} ms")
            print(f"  stage        : {n_fwd} fwd-side, {n_bwd} bwd-side  "
                  f"→ {'FWD-only (MoE new-shape recompile)' if n_bwd == 0 else 'mixed'}")
            print(f"  frequency    : every ~{median(ints):.0f} steps (min {min(ints)}, max {max(ints)})" if ints
                  else "  frequency    : (single spike)")
            print(f"  overhead     : {overhead/1000:.1f} s total = {100*overhead/total:.0f}% of wall-clock"
                  f"  → the #1 throughput lever (see §5 fixed-shape MoE padding)")
            worst = sorted(spikes, key=lambda t: -t[1])[:5]
            print("  worst steps  : " + ", ".join(f"{s}({v/1000:.1f}s)" for s, v in worst))

    # ---------------- HS / memory ----------------
    ff = col(recs, "profile/fetch_frac")
    fm = col(recs, "profile/fetch_ms")
    if ff:
        print("\n-- HS FETCH " + "-" * 66)
        print(f"fetch_frac : median {median(ff):.3f} | max {max(ff):.3f}   "
              f"→ {'HS NOT a bottleneck ✅' if median(ff) < 0.1 else '⚠️ HS is stalling — raise max_anchors or check serve'}")
        print(f"fetch_ms   : median {median(fm):.0f} | max {max(fm):.0f} ms")
    mr, ma = col(recs, "mem/max_reserved_gb"), col(recs, "mem/max_alloc_gb")
    if mr:
        print("\n-- MEMORY " + "-" * 68)
        print(f"reserved   : max {max(mr):.1f} GB   | alloc max {max(ma):.1f} GB"
              f"   ({100*max(mr)/64:.0f}% of a 64 GB A2 card)")

    # ---------------- auto notes ----------------
    print("\n-- NOTABLE " + "-" * 67)
    notes = []
    if fwd_r and bwd_r and bwd_r["steady_med"] > fwd_r["steady_med"]:
        notes.append(f"bwd ({bwd_r['steady_med']:.0f}ms) dominates fwd ({fwd_r['steady_med']:.0f}ms) in steady state.")
    if step_r and step_r["spikes"]:
        ov = sum(v - step_r["steady_med"] for _, v in step_r["spikes"])
        tot = sum(col(recs, "profile/step_ms"))
        if ov > 0.3 * tot:
            notes.append(f"recompile spikes eat {100*ov/tot:.0f}% of wall-clock — fixing them (fixed-shape "
                         f"MoE padding) would ~{tot/(tot-ov):.1f}× throughput.")
    if al and last_al < 3.94:
        notes.append(f"accept_len {last_al:.2f} < released-draft 3.94 — still training/warming; keep going.")
    cl = last_n_med("train/confidence_loss")
    pm, ar = last_n_med("train/confidence_pred_mean"), last_n_med("train/accept_rate")
    if pm is not None and ar is not None and abs(pm - ar) > 0.1:
        notes.append(f"confidence pred_mean {pm:.2f} vs accept_rate {ar:.2f} — calibration gap, watch it.")
    if not notes:
        notes.append("nothing anomalous flagged.")
    for n in notes:
        print(f"  • {n}")

    # ---------------- plots ----------------
    if args.out:
        _plots(recs, good, pos_keys, reps, args.out)
    print("=" * 78)


def _plots(recs, good, pos_keys, reps, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\n(plots skipped: matplotlib not installed — `pip install matplotlib`; console report above is complete)")
        return
    os.makedirs(out, exist_ok=True)

    def xy(key, recs_=recs):
        s = series(recs_, key)
        return [a for a, _ in s], [b for _, b in s]

    # 1) loss
    plt.figure(figsize=(9, 4))
    for k, lab in [("train/loss", "total"), ("train/ce_loss", "ce"),
                   ("train/tv_loss", "tv"), ("train/confidence_loss", "confidence")]:
        x, y = xy(k, good)
        if x:
            plt.plot(x, y, lw=0.8, label=lab)
    plt.xlabel("step"); plt.ylabel("loss"); plt.legend(); plt.title("Loss"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{out}/loss.png", dpi=120); plt.close()

    # 2) acceptance + per-position
    plt.figure(figsize=(9, 4))
    for k, lab in [("train/accept_len", "accept_len"), ("train/accept_rate", "accept_rate"),
                   ("train/full_acc", "full_acc")]:
        x, y = xy(k, good)
        if x:
            plt.plot(x, y, lw=0.9, label=lab)
    plt.axhline(3.94, ls="--", c="grey", lw=.8, label="released AL 3.94")
    plt.xlabel("step"); plt.ylabel("accept"); plt.legend(); plt.title("Acceptance"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{out}/acceptance.png", dpi=120); plt.close()

    if pos_keys:
        plt.figure(figsize=(9, 4))
        for i, k in enumerate(pos_keys):
            x, y = xy(k, good)
            if x:
                plt.plot(x, y, lw=0.8, label=f"pos{i+1}")
        plt.xlabel("step"); plt.ylabel("per-position acc"); plt.legend(); plt.title("Per-position draft accuracy")
        plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(f"{out}/position_acc.png", dpi=120); plt.close()

    # 3) confidence calibration
    plt.figure(figsize=(9, 4))
    for k, lab in [("train/confidence_pred_mean", "pred_mean"), ("train/accept_rate", "observed accept_rate"),
                   ("train/confidence_abs_error", "abs_error"), ("train/confidence_cumprod_bias", "cumprod_bias")]:
        x, y = xy(k, good)
        if x:
            plt.plot(x, y, lw=0.8, label=lab)
    plt.xlabel("step"); plt.ylabel("confidence"); plt.legend(); plt.title("Confidence calibration"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{out}/confidence.png", dpi=120); plt.close()

    # 4) timing (log-y so spikes + steady both visible)
    plt.figure(figsize=(9, 4))
    for k, lab in [("profile/fetch_ms", "fetch/HS"), ("profile/fwd_ms", "fwd"),
                   ("profile/bwd_ms", "bwd"), ("profile/opt_ms", "opt"), ("profile/step_ms", "step")]:
        x, y = xy(k)
        if x:
            plt.plot(x, y, lw=0.7, label=lab)
    plt.yscale("log"); plt.xlabel("step"); plt.ylabel("ms (log)"); plt.legend(ncol=5, fontsize=8)
    plt.title("Per-stage time (log-y — spikes are the recompiles)"); plt.grid(alpha=.3, which="both")
    plt.tight_layout(); plt.savefig(f"{out}/timing.png", dpi=120); plt.close()

    # 5) steady-state fwd/bwd distribution (spikes clipped)
    plt.figure(figsize=(9, 4))
    for k, lab in [("profile/fwd_ms", "fwd"), ("profile/bwd_ms", "bwd")]:
        r = reps[k]
        vals = [v for v in col(recs, k) if v <= 3 * r["steady_med"]]
        if vals:
            plt.hist(vals, bins=40, alpha=.6, label=f"{lab} (med {r['steady_med']:.0f}ms)")
    plt.xlabel("ms"); plt.ylabel("count"); plt.legend(); plt.title("Steady-state fwd/bwd distribution")
    plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(f"{out}/steady_hist.png", dpi=120); plt.close()

    print(f"\n📊 plots → {out}/  (loss, acceptance, position_acc, confidence, timing, steady_hist)")


if __name__ == "__main__":
    main()
