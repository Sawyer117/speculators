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
    python analyze_train_run.py                       # DEFAULT: newest *.log in $RUN (the run dir)
    python analyze_train_run.py <dir>                 # newest *.log with metrics in <dir>
    python analyze_train_run.py <logfile> [--out plots_dir]
    <cmd> 2>&1 | tee run.log ; python analyze_train_run.py run.log --out ./analysis
    python analyze_train_run.py -                     # read stdin, console only

The run dir defaults to $RUN or /home/a00652497/dspark_austin/run (matches train_dsv4_dspark.sh,
which writes $RUN/faithful_ep_<ts>.log + a rank0 mirror train_*.log). Override with RUN=... .
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from collections import Counter
from statistics import median, mean, pstdev

# where train_dsv4_dspark.sh writes logs (RUN=... in that script); override via env.
DEFAULT_RUN_DIR = os.environ.get("RUN", "/home/a00652497/dspark_austin/run")


def _has_metrics(path, tail_bytes=500_000) -> bool:
    """True if the file's tail contains trainer metric records (a real run, not startup/crash)."""
    try:
        with open(path, "rb") as fh:
            if os.path.getsize(path) > tail_bytes:
                fh.seek(-tail_bytes, os.SEEK_END)
            return b"global_step=" in fh.read()
    except OSError:
        return False


def resolve_log(path: str) -> str:
    """A file → itself; '-' → stdin; a directory (default) → newest *.log with metrics."""
    if path == "-":
        return "-"
    if os.path.isdir(path):
        cands = sorted(glob.glob(os.path.join(path, "*.log")), key=os.path.getmtime, reverse=True)
        if not cands:
            raise SystemExit(f"!! no *.log files in {path}  (set RUN=<dir> or pass a file)")
        for c in cands:
            if _has_metrics(c):
                age = (time.time() - os.path.getmtime(c)) / 60
                extra = "" if len(cands) == 1 else f"  (newest with metrics of {len(cands)} logs)"
                print(f">>> latest log in {path}:\n    {c}   [modified {age:.0f} min ago]{extra}\n")
                return c
        raise SystemExit(f"!! {len(cands)} *.log in {path} but none contain training metrics (global_step)")
    if not os.path.exists(path):
        raise SystemExit(f"!! not found: {path}  (default run dir is {DEFAULT_RUN_DIR}; set RUN= or pass a path)")
    return path

# key=value with NO space around '=' (so env dumps like "LOCAL_RANK = 0" are skipped)
_PAIR = re.compile(r"([A-Za-z0-9_/]+)=(-?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|nan|inf))")


def load(path: str) -> tuple[list[dict], str]:
    text = sys.stdin.read() if path == "-" else open(path, errors="ignore").read()
    recs, cur = [], {}
    for k, v in _PAIR.findall(text):
        cur[k] = v
        if k == "global_step":  # always the last field of a record → flush
            recs.append(cur)
            cur = {}
    return recs, text


def checkpoint_steps(text: str) -> set[int]:
    """Steps where a checkpoint SAVE happened (the save cost gets misread as a fetch/step spike).

    The 'Saving checkpoint' log line has no global_step of its own, so attribute it to the last
    step logged before it; the save shows up on that step or the next one."""
    steps: set[int] = set()
    for m in re.finditer(r"Saving checkpoint|Checkpoint saved", text):
        gs = re.findall(r"global_step=(\d+)", text[: m.start()])
        if gs:
            s = int(gs[-1])
            steps |= {s, s + 1}
    return steps


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
    ap.add_argument("logfile", nargs="?", default=DEFAULT_RUN_DIR,
                    help=f"log file, or a directory to auto-pick its newest *.log (default: {DEFAULT_RUN_DIR})")
    ap.add_argument("--out", default=None, help="folder for PNG plots (created if missing)")
    ap.add_argument("--spike-k", type=float, default=3.0, help="spike = stage > k*steady-median")
    ap.add_argument("--recent", type=int, default=500, help="window (steps) for the recent-dynamics trend")
    ap.add_argument("--skip", type=int, default=0,
                    help="drop global_steps < N (exclude shape-warmup / resume HS-regeneration) for a clean steady verdict")
    args = ap.parse_args()

    recs, raw_text = load(resolve_log(args.logfile))
    ckpt_steps = checkpoint_steps(raw_text)
    if not recs:
        print("!! no metric records parsed — is this a trainer.py rich-logger log?")
        return
    if args.skip > 0:
        kept = [r for r in recs if step_of(r) >= args.skip]
        if kept:
            print(f"(--skip {args.skip}: dropped {len(recs) - len(kept)} warmup/regen steps; analyzing {len(kept)})")
            recs = kept
    steps = [s for s in (step_of(r) for r in recs) if s >= 0]  # skip a leading partial record
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

    # ---------------- recent dynamics (is it STILL learning?) ----------------
    N = args.recent
    def wtrend(key, higher_better):
        s = [(step_of(r), f(r, key)) for r in good if f(r, key) is not None and step_of(r) >= 0]
        if len(s) < 40:
            return None
        n = N if len(s) >= 2 * N else max(15, len(s) // 3)
        recent = [v for _, v in s[-n:]]
        prior = [v for _, v in s[-2 * n:-n]] or [v for _, v in s[:-n]] or recent
        pr, rc, d = median(prior), median(recent), median([v for _, v in s[-n:]]) - median(prior)
        noise = 2 * (pstdev(recent) / max(1, len(recent) ** 0.5)) if len(recent) > 1 else 0
        if higher_better:
            verdict = "↑ improving" if d > noise else ("↓ WORSENING" if d < -noise else "→ plateaued")
        else:
            verdict = "↓ improving" if d < -noise else ("↑ WORSENING" if d > noise else "→ plateaued")
        return pr, rc, d, verdict, (s[-n][0], s[-1][0]), n
    print(f"\n-- RECENT DYNAMICS (last ~{N} steps vs the {N} before) " + "-" * 20)
    verdicts = {}
    for key, lab, hb in [("train/loss", "loss", False), ("train/accept_len", "accept_len", True),
                         ("train/full_acc", "full_acc", True)]:
        t = wtrend(key, hb)
        if t:
            pr, rc, d, verdict, span, n = t
            verdicts[lab] = verdict
            flag = "⚠️ " if "WORSENING" in verdict else ""
            print(f"{lab:11}: prior {fmt(pr)} → recent {fmt(rc)}  (Δ{d:+.3f})  {flag}{verdict}"
                  f"   [steps {span[0]}..{span[1]}]")
    if verdicts.get("loss") == "↑ WORSENING" and verdicts.get("accept_len") == "↓ WORSENING":
        print("  ⚠️ BOTH loss↑ and accept_len↓ over the recent window → possible divergence / overfit / lr too high.")
    elif all("plateaued" in v for v in verdicts.values()) and verdicts:
        print("  → plateaued on all metrics — near-converged, or stuck (try lr schedule / more data if far from 3.94).")

    # ---------------- timing (steady-state) ----------------
    print("\n-- TIMING (per stage: STEADY vs EFFECTIVE-avg incl spikes) " + "-" * 16)
    stages = ["profile/fetch_ms", "profile/fwd_ms", "profile/bwd_ms", "profile/opt_ms", "profile/step_ms"]
    _labels = {"profile/fetch_ms": "HS fetch", "profile/fwd_ms": "fwd", "profile/bwd_ms": "bwd",
               "profile/opt_ms": "opt", "profile/step_ms": "step"}
    reps = {s: spike_report(recs, s, args.spike_k) for s in stages}
    def stage_avg(key):
        vv = [(step_of(r), f(r, key)) for r in recs if f(r, key) is not None]
        if key == "profile/fetch_ms":  # exclude checkpoint-save steps (their cost is misread as fetch)
            vv = [(st, v) for st, v in vv if st not in ckpt_steps]
        return mean([v for _, v in vv]) if vv else None
    print(f"  {'stage':10} {'steady':>10}   {'effective avg':>14}   {'×':>5}")
    for s in stages:
        r = reps[s]
        if not r:
            continue
        avg = stage_avg(s) or r["steady_med"]
        note = "  (fetch: ckpt-misreads excluded)" if s == "profile/fetch_ms" else ""
        print(f"  {_labels[s]:10} {r['steady_med']:8.0f}ms   {avg:12.0f}ms   {avg/r['steady_med']:4.1f}×{note}")
    tp = col(recs, "profile/tokens_per_s")
    if tp:
        steady_tp = [v for v in tp if v > 0.5 * median(sorted(tp)[len(tp)//2:])]
        print(f"tokens/s   : {median(steady_tp):8.0f} (steady)")
    # EFFECTIVE (incl spikes) — the REAL average the run actually achieves.
    pairs = [(f(r, "profile/tokens_per_s"), f(r, "profile/step_ms")) for r in recs
             if f(r, "profile/tokens_per_s") and f(r, "profile/step_ms")]
    if pairs:
        tot_ms = sum(s for _, s in pairs)
        tot_tok = sum(t * s / 1000 for t, s in pairs)
        eff = tot_ms / len(pairs)
        steady = reps["profile/step_ms"]["steady_med"]
        real_tps = tot_tok / (tot_ms / 1000)
        print(f"EFFECTIVE  : {eff:8.0f} ms/step incl spikes ({eff/steady:.1f}× the steady {steady:.0f} ms)")
        print(f"             → REAL throughput ~{real_tps:.0f} tok/s (vs {median(steady_tp):.0f} steady)"
              f"  |  {tot_ms/1000/3600:.1f} h wall-clock so far")

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
            fetch_r = reps["profile/fetch_ms"]
            fwd_spk = {s for s, _ in (fwd_r["spikes"] if fwd_r else [])}
            bwd_spk = {s for s, _ in (bwd_r["spikes"] if bwd_r else [])}
            fetch_spk = {s for s, _ in (fetch_r["spikes"] if fetch_r else [])}
            # attribution priority: checkpoint-save (known, periodic) > HS-stall > recompile > bwd.
            # The save cost gets misread as fetch/step, so identify it FIRST from the log markers.
            def _cause(s):
                if s in ckpt_steps:  return "ckpt"
                if s in fetch_spk:   return "fetch"
                if s in fwd_spk:     return "fwd"
                if s in bwd_spk:     return "bwd"
                return "other"
            by = Counter(_cause(s) for s, _ in spikes)
            ov_ckpt = sum(v - smed for s, v in spikes if s in ckpt_steps)
            overhead = sum(v - smed for _, v in spikes)
            total = sum(col(recs, "profile/step_ms"))
            ints = [spikes[i][0] - spikes[i-1][0] for i in range(1, len(spikes))]
            print(f"step_ms spikes : {len(spikes)} steps > {args.spike_k}× steady ({smed:.0f} ms)")
            print(f"  magnitude    : max {max(mags):.0f} ms ({max(mags)/smed:.0f}×) | median spike {median(mags):.0f} ms")
            print(f"  cause        : {by['fwd']} MoE recompile (fwd), {by['ckpt']} checkpoint-save (EP-DCP gather), "
                  f"{by['fetch']} HS-stall (fetch), {by['bwd']} bwd")
            if by["ckpt"]:
                print(f"  ↳ checkpoints: {by['ckpt']} saves cost {ov_ckpt/1000:.0f} s ({100*ov_ckpt/total:.0f}% of wall-clock)"
                      f"  → raise --checkpoint-freq (save less often) to cut this")
            print(f"  frequency    : every ~{median(ints):.0f} steps (min {min(ints)}, max {max(ints)})" if ints
                  else "  frequency    : (single spike)")
            print(f"  overhead     : {overhead/1000:.1f} s total = {100*overhead/total:.0f}% of wall-clock"
                  f"  → see BOTTLENECK BREAKDOWN below for the recompile-vs-HS split")
            worst = sorted(spikes, key=lambda t: -t[1])[:5]
            print("  worst steps  : " + ", ".join(f"{s}({v/1000:.1f}s)" for s, v in worst))

    # ---------------- bottleneck breakdown: recompile vs HS vs ckpt (compile-worth-it verdict) ----------------
    print("\n-- BOTTLENECK BREAKDOWN (where wall-clock goes; is the compile fix worth it?) " + "-" * 1)
    total = sum(col(recs, "profile/step_ms")) or 1.0
    fwd_steady = reps["profile/fwd_ms"]["steady_med"] or 0.0
    fetch_steady = reps["profile/fetch_ms"]["steady_med"] or 0.0
    step_steady = reps["profile/step_ms"]["steady_med"] or 0.0

    def _excess(key, steady, only_nonckpt=False):
        tot = 0.0
        for r in recs:
            if only_nonckpt and step_of(r) in ckpt_steps:
                continue
            v = f(r, key)
            if v is not None and v > steady:
                tot += v - steady
        return tot

    recompile_ov = _excess("profile/fwd_ms", fwd_steady)                     # excess fwd time = grouped-GEMM recompile
    hs_ov = _excess("profile/fetch_ms", fetch_steady, only_nonckpt=True)     # excess HS-fetch = serve stall (ckpt saves excluded)
    ckpt_ov = sum(max(0.0, f(r, "profile/step_ms") - step_steady) for r in recs
                  if step_of(r) in ckpt_steps and f(r, "profile/step_ms") is not None)
    floor = max(0.0, total - recompile_ov - hs_ov - ckpt_ov)
    print(f"  {'component':26} {'time':>9}   {'% wall-clock':>12}")
    for label, v in [("steady compute (floor)", floor), ("recompile (fwd excess)", recompile_ov),
                     ("HS fetch stall (excess)", hs_ov), ("checkpoint saves", ckpt_ov)]:
        print(f"  {label:26} {v/1000:7.1f}s   {100*v/total:11.1f}%")
    rc, hs = 100 * recompile_ov / total, 100 * hs_ov / total
    print("  ── verdict ──")
    if recompile_ov >= 2 * hs_ov and rc >= 10:
        print(f"    RECOMPILE-bound (fwd {rc:.0f}% vs HS {hs:.0f}%): compile + maybe_mark_dynamic is the permanent")
        print(f"    fix and IS worth the inductor_npu_ext setup — ~{rc:.0f}% of wall-clock is recoverable.")
    elif hs_ov >= 2 * recompile_ov and hs >= 10:
        print(f"    HS/SERVING-bound (HS {hs:.0f}% vs fwd {rc:.0f}%): compile WON'T help — the serve can't dump HS")
        print(f"    fast enough. Fix the HS pipeline (serve throughput / prefetch), NOT the MoE kernel.")
    elif max(rc, hs) < 10:
        print(f"    NEITHER dominates (fwd {rc:.0f}%, HS {hs:.0f}% — both <10%): steady compute is the floor.")
        print(f"    compile's inductor_npu_ext setup is NOT worth it; bucketing already suffices.")
    else:
        print(f"    MIXED (fwd {rc:.0f}%, HS {hs:.0f}%): both matter. Try a bigger DSPARK_MOE_BUCKET (cheap) +")
        print(f"    check the serve; reach for compile only if recompile stays high AFTER the shape-warmup.")
    print("  ⚠ run with --skip <N> to drop the first epoch (shape-warmup + resume HS-regen inflate BOTH bars).")

    # ---------------- HS / memory ----------------
    ff = col(recs, "profile/fetch_frac")
    fm = col(recs, "profile/fetch_ms")
    if ff:
        print("\n-- HS FETCH " + "-" * 66)
        print(f"fetch_frac : median {median(ff):.3f} → {'HS NOT the typical bottleneck ✅' if median(ff) < 0.1 else '⚠️ HS stalling most steps'}")
        hi = [(step_of(r), f(r, "profile/fetch_frac")) for r in recs
              if f(r, "profile/fetch_frac") is not None and f(r, "profile/fetch_frac") > 0.5]
        if hi:
            ckpt_hi = [s for s, _ in hi if s in ckpt_steps]
            real = [s for s, _ in hi if s not in ckpt_steps]
            print(f"           : {len(hi)} step(s) with fetch>50% → {len(ckpt_hi)} are CHECKPOINT-SAVE steps "
                  f"(save cost misread as fetch), {len(real)} are real HS starvation.")
            if real:
                print(f"             real HS stalls at steps {real[:6]} — serve produced HS slower than train consumed (rolling buffer empty).")
            else:
                print(f"             → so there is NO real HS starvation; the giant 'fetch' spikes are all checkpoint saves.")
        mx = max(fm)
        print(f"fetch_ms   : median {median(fm):.0f} ms | max {(mx/1000):.1f} s" if mx >= 1000 else
              f"fetch_ms   : median {median(fm):.0f} ms | max {mx:.0f} ms")
    mr, ma = col(recs, "mem/max_reserved_gb"), col(recs, "mem/max_alloc_gb")
    if mr:
        print("\n-- MEMORY " + "-" * 68)
        alloc = f"{max(ma):.1f} GB" if ma else "—"
        print(f"reserved   : max {max(mr):.1f} GB   | alloc max {alloc}"
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
        _plots(recs, good, pos_keys, reps, ckpt_steps, args.out)
    print("=" * 78)


def _plots(recs, good, pos_keys, reps, ckpt_steps, out):
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

    # 2) acceptance + per-position (overview)
    plt.figure(figsize=(9, 4))
    for k, lab in [("train/accept_len", "accept_len"), ("train/accept_rate", "accept_rate"),
                   ("train/full_acc", "full_acc")]:
        x, y = xy(k, good)
        if x:
            plt.plot(x, y, lw=0.9, label=lab)
    plt.axhline(3.94, ls="--", c="grey", lw=.8, label="released AL 3.94")
    plt.xlabel("step"); plt.ylabel("accept"); plt.legend(); plt.title("Acceptance"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{out}/acceptance.png", dpi=120); plt.close()

    # 2b) accept_len DEDICATED: raw + smoothed + HIGHLIGHTED paper baseline
    x, y = xy("train/accept_len", good)
    if x:
        import numpy as np
        plt.figure(figsize=(9.5, 4.8))
        plt.plot(x, y, lw=0.5, alpha=0.30, color="#7A8AA8", label="raw (per step)")
        w = max(11, len(y) // 60)  # smoothing window
        if len(y) >= w:
            ys = np.convolve(np.asarray(y, float), np.ones(w) / w, mode="valid")
            off = (w - 1) // 2
            plt.plot(x[off:off + len(ys)], ys, lw=2.4, color="#2E6CF6",
                     label=f"smoothed ({w}-step moving avg)")
        # HIGHLIGHTED paper / released-draft target
        plt.axhline(3.94, ls="--", lw=2.4, color="#D62828",
                    label="released draft AL = 3.94 (paper / vllm-ascend PR #11196)")
        plt.annotate("target 3.94", xy=(x[0], 3.94), xytext=(x[0], 3.99),
                     color="#D62828", fontsize=11, weight="bold")
        cur = median(y[-min(50, len(y)):])
        plt.axhline(cur, ls=":", lw=1.2, color="#1B8A4E")
        plt.annotate(f"current ~{cur:.2f}", xy=(x[-1], cur), xytext=(x[-1], cur - 0.18),
                     color="#1B8A4E", fontsize=10, ha="right", weight="bold")
        plt.ylim(1.0, max(4.2, (max(y) if y else 4) + 0.2))
        plt.xlabel("step"); plt.ylabel("acceptance length")
        plt.title("Acceptance length — raw vs smoothed (target = released-draft 3.94)")
        plt.legend(loc="lower right"); plt.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(f"{out}/accept_len.png", dpi=120); plt.close()

    if pos_keys:
        # BAR chart of the CURRENT (last-N-step median) per-position accuracy, value-labeled.
        n = min(50, len(good))
        vals = []
        for k in pos_keys:
            v = col(good[-n:], k)
            vals.append(median(v) if v else 0.0)
        labels = [f"pos{i+1}" for i in range(len(pos_keys))]
        plt.figure(figsize=(8.5, 4.8))
        bars = plt.bar(labels, vals, width=0.62, color="#2E6CF6", zorder=3)
        for b, v in zip(bars, vals):
            plt.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=11, weight="bold", color="#1B2538")
        # released-draft per-position ACCEPT marginals (a related-but-different metric) as a shape ref
        rel = [0.81, 0.68, 0.58, 0.48, 0.39]
        if len(pos_keys) == len(rel):
            plt.plot(labels, rel, "o--", color="#D62828", lw=1.6, ms=6, zorder=4,
                     label="released draft accept marginal (ref)")
            for i, r in enumerate(rel):
                plt.text(i, r + 0.02, f"{r:.2f}", ha="center", color="#D62828", fontsize=9)
            plt.legend(loc="upper right")
        plt.ylim(0, 1.0)
        plt.ylabel("greedy accuracy  (argmax == target)")
        plt.title(f"Per-position draft accuracy — last {n} steps (decays p1→p{len(pos_keys)})")
        plt.grid(axis="y", alpha=.3, zorder=0)
        plt.tight_layout(); plt.savefig(f"{out}/position_acc.png", dpi=120); plt.close()

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

    # 5) per-stage timing BARS: steady (solid) vs effective-avg incl spikes (hollow/hatched), log-y
    import numpy as np
    stage_defs = [("profile/fetch_ms", "HS fetch"), ("profile/fwd_ms", "fwd"),
                  ("profile/bwd_ms", "bwd"), ("profile/opt_ms", "opt"), ("profile/step_ms", "step")]
    labels, sv, av = [], [], []
    for key, lab in stage_defs:
        r = reps.get(key)
        if not r:
            continue
        vv = [(step_of(rec), f(rec, key)) for rec in recs if f(rec, key) is not None]
        if key == "profile/fetch_ms":  # exclude checkpoint-save steps (misread as fetch)
            vv = [(st, v) for st, v in vv if st not in ckpt_steps]
        labels.append(lab)
        sv.append(r["steady_med"])
        av.append(mean([v for _, v in vv]) if vv else r["steady_med"])
    if labels:
        x = np.arange(len(labels)); w = 0.38
        plt.figure(figsize=(9.8, 5.0))
        b1 = plt.bar(x - w / 2, sv, w, color="#2E6CF6", zorder=3, label="steady (spikes excluded)")
        b2 = plt.bar(x + w / 2, av, w, facecolor="none", edgecolor="#D62828", lw=1.8,
                     hatch="////", zorder=3, label="effective avg (incl spikes)")
        for bars in (b1, b2):
            for b in bars:
                h = b.get_height()
                plt.text(b.get_x() + b.get_width() / 2, h * 1.06,
                         f"{h:.0f}" if h >= 10 else f"{h:.1f}", ha="center", va="bottom", fontsize=9)
        plt.yscale("log"); plt.xticks(x, labels); plt.ylabel("ms (log)")
        plt.title("Per-stage time — steady (solid) vs effective-avg incl spikes (hatched)")
        plt.legend(loc="upper left"); plt.grid(axis="y", alpha=.3, which="both", zorder=0)
        plt.tight_layout(); plt.savefig(f"{out}/timing_bars.png", dpi=120); plt.close()

    print(f"\n📊 plots → {out}/  (loss, acceptance, accept_len[raw+smoothed+target], position_acc[bars],\n"
          f"                    confidence, timing[lines], timing_bars[steady-vs-avg])")


if __name__ == "__main__":
    main()
