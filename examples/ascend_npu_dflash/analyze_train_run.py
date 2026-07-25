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

Usage (see --help for the full list + examples):
    python analyze_train_run.py                       # DEFAULT: newest *.log in $RUN (the run dir)
    python analyze_train_run.py <dir>                 # newest *.log with metrics in <dir>
    python analyze_train_run.py <logfile> [--out plots_dir]
    python analyze_train_run.py CURRENT --baseline OLD --out ./cmp   # COMPARE two runs
    <cmd> 2>&1 | tee run.log ; python analyze_train_run.py run.log --out ./analysis
    python analyze_train_run.py -                     # read stdin, console only

COMPARE MODE: pass --baseline <log> to overlay an older reference run on every plot
(CURRENT = solid, BASELINE = dashed / grouped bars). The text report stays CURRENT-only,
plus a compact 'VS BASELINE' headline delta table (accept_len / loss / tok_s / recompile% / HS%).

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
    # Derived per-record field: fwd_ms MINUS the align/all-gather straggler barrier. The align barrier
    # (trainer.py, added between the fetch and fwd marks) makes a serve/HS straggler wait get COUNTED
    # INSIDE fwd_ms (align_ms ⊂ fwd_ms), where the old breakdown mis-attributed it as a "recompile" fwd
    # spike. Subtracting align_ms isolates the TRUE draft-forward compute; the removed slice is
    # re-attributed to the HS-straggler bucket. On logs with no align_ms (pre-diagnostic) this is a
    # no-op (fwd_compute == fwd). This is the crux of the A2 "recompile 41%" mislabel.
    for r in recs:
        if "profile/fwd_ms" in r:
            try:
                _fwd = float(r["profile/fwd_ms"])
                _al = float(r.get("profile/align_ms", 0.0) or 0.0)
                r["profile/fwd_compute_ms"] = max(0.0, _fwd - _al)
            except (TypeError, ValueError):
                pass
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


def epoch_boundaries(recs) -> list[tuple[int, int]]:
    """Global_steps at which the parsed ``epoch`` field increments → (step, new_epoch) per boundary.
    ``step`` = the first global_step of the new epoch; ``new_epoch`` = its 0-indexed epoch value
    (so the log's "epoch 2/5 started" == epoch field 1 == the first boundary here). Empty for a
    single-epoch run → no clutter. Used to draw vertical epoch markers on the step-axis plots."""
    out, prev = [], None
    for r in recs:
        e, s = f(r, "epoch"), step_of(r)
        if e is None or s < 0:
            continue
        e = int(e)
        if prev is not None and e > prev:
            out.append((s, e))
        prev = e
    return out


# tqdm's "  N/M [elapsed<remaining, rate]" bar — M = len(train_loader) = full-epoch step count.
# M is epoch-invariant, so it's readable even mid-epoch-0 (before any epoch boundary exists).
_TQDM_TOTAL = re.compile(r"(\d+)/(\d+)\s*\[\d[\d:]*<")


def _steps_per_epoch(recs, raw_text: str = "") -> int | None:
    """Full-epoch step count (``len(train_loader)``). Preferred source: the tqdm ``N/M [t<t`` bar in
    the raw log (``M`` = steps/epoch, present from step 1 so it works while still in epoch 0). Fallback:
    the spacing between parsed ``epoch`` boundary increments. ``None`` if neither is available
    (→ the footer/console just show steps/wall and skip the per-epoch projection)."""
    totals = [int(m.group(2)) for m in _TQDM_TOTAL.finditer(raw_text or "")]
    if totals:
        return max(totals)  # a resumed epoch's bar is a shorter slice → max = the true full length
    eb = epoch_boundaries(recs)
    if eb:
        steps0 = [step_of(r) for r in recs if step_of(r) >= 0]
        pts = ([min(steps0)] if steps0 else []) + [s for s, _ in eb]
        diffs = [b - a for a, b in zip(pts, pts[1:]) if b > a]
        if diffs:
            return int(median(diffs))
    return None


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


def _default_label(path: str) -> str:
    """A short display name for a run — the log's filename without extension."""
    if path in ("-", None):
        return "current"
    base = os.path.basename(os.path.normpath(path))
    return os.path.splitext(base)[0] or base


def _load_and_skip(path: str, skip: int, quiet: bool = False):
    """resolve → load → drop steps < skip. Returns (recs, ckpt_steps, steps_per_epoch) — the last two
    are None when there are no metrics / no epoch length is discoverable."""
    recs, raw_text = load(resolve_log(path))
    if not recs:
        return None, None, None, None
    if skip > 0:
        kept = [r for r in recs if step_of(r) >= skip]
        if kept:
            if not quiet:
                print(f"(--skip {skip}: dropped {len(recs) - len(kept)} warmup/regen steps; analyzing {len(kept)})")
            recs = kept
    return recs, checkpoint_steps(raw_text), _steps_per_epoch(recs, raw_text), raw_text


def _headline(recs, ckpt_steps, spike_k=3.0) -> dict:
    """The handful of numbers we compare between two runs (cheap; recomputed per run)."""
    good = [r for r in recs if f(r, "train/loss") is not None and not isnan(f(r, "train/loss"))]
    al = [f(r, "train/accept_len") for r in good if f(r, "train/accept_len") is not None]
    loss = col(good[-20:], "train/loss")

    def _steady(key):
        rr = spike_report(recs, key, spike_k)
        return rr["steady_med"] if rr else None

    fwd_s, fetch_s, step_s = _steady("profile/fwd_compute_ms"), _steady("profile/fetch_ms"), _steady("profile/step_ms")
    total = sum(col(recs, "profile/step_ms")) or 1.0
    align_ov = sum(col(recs, "profile/align_ms"))   # HS-straggler barrier wait (all overhead; not spiky)

    def _excess(key, steady, only_nonckpt=False):
        if steady is None:
            return 0.0
        t = 0.0
        for r in recs:
            if only_nonckpt and step_of(r) in ckpt_steps:
                continue
            v = f(r, key)
            if v is not None and v > steady:
                t += v - steady
        return t

    tp = col(recs, "profile/tokens_per_s")
    tps = None
    if tp:
        thr = 0.5 * median(sorted(tp)[len(tp) // 2:])
        st = [v for v in tp if v > thr]
        tps = median(st) if st else median(tp)
    steps = [s for s in (step_of(r) for r in recs) if s >= 0]
    return {
        "accept_len": median(al[-20:]) if al else None,
        "accept_len_max": max(al) if al else None,
        "loss": median(loss) if loss else None,
        "step_ms": step_s,
        "tokens_s": tps,
        "recompile_pct": 100 * _excess("profile/fwd_compute_ms", fwd_s) / total,  # align removed → true fwd
        "hs_pct": 100 * _excess("profile/fetch_ms", fetch_s, only_nonckpt=True) / total,
        "align_pct": 100 * align_ov / total,   # HS-straggler wait (the ex-"recompile" on A2)
        "span": (min(steps) if steps else 0, max(steps) if steps else 0),
    }


def _print_vs_baseline(recs, ckpt_steps, base_recs, base_ckpt, cur_label, base_label, spike_k):
    """A compact headline delta table: CURRENT vs BASELINE (the plots carry the full comparison)."""
    hc = _headline(recs, ckpt_steps, spike_k)
    hb = _headline(base_recs, base_ckpt, spike_k)
    bl, cl = (base_label or "baseline")[:13], (cur_label or "current")[:13]
    print("\n-- VS BASELINE (headline; the full report below is CURRENT only) " + "-" * 13)
    print(f"  steps: {bl}={hb['span'][0]}..{hb['span'][1]}   {cl}={hc['span'][0]}..{hc['span'][1]}")
    print(f"  {'metric':16} {bl:>13} {cl:>13} {'Δ (cur−base)':>16}")

    def _row(label, key, higher_better, fmt_="{:.3f}", unit=""):
        vb, vc = hb.get(key), hc.get(key)
        if vb is None or vc is None:
            print(f"  {label:16} {'—':>13} {'—':>13}")
            return
        d = vc - vb
        if abs(d) / max(abs(vb), 1e-9) < 0.005:   # within 0.5% → noise, not a real change
            mark = "→ ~same"
        elif (d > 0) == higher_better:
            mark = "✅ better"
        else:
            mark = "⚠️  worse"
        dfmt = fmt_.replace("{:", "{:+")
        print(f"  {label:16} {(fmt_.format(vb) + unit):>13} {(fmt_.format(vc) + unit):>13}"
              f" {(dfmt.format(d) + unit):>12}  {mark}")

    _row("accept_len",    "accept_len",     True)
    _row("accept_len max","accept_len_max", True)
    _row("loss",          "loss",           False)
    _row("tokens/s",      "tokens_s",       True,  "{:.0f}")
    _row("step_ms steady","step_ms",        False, "{:.0f}", "ms")
    _row("recompile %",   "recompile_pct",  False, "{:.1f}", "%")
    _row("HS fetch %",    "hs_pct",         False, "{:.1f}", "%")
    _row("HS straggler %","align_pct",      False, "{:.1f}", "%")


def _print_multi_table(runs, spike_k):
    """One row per run (CURRENT = row 1) + Δ accept_len vs CURRENT. Any number of runs."""
    hs = [(r["label"], _headline(r["recs"], r["ckpt"], spike_k)) for r in runs]
    cols = [("accept_len", "accept_len", "{:.3f}"), ("acc_len_max", "accept_len_max", "{:.3f}"),
            ("loss", "loss", "{:.3f}"), ("tok/s", "tokens_s", "{:.0f}"),
            ("step_ms", "step_ms", "{:.0f}"), ("recompile%", "recompile_pct", "{:.1f}"),
            ("HS%", "hs_pct", "{:.1f}"), ("HSstrag%", "align_pct", "{:.1f}")]
    print("\n-- MULTI-RUN COMPARE (row 1 = CURRENT) " + "-" * 38)
    print(f"  {'run':18} {'steps':>11} " + " ".join(f"{c[0]:>11}" for c in cols))
    for lbl, h in hs:
        span = f"{h['span'][0]}..{h['span'][1]}"
        cells = [(fmt.format(h[key]) if h.get(key) is not None else "—") for _, key, fmt in cols]
        print(f"  {lbl[:18]:18} {span:>11} " + " ".join(f"{c:>11}" for c in cells))
    cur = hs[0][1]
    if cur.get("accept_len") is not None:
        deltas = []
        for lbl, h in hs[1:]:
            v = h.get("accept_len")
            if v is not None:
                d = cur["accept_len"] - v
                deltas.append(f"{lbl[:14]} {d:+.3f}{'✅' if d > 0 else ('⚠️' if d < 0 else '→')}")
        if deltas:
            print("  Δ accept_len (CURRENT − each): " + " | ".join(deltas))


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="analyze_train_run.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Analyze a DSV4-DSpark training run: quality + timing + spikes + bottleneck breakdown,\n"
            "printed as a console report and (with --out) PNG plots.\n\n"
            "COMPARE MODE: pass --baseline <log> to overlay a second, older run on every plot\n"
            "(CURRENT = solid, BASELINE = dashed). The text report stays CURRENT-only; a compact\n"
            "'VS BASELINE' headline table (accept_len / loss / tok_s / recompile% / HS%) is added."
        ),
        epilog=(
            "TERMS:\n"
            "  CURRENT  = the run you're evaluating now (positional; gets the full text report)\n"
            "  BASELINE = the older reference run to compare against (--baseline; overlaid on plots)\n\n"
            "EXAMPLES:\n"
            "  # newest *.log in $RUN, console only\n"
            "  python analyze_train_run.py\n\n"
            "  # one run + plots\n"
            "  python analyze_train_run.py run.log --out ./analysis\n\n"
            "  # compare anchor=384 (current) against anchor=196 (baseline)\n"
            "  python analyze_train_run.py new.log --baseline old.log --out ./cmp\n"
            "  python analyze_train_run.py new.log --baseline old.log \\\n"
            "         --label anchor384 --baseline-label anchor196 --out ./cmp --skip 500\n"
        ),
    )
    ap.add_argument("logfile", nargs="?", default=DEFAULT_RUN_DIR,
                    help=f"CURRENT run: a log file, or a dir to auto-pick its newest *.log (default: {DEFAULT_RUN_DIR})")
    ap.add_argument("--baseline", default=None, metavar="LOG", nargs="+",
                    help="one or more BASELINE runs to compare against. ONE -> head-to-head (delta table + "
                         "dashed overlays). MULTIPLE -> multi-run overlay (each run its own colour + a compare table).")
    ap.add_argument("--label", default=None, metavar="NAME",
                    help="display name for the CURRENT run (default: log filename)")
    ap.add_argument("--baseline-label", default=None, metavar="NAME", nargs="+",
                    help="display name(s) for the BASELINE run(s), positionally matched (default: filenames)")
    ap.add_argument("--full-baseline", action="store_true",
                    help="show the BASELINE's ENTIRE curve; default ALIGNS it to the CURRENT run's "
                         "step range (a short current run isn't buried under a long baseline)")
    ap.add_argument("--out", default=None, metavar="DIR", help="folder for PNG plots (created if missing)")
    ap.add_argument("--spike-k", type=float, default=3.0, help="spike = stage > k*steady-median (default 3.0)")
    ap.add_argument("--recent", type=int, default=500, help="window (steps) for the recent-dynamics trend")
    ap.add_argument("--skip", type=int, default=0,
                    help="drop global_steps < N (exclude shape-warmup / resume HS-regen); applied to BOTH runs")
    return ap


def fwd_profiler_report(text: str) -> None:
    """Attribute forward-time spikes to sub-ops from DSPARK_PROFILE_FWD / DSPARK_PROFILE_MOE prints.

    Backward compatible: if the log has NO profiler lines (profilers off, or an older run), prints
    nothing. Complements the recompile%/hs% headline (which splits fwd_ms vs fetch_ms), by naming
    WHICH compute sub-op (MLA / mHC-Sinkhorn / MoE) — and which MoE internal — carries the spike."""
    fwd = re.findall(r"\[FWD_PROF\]\s+([\w.]+):\s+(\d+)\s*ms", text)
    moe = re.findall(
        r"\[MOE-PROF r\d+\][^\n]*?permute\+bincount=(\d+)ms\s+experts_gemm=(\d+)ms\s+unpermute=(\d+)ms",
        text,
    )
    if not fwd and not moe:
        return  # no profiler data — stay silent (backward compatible with pre-profiler logs)

    print("\n-- FWD PROFILER (sub-op spike attribution) " + "-" * 35)

    if fwd:
        agg = {}
        for tag, ms in fwd:
            agg.setdefault(tag, []).append(int(ms))
        rows = sorted(agg.items(), key=lambda kv: sum(kv[1]), reverse=True)
        print(f"  block sub-ops (DSPARK_PROFILE_FWD): {len(fwd)} spike print(s)")
        for tag, xs in rows:
            print(f"    {tag:10} n={len(xs):>4}  max={max(xs):>6}ms  mean={sum(xs) // len(xs):>6}ms  "
                  f"Σ={sum(xs):>8}ms")
        print(f"  → dominant fwd sub-op: {rows[0][0]}   (MLA=latent attn, mHC=Sinkhorn hyper-conn, MoE=experts)")

    if moe:
        pm = [int(a) for a, _, _ in moe]
        gm = [int(b) for _, b, _ in moe]
        um = [int(c) for _, _, c in moe]
        wins = {"permute+bincount": 0, "experts_gemm": 0, "unpermute": 0}
        for a, b, c in moe:
            trio = {"permute+bincount": int(a), "experts_gemm": int(b), "unpermute": int(c)}
            wins[max(trio, key=trio.get)] += 1
        top = max(wins, key=wins.get)
        print(f"  MoE internals (DSPARK_PROFILE_MOE): {len(moe)} spike print(s)")
        for name, xs in (("permute+bincount", pm), ("experts_gemm", gm), ("unpermute", um)):
            print(f"    {name:16} max={max(xs):>6}ms  mean={sum(xs) // len(xs):>6}ms")
        print(f"  → MoE spike usually in: {top}  ({wins[top]}/{len(moe)})")

    print("  cross-check the free headline: fetch_ms/align_ms high = HS-fetch stall (not compute).")


def main() -> None:
    args = _build_parser().parse_args()

    recs, ckpt_steps, steps_per_epoch, raw_text = _load_and_skip(args.logfile, args.skip)
    if recs is None:
        print("!! no metric records parsed — is this a trainer.py rich-logger log?")
        return
    cur_label = args.label or _default_label(args.logfile)

    # optional BASELINE run(s) to compare against. ONE -> head-to-head (delta table + dashed
    # overlays). MULTIPLE -> multi-run overlay (each run its own colour + a compare table). Each
    # baseline is aligned to the CURRENT run's step range by default (--full-baseline = full curves).
    baselines: list[dict] = []
    if args.baseline:
        cur_steps0 = [s for s in (step_of(r) for r in recs) if s >= 0]
        cur_max = max(cur_steps0) if cur_steps0 else 0
        blabels = args.baseline_label or []
        for i, bpath in enumerate(args.baseline):
            blabel = blabels[i] if i < len(blabels) else _default_label(bpath)
            brecs, bckpt, _, _ = _load_and_skip(bpath, args.skip, quiet=True)
            if brecs is None:
                print(f"!! baseline '{bpath}' had no metrics — skipping")
                continue
            if not args.full_baseline:
                clipped = [r for r in brecs if 0 <= step_of(r) <= cur_max]
                if clipped and len(clipped) < len(brecs):
                    bfa = [f(r, "train/accept_len") for r in brecs
                           if f(r, "train/accept_len") is not None]
                    bref = f"; full run reached accept_len ~{median(bfa[-20:]):.2f}" if bfa else ""
                    print(f">>> baseline '{blabel}' aligned to current's step range (≤{cur_max}): "
                          f"{len(clipped)}/{len(brecs)} steps shown{bref}.")
                    brecs = clipped
            baselines.append({
                "label": blabel, "recs": brecs, "ckpt": bckpt,
                "good": [r for r in brecs
                         if f(r, "train/loss") is not None and not isnan(f(r, "train/loss"))],
            })
        if baselines:
            print()

    steps = [s for s in (step_of(r) for r in recs) if s >= 0]  # skip a leading partial record
    cmp_tag = f"   [CURRENT '{cur_label}' vs {len(baselines)} baseline(s)]" if baselines else ""
    print("=" * 78)
    print(f" TRAINING RUN ANALYSIS   {len(recs)} steps   (global_step {min(steps)}..{max(steps)}){cmp_tag}")
    print("=" * 78)
    if len(baselines) == 1:
        b = baselines[0]
        _print_vs_baseline(recs, ckpt_steps, b["recs"], b["ckpt"], cur_label, b["label"], args.spike_k)
    elif len(baselines) >= 2:
        _print_multi_table(
            [{"label": cur_label, "recs": recs, "ckpt": ckpt_steps}]
            + [{"label": b["label"], "recs": b["recs"], "ckpt": b["ckpt"]} for b in baselines],
            args.spike_k,
        )

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

    fwd_profiler_report(raw_text)

    # ---------------- recent dynamics (is it STILL learning?) ----------------
    N = args.recent

    def _slope_per_1k(pts):
        """Least-squares slope of value vs step, scaled to Δ per 1000 steps (signed)."""
        if len(pts) < 20:
            return None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        den = sum((x - mx) ** 2 for x in xs)
        if den == 0:
            return None
        return (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den) * 1000.0

    def wtrend(key, higher_better):
        s = [(step_of(r), f(r, key)) for r in good if f(r, key) is not None and step_of(r) >= 0]
        if len(s) < 40:
            return None
        n = N if len(s) >= 2 * N else max(15, len(s) // 3)
        recent = [v for _, v in s[-n:]]
        prior = [v for _, v in s[-2 * n:-n]] or [v for _, v in s[:-n]] or recent
        pr, rc = median(prior), median(recent)
        d = rc - pr
        noise = 2 * (pstdev(recent) / max(1, len(recent) ** 0.5)) if len(recent) > 1 else 0
        # LONG-HORIZON least-squares slope (Δ/1000 steps) over the last ~third of the run — the
        # short median-vs-median window can't see a slow creep against per-batch accept_len noise.
        length = min(len(s), max(2000, len(s) // 3))
        slope = _slope_per_1k(s[-length:])
        lh_change = (slope / 1000.0 * length) if slope is not None else 0.0  # implied Δ over that window
        good_dir = slope is not None and ((slope > 0) == higher_better)
        if (d > noise) if higher_better else (d < -noise):
            verdict = "↑ improving" if higher_better else "↓ improving"
        elif (d < -noise) if higher_better else (d > noise):
            verdict = "↓ WORSENING" if higher_better else "↑ WORSENING"
        elif slope is not None and abs(lh_change) > 2 * noise:
            # short window flat, but the long horizon has moved more than the short-window noise
            verdict = (f"→ short-flat, SLOW-CREEP {'↑' if higher_better else '↓'}" if good_dir
                       else f"→ short-flat, SLOW {'↓' if higher_better else '↑'}-DRIFT")
        else:
            verdict = "→ plateaued"
        return pr, rc, d, verdict, (s[-n][0], s[-1][0]), n, slope

    print(f"\n-- RECENT DYNAMICS (short: last ~{N} vs prior {N}; + long-horizon slope) " + "-" * 6)
    verdicts = {}
    for key, lab, hb in [("train/loss", "loss", False), ("train/accept_len", "accept_len", True),
                         ("train/full_acc", "full_acc", True)]:
        t = wtrend(key, hb)
        if t:
            pr, rc, d, verdict, span, n, slope = t
            verdicts[lab] = verdict
            flag = "⚠️ " if ("WORSENING" in verdict or "DRIFT" in verdict) else ""
            sl = f"  LH {slope:+.3f}/1k" if slope is not None else ""
            print(f"{lab:11}: prior {fmt(pr)} → recent {fmt(rc)}  (Δ{d:+.3f}){sl}  {flag}{verdict}"
                  f"   [steps {span[0]}..{span[1]}]")
    if verdicts.get("loss") == "↑ WORSENING" and verdicts.get("accept_len") == "↓ WORSENING":
        print("  ⚠️ BOTH loss↑ and accept_len↓ over the recent window → possible divergence / overfit / lr too high.")
    elif verdicts and any("SLOW-CREEP" in v for v in verdicts.values()):
        print("  → short window flat but still SLOWLY improving on the long horizon — NOT converged; keep going.")
    elif verdicts and all(v == "→ plateaued" for v in verdicts.values()):
        print("  → truly plateaued on all metrics — near-converged or stuck (lr schedule / more data if far from 3.94).")

    # ---------------- timing (steady-state) ----------------
    print("\n-- TIMING (per stage: STEADY vs EFFECTIVE-avg incl spikes) " + "-" * 16)
    stages = ["profile/fetch_ms", "profile/fwd_ms", "profile/fwd_compute_ms", "profile/bwd_ms",
              "profile/opt_ms", "profile/step_ms"]
    _labels = {"profile/fetch_ms": "HS fetch", "profile/fwd_ms": "fwd(+align)",
               "profile/fwd_compute_ms": "fwd compute", "profile/bwd_ms": "bwd",
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

    # ---------------- bottleneck breakdown: fwd-compute vs HS-straggler vs HS-fetch vs ckpt ----------------
    print("\n-- BOTTLENECK BREAKDOWN (where wall-clock goes) " + "-" * 48)
    total = sum(col(recs, "profile/step_ms")) or 1.0
    # ★ recompile is measured on fwd_compute (= fwd_ms − align_ms). The align/all-gather barrier sits
    #   INSIDE the fwd_ms window, so an HS straggler (a rank whose serve HS arrived late) inflates fwd_ms
    #   and USED to be mis-labelled "recompile". align_ms isolates that wait → its own bucket below.
    fwd_steady = reps["profile/fwd_compute_ms"]["steady_med"] or 0.0
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

    recompile_ov = _excess("profile/fwd_compute_ms", fwd_steady)             # excess TRUE fwd = grouped-GEMM recompile / AscendC stall
    align_ov = sum(col(recs, "profile/align_ms"))                            # ALL of it = HS-straggler barrier wait (serve too slow)
    hs_ov = _excess("profile/fetch_ms", fetch_steady, only_nonckpt=True)     # excess HS-fetch = local H2D / load stall (ckpt saves excluded)
    ckpt_ov = sum(max(0.0, f(r, "profile/step_ms") - step_steady) for r in recs
                  if step_of(r) in ckpt_steps and f(r, "profile/step_ms") is not None)
    floor = max(0.0, total - recompile_ov - align_ov - hs_ov - ckpt_ov)
    print(f"  {'component':30} {'time':>9}   {'% wall-clock':>12}")
    for label, v in [("steady compute (floor)", floor), ("recompile (true-fwd excess)", recompile_ov),
                     ("HS straggler (align barrier)", align_ov), ("HS fetch stall (excess)", hs_ov),
                     ("checkpoint saves", ckpt_ov)]:
        print(f"  {label:30} {v/1000:7.1f}s   {100*v/total:11.1f}%")
    rc, hs, al = 100 * recompile_ov / total, 100 * hs_ov / total, 100 * align_ov / total
    serve = hs_ov + align_ov                     # both trace to the serve/HS pipeline
    sv = 100 * serve / total
    print("  ── verdict ──")
    if serve >= 2 * recompile_ov and sv >= 10:
        print(f"    HS/SERVING-bound (serve {sv:.0f}% = straggler {al:.0f}% + fetch {hs:.0f}%  vs  true-fwd {rc:.0f}%):")
        print("    compile/kernel work WON'T help — the 115/116 serve can't dump HS fast enough, so the fast")
        print("    ranks idle at the EP all-gather barrier. Fix the HS PIPELINE: raise serve throughput")
        print("    (more DP / EAGER=0 graph / bigger batch), prefetch HS, or RAISE --max-anchors so each")
        print("    training step is heavier and matches the serve's HS rate (trades step time for serve idle).")
    elif recompile_ov >= 2 * serve and rc >= 10:
        print(f"    TRUE-FWD-bound (fwd {rc:.0f}% vs serve {sv:.0f}%): a real forward stall, NOT an HS straggler.")
        print("    With jit_compile=False + eager GMM already in, suspect a per-shape AscendC build in a")
        print("    DIFFERENT op (npu_moe_token_permute/unpermute, MLA). PIN it: DSPARK_PROFILE_FWD=1")
        print("    DSPARK_PROFILE_FWD_MS=0 DSPARK_PROFILE_MOE=1 on a SHORT no-RECOMPUTE MAX_ANCHORS=256 run")
        print("    (the profilers go silent inside the recompute checkpoint), then read the FWD PROFILER below.")
    elif max(rc, sv) < 10:
        print(f"    NEITHER dominates (fwd {rc:.0f}%, serve {sv:.0f}% — both <10%): steady compute is the floor.")
        print("    throughput is bound by the draft MoE fwd/bwd itself — EP / fused grouped-GEMM is the lever.")
    else:
        print(f"    MIXED (true-fwd {rc:.0f}%, serve {sv:.0f}% [straggler {al:.0f}% + fetch {hs:.0f}%]): both matter.")
        print("    Attack the bigger bar first; re-check align_ms after each change.")
    print("  ⚠ run with --skip <N> to drop the first epoch (shape-warmup + resume HS-regen inflate ALL bars).")

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
        if len(baselines) >= 2:
            # multi-run overlay (each run its own colour) — one metric per plot
            _plots_multi(
                [{"label": cur_label, "recs": recs, "good": good, "ckpt": ckpt_steps}]
                + [{"label": b["label"], "recs": b["recs"], "good": b["good"], "ckpt": b["ckpt"]}
                   for b in baselines],
                args.out,
            )
        else:
            base = None
            if len(baselines) == 1:
                b = baselines[0]
                base = {
                    "recs": b["recs"], "good": b["good"],
                    "pos_keys": detect_positions(b["recs"]),
                    "reps": {s: spike_report(b["recs"], s, args.spike_k) for s in
                             ["profile/fetch_ms", "profile/fwd_ms", "profile/bwd_ms", "profile/opt_ms", "profile/step_ms"]},
                    "ckpt_steps": b["ckpt"], "label": b["label"],
                }
            _plots(recs, good, pos_keys, reps, ckpt_steps, args.out, cur_label, base,
                   steps_per_epoch=steps_per_epoch)
    print("=" * 78)


# ── released-draft BASELINE reference (our #12006 serve, full DATASET=all, 2026-07-20) ──
# Source of truth: docs/deployment/ascend-npu-dsv4-dspark-eval-results.md (the `released draft` row).
# Shown as background reference lines on the accept_len + per-position plots (replaced the old single
# 3.94 official line). `avg` = unweighted mean over the 5 eval datasets.
RELEASED_ACCEPT_LEN = {"gsm8k": 4.658, "mt-bench": 3.294, "avg": 4.42}
# per-position CUMULATIVE accept rate S_k (= P(prefix 0..k accepted)) from the same eval:
_RELEASED_POS_CUM = {
    "gsm8k":    [0.9277, 0.8277, 0.7329, 0.6355, 0.5345],
    "mt-bench": [0.7921, 0.5855, 0.4145, 0.2932, 0.2087],
    "avg":      [0.9016, 0.7874, 0.6722, 0.5741, 0.4827],
}
_REF_STYLE = {"gsm8k": ("#D62828", "--"), "mt-bench": ("#8C2FBF", ":"), "avg": ("#1B8A4E", "-.")}


def _cum_to_marginal(cum):
    """S_k (cumulative survival) → c_k = S_k/S_{k-1} (per-slot marginal). The training position bars
    measure MARGINAL greedy accuracy (argmax==target per slot), so the reference must be marginal too
    — overlaying the cumulative S_k on marginal bars would falsely flatter the tail."""
    out, prev = [], 1.0
    for s in cum:
        out.append(s / prev if prev > 1e-9 else 0.0)
        prev = s
    return out


# per-slot MARGINAL, matched to the training bars' metric (derived from the cumulative eval numbers)
RELEASED_POS_MARGINAL = {k: _cum_to_marginal(v) for k, v in _RELEASED_POS_CUM.items()}


def _draw_accept_len_refs(plt):
    """3 horizontal released-accept_len refs (replaces the single 3.94 line), each tagged INLINE at
    the left edge so the line is self-identifying at a glance (not only via legend/title). Returns
    the top ref value (for ylim)."""
    ax = plt.gca()
    for name, al in RELEASED_ACCEPT_LEN.items():
        c, ls = _REF_STYLE[name]
        plt.axhline(al, ls=ls, lw=1.5, color=c, alpha=0.9, zorder=1,
                    label=f"released {name} AL {al:.2f}")
        # inline colour-matched tag at the LEFT edge (x = axes fraction, y = data): the training
        # curve is low/early there, so it never collides with these high (3.3–4.7) reference lines.
        ax.text(0.006, al, f"released {name} {al:.2f}", transform=ax.get_yaxis_transform(),
                color=c, fontsize=8, va="bottom", ha="left", weight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    return max(RELEASED_ACCEPT_LEN.values())


def _draw_pos_refs(plt, xs, npos):
    """3 released per-slot MARGINAL refs (replaces the single `rel` line), each tagged INLINE ON the
    line at a STAGGERED slot (gsm8k & avg coincide at the tail, so distinct x avoids overlapping
    text) so every line is self-identifying at a glance."""
    ax = plt.gca()
    names = [n for n in RELEASED_POS_MARGINAL if len(RELEASED_POS_MARGINAL[n]) >= npos]
    for i, name in enumerate(names):
        marg = RELEASED_POS_MARGINAL[name][:npos]
        c, ls = _REF_STYLE[name]
        plt.plot(xs, marg, marker="o", ls=ls, color=c, lw=1.5, ms=5, zorder=4,
                 label=f"released {name} (per-slot)")
        # label ON the line at a spread-out slot (i-th of len(names) → distinct x, no collisions)
        j = 0 if len(names) == 1 else round(i * (npos - 1) / (len(names) - 1))
        ax.annotate(f"released {name}", xy=(xs[j], marg[j]), xytext=(0, 8),
                    textcoords="offset points", color=c, fontsize=8, ha="center", va="bottom",
                    weight="bold", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))


def _timing_tag(recs, reps, spe) -> str:
    """The 3-number corner tag for the loss plot: step-time (steady) · steps · time-per-epoch.
    time/epoch needs steps/epoch (``spe``); if unknown it's just dropped (2 numbers)."""
    steady = (reps.get("profile/step_ms") or {}).get("steady_med")
    n = len([r for r in recs if step_of(r) >= 0])
    parts = ([f"step {steady:.0f} ms"] if steady else []) + [f"{n:,} steps"]
    if spe and steady:
        parts.append(f"~{spe * steady / 3.6e6:.1f} h/epoch  ({spe:,} steps/ep)")
    return "   ·   ".join(parts)


def _plots(recs, good, pos_keys, reps, ckpt_steps, out, label="current", base=None, steps_per_epoch=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.lines import Line2D
    except ImportError:
        print(f"\n(plots skipped: matplotlib not installed — `pip install matplotlib`; console report above is complete)")
        return
    os.makedirs(out, exist_ok=True)

    # optional BASELINE overlay (dashed / grouped). have_base gates every comparison branch.
    have_base = base is not None
    bgood = base["good"] if have_base else None
    brecs = base["recs"] if have_base else None
    breps = base["reps"] if have_base else None
    blabel = base["label"] if have_base else None

    def xy(key, recs_=recs):
        s = series(recs_, key)
        return [a for a, _ in s], [b for _, b in s]

    def _mark_ckpts_p(x, y, cks, color):
        """Dot + value at each checkpoint-save step (= epoch-half / epoch-end under CKPT_FREQ=0.5)."""
        if not cks or len(x) < 2:
            return
        import numpy as np  # noqa: PLC0415
        xa = np.asarray(x, float); ya = np.asarray(y, float); span = (xa[-1] - xa[0]) or 1.0
        for cs in sorted(cks):
            idx = int(np.argmin(np.abs(xa - cs)))
            if abs(xa[idx] - cs) > 0.02 * span:
                continue
            lo, hi = max(0, idx - 3), min(len(ya), idx + 4)
            val = float(median(list(ya[lo:hi])))
            plt.plot(xa[idx], val, "o", ms=5, color=color, mec="white", mew=0.8, zorder=6)
            plt.annotate(f"{val:.2f}", xy=(xa[idx], val), xytext=(0, 6), textcoords="offset points",
                         fontsize=6.5, color=color, ha="center", va="bottom", zorder=6,
                         bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))

    def _overlay(key, color, lab, cur_src, base_src):
        """current solid + (if comparing) baseline dashed, SAME color per metric."""
        x, y = xy(key, cur_src)
        if x:
            plt.plot(x, y, lw=1.0, color=color, label=lab)
        if have_base:
            xb, yb = xy(key, base_src)
            if xb:
                plt.plot(xb, yb, lw=1.0, color=color, ls="--", alpha=0.55)

    def _legend(**kw):
        """Set the legend; when comparing, append two proxy entries (solid=current / dashed=baseline)."""
        ax = plt.gca()
        h, l = ax.get_legend_handles_labels()
        if have_base:
            h = h + [Line2D([0], [0], color="0.25", ls="-", lw=2),
                     Line2D([0], [0], color="0.25", ls="--", lw=2)]
            l = l + [f"{label} (solid)", f"{blabel} (dashed)"]
        ax.legend(h, l, **kw)

    ttl_suffix = f"  ({label} solid vs {blabel} dashed)" if have_base else ""

    # epoch boundaries of the CURRENT run — vertical markers on every step-axis plot below.
    ep_bounds = epoch_boundaries(recs)

    def _epoch_lines():
        for s, e in ep_bounds:
            plt.axvline(s, color="0.55", ls=":", lw=0.9, alpha=0.75, zorder=0)
            plt.annotate(f"e{e}", xy=(s, 1.0), xycoords=("data", "axes fraction"),
                         xytext=(2, -2), textcoords="offset points",
                         color="0.4", fontsize=7, ha="left", va="top")

    # 1) loss  (+ a small 3-number tag in the corner: step-time · steps · time-per-epoch)
    plt.figure(figsize=(9, 4))
    for k, lab, c in [("train/loss", "total", "#222222"), ("train/ce_loss", "ce", "#2E6CF6"),
                      ("train/tv_loss", "tv", "#1B8A4E"), ("train/confidence_loss", "confidence", "#D62828")]:
        _overlay(k, c, lab, good, bgood)
    _epoch_lines()
    plt.xlabel("step"); plt.ylabel("loss"); plt.title("Loss" + ttl_suffix); plt.grid(alpha=.3); _legend()
    tag = _timing_tag(recs, reps, steps_per_epoch)
    if tag:
        plt.gca().text(0.015, 0.03, tag, transform=plt.gca().transAxes, fontsize=8.5,
                       family="monospace", va="bottom", ha="left", color="#333",
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85))
    plt.tight_layout(); plt.savefig(f"{out}/loss.png", dpi=120); plt.close()

    # 2) acceptance + per-position (overview)
    plt.figure(figsize=(9, 4))
    for k, lab, c in [("train/accept_len", "accept_len", "#2E6CF6"),
                      ("train/accept_rate", "accept_rate", "#1B8A4E"),
                      ("train/full_acc", "full_acc", "#D62828")]:
        _overlay(k, c, lab, good, bgood)
    _draw_accept_len_refs(plt)   # released gsm8k / mt-bench / avg accept_len (replaces single 3.94)
    _epoch_lines()
    plt.xlabel("step"); plt.ylabel("accept"); plt.title("Acceptance" + ttl_suffix); plt.grid(alpha=.3); _legend()
    plt.tight_layout(); plt.savefig(f"{out}/acceptance.png", dpi=120); plt.close()

    # 2b) accept_len DEDICATED (the hero comparison): raw + smoothed, both runs, + released target
    x, y = xy("train/accept_len", good)
    if x:
        plt.figure(figsize=(9.5, 4.8))

        def _smoothed(xx, yy, color, raw_color, name):
            plt.plot(xx, yy, lw=0.5, alpha=0.28, color=raw_color)
            w = max(11, len(yy) // 60)
            if len(yy) >= w:
                ys = np.convolve(np.asarray(yy, float), np.ones(w) / w, mode="valid")
                off = (w - 1) // 2
                plt.plot(xx[off:off + len(ys)], ys, lw=2.4, color=color, label=f"{name} (smoothed)")
            return median(yy[-min(50, len(yy)):])

        cur = _smoothed(x, y, "#2E6CF6", "#7A8AA8", label)
        _mark_ckpts_p(x, y, ckpt_steps, "#2E6CF6")
        base_y = []
        if have_base:
            xb, yb = xy("train/accept_len", bgood)
            if xb:
                base_y = yb
                bcur = _smoothed(xb, yb, "#E08A1E", "#C9A27A", blabel)
                _mark_ckpts_p(xb, yb, base.get("ckpt_steps"), "#E08A1E")
                plt.annotate(f"{blabel} ~{bcur:.2f}", xy=(xb[-1], bcur), xytext=(-8, -20),
                             textcoords="offset points", color="#E08A1E", fontsize=10, ha="right",
                             weight="bold", zorder=7,
                             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#E08A1E", alpha=0.92),
                             arrowprops=dict(arrowstyle="-", color="#E08A1E", lw=0.8))
        # y-axis: when the data is far below the 3.94 target (early training), ZOOM to the data
        # so the two runs are legible/separable; show the target as an off-scale note. Once
        # accept_len climbs near the target, show the full axis + the highlighted target line.
        ally = y + base_y
        data_top = (max(ally) if ally else 1.5) + 0.15
        ref_top = _draw_accept_len_refs(plt)             # released gsm8k/mt-bench/avg (replaces 3.94)
        plt.ylim(1.0, max(ref_top + 0.25, data_top))     # keep all 3 refs on-scale
        # our-run end marker: white-boxed + leader so the blue curve doesn't hide it (was occluded)
        plt.annotate(f"{label} ~{cur:.2f}", xy=(x[-1], cur), xytext=(-8, 18),
                     textcoords="offset points", color="#2E6CF6", fontsize=10, ha="right",
                     weight="bold", zorder=7,
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#2E6CF6", alpha=0.92),
                     arrowprops=dict(arrowstyle="-", color="#2E6CF6", lw=0.8))
        _epoch_lines()
        plt.xlabel("step"); plt.ylabel("acceptance length")
        plt.title((f"Acceptance length — {label} vs {blabel}" if have_base
                   else "Acceptance length — raw vs smoothed")
                  + "   (horizontal lines = released full-all baselines, tagged inline)")
        plt.legend(loc="lower right"); plt.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(f"{out}/accept_len.png", dpi=120); plt.close()

    if pos_keys:
        n = min(50, len(good))

        def _posvals(pk, g):
            gg = g[-min(50, len(g)):]
            return [(median(col(gg, k)) if col(gg, k) else 0.0) for k in pk]

        labels = [f"pos{i+1}" for i in range(len(pos_keys))]
        if have_base and base["pos_keys"]:
            # GROUPED bars: current vs baseline, side by side per position.
            cur_vals = _posvals(pos_keys, good)
            bvals_raw = _posvals(base["pos_keys"], bgood)
            bvals = (bvals_raw + [0.0] * len(labels))[:len(labels)]  # align to current's position count
            xx = np.arange(len(labels)); w = 0.4
            plt.figure(figsize=(9, 4.8))
            b1 = plt.bar(xx - w / 2, cur_vals, w, color="#2E6CF6", zorder=3, label=label)
            b2 = plt.bar(xx + w / 2, bvals, w, color="#E08A1E", zorder=3, label=blabel)
            for bars in (b1, b2):
                for b in bars:
                    h = b.get_height()
                    if h > 0:
                        plt.text(b.get_x() + b.get_width() / 2, h + 0.012, f"{h:.2f}",
                                 ha="center", va="bottom", fontsize=8, weight="bold")
            # released-draft per-slot MARGINAL accept (our #12006 full-all): gsm8k / mt-bench / avg
            _draw_pos_refs(plt, xx, len(labels))
            plt.ylim(0, 1.0); plt.xticks(xx, labels); plt.legend(loc="upper right")
            plt.ylabel("greedy accuracy  (argmax == target)")
            plt.title(f"Per-position draft accuracy — {label} vs {blabel} (last {n} steps)")
            plt.grid(axis="y", alpha=.3, zorder=0)
            plt.tight_layout(); plt.savefig(f"{out}/position_acc.png", dpi=120); plt.close()
        else:
            # BAR chart of the CURRENT (last-N-step median) per-position accuracy, value-labeled.
            vals = _posvals(pos_keys, good)
            plt.figure(figsize=(8.5, 4.8))
            bars = plt.bar(labels, vals, width=0.62, color="#2E6CF6", zorder=3)
            for b, v in zip(bars, vals):
                plt.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                         ha="center", va="bottom", fontsize=11, weight="bold", color="#1B2538")
            # released-draft per-slot MARGINAL accept (our #12006 full-all): gsm8k / mt-bench / avg
            _draw_pos_refs(plt, labels, len(pos_keys))
            plt.legend(loc="upper right")
            plt.ylim(0, 1.0)
            plt.ylabel("greedy accuracy  (argmax == target)")
            plt.title(f"Per-position draft accuracy — last {n} steps (decays p1→p{len(pos_keys)})")
            plt.grid(axis="y", alpha=.3, zorder=0)
            plt.tight_layout(); plt.savefig(f"{out}/position_acc.png", dpi=120); plt.close()

    # 3) confidence calibration
    plt.figure(figsize=(9, 4))
    for k, lab, c in [("train/confidence_pred_mean", "pred_mean", "#2E6CF6"),
                      ("train/accept_rate", "observed accept_rate", "#1B8A4E"),
                      ("train/confidence_abs_error", "abs_error", "#D62828"),
                      ("train/confidence_cumprod_bias", "cumprod_bias", "#8A2BE2")]:
        _overlay(k, c, lab, good, bgood)
    _epoch_lines()
    plt.xlabel("step"); plt.ylabel("confidence"); plt.title("Confidence calibration" + ttl_suffix)
    plt.grid(alpha=.3); _legend()
    plt.tight_layout(); plt.savefig(f"{out}/confidence.png", dpi=120); plt.close()

    # 4) timing (log-y so spikes + steady both visible)
    plt.figure(figsize=(9, 4))
    for k, lab, c in [("profile/fetch_ms", "fetch/HS", "#D62828"), ("profile/fwd_ms", "fwd", "#2E6CF6"),
                      ("profile/bwd_ms", "bwd", "#1B8A4E"), ("profile/opt_ms", "opt", "#8A2BE2"),
                      ("profile/step_ms", "step", "#222222")]:
        _overlay(k, c, lab, recs, brecs)
    _epoch_lines()
    plt.yscale("log"); plt.xlabel("step"); plt.ylabel("ms (log)"); _legend(ncol=5, fontsize=8)
    plt.title("Per-stage time (log-y — spikes are the recompiles)" + ttl_suffix); plt.grid(alpha=.3, which="both")
    plt.tight_layout(); plt.savefig(f"{out}/timing.png", dpi=120); plt.close()

    # 5) per-stage timing BARS
    stage_defs = [("profile/fetch_ms", "HS fetch"), ("profile/fwd_ms", "fwd"),
                  ("profile/bwd_ms", "bwd"), ("profile/opt_ms", "opt"), ("profile/step_ms", "step")]
    if have_base:
        # COMPARE: current STEADY vs baseline STEADY, grouped (did steady compute change?), log-y.
        labels, cs, bs = [], [], []
        for key, lab in stage_defs:
            r, rb = reps.get(key), (breps.get(key) if breps else None)
            if not r:
                continue
            labels.append(lab)
            cs.append(r["steady_med"])
            bs.append(rb["steady_med"] if rb else 0.0)
        if labels:
            x = np.arange(len(labels)); w = 0.38
            plt.figure(figsize=(9.8, 5.0))
            b1 = plt.bar(x - w / 2, cs, w, color="#2E6CF6", zorder=3, label=f"{label} steady")
            b2 = plt.bar(x + w / 2, bs, w, color="#E08A1E", zorder=3, label=f"{blabel} steady")
            for bars in (b1, b2):
                for b in bars:
                    h = b.get_height()
                    if h > 0:
                        plt.text(b.get_x() + b.get_width() / 2, h * 1.06,
                                 f"{h:.0f}" if h >= 10 else f"{h:.1f}", ha="center", va="bottom", fontsize=9)
            plt.yscale("log"); plt.xticks(x, labels); plt.ylabel("ms (log)")
            plt.title(f"Per-stage STEADY time — {label} vs {blabel}")
            plt.legend(loc="upper left"); plt.grid(axis="y", alpha=.3, which="both", zorder=0)
            plt.tight_layout(); plt.savefig(f"{out}/timing_bars.png", dpi=120); plt.close()
    else:
        # SINGLE run: steady (solid) vs effective-avg incl spikes (hollow/hatched), log-y.
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

    tail = f"   [baseline '{blabel}' overlaid: dashed lines / grouped bars]" if have_base else ""
    print(f"\n📊 plots → {out}/  (loss, acceptance, accept_len[raw+smoothed+target], position_acc[bars],\n"
          f"                    confidence, timing[lines], timing_bars){tail}")


def _plots_multi(runs, out):
    """Overlay N runs (CURRENT first) on the key comparison metrics — one metric per plot,
    each run its own colour + smoothed curve (raw faded behind). Used for >=2 baselines."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\n(plots skipped: matplotlib not installed — console report above is complete)")
        return
    os.makedirs(out, exist_ok=True)
    colors = ["#2E6CF6", "#E08A1E", "#1B8A4E", "#8A2BE2", "#D62828",
              "#00A3A3", "#B4661E", "#555555", "#C71585", "#2F6B2F"]
    for i, r in enumerate(runs):
        r["color"] = colors[i % len(colors)]

    def _xy(recs_, key):
        s = series(recs_, key)
        return [a for a, _ in s], [b for _, b in s]

    # epoch boundaries of the CURRENT run (runs[0]); marked on every step-axis plot.
    ep_bounds = epoch_boundaries(runs[0]["recs"]) if runs else []

    def _mark_ckpts(r, key, train=True):
        """Dot + value label at each checkpoint-save step — under CKPT_FREQ=0.5 those are the
        epoch-HALF and epoch-END points (= the ckpts that get converted + eval'd). Value = local
        median around the step so it tracks the smoothed curve, colored per run."""
        cks = sorted(s for s in (r.get("ckpt") or ()))
        if not cks:
            return
        x, y = _xy(r["good"] if train else r["recs"], key)
        if len(x) < 2:
            return
        xa = np.asarray(x, float); ya = np.asarray(y, float)
        span = (xa[-1] - xa[0]) or 1.0
        for cs in cks:
            idx = int(np.argmin(np.abs(xa - cs)))
            if abs(xa[idx] - cs) > 0.02 * span:      # this ckpt isn't within the run's plotted range
                continue
            lo, hi = max(0, idx - 3), min(len(ya), idx + 4)
            val = float(median(list(ya[lo:hi])))
            plt.plot(xa[idx], val, marker="o", ms=5, color=r["color"], mec="white", mew=0.8, zorder=6)
            plt.annotate(f"{val:.2f}", xy=(xa[idx], val), xytext=(0, 6), textcoords="offset points",
                         fontsize=6.5, color=r["color"], ha="center", va="bottom", zorder=6,
                         bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))

    def _overlay(key, fname, title, ylabel, *, train=True, logy=False, target=None, accept_len_refs=False):
        plt.figure(figsize=(9.5, 4.8))
        allv, drew = [], False
        for r in runs:
            x, y = _xy(r["good"] if train else r["recs"], key)
            if not x:
                continue
            drew = True
            allv += y
            c = r["color"]
            plt.plot(x, y, lw=0.5, alpha=0.18, color=c)
            w = max(11, len(y) // 60)
            if len(y) >= w:
                ys = np.convolve(np.asarray(y, float), np.ones(w) / w, mode="valid")
                off = (w - 1) // 2
                plt.plot(x[off:off + len(ys)], ys, lw=2.2, color=c,
                         label=f"{r['label']} ~{median(y[-min(50, len(y)):]):.3g}")
            else:
                plt.plot(x, y, lw=1.6, color=c, label=r["label"])
            if train:
                _mark_ckpts(r, key, train)   # epoch-half + epoch-end value dots
        if not drew:
            plt.close()
            return
        for s, e in ep_bounds:  # current-run epoch markers
            plt.axvline(s, color="0.55", ls=":", lw=0.9, alpha=0.75, zorder=0)
            plt.annotate(f"e{e}", xy=(s, 1.0), xycoords=("data", "axes fraction"),
                         xytext=(2, -2), textcoords="offset points",
                         color="0.4", fontsize=7, ha="left", va="top")
        if logy:
            plt.yscale("log")
        if accept_len_refs:
            top = (max(allv) if allv else 1.5) + 0.15
            ref_top = _draw_accept_len_refs(plt)      # released gsm8k/mt-bench/avg (replaces 3.94)
            plt.ylim(1.0, max(ref_top + 0.25, top))
        elif target is not None:
            top = (max(allv) if allv else 1.5) + 0.15
            if top < target - 0.34:
                plt.ylim(1.0, top)
                plt.annotate(f"↑ target = {target} (off-scale — early training)",
                             xy=(0.5, 0.97), xycoords="axes fraction", ha="center", va="top",
                             color="#D62828", fontsize=10, weight="bold")
            else:
                plt.axhline(target, ls="--", lw=2.0, color="#D62828", label=f"target {target}")
                plt.ylim(1.0, max(target + 0.2, top))
        plt.xlabel("step"); plt.ylabel(ylabel); plt.title(title)
        plt.grid(alpha=.3, which="both" if logy else "major"); plt.legend(loc="best", fontsize=9)
        plt.tight_layout(); plt.savefig(f"{out}/{fname}", dpi=120); plt.close()

    _overlay("train/accept_len", "accept_len.png", "Acceptance length — all runs", "acceptance length", accept_len_refs=True)
    _overlay("train/loss", "loss.png", "Total loss — all runs", "loss")
    _overlay("train/accept_rate", "accept_rate.png", "Accept rate — all runs", "accept_rate")
    _overlay("train/full_acc", "full_acc.png", "Full-block accuracy — all runs", "full_acc")
    _overlay("profile/grad_norm", "grad_norm.png", "grad_norm — all runs (log-y)", "grad_norm",
             train=False, logy=True)

    # per-position accuracy: grouped bars, all runs (last-50-step medians)
    pos_keys = detect_positions(runs[0]["recs"])
    if pos_keys:
        labels = [f"pos{i+1}" for i in range(len(pos_keys))]
        xx = np.arange(len(labels)); n = len(runs); w = 0.8 / n
        plt.figure(figsize=(10, 4.8))
        for j, r in enumerate(runs):
            g = r["good"][-50:]
            vals = [(median(col(g, k)) if col(g, k) else 0.0) for k in pos_keys]
            plt.bar(xx + (j - (n - 1) / 2) * w, vals, w, color=r["color"], zorder=3, label=r["label"])
        # released-draft per-slot MARGINAL accept (our #12006 full-all): gsm8k / mt-bench / avg
        _draw_pos_refs(plt, xx, len(pos_keys))
        plt.ylim(0, 1.0); plt.xticks(xx, labels); plt.legend(loc="upper right", fontsize=9)
        plt.ylabel("greedy accuracy  (argmax == target)")
        plt.title("Per-position draft accuracy — all runs (last 50 steps)")
        plt.grid(axis="y", alpha=.3, zorder=0)
        plt.tight_layout(); plt.savefig(f"{out}/position_acc.png", dpi=120); plt.close()

    print(f"\n📊 multi-run plots → {out}/  ({len(runs)} runs overlaid: accept_len, loss, accept_rate, "
          "full_acc, grad_norm[log], position_acc)")


if __name__ == "__main__":
    main()
