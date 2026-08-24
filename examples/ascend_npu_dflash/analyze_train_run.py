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
                # EP all-to-all straggler: a rank starved on HS reaches the MoE all-to-all LATE, so the
                # other ranks (incl rank0, whose record this is) WAIT there — and that wait lands INSIDE
                # fwd_ms, NOT align_ms. ([MOE-PROF] verdict 2026-07-27: the local GMM is 2-5ms stable; the
                # fwd "spike" IS the all-to-all wait.) Estimate = how much longer the SLOWEST rank fetched
                # than this one: max(0, fetch_ms_max − fetch_ms). ~0 on balanced steps, huge on a starved
                # step (fetch_ms_max 24419 vs own 26). Subtract it too → fwd_compute is the TRUE compute
                # (else this rank's wait is mislabelled "recompile"). No fetch_ms_max in log → a2a=0 (no-op).
                _fx = float(r.get("profile/fetch_ms", 0.0) or 0.0)
                _fxm = float(r.get("profile/fetch_ms_max", 0.0) or 0.0)
                _a2a = max(0.0, _fxm - _fx)
                r["profile/a2a_straggler_ms"] = _a2a
                r["profile/fwd_compute_ms"] = max(0.0, _fwd - _al - _a2a)
            except (TypeError, ValueError):
                pass
    return recs, text


def checkpoint_steps(text: str) -> set[int]:
    """Steps where a checkpoint SAVE happened (the save cost gets misread as a fetch/step spike).

    The 'Saving checkpoint' log line has no global_step of its own, so attribute it to the last
    step logged before it; the save shows up on that step or the next one.

    ⚠ ONE forward pass, deliberately. The obvious form -- for each save marker, re-scan
    ``text[:m.start()]`` for the last ``global_step=`` -- is O(#saves x len(text)), and BOTH
    factors grow linearly with the run, so the cost is QUADRATIC in step count. It was free at
    4.5k steps and hung the tool at 26k (and _load_and_skip pays it once per log, so a lead scan
    pays it three times). Interleaving both patterns in a single finditer is exactly equivalent:
    matches arrive in position order, so the most recently seen global_step IS the last one
    before the marker."""
    steps: set[int] = set()
    last: int | None = None
    for m in re.finditer(r"global_step=(\d+)|Saving checkpoint|Checkpoint saved", text):
        if m.group(1) is not None:
            last = int(m.group(1))
        elif last is not None:
            steps |= {last, last + 1}
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
_TQDM_TOTAL = re.compile(r"(\d[\d,]*)/(\d[\d,]*)\s*\[\s*\d[\d:]*\s*<")


def _steps_per_epoch(recs, raw_text: str = "") -> int | None:
    """Full-epoch step count (``len(train_loader)``). Preferred source: the tqdm ``N/M [t<t`` bar in
    the raw log (``M`` = steps/epoch, present from step 1 so it works while still in epoch 0). Fallback:
    the spacing between parsed ``epoch`` boundary increments. ``None`` if neither is available
    (→ the footer/console just show steps/wall and skip the per-epoch projection)."""
    # ⚠ ORDER MATTERS. Parsed epoch boundaries are ground truth, so prefer them whenever the run
    # has crossed at least one epoch. The tqdm bar is only the fallback for a run still INSIDE
    # epoch 0 (no boundary yet). Previously the bar was tried first and got poisoned by the
    # checkpoint-writer's own bar ("Writing model shards: … 1/1 [01:03<00:00") → steps_per_epoch=1
    # → the MoE plot's `while ep*steps_per_epoch <= xmax` loop drew ~124k epoch lines+annotations
    # (minutes of text layout), and the loss-plot footer read "1 steps/ep · ~0.0 h/epoch".
    eb = epoch_boundaries(recs)
    if eb:
        steps0 = [step_of(r) for r in recs if step_of(r) >= 0]
        pts = ([min(steps0)] if steps0 else []) + [s for s, _ in eb]
        diffs = [b - a for a, b in zip(pts, pts[1:]) if b > a]
        if diffs:
            return int(median(diffs))
    totals = [int(m.group(2).replace(",", "")) for m in _TQDM_TOTAL.finditer(raw_text or "")]
    totals = [t for t in totals if t > 1]  # drop the 1/1 shard-writer bars
    if totals:
        return max(totals)  # a resumed epoch's bar is a shorter slice → max = the true full length
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


def _load_and_skip(path: str, skip: int, quiet: bool = False, max_step: int = 0):
    """resolve → load → keep skip <= step <= max_step. Returns (recs, ckpt_steps, steps_per_epoch,
    raw_text) — the middle two are None when there are no metrics / no epoch length is discoverable."""
    recs, raw_text = load(resolve_log(path))
    if not recs:
        return None, None, None, None
    if skip > 0:
        kept = [r for r in recs if step_of(r) >= skip]
        if kept:
            if not quiet:
                print(f"(--skip {skip}: dropped {len(recs) - len(kept)} warmup/regen steps; analyzing {len(kept)})")
            recs = kept
    if max_step > 0:
        kept = [r for r in recs if step_of(r) <= max_step]
        if kept:
            if not quiet:
                print(f"(--max-step {max_step}: dropped {len(recs) - len(kept)} later steps; analyzing {len(kept)})")
            recs = kept
    return recs, checkpoint_steps(raw_text), _steps_per_epoch(recs, raw_text), raw_text


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


# Smoothing window for _mono_series / _series_noise, in RECORDS. Deliberately a constant.
# Deriving it from a run's own length (max(20, min(200, len//50))) made the window depend on
# how long the log happens to be, which broke the one thing this tool exists to do: a finished
# 38931-step run got win=200 while a live 4500-step one got win=90, so its noise estimate was
# 1.49x smaller for no reason but its length, and their error bars were not comparable. Worse,
# a given step's delta drifted as the run grew -- step 500 read +0.256, then +0.253, then
# +0.233 across four invocations of the same command on a growing log.
_SMOOTH_WIN = 100


def _series_noise(recs, series, key="train/accept_len", win=None):
    """Standard error of the smoothed curve: how much of it is just per-batch noise.

    Needed because the horizontal-distance metric below DIVIDES by the curve's slope, and on a
    saturating curve the slope goes to zero -- so once training flattens, a noise-sized change
    in value maps to a huge change in step. Without this band the metric confidently reports a
    four-figure "lead" for a run compared against an independent sample of ITSELF.
    """
    pts = [(step_of(r), f(r, key)) for r in recs
           if f(r, key) is not None and step_of(r) >= 0]
    if len(pts) < 40 or not series:
        return None
    pts.sort()
    lut = dict(series)
    res = []
    for st, v in pts:
        sm = lut.get(st)
        if sm is not None:
            res.append(v - sm)
    if len(res) < 40:
        return None
    if win is None:
        win = _SMOOTH_WIN
    # 1.253 = SE(median)/SE(mean) for a normal sample
    return 1.253 * pstdev(res) / (win ** 0.5)


def _mono_series(recs, key="train/accept_len", win=None):
    """(step, value) smoothed by a rolling median and forced monotone non-decreasing.

    accept_len is noisy per batch and the inversion below needs a function it can invert, so
    the running max is applied after smoothing rather than instead of it.
    """
    pts = [(step_of(r), f(r, key)) for r in recs
           if f(r, key) is not None and step_of(r) >= 0]
    if win is None:
        win = _SMOOTH_WIN
    if len(pts) < win * 2:
        return []
    pts.sort()
    out, run = [], None
    for i in range(win, len(pts) + 1):
        v = median([p[1] for p in pts[i - win:i]])
        run = v if run is None else max(run, v)
        out.append((pts[i - 1][0], run))
    return out


def _paired_delta(base_recs, arm_recs, key="train/accept_len", win=None):
    """PER-STEP difference, smoothed -- and its own noise, not two independent bands combined.

    These runs share a seed and a data order, so at any given step both saw the SAME batch.
    Whatever that batch does to accept_len it does to both, and subtracting cancels it. The
    combined-independent-bands estimate ignores that and is therefore far too wide: it asks
    "how much does each curve wobble", when the question is "how much does their DIFFERENCE
    wobble". With a +/-0.08 band we cannot resolve the +0.03..0.27 range that decides whether
    a head pays for itself at concurrency 1, so the instrument, not the head, is what is
    reporting 'indistinguishable'.

    Returns (series, se) where series is [(step, smoothed_diff)] and se is the standard error
    of that smoothed difference. Only steps present in BOTH runs are used.
    """
    win = win or _SMOOTH_WIN
    b = {step_of(r): f(r, key) for r in base_recs if f(r, key) is not None and step_of(r) >= 0}
    a = {step_of(r): f(r, key) for r in arm_recs if f(r, key) is not None and step_of(r) >= 0}
    common = sorted(set(a) & set(b))
    if len(common) < win * 2:
        return [], None
    d = [(st, a[st] - b[st]) for st in common]
    out, res = [], []
    for i in range(win, len(d) + 1):
        w = [x[1] for x in d[i - win:i]]
        m = median(w)
        out.append((d[i - 1][0], m))
        res.append(d[i - 1][1] - m)
    se = 1.253 * pstdev(res) / (win ** 0.5) if len(res) >= win else None
    return out, se


def _lead_steps(base_series, arm_value, at_step):
    """HOW MANY STEPS AHEAD the arm is: invert the baseline curve at the arm's value.

    Read directly as horizontal distance -- find the step at which the BASELINE first reaches
    what the arm has reached at ``at_step`` -- rather than as delta/slope. Slope estimation
    needs a trailing window, and on a saturating curve any window wide enough to be stable is
    also wide enough to overstate the local slope, which makes a pure head start look like a
    shrinking one. Inversion has no window and no bias: a curve shifted N steps left returns N
    at every sample point.

    Returns (lead, capped). ``capped`` means the arm is beyond anything the baseline reached in
    its logged range, so the true lead is at least what is reported.
    """
    if arm_value is None or not base_series:
        return None, False
    if arm_value <= base_series[0][1]:
        return arm_value - base_series[0][1], False  # behind before the curve starts: report ~0
    if arm_value > base_series[-1][1]:
        return base_series[-1][0] - at_step, True
    lo, hi = 0, len(base_series) - 1
    while lo < hi:                                   # first index whose value >= arm_value
        mid = (lo + hi) // 2
        if base_series[mid][1] < arm_value:
            lo = mid + 1
        else:
            hi = mid
    if lo == 0:
        return base_series[0][0] - at_step, False
    (s0, v0), (s1, v1) = base_series[lo - 1], base_series[lo]
    frac = 0.0 if v1 == v0 else (arm_value - v0) / (v1 - v0)
    return (s0 + frac * (s1 - s0)) - at_step, False


def _lead_scan(baseline_path, arms, at_steps, skip, spike_k, max_step=0):
    """Compare N experiment arms against ONE shared baseline, at several matched step counts.

    Two arms' deltas over a common baseline are comparable only when measured over the same
    steps, and one step count says almost nothing -- what discriminates is how each arm's lead
    EVOLVES. So this samples a ladder of step counts and reports, per arm, the raw delta and
    the delta expressed as an equivalent head start in steps.
    """
    # ⚠ max_step must reach BOTH the baseline and every arm. Arms run to different lengths
    # and the paired gains DECAY, so an untruncated comparison reads each arm at whatever step
    # it happens to have reached -- which flatters the short ones. Before this was wired up the
    # flag was accepted and silently ignored: the paired section still reported CONV at 81737
    # and CORRECTION at 38931 while the user had asked for 26320.
    base_recs, base_ck, _, _ = _load_and_skip(baseline_path, skip, quiet=True, max_step=max_step)
    if not base_recs:
        raise SystemExit(f"baseline has no metrics: {baseline_path}")
    # ⚠ The baseline needs the same "did it get there?" guard the arms have. _at() returns the
    # last point at or before S, so sampling past the baseline's own end silently compares an
    # arm's step-80000 value against the baseline's step-38931 ENDPOINT -- every step the arm
    # ran beyond the baseline's reach is then counted as lead. That is not a small distortion;
    # it is the difference between "our change works" and "our baseline is shorter".
    base_last = max(step_of(x) for x in base_recs)
    loaded = []
    for name, path in arms:
        r, ck, _, _ = _load_and_skip(path, skip, quiet=True, max_step=max_step)
        if not r:
            print(f"  ⚠ 跳过 {name}: 无指标 ({path})")
            continue
        loaded.append((name, r, ck))
    if not loaded:
        raise SystemExit("no usable arms")

    if not at_steps:
        longest = max(max(step_of(x) for x in r) for _, r, _ in loaded)
        ladder = [500, 1500, 3000, 5000, 10000, 20000, 40000, 80000]
        at_steps = [x for x in ladder if x <= longest] or [longest]
        if at_steps[-1] < longest * 0.8:
            at_steps.append(longest)

    print()
    print("=" * 78)
    print(f" LEAD SCAN   共同基线 = {os.path.basename(baseline_path)}  (跑到 {base_last} 步)"
          + (f"   [--max-step {max_step}: 各臂已截齐]" if max_step else ""))
    print("=" * 78)
    base_series = _mono_series(base_recs)
    if not base_series:
        raise SystemExit("baseline has too few accept_len points to invert")

    def _at(series, S):
        """The series value at the last point with step <= S (None if it starts after S)."""
        v = None
        for st, val in series:
            if st > S:
                break
            v = val
        return v

    arm_series = {name: _mono_series(recs) for name, recs, _ in loaded}
    paired = {name: _paired_delta(base_recs, recs) for name, recs, _ in loaded}
    arm_noise = {name: _series_noise(recs, arm_series[name]) for name, recs, _ in loaded}
    base_noise = _series_noise(base_recs, base_series) or 0.0
    arm_last = {name: max(step_of(x) for x in recs) for name, recs, _ in loaded}
    trend = {name: [] for name, _, _ in loaded}
    for S in at_steps:
        if S > base_last:
            print()
            print(f"  ── @ step ≤ {S}   ⚠ 跳过:基线只跑到 {base_last} 步 ──")
            print(f"     在基线终点之外采样 = 拿基线的终点和臂的当前值比,"
                  f"臂多跑的 {S - base_last} 步会被整个算成领先。")
            continue
        bv = _at(base_series, S)
        if bv is None:
            continue
        print()
        print(f"  ── @ step ≤ {S}   基线 accept_len {bv:.3f} ──")
        print(f"     {'arm':<12} {'accept_len':>10} {'★ delta':>8}{'':<7} {'d_loss':>8} {'lead(steps)':>16}")
        for name, recs, ck in loaded:
            # An arm that has not REACHED S has no value there. Comparing its endpoint against
            # the baseline at S is not a measurement -- it is what produced the nonsense column
            # of growing negative leads the first time this ran.
            if arm_last[name] < S * 0.95:
                print(f"     {name:<12} {'(未跑到)':>10}")
                continue
            av = _at(arm_series[name], S)
            if av is None:
                continue
            a = [r for r in recs if 0 <= step_of(r) <= S]
            b = [r for r in base_recs if 0 <= step_of(r) <= S]
            dl = None
            if len(a) >= 40 and len(b) >= 40:
                ha, hb = _headline(a, ck, spike_k), _headline(b, base_ck, spike_k)
                if ha["loss"] and hb["loss"]:
                    dl = ha["loss"] - hb["loss"]
            d = av - bv
            lead, capped = _lead_steps(base_series, av, S)
            # Propagate the value noise through the inversion: where the curve is flat the band
            # blows up, and a lead wider than its own error bar is not a measurement.
            se = arm_noise.get(name) or 0.0
            band = None
            if se and lead is not None and not capped:
                lo, _ = _lead_steps(base_series, av - 2 * se, S)
                hi, hc = _lead_steps(base_series, av + 2 * se, S)
                if lo is not None and hi is not None and not hc:
                    band = abs(hi - lo) / 2
            usable = band is not None and band < max(150.0, 0.8 * abs(lead or 0))
            dband = 2.0 * ((se or 0.0) ** 2 + base_noise ** 2) ** 0.5
            trend[name].append((S, d, dband, lead, capped or not usable, band))
            if lead is None:
                lead_s = "-"
            elif capped:
                lead_s = f">{lead:.0f}"
            elif band is None:
                lead_s = f"{lead:.0f}?"
            else:
                lead_s = f"{lead:.0f}±{band:.0f}" + ("" if usable else " ?")
            print(f"     {name:<12} {av:>10.3f} {d:>+8.3f}±{dband:<6.3f}"
                  + (f"{dl:>+8.3f}" if dl is not None else f"{'-':>8}")
                  + f" {lead_s:>16}")

    print()
    # ── PAIRED difference: the estimate that can actually resolve the range we care about ──
    print()
    print("  ── ★ 配对差（同种子同数据 ⟹ 逐步相减,批次噪声抵消）──")
    print("     上面那个 ± 是两条独立曲线的带子合成,问的是「每条曲线抖多少」;")
    print("     真正该问的是「它们的差抖多少」。配对差直接测后者,窄得多。")
    any_paired = False
    for name, _, _ in loaded:
        ser, se = paired.get(name, ([], None))
        if not ser:
            print(f"    {name:<12} 样本不足")
            continue
        any_paired = True
        band = 2 * se if se else None
        head = ser[len(ser) // 20][1] if len(ser) > 20 else ser[0][1]
        tail = ser[-1][1]
        at = ser[-1][0]
        verdict = "—"
        if band is not None:
            # ⚠ Significance is not worth. A paired band lands around +/-0.005, so almost any
            # systematic difference clears it -- and reading "**真有差异**" as "ship it" is
            # exactly the trap a tighter ruler creates. State the magnitude against something
            # real: at concurrency 1 a codebook-reading head must buy +0.03..0.27 tokens to
            # pay for its own memory traffic (worklog 12.1). A head that costs nothing
            # sequential -- the block convolution -- has a much lower bar, so the economics
            # are per-arm and this only flags the common case.
            if abs(tail) <= band:
                verdict = "带内 ⟹ 与基线无异"
            elif abs(tail) < 0.03:
                verdict = f"显著但微小(<0.03 回本线) ⟹ **别据此上线**"
            else:
                verdict = "★ 显著且过回本线量级"
            if abs(head) > abs(tail) * 1.3:
                verdict += f"；且在衰减 {head:+.4f}→{tail:+.4f}"
        print(f"    {name:<12} @{at:<6} 末段 {tail:+.4f} ± {band:.4f}   (早段 {head:+.4f})   {verdict}"
              if band is not None else
              f"    {name:<12} @{at:<6} 末段 {tail:+.4f}   (早段 {head:+.4f})")
    if any_paired:
        print("     ⚠ 前提:两条 run 同种子、同数据、同 batch 顺序。任何一条不成立,配对差就无效,")
        print("        这时以上面那个保守的独立带为准。")
    print()

    print("  ── 判读：delta 的走势（收敛到 0 = 同一渐近线；走平在非零 = 渐近线之差）──")
    for name, pts in trend.items():
        # ⚠ THE VERDICT IS ON delta, NOT ON THE LEAD IN STEPS. The lead divides by the curve's
        # slope, so as training saturates it both explodes and self-censors: on the Correction
        # run -- known to converge to nothing -- every point past step 8000 was dropped for a
        # too-wide error bar, and the surviving early points read "level lift" purely because a
        # FLAT delta of +0.06 maps to an ever-larger horizontal distance as the slope decays.
        # delta has no division, stays readable to the end, and says what we actually want:
        #     converging to the SAME asymptote  <=>  delta -> 0
        #     different asymptotes              <=>  delta -> a nonzero constant, = the gap
        usable = [(S, d, db) for S, d, db, *_ in pts if d is not None]
        if len(usable) < 3:
            print(f"    {name:<12} 采样点不足({len(usable)})，再多跑一段")
            continue
        print(f"    {name:<12} " + "  ".join(f"{S}:{d:+.3f}" for S, d, _ in usable))
        tail = usable[-3:]
        d_last = median([d for _, d, _ in tail])
        b_last = median([db for _, _, db in tail])
        head = usable[: max(1, len(usable) // 3)]
        d_head = median([d for _, d, _ in head])
        base_now = _at(base_series, usable[-1][0]) or 1.0
        settled = abs(d_last - median([d for _, d, _ in usable[-min(6, len(usable)):-3]] or [d_last])) <= b_last

        if abs(d_last) <= b_last:
            print(f"      => 末段 delta {d_last:+.3f} 在误差带 ±{b_last:.3f} 内 —— "
                  "**收敛到同一渐近线**，无净增益。")
        elif abs(d_last) < 1.5 * b_last:
            # ⚠ A verdict decided by clearing one's own error bar by a few percent is a
            # threshold artefact, not a finding. Two arms landed on opposite sides of this
            # line at 0.18 sigma apart from each other; say so instead of picking a winner.
            print(f"      => 末段 delta {d_last:+.3f}，仅略超误差带 ±{b_last:.3f}"
                  f"（{d_last / b_last:.2f}x）—— **边缘,判不了**。"
                  "跨过带 10~50% 是阈值效应,不是结论;继续采样。")
        elif d_last < 0:
            print(f"      => 末段 delta {d_last:+.3f}（带 ±{b_last:.3f}）—— **比基线差**。")
        else:
            pct = 100.0 * d_last / base_now
            shrunk = "，且已从 {:+.3f} 衰减".format(d_head) if d_head - d_last > b_last else ""
            print(f"      => 末段 delta {d_last:+.3f}（带 ±{b_last:.3f}，占当前 {pct:.1f}%）{shrunk}")
            print("         " + ("已走平 ⟹ 渐近线确实高出这么多。" if settled
                                 else "仍在变化 ⟹ 尚未定型，继续采样。")
                  + f"  值不值得,拿它和 §8 的回本门槛比。")

    # ARM vS ARM. Each arm's verdict is against the shared baseline, so two arms can land on
    # opposite sides of a threshold while being indistinguishable FROM EACH OTHER -- which is
    # the comparison actually being asked for when more than one arm is passed.
    last = {}
    for name, pts in trend.items():
        u = [(S, d, db) for S, d, db, *_ in pts if d is not None]
        if u:
            t = u[-3:]
            last[name] = (u[-1][0], median([d for _, d, _ in t]), median([b for _, _, b in t]))
    names = [n for n, _, _ in loaded if n in last]
    if len(names) >= 2:
        print()
        print("  ── 两臂直接比较（末段）──")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                (_, da, ba), (_, db_, bb) = last[a], last[b]
                diff = da - db_
                comb = (ba ** 2 + bb ** 2) ** 0.5
                sig = abs(diff) / comb if comb else float("inf")
                print(f"    {a} − {b} = {diff:+.3f} ± {comb:.3f}   ({sig:.2f}σ) "
                      + ("⟹ **无法区分**" if sig < 1.0 else
                         "⟹ 有差别但不牢靠" if sig < 2.0 else "⟹ 可区分"))

    print()
    print("  ⚠ 这是训练侧 SOFT accept_len 的代理判据，不是裁决。裁决只能是转换后的服务端评测。")
    print("  ⚠ 各臂之间往往不止一个变量不同（Correction 还带 --no-confidence-detach-features")
    print("     和整块 --correction-*/--dflash-*），所以这是「哪次实验推动更大」，不是「哪个头更好」。")


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
    align_ov = sum(col(recs, "profile/align_ms")) + sum(col(recs, "profile/a2a_straggler_ms"))  # HS-straggler: align barrier + EP all-to-all wait

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


def _print_selection_headroom(good, col, median):
    """recall@k vs recall@1 per position + oracle vs hard accept_len.

    Reads the metrics emitted by examples/ascend_npu_dflash/recall_headroom_probe.py; stays
    silent on ordinary runs, which do not have them. The question it answers: when the draft's
    argmax is wrong, is the target's token still among the top-k candidates the draft already
    produced? If yes, a selector can recover it WITHOUT retraining the draft -- that is the
    whole premise of DFlash2-style path selection.

    ``position_{k}_acc`` IS recall@1 (both are "argmax == target"), so it is the correct
    baseline column and needs no separate metric. Likewise ``hard_accept_len`` is the greedy
    accept length and ``oracle_accept_len_K`` the same quantity with perfect selection inside
    top-K, so their difference is the headroom in tokens.

    ⚠ Teacher-forced: the Markov head is fed the TRUE previous token here, where the serve
    feeds the draft's own pick. These figures are therefore an UPPER bound -- decisive when
    they come back small, only suggestive when large.
    """
    ks, npos = [], 0
    for r in good[-1:] or good:
        for k in r:
            m = re.match(r"train/position_(\d+)_recall(\d+)$", k)
            if m:
                npos = max(npos, int(m.group(1)) + 1)
                if int(m.group(2)) not in ks:
                    ks.append(int(m.group(2)))
    if not ks or not npos:
        return  # not a probe run
    ks.sort()

    def med(key, n=200):
        v = col(good[-n:], key)
        return median(v) if v else None

    print("\n-- SELECTION HEADROOM (recall@k vs argmax; probe run only) " + "-" * 18)
    print("  是否'草稿其实知道、只是 argmax 选错了'。recall@1 == position_k_acc。")
    hdr = "  位置  " + "recall@1" + "".join(f"  recall@{k:<2}" for k in ks) + "   缺口(@%d−@1)" % ks[-1]
    print(hdr)
    gaps = []
    for p in range(npos):
        a = med(f"train/position_{p}_acc")
        if a is None:
            continue
        cells, top = [], None
        for k in ks:
            v = med(f"train/position_{p}_recall{k}")
            cells.append("    —   " if v is None else f"  {v:6.3f} ")
            if k == ks[-1]:
                top = v
        gap = None if top is None else (top - a) * 100
        if gap is not None:
            gaps.append((p, gap))
        print(f"  pos{p}   {a:6.3f} " + "".join(cells)
              + ("     —" if gap is None else f"   {gap:+5.1f} 点"))

    hard = med("train/hard_accept_len")
    for k in ks:
        orc = med(f"train/oracle_accept_len_{k}")
        if hard is not None and orc is not None:
            print(f"  accept_len: hard {hard:.3f}  →  oracle@{k:<2} {orc:.3f}   {orc - hard:+.2f} token")

    # 名次分布:小 k 能拿回多少,决定 selector 的 topk 成本
    tail = [p for p, _ in gaps if p >= max(0, npos - 2)]
    if tail and len(ks) > 1:
        p = tail[-1]
        a = med(f"train/position_{p}_acc")
        full = med(f"train/position_{p}_recall{ks[-1]}")
        if a is not None and full is not None and full > a:
            print(f"  末位(pos{p})的缺口有多少落在浅名次 —— 决定 selector 用多大的 k:")
            for k in ks:
                v = med(f"train/position_{p}_recall{k}")
                if v is not None:
                    print(f"     k={k:<2} 拿回 {(v - a) * 100:+5.1f} 点 = 全部缺口的 {(v - a) / (full - a) * 100:3.0f}%")

    if gaps:
        worst = max(g for _, g in gaps[-2:]) if len(gaps) >= 2 else gaps[-1][1]
        print("  ── 判据(在看到数字之前钉死)──")
        if worst >= 10:
            print(f"    末段缺口 {worst:+.1f} 点 ≥ 10 ⟹ 头寸真实:argmax 正在丢掉草稿已经排出来的 token。")
            print("    值得为 path selection 计算成本(服务侧改动 + host 侧 launch 开销)。")
        elif worst <= 3:
            print(f"    末段缺口 {worst:+.1f} 点 ≤ 3 ⟹ 草稿是真不知道那个 token,不是排错。")
            print("    path selection 对本模型无效;只剩块宽对齐重训这条路。")
        else:
            print(f"    末段缺口 {worst:+.1f} 点落在 3–10 的中间地带:看上面的名次分布再定。")
        print("    ⚠ 上界:此处 Markov 头吃的是真前驱 token,服务端吃的是草稿自己的选择。")


def _print_select_ablation(good, col, median):
    """Read select_ablation_probe.py's four corners of (markov on/off) x (select on/off).

    Two readings, and conflating them is the trap this section exists to prevent:

      * ``on - sel_off`` is EXACT and answers "what does the vllm-ascend patch buy on this
        checkpoint": same weights, same batch, one additive term removed. ``sel_off`` is
        literally what an unpatched serve computes.
      * It does NOT answer "was training with a select head worth it". ``sel_off`` is this
        model with a limb removed, not a vanilla-trained model -- the backbone trained
        knowing the limb was there and may have leaned on it. Only the paired
        SELECT-vs-ROPEFIX comparison settles that; the mk_gain column says which story fits.

    ``*_bias_rms`` sits next to the gains because a near-zero gain means opposite things
    depending on whether the term is still at its zero init or has grown and does nothing.
    """
    if not any("train/sel_gain" in r for r in good[-1:] or good):
        return

    def med(key, n=200):
        v = col(good[-n:], key)
        return median(v) if v else None

    print("\n-- SELECT ABLATION (四角:markov x select,同权重同 batch) " + "-" * 14)
    corners = [
        ("on", "两项都在", "  ← 模型实际算的"),
        ("sel_off", "只有 markov", "  ← 未打补丁的服务端算的就是这个"),
        ("mk_off", "只有 select", ""),
        ("both_off", "都去掉", "  ← 裸骨干"),
    ]
    al = {c: med(f"train/sel_{c}_accept_len") for c, _, _ in corners}
    print(f"\n  {'角':<12} {'accept_len':>11} {'相对 on':>10}")
    for c, label, tag in corners:
        if al[c] is None:
            continue
        d = "" if c == "on" else f"{al[c]-al['on']:>+10.3f}"
        print(f"  {label:<12} {al[c]:>11.3f} " + (f"{d}" if d else f"{'—':>10}") + tag)

    sg, mg, bg = med("train/sel_gain"), med("train/mk_gain"), med("train/both_gain")
    srms, mrms = med("train/sel_bias_rms"), med("train/mk_bias_rms")
    w, l = med("train/sel_win"), med("train/sel_loss")
    print()
    if sg is not None:
        print(f"  ★ sel_gain (on − sel_off) = {sg:+.3f} token   ← 服务端补丁能买到的")
    if w is not None:
        print(f"    逐块 赢 {w:.1%} / 输 {l:.1%}")
    if mg is not None:
        print(f"    mk_gain  (on − mk_off)   = {mg:+.3f}      both_gain = {bg:+.3f}")
    if srms is not None:
        print(f"    bias RMS  select {srms:.4f}   markov {mrms:.4f}")

    print("\n  ── 判读 ──")
    if srms is not None and srms < 1e-3:
        print("    select 项仍≈零初始化 ⟹ 还没长起来,gain 无论多少都不能下结论。继续跑。")
    elif sg is None:
        pass
    else:
        if sg > 0.02:
            print(f"    {sg:+.3f} ⟹ **该项在这个检查点上是正的**,服务端补丁有东西可买。")
        elif sg < -0.02:
            print(f"    {sg:+.3f} ⟹ 该项已长起来却在伤害接受长度。查初始化与学习率。")
        else:
            print(f"    {sg:+.3f} ⟹ 已训练但无净效果 ⟹ 服务端补丁不值得做。")
        if mg is not None and sg > 0.02:
            share = sg / (sg + mg) if (sg + mg) > 1e-9 else float("nan")
            print(f"    分工:select 占两项合计的 {share:.0%}。"
                  + ("接近对半 ⟹ 骨干确实把活分给了它。" if 0.3 < share < 0.7
                     else "偏低 ⟹ 它只承担了边角。" if share <= 0.3
                     else "偏高 ⟹ 它接管了大部分转移建模,注意 markov 侧是否被掏空。"))
    print("    ⚠ 这一段回答的是「补丁值多少」,**不是**「带 select 训练是否更好」——")
    print("       sel_off 是本模型被截肢,不是 vanilla 模型。后者只有 VS BASELINE 那段能答。")
    print("    ⚠ teacher-forced:两个头看到的都是真前驱。本模型实测曝光偏差 0.014,温和上界。")


def _print_decoder_ablation(good, col, median):
    """Read decoder_ablation_probe.py's replay of four block decoders on the same blocks.

    Silent on runs without those keys. What it must make impossible to misread:

      * ``today`` is the SERVE'S CURRENT RULE (full-vocab argmax of base+bias, committing
        each step) replayed offline -- so it is the baseline every other number is measured
        against, and its agreement with the run's own ``hard_accept_len`` is the fidelity
        gate. If those two disagree, the replay does not match the real decoder and every
        figure below is void; that check therefore runs first and shouts.

      * ``restrict`` is the same rule with candidates pruned to top-k. It is a strict
        handicap on ``today`` (same commit point, smaller search set) and exists only to
        PRICE that pruning -- which viterbi and decay also pay. Hence the decomposition
        the verdict below turns on:
            gain = (what joint decoding wins) - (what top-k pruning costs)
        A negative gain whose magnitude is under ``restrict_cost`` therefore does NOT mean
        joint decoding failed; it means k is too small and the next run should raise it.
    """
    names = ["today", "restrict", "viterbi", "decay", "viterbiN", "decayN"]
    if not any(f"train/dec_{n}_accept_len" in r for r in good[-1:] or good for n in names):
        return

    def med(key, n=200):
        v = col(good[-n:], key)
        return median(v) if v else None

    print("\n-- DECODER ABLATION (offline replay; probe run only) " + "-" * 24)
    al = {n: med(f"train/dec_{n}_accept_len") for n in names}
    hard = med("train/hard_accept_len")

    # 保真门:重放的 today 必须复现真实解码器
    if al["today"] is not None and hard is not None:
        d = abs(al["today"] - hard)
        if d > 0.05:
            print(f"  ✗✗ 保真门失败:重放的 today {al['today']:.3f} vs 实际 hard_accept_len {hard:.3f}"
                  f"(差 {d:.3f})")
            print("     重放与真实解码路径不一致,下面所有数字作废。先查 base 还原与 seed 取法。")
        else:
            print(f"  ✓ 保真门:重放 today {al['today']:.3f} ≈ hard_accept_len {hard:.3f}(差 {d:.3f})"
                  f" ⟹ 重放可信")

    print(f"\n  {'解码器':<10} {'accept_len':>11} {'相对 today':>11} {'赢':>7} {'输':>7}")
    for n in names:
        if al[n] is None:
            continue
        g = med(f"train/dec_{n}_gain")
        w = med(f"train/dec_{n}_win")
        l = med(f"train/dec_{n}_loss")
        tag = {"today": "  ← 基线(服务端现行)", "restrict": "  ← 剪枝对照",
               "viterbiN": "  ← 同上,按位归一", "decayN": "  ← 同上,按位归一"}.get(n, "")
        print(f"  {n:<10} {al[n]:>11.3f} "
              + (f"{g:>+11.3f}" if g is not None else f"{'—':>11}")
              + (f" {w:>6.1%} {l:>6.1%}" if w is not None else f" {'—':>6} {'—':>6}")
              + tag)

    cost = med("train/dec_restrict_cost")
    if cost is not None:
        print(f"\n  ★ top-k 剪枝成本(today − restrict)= {cost:+.3f} token")
        if cost < -1e-6:
            print("     ⚠ 为负:不可能 —— restrict 是 today 的严格削弱。查重放实现。")

    print("\n  ── 判读 ──")
    note = {
        "viterbi": "它最大化整块链分,而前缀接受付钱的是 Σ_t P(0..t 全对) —— 目标函数本就不同。",
        "decay":   "位置权重已压低后位、靠近前缀接受;若仍为负,说明衰减还不够或候选集不对。",
        "viterbiN": "按位归一后链分才是 log P(路径);若这一档才转正,先前的负值来自 logit 量纲而非目标错配。",
        "decayN":  "归一 + 位置权重,最贴近前缀接受的一档 —— 这是四个联合解码里先验最强的。",
    }
    for n in ("viterbi", "decay", "viterbiN", "decayN"):
        g = med(f"train/dec_{n}_gain")
        if g is None:
            continue
        if g > 0.02:
            print(f"    {n}: {g:+.3f} ⟹ **净赢**。联合解码的收益已盖过 k 的剪枝损失。")
        elif cost is not None and g < 0 and abs(g) < cost:
            print(f"    {n}: {g:+.3f},而剪枝独自就要 −{cost:.3f} ⟹ **联合解码本身是正收益,被 k 吃掉了**。")
            print("       该加大 DECODER_K 重跑,而不是判死刑。")
        elif g < -0.02:
            print(f"    {n}: {g:+.3f},超出剪枝损失 {cost if cost is not None else float('nan'):.3f}"
                  f" ⟹ 联合解码本身在伤害接受长度。")
        else:
            print(f"    {n}: {g:+.3f} ⟹ 与 today 打平(±0.02 内)。")
        print(f"       {note[n]}")

    # 归一化到底值多少:同一位置权重下,raw logits 求和 vs log-prob 求和
    pairs = [("viterbi", "viterbiN"), ("decay", "decayN")]
    deltas = [(a, b, med(f"train/dec_{b}_accept_len"), med(f"train/dec_{a}_accept_len"))
              for a, b in pairs]
    deltas = [(a, b, x - y) for a, b, x, y in deltas if x is not None and y is not None]
    if deltas:
        print("\n  ── 按位归一的效果(logsumexp 后再求和)──")
        for a, b, d in deltas:
            print(f"    {b} − {a} = {d:+.3f} token")
        if max(d for _, _, d in deltas) > 0.02:
            print("    ⟹ 确认:此前的链分把未归一的 logits 跨位置相加,按位置的 logit 量纲加权,这是伪影。")
        else:
            print("    ⟹ 归一化没救回来 ⟹ 负收益是真的目标错配,不是量纲伪影。")
    print("    ⚠ teacher-forced 上界:块的起始上下文是真前缀,服务端是目标模型验证过的那个。")


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
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=LOG",
                    help="LEAD SCAN mode: an experiment arm to compare against the shared "
                         "--baseline. Repeatable; the positional logfile is then unused")
    ap.add_argument("--at", type=int, nargs="+", default=None, metavar="STEP",
                    help="lead-scan: step counts to sample at (default: an automatic ladder)")
    ap.add_argument("--max-step", type=int, default=0, metavar="N",
                    help="drop global_steps > N; applied to BOTH runs. Use it to compare a long "
                         "finished run against a short live one AT THE SAME STEP COUNT -- two "
                         "experiments' deltas over the same baseline are only comparable when "
                         "measured over the same steps, since a delta drifts as the curve flattens")
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
        # Split each sub-op into STEADY (median) vs SPIKE (prints > 3x steady) and sum the spike
        # EXCESS-over-steady = the per-shape rebuild tax. Separates the recompile DRIVER (big spikeΣ,
        # e.g. MoE grouped-GEMM re-building per varying per-expert token count) from a real STEADY
        # cost (big steady, e.g. MLA eager sink-einsum). Derived from [FWD_PROF] ALONE → works when
        # [MOE-PROF] is absent (EP/compile path where its hook doesn't fire, or older logs).
        st = {}
        for tag, xs in agg.items():
            med = sorted(xs)[len(xs) // 2]
            spk = [x for x in xs if med > 0 and x > 3 * med]
            st[tag] = dict(xs=xs, med=med, nspk=len(spk),
                           spk=sum(x - med for x in spk), tot=sum(xs))
        rows = sorted(st.items(), key=lambda kv: kv[1]["tot"], reverse=True)
        print(f"  block sub-ops (DSPARK_PROFILE_FWD): {len(fwd)} print(s)  "
              f"[steady=median · spikeΣ=excess-over-steady = the per-shape rebuild tax]")
        for tag, s in rows:
            print(f"    {tag:10} n={len(s['xs']):>4}  steady={s['med']:>5}ms  max={max(s['xs']):>6}ms  "
                  f"spikes={s['nspk']:>3}(>3×)  spikeΣ={s['spk']:>8}ms  Σ={s['tot']:>8}ms")
        drv = max(st.items(), key=lambda kv: kv[1]["spk"])                       # biggest rebuild tax
        stdy = max(st.items(), key=lambda kv: kv[1]["med"] * len(kv[1]["xs"]))   # biggest steady*n
        print(f"  → recompile DRIVER (biggest spikeΣ) = {drv[0]}  |  STEADY-cost leader = {stdy[0]}   "
              f"(MLA=latent attn · mHC=Sinkhorn hyper-conn · MoE=experts)")

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


def moe_load_report(text: str) -> None:
    """MoE expert-load balance over the run from ``[MOE-LOAD Ln]`` prints (DSPARK_LOG_EXPERT_LOAD=1).
    Silent if absent. Shows per-layer used/dead experts + normalized entropy EARLY->LATE, so a router
    COLLAPSE (entropy falling / dead experts growing over training) is visible. entropy 1.0 = uniform
    (all 256 experts used), low = collapsed to a few -> the 256-expert capacity is wasted."""
    rows = re.findall(
        r"\[MOE-LOAD L(\d+)\]\s+used=(\d+)/(\d+)\s+dead=(\d+)\s+top\d+=([\d.]+)\s+entropy=([\d.]+)",
        text,
    )
    if not rows:
        return  # diagnostic off / older log — stay silent
    print("\n-- MoE EXPERT-LOAD BALANCE (DSPARK_LOG_EXPERT_LOAD) " + "-" * 27)
    by_layer: dict[int, list] = {}
    for lyr, used, E, dead, top, ent in rows:
        by_layer.setdefault(int(lyr), []).append((int(used), int(E), float(ent)))
    # hot-expert-ID overlap across prints — the CRUCIAL check: single-step used/entropy is over ONLY
    # ~180 tokens/rank, so even a balanced router looks sparse PER STEP. What matters is whether the
    # SAME experts are hot every step (union ~ top-k => a FIXED collapsed subset) or the hot set ROTATES
    # (union large => different inputs hit different experts => over the dataset ALL experts get used =
    # not a real collapse, the per-step entropy is a small-batch artifact).
    hot_by_layer: dict[int, list] = {}
    for lyr, hs in re.findall(r"\[MOE-LOAD L(\d+)\][^\n]*?hot=\[([\d,\s]*)\]", text):
        ids = {int(x) for x in hs.replace(" ", "").split(",") if x}
        if ids:
            hot_by_layer.setdefault(int(lyr), []).append(ids)

    print(f"  {'layer':6} {'#print':>6} {'used/E first->last':>20} {'entropy first->last (min)':>28} {'eff.experts now':>16}")
    entropy_low = False
    for lyr in sorted(by_layer):
        seq = by_layer[lyr]
        (u0, E, e0), (uL, _, eL) = seq[0], seq[-1]
        emin = min(s[2] for s in seq)
        neff = round(E ** eL)    # effective experts carrying the mass NOW = E^entropy (the honest count, not 'used')
        print(f"  L{lyr:<5} {len(seq):>6} {f'{u0}->{uL}/{E}':>20} {f'{e0:.2f}->{eL:.2f} (min {emin:.2f})':>28} {f'~{neff}/{E}':>16}")
        if eL < 0.65:            # eff.experts < ~15% of E -> per-step collapsed ('used' is misleading -> ignore it)
            entropy_low = True

    fixed_collapse = False
    rotating = False
    if hot_by_layer and any(len(v) >= 2 for v in hot_by_layer.values()):
        print(f"\n  {'layer':6} {'#hot-sets':>9} {'union (ever-hot pool)':>22} {'core':>5} {'pool growth (2nd half)':>24}")
        for lyr in sorted(hot_by_layer):
            sets = hot_by_layer[lyr]
            union = set().union(*sets)
            inter = set.intersection(*sets)
            E = by_layer.get(lyr, [(0, 256, 0)])[0][1]
            k = len(sets[0])  # top-k size
            # SATURATION: does the 2nd half of training keep adding NEW experts to the pool (rotating,
            # heading to full coverage) or has the pool stopped growing (a FIXED subset)? This is the real
            # fixed-vs-rotating test — a saturated pool of 65 is just as collapsed as one of 16.
            half = len(sets) // 2
            u_early = set().union(*sets[:half]) if half else set()
            new_late = len(union - u_early)   # experts the SECOND half added to the ever-hot pool
            saturated = len(sets) >= 20 and new_late <= max(3, 0.08 * len(union))
            growth = f"+{new_late} (SATURATED)" if saturated else f"+{new_late} (still growing)"
            print(f"  L{lyr:<5} {len(sets):>9} {f'{len(union)}/{E}':>22} {len(inter):>5} {growth:>24}")
            if len(union) >= E * 0.5:                       # broad pool -> dataset coverage OK (not a collapse)
                rotating = True
            elif len(union) <= 2 * k or saturated:          # tiny pool, OR a sub-half pool that STOPPED growing
                fixed_collapse = True                        # (still-growing small pool -> neither -> keep watching)

    print("  ── verdict ──")
    if entropy_low and fixed_collapse:
        print("    ⛔ TRUE COLLAPSE: low per-step effective-experts AND the ever-hot pool has SATURATED")
        print("    (a FIXED subset — the 2nd half of training added ~no new experts), so most of the")
        print("    E-expert capacity is permanently idle => caps accept_len + drives over-train decline.")
        print("    FIX = noaux_tc load-balance bias (DSPARK_MOE_BALANCE) and/or lower LR (less collapse")
        print("    pressure). ⚠ Confirm TRAINING-drift not DATA-limited: is it already low at step 0 (INIT)?")
    elif entropy_low and rotating:
        print("    ⚠ PER-STEP SPARSE BUT ROTATING: entropy is low per step, but the hot experts CHANGE")
        print("    across steps (large union) -> over the dataset most experts DO get used. The low")
        print("    per-step number is a small-batch artifact, NOT a fixed collapse -> load balancing is")
        print("    likely NOT the bottleneck; look elsewhere. (Need many [MOE-LOAD] prints to be sure.)")
    elif entropy_low:
        print("    low per-step entropy, but too few prints to judge fixed-vs-rotating hot set — run")
        print("    longer with DSPARK_LOG_EXPERT_LOAD=1 (more [MOE-LOAD] prints) to compute the union.")
    else:
        print("    ✓ balanced (high entropy, most experts used) — load balancing is NOT the bottleneck.")



def loss_imbalance_report(recs, text: str) -> None:
    """Per-rank SUPERVISED-TOKEN imbalance over the run (trainer ``profile/sup_tokens_ranks``).

    Silent on logs without the instrumentation. This is the premise of the global loss
    normalization (``DSPARK_GLOBAL_LOSS_REDUCE`` / upstream PR #942): normalizing the masked
    loss per-rank weights rank r's tokens by ``1/(R*n_r)``, while the token-weighted objective
    weights every token by ``1/N``. The ratio ``mean(n)/n_r`` is exactly how far off rank r's
    weight is — **1.0 everywhere means the two objectives coincide and the fix is a no-op**, so
    this report is what says whether the fix can matter at all on this run.

    Everything is recomputed from the RAW per-rank counts rather than read from the trainer's
    derived fields, so a log written before those fields were fixed still analyses correctly.
    """
    steps = [
        [int(x) for x in line.replace(" ", "").split(",") if x]
        for line in re.findall(r"sup_tokens_ranks=\[([\d,\s]*)\]", text)
    ]
    steps = [v for v in steps if v and sum(v) > 0]
    if not steps:
        return  # instrumentation off / older log — stay silent

    print("\n-- PER-RANK SUPERVISED-TOKEN IMBALANCE " + "-" * 41)
    hi, lo, spread, zero_steps = [], [], [], 0
    for v in steps:
        mean = sum(v) / len(v)
        nz = [t for t in v if t > 0]
        if len(nz) < len(v):
            zero_steps += 1
        skew = [mean / t for t in nz]          # zero-token ranks excluded: mean/0 is not a ratio
        hi.append(max(skew)); lo.append(min(skew))
        spread.append((max(v) - min(v)) / mean)

    n = len(steps)
    med = lambda v: sorted(v)[len(v) // 2]
    print(f"  logged steps: {n}   ranks: {len(steps[0])}")
    # median first: the mean of skew_max is dragged by rare near-empty ranks, so the
    # median is the representative figure and the max is a tail illustration.
    print(f"  weight skew mean(n)/n_r    most OVER-weighted rank: median {med(hi):.3f}  "
          f"mean {sum(hi)/n:.3f}  worst {max(hi):.3f}")
    print(f"                             most UNDER-weighted rank: median {med(lo):.3f}  "
          f"mean {sum(lo)/n:.3f}  worst {min(lo):.3f}")
    print(f"  token spread (max-min)/mean: mean {sum(spread)/n:.3f}  worst {max(spread):.3f}")
    if zero_steps:
        print(f"  ⚠ {zero_steps}/{n} steps had a rank with ZERO supervised tokens — the extreme case:")
        print("    it contributes nothing to the loss yet still takes 1/R of the mean-of-ratios.")

    # A chronically light rank is a permanently down-weighted slice of the data, which the
    # per-step spread hides — cumulative totals are the number that matters.
    per_rank: dict[int, int] = {}
    for v in steps:
        for i, t in enumerate(v):
            per_rank[i] = per_rank.get(i, 0) + t
    gmean = sum(per_rank.values()) / len(per_rank)
    worst, best = min(per_rank, key=per_rank.get), max(per_rank, key=per_rank.get)
    print("  cumulative tokens/rank: " + " ".join(f"r{i}={per_rank[i]}" for i in sorted(per_rank)))
    print(f"  ⟹ cumulatively rank {worst} carries {100*per_rank[worst]/gmean-100:+.1f}% vs mean, "
          f"rank {best} {100*per_rank[best]/gmean-100:+.1f}%")
    # NOISE vs SYSTEMATIC BIAS. If the per-step deviations were zero-mean and independent,
    # the cumulative relative deviation would shrink like 1/sqrt(steps) — halving the sample
    # should inflate it by ~sqrt(2). If instead each half shows the SAME deviation, the
    # imbalance is a fixed property of the sampler and never averages out.
    half = len(steps) // 2
    if half >= 20:
        def _dev(chunk):
            tot: dict[int, int] = {}
            for v in chunk:
                for i, t in enumerate(v):
                    tot[i] = tot.get(i, 0) + t
            m = sum(tot.values()) / len(tot)
            return {i: 100 * tot[i] / m - 100 for i in tot}, tot
        d1, t1 = _dev(steps[:half])
        d2, t2 = _dev(steps[half:])
        hv = max(d1, key=d1.get)
        print(f"  systematic? first half r{hv} {d1[hv]:+.1f}%   second half r{hv} {d2[hv]:+.1f}%")
        # Correlate the two halves' per-rank deviation VECTORS. Requiring the full rank
        # ordering to match was too strict: the middle ranks sit within ~1% of each other,
        # so one swap among near-ties flipped the verdict to "noise" while both halves were
        # showing the same +11.5% skew. Correlation is robust to those ties.
        ks = sorted(d1)
        a = [d1[k] for k in ks]
        b = [d2[k] for k in ks]
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        va = sum((x - ma) ** 2 for x in a) ** 0.5
        vb = sum((y - mb) ** 2 for y in b) ** 0.5
        corr = cov / (va * vb) if va and vb else 0.0
        same_mag = abs(d1[hv] - d2[hv]) < 0.35 * max(abs(d1[hv]), 1e-9)
        print(f"    per-rank deviation correlation between halves: {corr:+.3f}")
        if corr > 0.8 and same_mag:
            print("    ⟹ SYSTEMATIC: both halves show the same skew, on the same ranks.")
            print("      Zero-mean noise would shrink ~1/sqrt(n); this does not, so it never averages out.")
        elif corr > 0.8:
            print("    ⟹ same ranks are consistently heavy/light, but the magnitude moved — partly systematic.")
        else:
            print("    ⟹ looks like sampling noise: the halves disagree, so it averages out with more steps.")

    ratio = per_rank[best] / max(per_rank[worst], 1)
    if ratio < 1.01:
        print("  ⟹ balanced within 1% cumulatively — per-rank and global normalization are")
        print("    effectively the same objective here, so the fix would be a no-op on this run.")
    else:
        print(f"  ⟹ NOT balanced (heaviest/lightest = {ratio:.3f} cumulatively): the per-rank objective")
        print("    is a mean-of-ratios, not token-weighted. This is what DSPARK_GLOBAL_LOSS_REDUCE=1 fixes.")
    print(f"  global loss reduce in this run: {'ON' if '[LOSS-REDUCE]' in text else 'OFF (per-rank)'}")


def hs_split_report(text: str) -> None:
    """HS-fetch 3-phase split from ``[HS-SPLIT]`` prints (DSPARK_HS_SPLIT=1). Silent if absent.
    Each SLOW fetch (> DSPARK_HS_SPLIT_MS) is split into: create (the completions.create() round-trip
    = serve prefill/compute + HTTP), wait (create -> file appears = serve dumper NFS write + dirent
    visibility), read (load_file of the ~132MB safetensors = trainer NFS read). This PINS a per-rank
    HS straggler to the SERVE (create/wait) vs the trainer's NFS read (read) — settling the
    'is it NFS/NUMA or the serve?' question directly instead of guessing topology."""
    rows = re.findall(
        r"\[HS-SPLIT\] rank=(\S+) idx=\S+ total=\d+ms create=(\d+) wait=(\d+) read=(\d+)",
        text,
    )
    if not rows:
        return  # DSPARK_HS_SPLIT off / older log — stay silent

    def _med(a):
        if not a:
            return 0
        s = sorted(a)
        return int(s[len(s) // 2])

    by_rank: dict[str, dict] = {}
    all_c: list = []
    all_w: list = []
    all_r: list = []
    for rk, c, w, rd in rows:
        c, w, rd = int(c), int(w), int(rd)
        d = by_rank.setdefault(rk, {"create": [], "wait": [], "read": []})
        d["create"].append(c)
        d["wait"].append(w)
        d["read"].append(rd)
        all_c.append(c)
        all_w.append(w)
        all_r.append(rd)

    print("\n-- HS FETCH 3-PHASE SPLIT (DSPARK_HS_SPLIT) " + "-" * 35)
    print(f"  {len(rows)} slow fetches logged (> DSPARK_HS_SPLIT_MS). medians (ms):")
    print(f"  {'rank':>5} {'n':>6} {'create':>9} {'wait':>7} {'read':>7}")
    for rk in sorted(by_rank, key=lambda x: int(x) if x.isdigit() else 99):
        d = by_rank[rk]
        print(f"  {rk:>5} {len(d['read']):>6} {_med(d['create']):>9} {_med(d['wait']):>7} {_med(d['read']):>7}")
    mc, mw, mr = _med(all_c), _med(all_w), _med(all_r)
    tot = max(1, mc + mw + mr)
    print(f"  {'ALL':>5} {len(rows):>6} {mc:>9} {mw:>7} {mr:>7}   "
          f"→ create {100 * mc // tot}% / wait {100 * mw // tot}% / read {100 * mr // tot}%")
    dom = max((("create", mc), ("wait", mw), ("read", mr)), key=lambda kv: kv[1])[0]
    print("  ── verdict ──")
    if dom == "create":
        print("    ★ SERVE-BOUND: the straggler is `create` = the completions.create() round-trip")
        print("    (serve prefill/compute + HTTP), NOT NFS I/O (wait/read are tiny). The serve can't")
        print("    prefill+return HS fast enough — over-subscribed (NPROC*NUM_WORKERS concurrent")
        print("    prefills vs the serve's max-num-batched-tokens). FIX = serve throughput (more DP /")
        print("    bigger max-num-batched-tokens / w8a8), fewer concurrent fetchers, or RAISE")
        print("    --max-anchors (heavier step lets the serve keep up). NUMA-pin does NOTHING here.")
    elif dom == "wait":
        print("    SERVE-WRITE-bound: `wait` (create -> file appears) dominates = the serve dumper's")
        print("    NFS write / dirent-visibility lag. FIX = async dump / stage to local disk + sidecar /")
        print("    NFS mount tuning (async, larger wsize).")
    else:
        print("    NFS-READ-bound: `read` (load_file) dominates = the TRAINER's NFS read of the ~132MB")
        print("    safetensors is slow (NUMA-far from the NIC / congested NFS). FIX = NUMA-pin the")
        print("    dataloader workers to the NIC's NUMA node, or read-once + HCCS broadcast.")


def main() -> None:
    args = _build_parser().parse_args()

    if args.arm:
        if not args.baseline:
            raise SystemExit("--arm needs a shared --baseline")
        pairs = []
        for a in args.arm:
            if "=" not in a:
                raise SystemExit(f"--arm wants NAME=LOG, got {a!r}")
            n, _, p = a.partition("=")
            pairs.append((n, p))
        _lead_scan(args.baseline[0], pairs, args.at, args.skip, args.spike_k, args.max_step)
        return

    recs, ckpt_steps, steps_per_epoch, raw_text = _load_and_skip(
        args.logfile, args.skip, max_step=args.max_step)
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
            brecs, bckpt, _, _ = _load_and_skip(bpath, args.skip, quiet=True, max_step=args.max_step)
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
              f"   [block ceiling = block_size. ⚠ this is the SOFT/analytical accept_len "
              f"(1-d_TV overlap); hard_accept_len is the greedy one comparable to a serve eval. "
              f"Same-serve released draft = {RELEASED_ACCEPT_LEN['avg']} avg / "
              f"{RELEASED_ACCEPT_LEN['gsm8k']} gsm8k; 3.94 is only the cross-stack floor]")
    for k, lab in [("train/accept_rate", "accept_rate"), ("train/full_acc", "full_acc")]:
        v = last_n_med(k)
        if v is not None:
            print(f"{lab:15}: {fmt(v)}")
    _print_selection_headroom(good, col, median)
    _print_select_ablation(good, col, median)
    _print_decoder_ablation(good, col, median)

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
    moe_load_report(raw_text)
    hs_split_report(raw_text)
    loss_imbalance_report(recs, raw_text)

    # ---------------- recent dynamics (is it STILL learning?) ----------------
    N = args.recent

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
        print("  → truly plateaued on all metrics — near-converged or stuck (lr schedule / more data "
              f"if far from the {RELEASED_ACCEPT_LEN['avg']} same-serve released bar).")

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
        # ⚠ hoist the threshold OUT of the comprehension. Inline it was re-sorted + re-medianed
        # once PER ELEMENT — O(n² log n). On a 124,480-step run that is ~17 min of pure waste
        # (8.3 ms × 124,480); hoisted it is ~1 ms. Same value, same result.
        _tp_thr = 0.5 * median(sorted(tp)[len(tp) // 2:])
        steady_tp = [v for v in tp if v > _tp_thr]
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

    recompile_ov = _excess("profile/fwd_compute_ms", fwd_steady)             # excess TRUE fwd (align + all-to-all straggler already removed) = grouped-GEMM recompile / AscendC stall
    align_ov = sum(col(recs, "profile/align_ms"))                            # explicit all-gather barrier wait
    a2a_ov = sum(col(recs, "profile/a2a_straggler_ms"))                      # EP all-to-all wait for an HS-STARVED rank (was mislabelled "recompile" — [MOE-PROF] verdict)
    hs_ov = _excess("profile/fetch_ms", fetch_steady, only_nonckpt=True)     # excess HS-fetch = local H2D / load stall (ckpt saves excluded)
    ckpt_ov = sum(max(0.0, f(r, "profile/step_ms") - step_steady) for r in recs
                  if step_of(r) in ckpt_steps and f(r, "profile/step_ms") is not None)
    floor = max(0.0, total - recompile_ov - align_ov - a2a_ov - hs_ov - ckpt_ov)
    print(f"  {'component':30} {'time':>9}   {'% wall-clock':>12}")
    for label, v in [("steady compute (floor)", floor), ("recompile (true-fwd excess)", recompile_ov),
                     ("HS straggler (align barrier)", align_ov), ("HS straggler (EP all-to-all)", a2a_ov),
                     ("HS fetch stall (excess)", hs_ov), ("checkpoint saves", ckpt_ov)]:
        print(f"  {label:30} {v/1000:7.1f}s   {100*v/total:11.1f}%")
    rc, hs, al = 100 * recompile_ov / total, 100 * hs_ov / total, 100 * (align_ov + a2a_ov) / total
    serve = hs_ov + align_ov + a2a_ov            # all trace to the serve/HS pipeline
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
    if al and last_al < RELEASED_ACCEPT_LEN["avg"]:
        notes.append(
            f"accept_len {last_al:.2f} (SOFT) < the {RELEASED_ACCEPT_LEN['avg']} same-serve released "
            "bar — still training; note the serve-side number is measured differently, so compare "
            "hard_accept_len, and settle it with a real eval."
        )
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
        _plot_moe(raw_text, args.out, steps_per_epoch)   # expert-load plot (both modes; silent if no [MOE-LOAD])
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


def _plot_moe(text, out, steps_per_epoch=None):
    """MoE expert-load plots from ``[MOE-LOAD]`` prints — built to be read AT A GLANCE by non-experts.

    Panel A: 'effective experts working' = ``E**entropy`` (the honest capacity number) over training,
    per layer. Bold line = actually working; faint dotted = merely 'touched' by >=1 token (looks fine
    but misleads). A collapse shows as the bold line sliding toward the red zone.
    Panel B: a final-state capacity gauge per layer ('18 of 256 working (7%)').
    Silent if the log has no ``[MOE-LOAD]`` lines (``DSPARK_LOG_EXPERT_LOAD`` was off)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        return
    step_re = re.compile(r"global_step=(\d+)")
    moe_re = re.compile(
        r"\[MOE-LOAD L(\d+)\]\s+used=(\d+)/(\d+)\s+dead=\d+\s+top\d+=[\d.]+\s+entropy=([\d.]+)")
    # each [MOE-LOAD] block is printed just BEFORE its step's trainer log -> attach the NEXT global_step.
    series: dict[int, list] = {}
    pending: list = []
    for line in text.splitlines():
        mm = moe_re.search(line)
        if mm:
            pending.append((int(mm.group(1)), int(mm.group(2)), int(mm.group(3)), float(mm.group(4))))
            continue
        ms = step_re.search(line)
        if ms and pending:
            st = int(ms.group(1))
            for lyr, used, E, ent in pending:
                series.setdefault(lyr, []).append((st, ent, used, E))
            pending = []
    if not series:
        return
    os.makedirs(out, exist_ok=True)
    layers = sorted(series)
    colors = ["#2E6CF6", "#1B8A4E", "#D62828", "#7A3FF2", "#E08A1E", "#00A5B5"]
    E = series[layers[0]][0][3]

    fig, (axA, axB) = plt.subplots(
        2, 1, figsize=(9.5, 7.4), gridspec_kw={"height_ratios": [2.3, 1.0]})

    # ---- Panel A: effective experts over training ----
    for i, lyr in enumerate(layers):
        seq = series[lyr]
        xs = [s for s, _, _, _ in seq]
        neff = [E ** e for _, e, _, _ in seq]         # effective experts = E^normalized-entropy
        c = colors[i % len(colors)]
        axA.plot(xs, neff, lw=2.2, color=c, marker="o", ms=3, label=f"layer {lyr}")
        axA.annotate(f"{neff[-1]:.0f}", xy=(xs[-1], neff[-1]),
                     xytext=(5, (i - (len(layers) - 1) / 2) * 9), textcoords="offset points",
                     fontsize=8.5, color=c, va="center", fontweight="bold")
    axA.axhline(E, ls="--", lw=1.2, color="0.35")
    axA.annotate(f"all {E} experts sharing the work = ideal", xy=(0.01, E),
                 xycoords=("axes fraction", "data"), xytext=(0, -3), textcoords="offset points",
                 fontsize=8, color="0.35", va="top")
    axA.axhspan(0, E * 0.12, color="#D62828", alpha=0.06)
    axA.annotate("collapsed — capacity wasted", xy=(0.5, E * 0.06), xycoords=("axes fraction", "data"),
                 fontsize=8, color="#B02020", ha="center", va="center", alpha=0.85)
    if steps_per_epoch:
        xmax = max(s for lyr in layers for s, _, _, _ in series[lyr])
        ep = 1
        while ep * steps_per_epoch <= xmax * 1.001 and ep <= 64:  # cap: never let a bad
            # steps_per_epoch turn this into tens of thousands of text artists
            axA.axvline(ep * steps_per_epoch, color="0.6", ls=":", lw=0.8, alpha=0.6)
            axA.annotate(f"e{ep}", xy=(ep * steps_per_epoch, 1.0), xycoords=("data", "axes fraction"),
                         xytext=(2, -2), textcoords="offset points", color="0.45", fontsize=7, va="top")
            ep += 1
    axA.set_ylim(0, E * 1.10)
    axA.set_xlabel("training step")
    axA.set_ylabel(f"experts (out of {E})")
    fig.suptitle("MoE — how many of the experts are ACTUALLY working", fontsize=13, fontweight="bold", y=0.995)
    axA.set_title(f"effective experts = {E}^entropy — how many really share the load (high = healthy, low = collapsed)",
                  fontsize=8.5, color="0.4")
    axA.grid(alpha=0.3)
    axA.legend(loc="center right", fontsize=8.5)

    # ---- Panel B: final-state capacity gauge per layer ----
    finals = [(lyr, E ** series[lyr][-1][1]) for lyr in layers]
    ypos = list(range(len(finals)))
    for (lyr, nf), y in zip(finals, ypos):
        frac = nf / E
        col = "#D62828" if frac < 0.15 else "#E08A1E" if frac < 0.40 else "#1B8A4E"
        axB.barh(y, E, color="0.90", height=0.6, zorder=1)          # total capacity (grey)
        axB.barh(y, nf, color=col, height=0.6, zorder=2)            # working (colored)
        axB.text(E * 0.985, y, f"{nf:.0f} of {E} working  ({100 * frac:.0f}%)", ha="right", va="center",
                 fontsize=9.5, color="#1a1a1a", fontweight="bold", zorder=3)
    axB.set_yticks(ypos); axB.set_yticklabels([f"layer {l}" for l, _ in finals], fontsize=9)
    axB.set_xlim(0, E); axB.invert_yaxis()
    axB.set_xlabel(f"experts working (of {E})  —  latest state")
    axB.set_title("Current expert utilization per layer  (green = healthy, red = collapsed)", fontsize=10)
    axB.grid(axis="x", alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(f"{out}/moe_experts.png", dpi=120)
    plt.close(fig)
    print(f"  · MoE expert-load plot -> {out}/moe_experts.png")


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
