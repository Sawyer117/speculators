#!/usr/bin/env python3
"""Analyze a DFlash training log: parse loss / per-position acceptance / lr /
throughput, print a summary, dump a CSV, and render charts.

Robust to the repo's rich-logged format (multi-line records, optional leading
`[HH:MM:SS]` from a `tail -f | awk` pipe, doubled timestamps). It keys records
on `global_step=`, which is the last token of each step's block.

Usage:
    python analyze_train_log.py <logfile|glob|dir>            # auto-picks newest
    python analyze_train_log.py outputs/.../logs/train_4b_*.log
    python analyze_train_log.py train.log --out ./logreport --last 2000

Outputs (under --out, default "<logfile>.analysis/"):
    metrics.csv         one row per logged step
    loss.png            loss (raw + EMA) vs step
    full_acc.png        full_acc (raw + EMA) vs step
    position_curve.png  acceptance vs position (early vs late window)
    position_heatmap.png  position acc over training (step buckets)
    accept_length.png   expected accept length proxy vs step
    lr.png              learning-rate schedule
"""
# SPDX-License-Identifier: Apache-2.0
import argparse
import csv
import glob
import os
import re
import sys

# numeric token: ints, floats, scientific
_KV = re.compile(r"([A-Za-z][\w/]*)=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
_TS = re.compile(r"\[(\d{2}):(\d{2}):(\d{2})\]")


def _resolve_log(arg: str) -> str:
    """Accept a file, a glob, or a dir; return the newest matching .log file."""
    if os.path.isdir(arg):
        cands = glob.glob(os.path.join(arg, "*.log"))
    else:
        cands = glob.glob(arg)
    if not cands:
        if os.path.exists(arg):
            return arg
        sys.exit(f"No log file matches: {arg}")
    return max(cands, key=os.path.getmtime)


def parse_log(path: str):
    """Return (records, position_keys). Each record is a dict of floats plus
    't' = seconds-since-first-line (monotonic, handles midnight wrap)."""
    records = []
    cur: dict = {}
    pos_nums = set()
    day = 0
    prev_secs = None
    cur_t = None

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _TS.search(line)
            if m:
                h, mi, s = (int(x) for x in m.groups())
                secs = h * 3600 + mi * 60 + s
                if prev_secs is not None and secs < prev_secs - 5:
                    day += 1  # wrapped past midnight
                prev_secs = secs
                cur_t = day * 86400 + secs

            for key, val in _KV.findall(line):
                if key == "global_step":
                    cur["global_step"] = int(float(val))
                    cur["t"] = cur_t
                    if "loss" in cur or "full_acc" in cur:
                        records.append(cur)
                    cur = {}
                elif key == "epoch":
                    cur["epoch"] = int(float(val))
                elif key == "lr":
                    cur["lr"] = float(val)
                elif key == "train/loss":
                    cur["loss"] = float(val)
                elif key == "train/full_acc":
                    cur["full_acc"] = float(val)
                else:
                    pm = re.fullmatch(r"train/position_(\d+)_acc", key)
                    if pm:
                        n = int(pm.group(1))
                        pos_nums.add(n)
                        cur[f"pos_{n}"] = float(val)

    pos_keys = [f"pos_{n}" for n in sorted(pos_nums)]
    return records, pos_keys


def _ema(xs, alpha=0.05):
    out, acc = [], None
    for x in xs:
        acc = x if acc is None else alpha * x + (1 - alpha) * acc
        out.append(acc)
    return out


def _expected_accept_len(rec, pos_keys):
    """Proxy for speculative accept length: sum_k prod_{j<=k} p_j over positions
    (independence approximation using marginal per-position accuracies)."""
    prod, total = 1.0, 0.0
    for k in pos_keys:
        p = rec.get(k)
        if p is None:
            break
        prod *= p
        total += prod
    return total


def main():
    ap = argparse.ArgumentParser(description="Analyze a DFlash training log.")
    ap.add_argument("log", help="log file, glob, or directory (newest .log used)")
    ap.add_argument("--out", default=None, help="output dir (default <log>.analysis)")
    ap.add_argument("--last", type=int, default=2000,
                    help="window size (steps) for 'late' per-position stats")
    ap.add_argument("--ema", type=float, default=0.02, help="EMA alpha for smoothing")
    ap.add_argument("--epochs", type=int, default=1,
                    help="analyze only the first N epochs present in the log "
                         "(default 1; runs here are single-epoch). 0 or "
                         "--all-epochs = analyze everything.")
    ap.add_argument("--all-epochs", action="store_true",
                    help="analyze all epochs (overrides --epochs)")
    args = ap.parse_args()

    path = _resolve_log(args.log)
    records, pos_keys = parse_log(path)
    if not records:
        sys.exit(f"Parsed 0 step records from {path} — is it a DFlash train log?")

    # restrict to the first N epochs PRESENT in the log (robust to tailing a
    # mid-run log where epoch 0 has scrolled off: takes the lowest N present).
    present_epochs = sorted({r["epoch"] for r in records if "epoch" in r})
    analyzed_epochs = present_epochs
    if not args.all_epochs and args.epochs > 0 and present_epochs:
        analyzed_epochs = present_epochs[: args.epochs]
        keep = set(analyzed_epochs)
        filtered = [r for r in records if r.get("epoch") in keep]
        if filtered:
            records = filtered
        else:
            print("[warn] epoch filter matched 0 records; analyzing all instead")
            analyzed_epochs = present_epochs

    outdir = args.out or (path + ".analysis")
    os.makedirs(outdir, exist_ok=True)

    steps = [r.get("global_step") for r in records]
    loss = [r.get("loss") for r in records]
    full = [r.get("full_acc") for r in records]
    lr = [r.get("lr") for r in records]

    # ---- throughput (it/s) from step/time deltas, robust median ----
    rates = []
    for a, b in zip(records, records[1:]):
        if a.get("t") is not None and b.get("t") is not None:
            dt, ds = b["t"] - a["t"], b["global_step"] - a["global_step"]
            if dt > 0 and ds > 0:
                rates.append(ds / dt)
    rates.sort()
    med_rate = rates[len(rates) // 2] if rates else None

    # ---- late-window per-position means ----
    late = records[-args.last:]
    early = records[: max(1, len(records) // 20)]  # first ~5%

    def _pos_mean(rs, key):
        vals = [r[key] for r in rs if key in r]
        return sum(vals) / len(vals) if vals else None

    late_curve = [(_pos_mean(late, k)) for k in pos_keys]
    early_curve = [(_pos_mean(early, k)) for k in pos_keys]

    # ---- console summary ----
    def _fmt(x, p=3):
        return "n/a" if x is None else f"{x:.{p}f}"

    lossv = [x for x in loss if x is not None]
    fullv = [x for x in full if x is not None]
    print("=" * 64)
    print(f"DFlash log analysis  ::  {path}")
    print("=" * 64)
    print(f"steps parsed     : {len(records)}  (global_step {steps[0]} → {steps[-1]})")
    _ep_note = "" if args.all_epochs else f"  (present: {present_epochs}; --all-epochs for all)"
    print(f"epochs analyzed  : {analyzed_epochs}{_ep_note}")
    print(f"block positions  : {len(pos_keys)}  ({pos_keys[0]}..{pos_keys[-1]})"
          if pos_keys else "block positions  : none found")
    if med_rate:
        print(f"throughput       : ~{med_rate:.2f} it/s (median)  | "
              f"~{med_rate*3600:.0f} steps/h")
    print(f"loss             : last {_fmt(loss[-1])} | min {_fmt(min(lossv))} | "
          f"mean(last {len(late)}) {_fmt(sum(x for x in (r.get('loss') for r in late) if x is not None)/max(1,sum(1 for r in late if r.get('loss') is not None)))}")
    print(f"full_acc         : last {_fmt(full[-1])} | max {_fmt(max(fullv))}")
    if pos_keys:
        el_late = sum(_expected_accept_len(r, pos_keys) for r in late) / len(late)
        el_early = sum(_expected_accept_len(r, pos_keys) for r in early) / len(early)
        print(f"E[accept len]    : early {_fmt(el_early,2)} → late {_fmt(el_late,2)} "
              f"tokens/block (independence proxy, block={len(pos_keys)+1})")
        print("per-position acc (late window):")
        for k, v in zip(pos_keys, late_curve):
            bar = "█" * int((v or 0) * 40)
            print(f"   {k:>7} {_fmt(v)} {bar}")

    # ---- CSV ----
    csv_path = os.path.join(outdir, "metrics.csv")
    cols = ["global_step", "epoch", "t", "loss", "full_acc", "lr"] + pos_keys
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in records:
            w.writerow([r.get(c, "") for c in cols])
    print("=" * 64)
    print(f"wrote {csv_path}")

    # ---- plots ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[plots skipped] matplotlib unavailable: {e}")
        return

    def _save(fig, name):
        p = os.path.join(outdir, name)
        fig.tight_layout()
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print(f"wrote {p}")

    # loss
    if lossv:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(steps, loss, lw=0.4, alpha=0.3, color="tab:blue", label="raw")
        ax.plot(steps, _ema(loss, args.ema), lw=1.8, color="tab:blue", label="EMA")
        ax.set(xlabel="global_step", ylabel="loss", title="Training loss")
        ax.grid(alpha=0.3); ax.legend()
        _save(fig, "loss.png")

    # full_acc
    if fullv:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(steps, full, lw=0.4, alpha=0.3, color="tab:green", label="raw")
        ax.plot(steps, _ema(full, args.ema), lw=1.8, color="tab:green", label="EMA")
        ax.set(xlabel="global_step", ylabel="full_acc", title="Full-block accuracy")
        ax.grid(alpha=0.3); ax.legend()
        _save(fig, "full_acc.png")

    # position curve: early vs late
    if pos_keys:
        xs = list(range(1, len(pos_keys) + 1))
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if any(v is not None for v in early_curve):
            ax.plot(xs, early_curve, "o--", color="tab:gray",
                    label=f"early ({len(early)} steps)")
        ax.plot(xs, late_curve, "o-", color="tab:red",
                label=f"late ({len(late)} steps)")
        ax.set(xlabel="block position", ylabel="acceptance / accuracy",
               title="Per-position acceptance (early vs late)")
        ax.set_xticks(xs); ax.grid(alpha=0.3); ax.legend(); ax.set_ylim(0, 1)
        _save(fig, "position_curve.png")

        # heatmap over training
        import numpy as np
        nb = min(120, len(records))
        idx = np.linspace(0, len(records) - 1, nb).astype(int)
        mat = np.full((len(pos_keys), nb), np.nan)
        bx = []
        for j, i in enumerate(idx):
            bx.append(records[i].get("global_step"))
            for pi, k in enumerate(pos_keys):
                if k in records[i]:
                    mat[pi, j] = records[i][k]
        fig, ax = plt.subplots(figsize=(10, 4.5))
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis",
                       vmin=0, vmax=1, extent=[bx[0], bx[-1], 1, len(pos_keys)])
        ax.set(xlabel="global_step", ylabel="block position",
               title="Per-position acceptance over training")
        fig.colorbar(im, ax=ax, label="acc")
        _save(fig, "position_heatmap.png")

        # expected accept length over training
        el = [_expected_accept_len(r, pos_keys) for r in records]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(steps, el, lw=0.4, alpha=0.3, color="tab:purple", label="raw")
        ax.plot(steps, _ema(el, args.ema), lw=1.8, color="tab:purple", label="EMA")
        ax.set(xlabel="global_step", ylabel="E[accepted tokens] / block",
               title="Expected accept length (independence proxy)")
        ax.grid(alpha=0.3); ax.legend()
        _save(fig, "accept_length.png")

    # lr
    if any(x is not None for x in lr):
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.plot(steps, lr, lw=1.4, color="tab:orange")
        ax.set(xlabel="global_step", ylabel="lr", title="Learning-rate schedule")
        ax.grid(alpha=0.3)
        _save(fig, "lr.png")


if __name__ == "__main__":
    main()
