#!/usr/bin/env python3
"""Plot DFlash training curves (loss + whatever else was logged) into a PNG.

TensorBoard-first (per-step, accurate; the trainer's `--logger tensorboard` writes
events.out.tfevents* under <repo>/logs/<run_name>/). Falls back to parsing the nohup
text log's `global_step=` + `loss=` if no tfevents are found.

Plots every scalar tag in a grid (loss first, with EMA smoothing like TensorBoard's
smoothing slider) and saves a PNG you can scp/open.

Usage:
    python plot_train_loss.py logs/                       # newest run under logs/
    python plot_train_loss.py logs/2026-06-27T...         # a specific run dir
    python plot_train_loss.py logs/ --out /tmp/train.png --ema 0.9
    python plot_train_loss.py outputs/.../logs/train_4b_*.log --text   # text-log fallback
"""
# SPDX-License-Identifier: Apache-2.0
import argparse
import glob
import os
import re
import sys

_TS = re.compile(r"\bglobal_step=(\d+)")
_LOSS = re.compile(r"\bloss=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def _find_tfevents(path: str):
    if os.path.isfile(path) and "tfevents" in os.path.basename(path):
        return path
    if os.path.isdir(path):
        cands = glob.glob(os.path.join(path, "**", "events.out.tfevents*"), recursive=True)
        if cands:
            return max(cands, key=os.path.getmtime)
    return None


def _read_tfevents(evfile: str):
    """Return ({tag: [(step, value)]}, rundir) or (None, _) on failure."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (  # noqa: PLC0415
            EventAccumulator,
        )
    except Exception:  # noqa: BLE001
        return None, None
    rundir = os.path.dirname(evfile) or "."
    try:
        ea = EventAccumulator(rundir, size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        series = {t: [(int(e.step), float(e.value)) for e in ea.Scalars(t)] for t in tags}
        return (series or None), rundir
    except Exception as e:  # noqa: BLE001
        print(f"[warn] tensorboard read failed ({type(e).__name__}); trying text log")
        return None, None


def _read_textlog(path: str):
    """Parse `global_step=` + `loss=` from the nohup text log → {'train/loss': pairs}."""
    pts = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            sm, lm = _TS.search(line), _LOSS.search(line)
            if sm and lm:
                pts.append((int(sm.group(1)), float(lm.group(1))))
    return {"train/loss": pts} if pts else {}


def _ema(ys, alpha):
    if not ys:
        return ys
    out = [ys[0]]
    for y in ys[1:]:
        out.append(alpha * out[-1] + (1 - alpha) * y)
    return out


def main():
    ap = argparse.ArgumentParser(description="Plot DFlash training curves to a PNG.")
    ap.add_argument("path", help="logs/ dir, a run dir, a tfevents file, or a .log")
    ap.add_argument("--out", default=None, help="output PNG (default <rundir>/train_curves.png)")
    ap.add_argument("--ema", type=float, default=0.9, help="EMA smoothing for loss (0=off)")
    ap.add_argument("--text", action="store_true", help="force text-log parse (skip tensorboard)")
    args = ap.parse_args()

    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        sys.exit("need matplotlib — `pip install matplotlib`")

    series, rundir = {}, "."
    if not args.text:
        ev = _find_tfevents(args.path)
        if ev:
            got, rd = _read_tfevents(ev)
            if got:
                series, rundir = got, rd
    if not series:
        logs = [args.path] if os.path.isfile(args.path) and args.path.endswith(".log") else []
        if os.path.isdir(args.path):
            logs = glob.glob(os.path.join(args.path, "**", "*.log"), recursive=True)
        logs = [x for x in logs if x.endswith(".log")]
        if logs:
            newest = max(logs, key=os.path.getmtime)
            series = _read_textlog(newest)
            rundir = os.path.dirname(newest) or "."
    if not series:
        sys.exit("no tfevents scalars and no parseable loss in a .log under that path")

    # loss first, then the rest alphabetically
    tags = sorted(series, key=lambda t: (0 if "loss" in t.lower() else 1, t))
    n = len(tags)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 3.2 * rows), squeeze=False)

    for idx, t in enumerate(tags):
        ax = axes[idx // cols][idx % cols]
        pts = sorted(set(series[t]))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, lw=0.7, alpha=0.35, label="raw")
        if "loss" in t.lower() and args.ema > 0 and len(ys) > 2:
            ax.plot(xs, _ema(ys, args.ema), lw=1.7, label=f"ema {args.ema}")
        ax.set_title(t)
        ax.set_xlabel("global_step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.tight_layout()
    out = args.out or os.path.join(rundir, "train_curves.png")
    fig.savefig(out, dpi=120)

    loss_tag = next((t for t in tags if "loss" in t.lower()), None)
    if loss_tag and series[loss_tag]:
        ys = [v for _, v in sorted(set(series[loss_tag]))]
        steps = [s for s, _ in sorted(set(series[loss_tag]))]
        print(f"loss: first={ys[0]:.4f} last={ys[-1]:.4f} min={min(ys):.4f} "
              f"over steps {steps[0]}..{steps[-1]}")
    print(f"saved: {out}  ({n} scalar(s): {', '.join(tags)})")


if __name__ == "__main__":
    main()
