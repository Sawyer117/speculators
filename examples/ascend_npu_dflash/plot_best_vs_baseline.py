#!/usr/bin/env python3
"""Static "our best vs released baseline" accept-length TABLE, rendered to a PNG.

Unlike analyze_train_run.py (which plots a LIVE training log), this figure reads NO log — it renders
two hardcoded result rows so the image is STABLE:
  * BASELINE never changes (the released draft, measured once).
  * BEST only moves when a new run beats it and we EDIT the BEST block below.
So it does not wiggle per-log; it's the fixed scoreboard. Numbers are the per-dataset accept_len from
docs/deployment/ascend-npu-dsv4-dspark-eval-results.md — keep the two in sync (this file is a mirror).

    python plot_best_vs_baseline.py                    # -> ./best_vs_baseline.png
    python plot_best_vs_baseline.py --out ~/scoreboard.png
"""
# SPDX-License-Identifier: Apache-2.0
import argparse

DATASETS = ["gsm8k", "math500", "humaneval", "mbpp", "mt-bench"]

# ── released draft, full DATASET=all on our #12006 serve — the FIXED bar. Do NOT change unless the
#    released baseline itself is re-measured. (source: the eval-results ledger, `released draft` row)
BASELINE = {"gsm8k": 4.658, "math500": 4.661, "humaneval": 4.942, "mbpp": 4.535, "mt-bench": 3.294}

# ── our current BEST trained draft. ✏️ UPDATE THIS BLOCK (values + BEST_LABEL) ONLY when a new run
#    beats it — that is the only thing that should ever move this scoreboard.
BEST_LABEL = "epoch4-17w  (2026-07-19)"
BEST = {"gsm8k": 3.404, "math500": 3.265, "humaneval": 3.312, "mbpp": 3.058, "mt-bench": 2.344}


def _avg(d):
    return sum(d[k] for k in DATASETS) / len(DATASETS)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="best_vs_baseline.png", help="output PNG (default ./best_vs_baseline.png)")
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import colors
    except ImportError:
        raise SystemExit("need matplotlib — pip install matplotlib")

    rows = DATASETS + ["average"]
    base = dict(BASELINE, average=_avg(BASELINE))
    best = dict(BEST, average=_avg(BEST))

    cmap = plt.get_cmap("RdYlGn")
    norm = colors.Normalize(vmin=50, vmax=100)   # shade the "% of baseline" column over 50–100%

    col_labels = ["Dataset", "Released\n(baseline)", f"Our best\n{BEST_LABEL}", "% of\nbaseline", "gap"]
    cell_text, cell_colors = [], []
    for r in rows:
        pct = 100.0 * best[r] / base[r]
        gap = best[r] - base[r]
        cell_text.append([r, f"{base[r]:.3f}", f"{best[r]:.3f}", f"{pct:.0f}%", f"{gap:+.3f}"])
        cell_colors.append(["#eef2f7", "#eef2f7", "#ffffff", colors.to_hex(cmap(norm(pct))), "#ffffff"])

    fig, ax = plt.subplots(figsize=(8.2, 0.9 + 0.52 * (len(rows) + 1)))
    ax.axis("off")
    ax.set_title("DSV4-DSpark  —  accept length: our best vs released baseline",
                 fontsize=13, fontweight="bold", pad=16)

    tbl = ax.table(cellText=cell_text, colLabels=col_labels, cellColours=cell_colors,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1, 1.7)
    ncol = len(col_labels)
    for c in range(ncol):
        tbl[0, c].set_text_props(fontweight="bold", color="white")   # header
        tbl[0, c].set_facecolor("#33415a")
        tbl[len(rows), c].set_text_props(fontweight="bold")          # average row (last)
    for r in range(len(rows) + 1):                                   # thin cell borders
        for c in range(ncol):
            tbl[r, c].set_edgecolor("#c8d0da")

    fig.text(0.5, 0.03,
             "baseline = released draft, full DATASET=all on the #12006 serve (num_spec=5, greedy).  "
             "Source: docs/deployment/ascend-npu-dsv4-dspark-eval-results.md",
             ha="center", fontsize=7, color="0.45")
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved: {args.out}")
    print(f"  baseline avg {base['average']:.3f}  |  best avg {best['average']:.3f}  "
          f"({100 * best['average'] / base['average']:.0f}% of baseline)")


if __name__ == "__main__":
    main()
