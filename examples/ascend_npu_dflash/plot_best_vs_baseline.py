#!/usr/bin/env python3
"""Static "our runs vs released baseline" scoreboard TABLE (accept_len + per-position), to a PNG.

Unlike analyze_train_run.py (which plots a LIVE training log), this figure reads NO log — it renders
hardcoded result rows so the image is STABLE:
  * BASELINE never changes (the released draft, measured once).
  * RUNS is the list of OUR eval'd checkpoints; each renders one row under released per dataset.
So it does not wiggle per-log; it's the fixed scoreboard. Numbers mirror the per-dataset accept_len
AND per-position accept rate in docs/deployment/ascend-npu-dsv4-dspark-eval-results.md — keep in sync.

Layout: per dataset, one Released row then one row per RUN — columns accept_len + pos0..pos4. Every
run cell is shaded RdYlGn by its % of the baseline cell, so the tail collapse (pos3/pos4) reds out at
a glance while pos0-2 stay green.

    python plot_best_vs_baseline.py                    # -> ./best_vs_baseline.png
    python plot_best_vs_baseline.py --out ~/scoreboard.png
"""
# SPDX-License-Identifier: Apache-2.0
import argparse

DATASETS = ["gsm8k", "math500", "humaneval", "mbpp", "mt-bench"]

# ── released draft, full DATASET=all on our #12006 serve — the FIXED bar. Do NOT change unless the
#    released baseline itself is re-measured. (source: the eval-results ledger, `released draft` row)
BASELINE = {"gsm8k": 4.658, "math500": 4.661, "humaneval": 4.942, "mbpp": 4.535, "mt-bench": 3.294}
BASELINE_POS = {  # per-position CUMULATIVE accept rate S_k (%), pos0..pos4
    "gsm8k":     [92.77, 82.77, 73.29, 63.55, 53.45],
    "math500":   [91.78, 82.53, 73.01, 63.83, 54.93],
    "humaneval": [95.60, 88.94, 78.02, 70.16, 61.47],
    "mbpp":      [91.42, 80.91, 70.33, 60.20, 50.61],
    "mt-bench":  [79.21, 58.55, 41.45, 29.32, 20.87],
}

# ── OUR trained runs, oldest→newest. ✏️ APPEND a dict here when a new ckpt is eval'd (keep in sync
#    with the eval-results ledger). Current scoreboard = the window-128 / 77W run (epoch curve).
#    (The old window-2048 `epoch4-17w` run was removed — different draft config, not comparable.)
RUNS = [
    {
        "label": "ep0-77w (w128)",
        "al": {"gsm8k": 3.186, "math500": 3.041, "humaneval": 3.079, "mbpp": 2.868, "mt-bench": 2.255},
        "pos": {
            "gsm8k":     [87.91, 73.10, 39.21, 13.65, 4.73],
            "math500":   [84.81, 68.76, 35.60, 11.40, 3.58],
            "humaneval": [90.27, 74.20, 31.85, 9.54, 2.01],
            "mbpp":      [84.40, 64.07, 28.51, 7.93, 1.93],
            "mt-bench":  [67.61, 39.52, 14.15, 3.46, 0.77],
        },
    },
    {
        "label": "ep2.5-77w (w128, noise0.05) ⚠176",
        "al": {"gsm8k": 3.203, "math500": 3.096, "humaneval": 3.106, "mbpp": 2.928, "mt-bench": 2.272},
        "pos": {
            "gsm8k":     [88.53, 73.16, 39.37, 14.21, 5.07],
            "math500":   [86.00, 69.90, 37.38, 12.45, 3.89],
            "humaneval": [90.22, 74.45, 32.98, 10.46, 2.47],
            "mbpp":      [86.36, 66.43, 29.55, 8.37, 2.07],
            "mt-bench":  [68.33, 39.72, 14.55, 3.77, 0.82],
        },
    },
]


def _avg(d):
    return sum(d[k] for k in DATASETS) / len(DATASETS)


def _avg_pos(pos):
    return [sum(pos[k][i] for k in DATASETS) / len(DATASETS) for i in range(5)]


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
    base_al = dict(BASELINE, average=_avg(BASELINE))
    base_pos = dict(BASELINE_POS, average=_avg_pos(BASELINE_POS))
    runs = [{"label": r["label"],
             "al": dict(r["al"], average=_avg(r["al"])),
             "pos": dict(r["pos"], average=_avg_pos(r["pos"]))} for r in RUNS]
    rows_per_group = 1 + len(runs)   # released + one row per run

    cmap = plt.get_cmap("RdYlGn")
    norm = colors.Normalize(vmin=0, vmax=100)   # shade run cells 0–100% of baseline

    col_labels = ["Dataset", "Run", "accept_len", "pos0", "pos1", "pos2", "pos3", "pos4"]
    NEUT = "#eef2f7"      # released (baseline) row cells
    cell_text, cell_colors = [], []
    for d in rows:
        # released row (fixed reference)
        cell_text.append([d, "released", f"{base_al[d]:.3f}"] + [f"{v:.1f}" for v in base_pos[d]])
        cell_colors.append([NEUT] * len(col_labels))
        # one row per run — accept_len shows the % inline; every metric cell shaded by its own %
        for run in runs:
            pct_al = 100.0 * run["al"][d] / base_al[d] if base_al[d] else 0.0
            cells = [f"{run['al'][d]:.3f}  ({pct_al:.0f}%)"] + [f"{v:.1f}" for v in run["pos"][d]]
            shaded = [colors.to_hex(cmap(norm(pct_al)))]
            for i in range(5):
                p = 100.0 * run["pos"][d][i] / base_pos[d][i] if base_pos[d][i] else 0.0
                shaded.append(colors.to_hex(cmap(norm(p))))
            cell_text.append(["", run["label"]] + cells)
            cell_colors.append(["#ffffff", "#ffffff"] + shaded)

    fig, ax = plt.subplots(figsize=(11.5, 0.9 + 0.42 * (len(cell_text) + 1)))
    ax.axis("off")
    ax.set_title("DSV4-DSpark  —  our runs (w128, 77W) vs released baseline  (accept_len + per-position accept rate %)",
                 fontsize=13, fontweight="bold", pad=16)

    tbl = ax.table(cellText=cell_text, colLabels=col_labels, cellColours=cell_colors,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.55)

    ncol = len(col_labels)
    for c in range(ncol):                                    # header
        tbl[0, c].set_text_props(fontweight="bold", color="white")
        tbl[0, c].set_facecolor("#33415a")
    for ri in range(len(cell_text)):                         # dataset name bold; run label italic
        if cell_text[ri][0]:
            tbl[ri + 1, 0].set_text_props(fontweight="bold")
        if cell_text[ri][1] != "released":
            tbl[ri + 1, 1].set_text_props(fontstyle="italic")
    for ri in range(len(cell_text) - rows_per_group, len(cell_text)):   # average group bold
        for c in range(ncol):
            tbl[ri + 1, c].set_text_props(fontweight="bold")
    for r in range(len(cell_text) + 1):                      # thin borders + thick separator above each group
        for c in range(ncol):
            cell = tbl[r, c]
            cell.set_edgecolor("#c8d0da")
            thick = (r > 0 and (r - 1) % rows_per_group == 0)
            cell.set_linewidth(2.0 if thick else 0.5)

    fig.text(0.5, 0.03,
             "cell shading = our run as % of the released baseline in that cell (green→red).  "
             "per-position = cumulative accept rate S_k (%).  "
             "baseline = released draft, full DATASET=all, #12006 serve, num_spec=5, greedy.  "
             "⚠ ep2.5 on 176 A3-single serve (calibration pending).  "
             "Source: docs/deployment/ascend-npu-dsv4-dspark-eval-results.md",
             ha="center", fontsize=6.0, color="0.45")
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved: {args.out}")
    for run in runs:
        print(f"  {run['label']}: avg {run['al']['average']:.3f} = "
              f"{100 * run['al']['average'] / base_al['average']:.0f}% of released {base_al['average']:.3f}")


if __name__ == "__main__":
    main()
