#!/usr/bin/env python3
"""Static "our best vs released baseline" scoreboard TABLE (accept_len + per-position), to a PNG.

Unlike analyze_train_run.py (which plots a LIVE training log), this figure reads NO log — it renders
hardcoded result rows so the image is STABLE:
  * BASELINE never changes (the released draft, measured once).
  * BEST only moves when a new run beats it and we EDIT the BEST* blocks below.
So it does not wiggle per-log; it's the fixed scoreboard. Numbers mirror the per-dataset accept_len
AND per-position accept rate in docs/deployment/ascend-npu-dsv4-dspark-eval-results.md — keep in sync.

Layout: one row per (dataset, run) — Released then Our best — with columns accept_len + pos0..pos4.
Every "Our best" cell is shaded RdYlGn by its % of the baseline cell, so the tail collapse
(pos3/pos4) reds out at a glance while pos0–2 stay green.

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

# ── our current BEST trained draft. ✏️ UPDATE THESE THREE (label + BEST + BEST_POS) ONLY when a new
#    run beats it — that is the only thing that should ever move this scoreboard.
BEST_LABEL = "epoch4-17w (2026-07-19)"
BEST = {"gsm8k": 3.404, "math500": 3.265, "humaneval": 3.312, "mbpp": 3.058, "mt-bench": 2.344}
BEST_POS = {
    "gsm8k":     [90.44, 71.96, 53.87, 19.18, 4.95],
    "math500":   [87.82, 66.70, 50.16, 17.39, 4.43],
    "humaneval": [92.75, 72.15, 49.56, 13.70, 3.00],
    "mbpp":      [87.30, 62.25, 41.31, 12.36, 2.63],
    "mt-bench":  [69.45, 38.90, 19.86, 5.28, 0.91],
}


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

    # assemble per-row data incl. the computed average row
    base_al = dict(BASELINE, average=_avg(BASELINE))
    best_al = dict(BEST, average=_avg(BEST))
    base_pos = dict(BASELINE_POS, average=_avg_pos(BASELINE_POS))
    best_pos = dict(BEST_POS, average=_avg_pos(BEST_POS))
    rows = DATASETS + ["average"]

    cmap = plt.get_cmap("RdYlGn")
    norm = colors.Normalize(vmin=0, vmax=100)   # shade best cells 0–100% of baseline

    col_labels = ["Dataset", "Run", "accept_len", "pos0", "pos1", "pos2", "pos3", "pos4"]
    cell_text, cell_colors = [], []
    NEUT = "#eef2f7"      # released (baseline) row cells
    for d in rows:
        pct_al = 100.0 * best_al[d] / base_al[d]
        # released row (fixed reference)
        cell_text.append([d, "released", f"{base_al[d]:.3f}"] + [f"{v:.1f}" for v in base_pos[d]])
        cell_colors.append([NEUT] * len(col_labels))
        # our-best row — accept_len shows the % inline; every metric cell shaded by its own %
        best_cells = [f"{best_al[d]:.3f}  ({pct_al:.0f}%)"] + [f"{v:.1f}" for v in best_pos[d]]
        shaded = [colors.to_hex(cmap(norm(pct_al)))]
        for i in range(5):
            p = 100.0 * best_pos[d][i] / base_pos[d][i] if base_pos[d][i] else 0.0
            shaded.append(colors.to_hex(cmap(norm(p))))
        cell_text.append(["", "our best"] + best_cells)
        cell_colors.append(["#ffffff", "#ffffff"] + shaded)

    fig, ax = plt.subplots(figsize=(11.0, 0.9 + 0.42 * (len(cell_text) + 1)))
    ax.axis("off")
    ax.set_title("DSV4-DSpark  —  our best vs released baseline  (accept_len + per-position accept rate %)",
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
    for ri in range(len(cell_text)):                         # dataset name + best-run cells bold
        tr = tbl[ri + 1, 0]
        if cell_text[ri][0]:
            tr.set_text_props(fontweight="bold")
        if cell_text[ri][1] == "our best":
            tbl[ri + 1, 1].set_text_props(fontstyle="italic")
    for c in range(ncol):                                    # average group (last 2 rows) bold
        tbl[len(cell_text) - 1, c].set_text_props(fontweight="bold")
        tbl[len(cell_text), c].set_text_props(fontweight="bold")
    for r in range(len(cell_text) + 1):                      # thin borders + group separators
        for c in range(ncol):
            cell = tbl[r, c]
            cell.set_edgecolor("#c8d0da")
            cell.set_linewidth(2.0 if (r > 0 and r % 2 == 1) else 0.5)  # thicker line above each dataset

    fig.text(0.5, 0.03,
             "cell shading = Our best as % of the released baseline in that cell (green→red).  "
             "per-position = cumulative accept rate S_k (%).  "
             "baseline = released draft, full DATASET=all, #12006 serve, num_spec=5, greedy.  "
             "Source: docs/deployment/ascend-npu-dsv4-dspark-eval-results.md",
             ha="center", fontsize=6.5, color="0.45")
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved: {args.out}")
    print(f"  best avg {best_al['average']:.3f} = {100 * best_al['average'] / base_al['average']:.0f}% "
          f"of released {base_al['average']:.3f};  "
          f"avg per-pos best {[round(v) for v in best_pos['average']]} vs base "
          f"{[round(v) for v in base_pos['average']]}")


if __name__ == "__main__":
    main()
