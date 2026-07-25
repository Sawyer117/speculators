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

# ── OUR trained runs (non-causal 77W). The report comparison: the DOUBLE-norm teacher (accidental
#    sharpening — beat single-norm early but aims 16% off the real verifier = an optimization, "复现之外")
#    vs the SINGLE-norm "f1" teacher (reproduction-faithful = the real verifier's distribution, same target
#    as released). Story: at 1.0ep dnorm>f1 (the surprise regression), but single-norm f1 SCALES and its
#    1.5ep already passes the dnorm 1.0ep. ✏️ APPEND a dict when a new ckpt is eval'd (sync with the ledger).
#    (Older CAUSAL rows removed — broken attention task; archived in the ledger.)
RUNS = [
    {
        "label": "dnorm ep0mid (0.5ep)",
        "al": {"gsm8k": 4.032, "math500": 3.721, "humaneval": 3.856, "mbpp": 3.586, "mt-bench": 2.482},
        "pos": {
            "gsm8k":     [88.54, 74.73, 58.88, 46.05, 35.00],
            "math500":   [84.57, 69.23, 52.52, 38.75, 27.05],
            "humaneval": [88.12, 74.83, 55.05, 40.23, 27.43],
            "mbpp":      [85.29, 67.76, 48.11, 34.06, 23.40],
            "mt-bench":  [67.41, 39.89, 21.83, 12.21, 6.86],
        },
    },
    {
        "label": "dnorm ep0end (1.0ep)",
        "al": {"gsm8k": 4.006, "math500": 3.753, "humaneval": 3.970, "mbpp": 3.666, "mt-bench": 2.539},
        "pos": {
            "gsm8k":     [89.53, 77.05, 56.63, 43.46, 33.90],
            "math500":   [86.74, 72.47, 51.73, 37.14, 27.17],
            "humaneval": [92.30, 80.42, 55.33, 39.48, 29.49],
            "mbpp":      [86.96, 70.96, 48.55, 34.99, 25.18],
            "mt-bench":  [68.51, 42.41, 22.52, 12.91, 7.55],
        },
    },
    {
        "label": "f1 ep0end (1.0ep)",
        "al": {"gsm8k": 3.811, "math500": 3.511, "humaneval": 3.680, "mbpp": 3.407, "mt-bench": 2.460},
        "pos": {
            "gsm8k":     [89.20, 76.24, 51.46, 36.13, 28.11],
            "math500":   [86.38, 71.11, 43.67, 28.84, 21.07],
            "humaneval": [92.70, 80.53, 45.47, 29.28, 20.07],
            "mbpp":      [86.98, 69.49, 40.11, 26.07, 18.09],
            "mt-bench":  [68.97, 41.96, 19.46, 9.97, 5.62],
        },
    },
    {
        "label": "f1 ep1mid (1.5ep)",
        "al": {"gsm8k": 4.069, "math500": 3.757, "humaneval": 4.100, "mbpp": 3.674, "mt-bench": 2.549},
        "pos": {
            "gsm8k":     [90.54, 77.07, 59.39, 44.48, 35.42],
            "math500":   [87.26, 70.84, 53.46, 36.97, 27.19],
            "humaneval": [93.86, 81.49, 59.83, 42.43, 32.37],
            "mbpp":      [88.24, 71.39, 49.57, 33.86, 24.38],
            "mt-bench":  [70.35, 42.78, 22.64, 12.14, 7.03],
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
    ax.set_title("DSV4-DSpark 77W  —  double-norm vs single-norm(f1) teacher, epoch curve vs released  (accept_len + per-position accept rate %)",
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
             "f1 = single-norm teacher (reproduction-faithful); ep0end/ep1mid = 1.0/1.5 epoch on 176 A3-single serve.  "
             "Source: docs/deployment/ascend-npu-dsv4-dspark-eval-results.md",
             ha="center", fontsize=6.0, color="0.45")
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved: {args.out}")
    for run in runs:
        print(f"  {run['label']}: avg {run['al']['average']:.3f} = "
              f"{100 * run['al']['average'] / base_al['average']:.0f}% of released {base_al['average']:.3f}")


if __name__ == "__main__":
    main()
