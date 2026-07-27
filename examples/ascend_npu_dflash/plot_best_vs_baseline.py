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

# ── OUR trained runs — FAITHFUL ONLY (non-causal, SINGLE-norm "f1" teacher = the real verifier's
#    distribution, same target as released). The earlier NON-aligned experiments are DELIBERATELY EXCLUDED:
#    the DOUBLE-norm (`dnorm`) teacher rows (an accidental ~w² sharpening that aims ~16% off the real
#    verifier — an optimization, not a reproduction) and the older CAUSAL rows (broken attention task).
#    Both are archived in the ledger, kept OUT of this scoreboard on purpose.
#    Two groups, shown AS-IS (over-train collapse included — reported faithfully, not hidden):
#      (A) f1 no-balance REPRODUCTION curve: 1.0→1.5(PEAK)→2.0→2.5→3.0ep — peaks at 1.5ep (3.63/82%)
#          then over-trains under LR 3e-4 down to 3.0ep (3.01/68%). 3.0ep = accept_len only (its full
#          per-position wasn't recorded → pos cells render "—"; no fabrication).
#      (B) f1 + noaux_tc load balance (DSPARK_MOE_BALANCE=1 @ 5e-3): 0.5ep, 1.0ep.
#    ✏️ APPEND a dict when a new ckpt is eval'd. Source: docs/deployment/ascend-npu-dsv4-dspark-eval-results.md
RUNS = [
    # ── (A) f1 single-norm reproduction curve (peak @1.5ep, then over-train decline) ──
    {
        "label": "f1 1.0ep",
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
        "label": "f1 1.5ep ★peak",
        "al": {"gsm8k": 4.069, "math500": 3.757, "humaneval": 4.100, "mbpp": 3.674, "mt-bench": 2.549},
        "pos": {
            "gsm8k":     [90.54, 77.07, 59.39, 44.48, 35.42],
            "math500":   [87.26, 70.84, 53.46, 36.97, 27.19],
            "humaneval": [93.86, 81.49, 59.83, 42.43, 32.37],
            "mbpp":      [88.24, 71.39, 49.57, 33.86, 24.38],
            "mt-bench":  [70.35, 42.78, 22.64, 12.14, 7.03],
        },
    },
    {
        # 2.0ep: first down-tick (−0.067 mean vs 1.5ep peak).
        "label": "f1 2.0ep",
        "al": {"gsm8k": 4.059, "math500": 3.641, "humaneval": 4.007, "mbpp": 3.594, "mt-bench": 2.513},
        "pos": {
            "gsm8k":     [89.42, 74.11, 58.83, 46.69, 36.81],
            "math500":   [85.18, 65.62, 50.43, 36.27, 26.57],
            "humaneval": [91.41, 77.46, 57.19, 42.77, 31.90],
            "mbpp":      [85.83, 66.77, 47.71, 34.50, 24.61],
            "mt-bench":  [68.55, 40.23, 22.42, 12.71, 7.44],
        },
    },
    {
        # 2.5ep: regression confirmed (−0.35 vs 2.0ep). Over-train under LR 3e-4 (single-norm).
        "label": "f1 2.5ep",
        "al": {"gsm8k": 3.703, "math500": 3.293, "humaneval": 3.552, "mbpp": 3.200, "mt-bench": 2.320},
        "pos": {
            "gsm8k":     [83.72, 65.15, 51.10, 39.59, 30.75],
            "math500":   [78.56, 56.57, 42.68, 30.03, 21.50],
            "humaneval": [82.94, 64.64, 48.17, 34.17, 25.25],
            "mbpp":      [78.14, 55.34, 39.40, 27.93, 19.18],
            "mt-bench":  [62.13, 33.76, 18.90, 10.94, 6.26],
        },
    },
    {
        # 3.0ep: deepest point, decline monotonic since 1.5ep peak (3.63→3.01, 68% of released). Shown
        # faithfully. accept_len only — full per-position not recorded in the ledger (pos renders "—").
        "label": "f1 3.0ep ↓collapse",
        "al": {"gsm8k": 3.493, "math500": 3.065, "humaneval": 3.334, "mbpp": 2.956, "mt-bench": 2.198},
        "pos": None,
    },
    # ── (B) f1 + noaux_tc load balance (DSPARK_MOE_BALANCE=1 @ 5e-3) ──
    {
        # 0.5ep already > no-balance f1 1.0ep (3.37), near its 1.5ep peak — clean +balance A/B.
        "label": "f1+bal 0.5ep",
        "al": {"gsm8k": 3.998, "math500": 3.700, "humaneval": 3.885, "mbpp": 3.595, "mt-bench": 2.457},
        "pos": {
            "gsm8k":     [87.94, 73.42, 58.15, 45.43, 34.86],
            "math500":   [84.29, 68.01, 52.00, 38.47, 27.25],
            "humaneval": [88.90, 74.89, 55.58, 40.42, 28.67],
            "mbpp":      [84.96, 66.99, 48.69, 34.92, 23.96],
            "mt-bench":  [66.17, 39.06, 21.45, 12.05, 6.96],
        },
    },
    {
        # 1.0ep balance: gsm8k still climbing (3.998→4.087); mean flat ~3.52 (harder sets drag). The
        # definitive "does balance avoid the no-balance 1.5→3.0ep collapse" test — watch 1.5ep next.
        "label": "f1+bal 1.0ep",
        "al": {"gsm8k": 4.087, "math500": 3.638, "humaneval": 3.846, "mbpp": 3.530, "mt-bench": 2.488},
        "pos": {
            "gsm8k":     [89.25, 76.00, 60.24, 46.72, 36.48],
            "math500":   [85.50, 69.27, 49.81, 34.34, 24.84],
            "humaneval": [90.38, 76.40, 53.36, 37.26, 27.24],
            "mbpp":      [86.40, 68.45, 45.43, 30.85, 21.87],
            "mt-bench":  [68.61, 41.15, 21.25, 11.45, 6.34],
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
    ap.add_argument("--group", choices=["all", "nobalance", "balance"], default="all",
                    help="which runs to render: all (default) | nobalance (f1 reproduction curve only) | "
                         "balance (f1+bal only). Run twice for two separate figures.")
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
    # --group filter: a run is "balance" iff its label carries "bal" (f1+bal…); else "nobalance".
    def _is_bal(r):
        return "bal" in r["label"]
    sel = [r for r in RUNS
           if args.group == "all" or (_is_bal(r) if args.group == "balance" else not _is_bal(r))]
    runs = [{"label": r["label"],
             "al": dict(r["al"], average=_avg(r["al"])),
             "pos": (dict(r["pos"], average=_avg_pos(r["pos"])) if r.get("pos") else None)}
            for r in sel]
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
        # one row per run — accept_len shows the % inline; every metric cell shaded by its own %.
        # A run with pos=None (per-position not recorded) shows accept_len only; pos cells render "—".
        for run in runs:
            pct_al = 100.0 * run["al"][d] / base_al[d] if base_al[d] else 0.0
            if run["pos"] is None:
                cells = [f"{run['al'][d]:.3f}  ({pct_al:.0f}%)"] + ["—"] * 5
                shaded = [colors.to_hex(cmap(norm(pct_al)))] + ["#f4f4f4"] * 5
            else:
                cells = [f"{run['al'][d]:.3f}  ({pct_al:.0f}%)"] + [f"{v:.1f}" for v in run["pos"][d]]
                shaded = [colors.to_hex(cmap(norm(pct_al)))]
                for i in range(5):
                    p = 100.0 * run["pos"][d][i] / base_pos[d][i] if base_pos[d][i] else 0.0
                    shaded.append(colors.to_hex(cmap(norm(p))))
            cell_text.append(["", run["label"]] + cells)
            cell_colors.append(["#ffffff", "#ffffff"] + shaded)

    fig, ax = plt.subplots(figsize=(11.5, 0.9 + 0.42 * (len(cell_text) + 1)))
    ax.axis("off")
    _gtag = {
        "all": "f1 reproduction curve (incl. over-train collapse) + noaux load-balance",
        "nobalance": "f1 single-norm reproduction curve (incl. over-train collapse)",
        "balance": "f1 + noaux load-balance (DSPARK_MOE_BALANCE @ 5e-3)",
    }[args.group]
    ax.set_title(f"DSV4-DSpark 77W  —  {_gtag}, vs released  (accept_len + per-position %)",
                 fontsize=12, fontweight="bold", pad=16)

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
             "FAITHFUL runs only — double-norm(dnorm) + causal experiments EXCLUDED (archived in ledger).  "
             "cell shading = our run as % of released in that cell (green→red).  per-position = cumulative accept rate S_k (%); \"—\" = not recorded.  "
             "baseline = released draft, full DATASET=all, #12006 serve, num_spec=5, greedy.  "
             "f1 = single-norm teacher (reproduction-faithful); curve peaks 1.5ep then over-trains to 3.0ep (shown as-is).  bal5e3 = +DSPARK_MOE_BALANCE.  176 A3-single serve.  "
             "Source: docs/deployment/ascend-npu-dsv4-dspark-eval-results.md",
             ha="center", fontsize=5.6, color="0.45")
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved: {args.out}")
    for run in runs:
        print(f"  {run['label']}: avg {run['al']['average']:.3f} = "
              f"{100 * run['al']['average'] / base_al['average']:.0f}% of released {base_al['average']:.3f}")


if __name__ == "__main__":
    main()
