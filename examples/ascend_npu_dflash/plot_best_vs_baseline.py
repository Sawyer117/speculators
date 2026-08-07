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
    # ── (D) ★ RoPE-FIX line (real cos/sin rotary, feb0066) — the current best, near-parity with
    #    released. Same recipe as the degenerate bal1e3 line (only variable = RoPE). STILL CLIMBING:
    #    0.5ep 3.84 → 3.0ep 4.35 (98.4% of released); gsm8k 3.0ep 4.796 = 103.0% of released. Use
    #    `--group ropefix` for a clean "released vs current" figure. Source = the ropefix rows + the
    #    saved eval logs (eval_ep{0p5,1p5}_ropefix_all.txt). Rendered as the "current best" line
    #    (default `--group best`); the title says "current best", not "RoPE-fix".
    {
        "label": "current best 0.5ep",
        "al": {"gsm8k": 4.309, "math500": 4.068, "humaneval": 4.298, "mbpp": 3.908, "mt-bench": 2.627},
        "pos": {
            "gsm8k":     [90.41, 78.54, 65.59, 53.64, 42.76],
            "math500":   [87.65, 74.57, 60.74, 47.80, 36.04],
            "humaneval": [92.39, 80.65, 65.76, 51.79, 39.26],
            "mbpp":      [87.35, 72.01, 56.38, 43.25, 31.85],
            "mt-bench":  [68.95, 42.98, 25.90, 15.51, 9.33],
        },
    },
    {
        "label": "current best 1.0ep",
        "al": {"gsm8k": 4.493, "math500": 4.241, "humaneval": 4.585, "mbpp": 4.167, "mt-bench": 2.796},
        "pos": {
            "gsm8k":     [91.50, 80.80, 69.55, 58.71, 48.75],
            "math500":   [88.50, 76.52, 64.26, 52.71, 42.08],
            "humaneval": [93.94, 84.49, 71.47, 60.14, 48.51],
            "mbpp":      [89.04, 76.07, 62.09, 50.12, 39.37],
            "mt-bench":  [72.18, 47.01, 29.36, 18.87, 12.20],
        },
    },
    {
        "label": "current best 1.5ep",
        "al": {"gsm8k": 4.628, "math500": 4.408, "humaneval": 4.706, "mbpp": 4.300, "mt-bench": 2.865},
        "pos": {
            "gsm8k":     [92.23, 82.56, 72.36, 62.47, 53.14],
            "math500":   [89.96, 79.17, 67.92, 57.05, 46.66],
            "humaneval": [94.43, 85.35, 74.31, 64.33, 52.15],
            "mbpp":      [89.77, 77.90, 65.28, 53.79, 43.21],
            "mt-bench":  [73.22, 48.24, 31.21, 20.48, 13.38],
        },
    },
    {
        # 2.0ep: gsm8k 4.701 passes released 4.658; mean 4.25 = 96.3%. Climb decelerating.
        "label": "current best 2.0ep",
        "al": {"gsm8k": 4.701, "math500": 4.431, "humaneval": 4.745, "mbpp": 4.412, "mt-bench": 2.980},
        "pos": {
            "gsm8k":     [92.60, 83.67, 73.96, 64.58, 55.30],
            "math500":   [89.94, 79.35, 68.37, 57.77, 47.64],
            "humaneval": [93.97, 85.44, 74.84, 65.03, 55.25],
            "mbpp":      [90.57, 79.83, 67.68, 56.62, 46.50],
            "mt-bench":  [74.51, 51.09, 34.11, 22.94, 15.32],
        },
    },
    {
        # 2.5ep: mean 4.29 = 97.2%; gsm8k 4.753 = 102.0% of released. Non-chat 4 avg = 98.3%.
        # 3.0ep (below): mean 4.35 = 98.4%; non-chat 4 avg 4.671 = 99.4%; gsm8k conditional c0-c4 ALL
        # above released for the first time. Climb did NOT flatten: deltas +0.22/+0.12/+0.07/+0.04/+0.06.
        "label": "current best 2.5ep",
        "al": {"gsm8k": 4.753, "math500": 4.485, "humaneval": 4.818, "mbpp": 4.428, "mt-bench": 2.988},
        "pos": {
            "gsm8k":     [92.73, 84.16, 75.14, 66.08, 57.16],
            "math500":   [90.29, 79.79, 69.49, 59.41, 49.54],
            "humaneval": [94.35, 86.74, 76.65, 67.18, 56.85],
            "mbpp":      [90.91, 79.77, 67.90, 56.99, 47.19],
            "mt-bench":  [75.20, 51.49, 34.10, 22.70, 15.28],
        },
    },
    {
        "label": "current best 3.0ep",
        "al": {"gsm8k": 4.796, "math500": 4.519, "humaneval": 4.855, "mbpp": 4.512, "mt-bench": 3.046},
        "pos": {
            "gsm8k":     [93.28, 84.98, 75.96, 67.03, 58.34],
            "math500":   [90.56, 80.45, 70.20, 60.12, 50.62],
            "humaneval": [94.12, 86.92, 77.05, 68.29, 59.09],
            "mbpp":      [91.49, 81.06, 69.60, 59.30, 49.79],
            "mt-bench":  [75.66, 52.63, 35.47, 24.22, 16.66],
        },
    },
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
    {
        # 1.5ep balance: VERDICT — balance is NOT the lever. Curve 0.5→1.0→1.5ep = 3.53→3.52→3.47
        # (slow decline from the start, never climbed); at the SAME 1.5ep no-balance f1 PEAK 3.63 wins.
        # Both arms over-train under LR 3e-4 → decline is LR-driven, not collapse. Next lever = lower LR.
        "label": "f1+bal 1.5ep ↓",
        "al": {"gsm8k": 3.963, "math500": 3.675, "humaneval": 3.740, "mbpp": 3.482, "mt-bench": 2.475},
        "pos": {
            "gsm8k":     [89.48, 76.93, 53.68, 42.44, 33.77],
            "math500":   [86.99, 72.58, 47.56, 34.74, 25.64],
            "humaneval": [91.89, 79.56, 44.66, 33.09, 24.76],
            "mbpp":      [86.87, 69.87, 41.12, 29.33, 21.00],
            "mt-bench":  [68.76, 42.15, 19.31, 10.89, 6.41],
        },
    },
    # ── (C) current best-per-epoch balance line: fresh-router noaux_tc @ 1e-3, A2 DP8 / LR 2e-4 /
    #    anchor512, pre-dedup 77W (arrow_0720_77w). A DIFFERENT experiment than the (B) f1+bal @5e-3
    #    A3 A/B above — do NOT read (B)+(C) as one curve. accept_len + per-position are recovered
    #    from the saved eval-run stdout (the ledger table only kept the gsm8k tail; the full pos0-4
    #    was in the eval logs all along and is backfilled here).
    {
        # 0.5ep (epoch0-mid, /0 step 12388): BEST balance mean yet (3.56); gsm8k 4.050 top-of-class for
        # 0.5ep. NB the GLOBAL best across all runs is still no-balance f1 1.5ep (3.63, --group nobalance).
        "label": "bal1e3 0.5ep ★best-bal",
        "al": {"gsm8k": 4.050, "math500": 3.784, "humaneval": 3.890, "mbpp": 3.591, "mt-bench": 2.466},
        "pos": {  # recovered from the saved eval log (eval_ep0p5.txt)
            "gsm8k":     [88.80, 75.08, 59.78, 46.29, 35.02],
            "math500":   [85.20, 70.28, 54.44, 40.08, 28.44],
            "humaneval": [90.07, 75.90, 55.26, 40.21, 27.56],
            "mbpp":      [85.31, 67.22, 48.59, 34.35, 23.61],
            "mt-bench":  [66.31, 39.33, 21.83, 12.17, 6.96],
        },
    },
    {
        # 1.0ep (epoch1.0): FLAT vs its own 0.5ep (3.56 → 3.52) over one half-epoch — balance front-loads
        # then sits flat; the peak is likely still ahead at 1.5ep (the dirty run was killed at 1.0ep).
        "label": "bal1e3 1.0ep",
        "al": {"gsm8k": 4.017, "math500": 3.612, "humaneval": 3.938, "mbpp": 3.542, "mt-bench": 2.480},
        "pos": {  # recovered from the saved eval log (eval_ep1p0dedup.txt)
            "gsm8k":     [89.20, 76.14, 60.12, 43.65, 32.60],
            "math500":   [85.37, 68.76, 50.32, 33.39, 23.41],
            "humaneval": [90.89, 78.60, 57.45, 39.51, 27.40],
            "mbpp":      [85.91, 67.21, 47.54, 31.80, 21.77],
            "mt-bench":  [67.88, 40.52, 21.79, 11.27, 6.49],
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
    ap.add_argument("--group", choices=["all", "best", "nobalance", "balance"], default="best",
                    help="which runs to render: best (default — the current-best line vs released) | "
                         "all (full historical scoreboard) | nobalance (f1 reproduction curve) | "
                         "balance (f1+bal only). Run with different values for separate figures.")
    ap.add_argument("--latest", action="store_true",
                    help="render ONLY the last checkpoint of the selected group (released vs one row) "
                         "— the short summary figure for a report's first page.")
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
    # --group filter: ropefix (real-RoPE line) | balance (label carries "bal") | nobalance | all.
    def _grp(r):
        if "current best" in r["label"]:
            return "best"
        return "balance" if "bal" in r["label"] else "nobalance"
    sel = [r for r in RUNS if args.group == "all" or _grp(r) == args.group]
    if args.latest:
        # RUNS is authored oldest→newest within a group, so the last entry is the newest ckpt.
        sel = sel[-1:]
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
        "best": "current best",
        "all": "f1 reproduction curve (incl. over-train collapse) + noaux load-balance + current best",
        "nobalance": "f1 single-norm reproduction curve (incl. over-train collapse)",
        "balance": "f1 + noaux load-balance (DSPARK_MOE_BALANCE @ 5e-3)",
    }[args.group]
    if args.latest and runs:
        _gtag = f"{_gtag} ({runs[0]['label'].split(maxsplit=2)[-1]})"
    ax.set_title(f"DSV4-DSpark 77W  —  {_gtag}, vs released  (accept_len + per-position %)",
                 fontsize=12, fontweight="bold", pad=16)

    # Explicit column widths: the Run column holds the longest strings ("current best 2.5ep"),
    # and matplotlib's default equal widths clip it under the neighbouring accept_len cell.
    _w = [0.115, 0.185, 0.150] + [0.0925] * 5          # Dataset, Run, accept_len, pos0..pos4
    tbl = ax.table(cellText=cell_text, colLabels=col_labels, cellColours=cell_colors,
                   colWidths=_w, cellLoc="center", loc="center")
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

    # Footnote: --latest is the shareable summary figure, so it carries only what the figure needs
    # to be read. The full scoreboard keeps the provenance notes about which run groups are shown.
    if args.latest:
        _note = (
            "cell shading = our run as % of the released draft in that cell.  "
            "per-position = cumulative accept rate S_k (%).  "
            "baseline = released DSpark draft, full DATASET=all, num_spec=5, greedy, same serve."
        )
    else:
        _note = (
            "FAITHFUL runs only — double-norm(dnorm) + causal experiments EXCLUDED (archived in ledger).  "
            "cell shading = our run as % of released in that cell (green→red).  per-position = cumulative accept rate S_k (%); \"—\" = not recorded.  "
            "baseline = released draft, full DATASET=all, #12006 serve, num_spec=5, greedy.  "
            "f1 = single-norm teacher (reproduction-faithful); curve peaks 1.5ep then over-trains to 3.0ep (shown as-is).  bal5e3 = +DSPARK_MOE_BALANCE.  176 A3-single serve.  "
            "Source: docs/deployment/ascend-npu-dsv4-dspark-eval-results.md"
        )
    fig.text(0.5, 0.03, _note, ha="center", fontsize=6.4, color="0.45")
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved: {args.out}")
    for run in runs:
        print(f"  {run['label']}: avg {run['al']['average']:.3f} = "
              f"{100 * run['al']['average'] / base_al['average']:.0f}% of released {base_al['average']:.3f}")


if __name__ == "__main__":
    main()
