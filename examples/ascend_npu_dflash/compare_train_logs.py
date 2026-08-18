#!/usr/bin/env python3
"""Join two DSV4-DSpark training logs on global_step and diff the metrics.

    python examples/ascend_npu_dflash/compare_train_logs.py \
        <RUN>/faithful_ep_<baseline-TS>.log <RUN>/faithful_ep_<new-TS>.log BASE CORR

Two runs are comparable step-for-step only when everything except the change
under test matches -- same data, block, LR, schedule, seed, world size. That
holds for the Correction-head run against `ep5p0-ropefix`: the launcher pins all
of those and only EXTRA_ARGS differs.

★ What the early steps are actually testing. Every part the Correction head and
the DFlash backbone hooks add is ZERO-INITIALISED (`tanh(0) == 0`, zeroed
`correction_up` and block-position embedding), so at step 0 the model should be
arithmetically the same model as the baseline. Early-step `train/loss` therefore
has to track the baseline closely; a visible gap in the first tens of steps means
some "zero init" is not zero, which is far cheaper to find here than to diagnose
from a diverged curve on day two.

⚠ `train/loss` is the TOTAL. This run adds `--correction-hidden-aux-loss`
(weight 0.1), so a small positive offset is expected and is not by itself a
problem. `train/accept_len`, `train/full_acc` and the per-position accuracies
carry no such extra term -- lean on those for the like-for-like read.
"""
import re
import sys

KEYS = [
    "train/loss", "train/confidence_loss", "train/accept_len", "train/full_acc",
    "train/position_0_acc", "train/position_1_acc", "train/position_2_acc",
    "train/position_3_acc", "train/position_4_acc",
    "profile/grad_norm", "profile/step_ms", "profile/fwd_ms", "profile/bwd_ms",
]


def parse(path: str) -> dict[int, dict[str, float]]:
    """Map global_step -> metrics. The logger wraps one step over many lines."""
    txt = open(path, errors="replace").read().replace("\n", " ")
    txt = re.sub(r"trainer\.py:\d+", " ", txt)          # drop the rich gutter
    parts = re.split(r"global_step=(\d+)", txt)          # [body, step, body, step, ...]
    bodies = [parts[i] for i in range(0, len(parts) - 1, 2)]
    steps = [int(parts[i]) for i in range(1, len(parts), 2)]
    rows = {}
    for step, body in zip(steps, bodies):
        row = {}
        for key in KEYS:
            m = re.search(re.escape(key) + r"=([-\d.e+]+)", body)
            if m:
                row[key] = float(m.group(1))
        if row:
            rows[step] = row
    return rows


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    a, b = parse(sys.argv[1]), parse(sys.argv[2])
    la = sys.argv[3] if len(sys.argv) > 3 else "A"
    lb = sys.argv[4] if len(sys.argv) > 4 else "B"
    common = sorted(set(a) & set(b))
    if not common:
        span = lambda d: f"{min(d)}..{max(d)}" if d else "<none>"  # noqa: E731
        print(f"no overlapping steps  ({la}: {span(a)}, {lb}: {span(b)})")
        return 1

    print(f"{len(common)} overlapping steps: {common[0]}..{common[-1]}\n")
    for key in KEYS:
        xs = [(s, a[s][key], b[s][key]) for s in common if key in a[s] and key in b[s]]
        if not xs:
            continue
        worst = max(xs, key=lambda t: abs(t[2] - t[1]))
        ma = sum(x for _, x, _ in xs) / len(xs)
        mb = sum(y for _, _, y in xs) / len(xs)
        print(f"{key:<26} {la}={ma:>10.4f}  {lb}={mb:>10.4f}  "
              f"Δmean={mb - ma:>+10.4f}  Δmax={worst[2] - worst[1]:>+10.4f} @{worst[0]}")

    print(f"\nfirst 12 common steps — train/loss (zero-init check):")
    print(f"  {'step':>6} {la:>11} {lb:>11} {'Δ':>11}")
    for s in common[:12]:
        if "train/loss" in a[s] and "train/loss" in b[s]:
            x, y = a[s]["train/loss"], b[s]["train/loss"]
            print(f"  {s:>6} {x:>11.4f} {y:>11.4f} {y - x:>+11.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
