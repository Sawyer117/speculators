#!/usr/bin/env python3
"""Split a DSV4-DSpark training log into one small CSV per metric family.

WHY. A run log is ~253 MB (124k steps). Compressed it is still ~11 MB, which a gateway
that caps a single request at ~100 KB will not take, and which nobody wants in git
history anyway. But the log is ~100% metrics: split by family, each file is small,
self-describing, plottable on its own, and needs no reassembly.

⚠ Do NOT try to distill by grepping one key. The logger WRAPS one step's record across
~26 physical lines; `grep global_step=` keeps the single line carrying it and silently
drops train/loss, accept_len, step_ms and the rest (measured: 124,480 -> 0). This parser
reassembles the wrapped record first, which is the whole point of it existing.

Record shape:
    [23:39:14] INFO     train/confidence_loss=0.299,          trainer.py:592
                        train/loss=0.870, train/ce_loss=1.344,
                        ...
                        lr=2.00e-04, global_step=7635
plus single-line MoE counters:
    [MOE-LOAD L0] used=158/256 dead=98 top16=0.77 entropy=0.601  hot=[80, 52, ...]

USAGE
    python split_train_log.py <logfile> [--out DIR] [--every N] [--gzip]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import re
import sys

# A record starts at a timestamp OR at a [MOE-LOAD Lx] line. Leaving MOE out of this
# is a silent data-loss bug: the MOE line does not match, so it gets APPENDED to the
# preceding step's record, and that step's metrics are then written as a MoE row and
# lost. Measured when it was wrong: 118,257 steps recovered out of 124,480.
REC_START = re.compile(r"^\[(?:\d\d:\d\d:\d\d\]|MOE-LOAD )")
SRC_COL = re.compile(r"\s+[A-Za-z_]+\.py:\d+\s*$")     # rich's right-hand source column
PAIR = re.compile(r"([A-Za-z_][\w/]*)=(\[[^\]]*\]|[^,\s]+)")
MOE = re.compile(
    r"\[MOE-LOAD L(\d+)\]\s+used=(\d+)/(\d+)\s+dead=(\d+)\s+top16=([\d.]+)\s+"
    r"entropy=([\d.]+)\s+hot=\[([^\]]*)\]"
)

# family -> columns, in the order they should appear. A key absent from a record is "".
FAMILIES: dict[str, list[str]] = {
    "loss":       ["train/loss", "train/ce_loss", "train/tv_loss", "train/confidence_loss"],
    "accept":     ["train/accept_rate", "train/accept_len", "train/hard_accept_len",
                   "train/full_acc", "train/position_0_acc", "train/position_1_acc",
                   "train/position_2_acc", "train/position_3_acc", "train/position_4_acc"],
    "confidence": ["train/confidence_abs_error", "train/confidence_pred_mean",
                   "train/confidence_cumprod_bias"],
    "timing":     ["profile/fetch_ms", "profile/fwd_ms", "profile/bwd_ms", "profile/opt_ms",
                   "profile/step_ms", "profile/tokens_per_s", "profile/fetch_frac",
                   "profile/align_ms", "profile/fetch_ms_max", "profile/grad_norm"],
    "sched":      ["lr", "epoch"],
    "ranks":      ["profile/fetch_ms_ranks"],
}


def short(col: str) -> str:
    return col.split("/", 1)[-1]


def records(fh):
    """Yield one joined record string per wrapped log record."""
    buf: list[str] = []
    for line in fh:
        if REC_START.match(line):
            if buf:
                yield "".join(buf)
            buf = [SRC_COL.sub("", line.rstrip("\n")) + " "]
        elif buf:
            buf.append(SRC_COL.sub("", line.rstrip("\n")).strip() + " ")
    if buf:
        yield "".join(buf)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile")
    ap.add_argument("--out", default=None, help="output dir (default: <logname>.split/)")
    ap.add_argument("--every", type=int, default=1, help="keep 1 step in N (default 1 = all)")
    ap.add_argument("--gzip", action="store_true", help="write .csv.gz")
    args = ap.parse_args()

    out = args.out or os.path.basename(args.logfile) + ".split"
    os.makedirs(out, exist_ok=True)
    opener = (lambda p: gzip.open(p + ".gz", "wt", newline="")) if args.gzip \
        else (lambda p: open(p, "w", newline=""))

    files, writers = {}, {}
    for fam, cols in FAMILIES.items():
        files[fam] = opener(os.path.join(out, f"{fam}.csv"))
        writers[fam] = csv.writer(files[fam])
        writers[fam].writerow(["step"] + [short(c) for c in cols])
    files["moe"] = opener(os.path.join(out, "moe_load.csv"))
    writers["moe"] = csv.writer(files["moe"])
    writers["moe"].writerow(["step", "layer", "used", "total", "dead", "top16", "entropy", "hot"])

    # `keep` tracks whether the LAST step survived --every, so the MoE rows that follow
    # it are subsampled with it. Without this they are always written and moe_load.csv
    # stays full size no matter what --every says.
    last_step, n_rec, n_moe, keep = "", 0, 0, True
    with open(args.logfile, errors="replace") as fh:
        for rec in records(fh):
            hits = MOE.findall(rec)          # one line per layer (L0/L1/L2)
            if hits:
                if not keep:
                    continue
                for g in hits:
                    n_moe += 1
                    writers["moe"].writerow([last_step, *g[:6], g[6].replace(" ", "")])
                continue
            kv = dict(PAIR.findall(rec))
            step = kv.get("global_step")
            if step is None:
                continue
            keep = not (args.every > 1 and int(step) % args.every)
            last_step = step
            if not keep:
                continue
            n_rec += 1
            for fam, cols in FAMILIES.items():
                row = [kv.get(c, "") for c in cols]
                if any(row):
                    writers[fam].writerow([step] + [v.replace(" ", "") for v in row])
    for f in files.values():
        f.close()

    print(f"{n_rec} 个训练步 · {n_moe} 条 MoE 记录  ->  {out}/")
    total = 0
    for name in sorted(os.listdir(out)):
        sz = os.path.getsize(os.path.join(out, name))
        total += sz
        print(f"  {sz:>10,}  {name}")
    print(f"  {total:>10,}  合计")
    return 0


if __name__ == "__main__":
    sys.exit(main())
