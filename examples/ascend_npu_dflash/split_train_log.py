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
    python split_train_log.py <logfile> --out DIR --gzip --resume     # 追加,只读新增字节

--resume makes this incremental: it remembers the byte offset it stopped at in
``<out>/.split_state.json`` and APPENDS only what the log has grown by. Re-splitting a live
491 MB log from scratch every time you want a fresh plot costs minutes and rewrites files git
already has; with --resume a refresh costs the new bytes only. Safe on a log that is still
being written — the final, possibly half-flushed record is withheld and re-read next time.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
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
    # ⚠️ position_* is filled in at runtime, NOT listed here. It used to be hardcoded to
    # position_0..4 from the block-5 era, which silently DROPPED positions 5-14 on a
    # block-15 run -- the CSV looked complete and was missing two thirds of the per-position
    # data. Columns absent from FAMILIES are not written, and nothing warns.
    "accept":     ["train/accept_rate", "train/accept_len", "train/hard_accept_len",
                   "train/full_acc"],
    "confidence": ["train/confidence_abs_error", "train/confidence_pred_mean",
                   "train/confidence_cumprod_bias"],
    "timing":     ["profile/fetch_ms", "profile/fwd_ms", "profile/bwd_ms", "profile/opt_ms",
                   "profile/step_ms", "profile/tokens_per_s", "profile/fetch_frac",
                   "profile/align_ms", "profile/fetch_ms_max", "profile/grad_norm"],
    "sched":      ["lr", "epoch"],
    "ranks":      ["profile/fetch_ms_ranks"],
}


STATE = ".split_state.json"


def _read_state(out: str, logfile: str) -> dict | None:
    """Load the resume state, or None if there is none / it does not match this log.

    Mismatch cases that MUST fall back to a full rebuild rather than append:
      * different source file (the state is per-log, the dir is not)
      * the log is SHORTER than the offset — it was rotated, truncated or replaced, so the
        offset now points into unrelated bytes
      * a different --every / --gzip, which would make the appended rows inconsistent with
        the ones already written
    """
    path = os.path.join(out, STATE)
    try:
        with open(path) as fh:
            st = json.load(fh)
    except (OSError, ValueError):
        return None
    if st.get("src") != os.path.abspath(logfile):
        print(f"⚠️ {path} 记的是 {st.get('src')},不是本次的日志 —— 全量重建")
        return None
    if os.path.getsize(logfile) < st.get("offset", 0):
        print("⚠️ 日志比上次记录的还短(轮转/截断/换文件) —— 全量重建")
        return None
    return st


def short(col: str) -> str:
    return col.split("/", 1)[-1]


def records(fh, hold_last: bool = False):
    """Yield ``(record, offset)`` per wrapped log record; ``offset`` = byte position of the
    record's first byte. The LAST item is always ``(None, offset)`` — the byte to resume from.

    ``fh`` must be opened in BINARY mode. ``tell()`` is disabled inside a text-mode iteration
    (``OSError: telling position disabled by next() call``), and the offset is the whole point.

    ``hold_last=True`` withholds the final buffered record and reports ITS start offset as the
    resume point. A live log's last record is usually half-written — the logger wraps one step
    across ~26 physical lines — so resuming from its start re-reads it whole instead of
    splitting one step into two half-rows.
    """
    buf: list[str] = []
    pos = fh.tell()
    start = pos                       # byte offset of the record currently in `buf`
    for raw in fh:
        line = raw.decode("utf-8", "replace")
        nxt = pos + len(raw)
        if REC_START.match(line):
            if buf:
                yield "".join(buf), start
            buf = [SRC_COL.sub("", line.rstrip("\n")) + " "]
            start = pos
        elif buf:
            buf.append(SRC_COL.sub("", line.rstrip("\n")).strip() + " ")
        pos = nxt
    if buf and not hold_last:
        yield "".join(buf), start
        start = pos
    elif not buf:
        start = pos
    yield None, start


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile")
    ap.add_argument("--out", default=None, help="output dir (default: <logname>.split/)")
    ap.add_argument("--every", type=int, default=1, help="keep 1 step in N (default 1 = all)")
    ap.add_argument("--gzip", action="store_true", help="write .csv.gz")
    ap.add_argument("--resume", action="store_true",
                    help="增量:只读上次停下之后新增的字节,追加到已有 CSV(状态存 <out>/.split_state.json)")
    args = ap.parse_args()

    out = args.out or os.path.basename(args.logfile) + ".split"
    os.makedirs(out, exist_ok=True)

    state = _read_state(out, args.logfile) if args.resume else None
    if state and (state.get("every") != args.every or state.get("gzip") != args.gzip):
        print(f"⚠️ 上次是 --every {state.get('every')} --gzip {state.get('gzip')},本次不同 —— 全量重建")
        state = None
    append = state is not None

    # How many draft positions this run logs (block_size varies per run) — extend the accept
    # family to match. On --resume this comes from the STATE, not a rescan: the column set is
    # frozen by the header already written, and re-detecting could widen it mid-file and shift
    # every later row's columns silently.
    if append:
        n_pos = state["n_pos"]
    else:
        n_pos = 0
        with open(args.logfile, errors="ignore") as fh:
            for i, line in enumerate(fh):
                for m in re.finditer(r"train/position_(\d+)_acc", line):
                    n_pos = max(n_pos, int(m.group(1)) + 1)
                if n_pos and i > 5000:
                    break
    if n_pos:
        FAMILIES["accept"] += [f"train/position_{k}_acc" for k in range(n_pos)]
        src = "沿用上次" if append else "检测到"
        print(f"{src} {n_pos} 个草稿位置 -> accept.csv 写 position_0..{n_pos - 1}_acc")
    else:
        print("⚠️ 日志里没有 train/position_*_acc —— accept.csv 将不含逐位数据")

    mode = "at" if append else "wt"
    opener = (lambda p: gzip.open(p + ".gz", mode, newline="")) if args.gzip \
        else (lambda p: open(p, mode[0], newline=""))

    files, writers = {}, {}
    for fam, cols in FAMILIES.items():
        files[fam] = opener(os.path.join(out, f"{fam}.csv"))
        writers[fam] = csv.writer(files[fam])
        if not append:
            writers[fam].writerow(["step"] + [short(c) for c in cols])
    files["moe"] = opener(os.path.join(out, "moe_load.csv"))
    writers["moe"] = csv.writer(files["moe"])
    if not append:
        writers["moe"].writerow(["step", "layer", "used", "total", "dead", "top16", "entropy", "hot"])

    # `keep` tracks whether the LAST step survived --every, so the MoE rows that follow
    # it are subsampled with it. Without this they are always written and moe_load.csv
    # stays full size no matter what --every says. Both it and last_step are carried across
    # a --resume, because the offset can land between a step record and its MoE lines.
    last_step = state["last_step"] if append else ""
    keep = state["keep"] if append else True
    n_rec = n_moe = 0
    start = state["offset"] if append else 0
    end = start
    size = os.path.getsize(args.logfile)
    if append:
        print(f">>> 增量:从第 {start:,} 字节续读(日志现有 {size:,} 字节,新增 {size - start:,})")
    with open(args.logfile, "rb") as fh:
        fh.seek(start)
        for rec, off in records(fh, hold_last=True):
            if rec is None:                  # sentinel: byte to resume from next time
                end = off
                break
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

    total_rec = n_rec + (state["rows"] if append else 0)
    with open(os.path.join(out, STATE), "w") as fh:
        json.dump({"src": os.path.abspath(args.logfile), "offset": end, "n_pos": n_pos,
                   "rows": total_rec, "last_step": last_step, "keep": keep,
                   "every": args.every, "gzip": args.gzip}, fh, indent=1)

    verb = "新增" if append else "写入"
    tail = f"(累计 {total_rec:,})" if append else ""
    print(f"{verb} {n_rec} 个训练步 · {n_moe} 条 MoE 记录 {tail} -> {out}/   [last_step={last_step or '—'}]")
    total = 0
    for name in sorted(os.listdir(out)):
        if name == STATE:
            continue
        sz = os.path.getsize(os.path.join(out, name))
        total += sz
        print(f"  {sz:>10,}  {name}")
    print(f"  {total:>10,}  合计")
    return 0


if __name__ == "__main__":
    sys.exit(main())
