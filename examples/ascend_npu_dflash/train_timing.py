#!/usr/bin/env python3
"""Quick timing read-out for a DFlash training run: per-step time (s/step) and
total / remaining wall-clock. A focused companion to analyze_train_log.py (which
also charts loss / per-position acceptance).

Two independent sources, whichever is present:
  1) the tqdm progress bar   "N/TOT [elapsed<remaining, RATE]"   (most direct)
  2) [HH:MM:SS] line stamps + global_step  ->  median s/step      (cross-check)

Usage:
    python train_timing.py <logfile|glob|dir>          # newest match is used
    python train_timing.py outputs/.../logs/train_4b_*.log
    watch -n 30 'python train_timing.py outputs/.../logs/train_4b_*.log'   # live
"""
# SPDX-License-Identifier: Apache-2.0
import argparse
import glob
import os
import re
import sys

# a tqdm duration, e.g. "12:34", "1:23:45", or ">24h" as "1 day, 9:23:45"
_DUR = r"(?:\d+\s*day[s]?,\s*)?\d{1,3}:\d{2}(?::\d{2})?"
# full tqdm bar:  "  9000/76319 [12:34<1:23:45,  1.85s/it]"  (rate is it/s OR s/it)
_TQDM = re.compile(
    r"(\d+)\s*/\s*(\d+)\s*\[(" + _DUR + r")<(" + _DUR + r"),\s*([\d.]+)\s*(it/s|s/it)\]"
)
_TS = re.compile(r"\[(\d{2}):(\d{2}):(\d{2})\]")
_STEP = re.compile(r"global_step=(\d+)")


def _hms(seconds: float) -> str:
    s = int(seconds)
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{sec:02d}s"


def _to_secs(s: str) -> int:
    """Parse a tqdm duration ('MM:SS', 'H:MM:SS', or '1 day, H:MM:SS') to seconds."""
    total = 0
    dm = re.match(r"\s*(\d+)\s*day[s]?,\s*(.*)", s)
    if dm:
        total += int(dm.group(1)) * 86400
        s = dm.group(2)
    parts = [int(x) for x in s.strip().split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts
    return total + h * 3600 + m * 60 + sec


def _resolve(args_log):
    cands = []
    for a in args_log:
        if os.path.isdir(a):
            cands += glob.glob(os.path.join(a, "*.log"))
        else:
            g = glob.glob(a)
            cands += g if g else ([a] if os.path.exists(a) else [])
    cands = sorted(set(cands))
    if not cands:
        sys.exit(f"No log file matches: {' '.join(args_log)}")
    newest = max(cands, key=os.path.getmtime)
    if len(cands) > 1:
        print(f"[info] {len(cands)} logs matched; using newest: {newest}")
    return newest


def main():
    ap = argparse.ArgumentParser(description="DFlash training timing read-out.")
    ap.add_argument("log", nargs="+", help="log file(s), glob, or dir; newest is used")
    args = ap.parse_args()
    path = _resolve(args.log)
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    print("=" * 60)
    print(f"DFlash 训练用时  ::  {path}")
    print("=" * 60)

    # ---- source 1: last tqdm bar (most direct) ----
    last = None
    for m in _TQDM.finditer(text):
        last = m
    if last:
        done, total = int(last.group(1)), int(last.group(2))
        el, rem = _to_secs(last.group(3)), _to_secs(last.group(4))
        rate, unit = float(last.group(5)), last.group(6)
        sec_per_step = (1.0 / rate) if unit == "it/s" else rate
        pct = (100 * done / total) if total else 0.0
        print(f"进度       : {done:,} / {total:,} steps  ({pct:.1f}%)")
        print(f"单步时间   : {sec_per_step:.2f} s/step    ({rate:g} {unit})")
        print(f"已用       : {_hms(el)}")
        print(f"剩余       : ~{_hms(rem)}")
        print(f"预计总时长 : ~{_hms(el + rem)}   (本 epoch)")
    else:
        print("进度       : 日志里还没有 tqdm 进度条(等几步再看)")

    # ---- source 2: [HH:MM:SS] timestamps + global_step (cross-check) ----
    steps_t = []
    cur_t = None
    prev = None
    day = 0
    for line in text.splitlines():
        tm = _TS.search(line)
        if tm:
            h, mi, s = (int(x) for x in tm.groups())
            secs = h * 3600 + mi * 60 + s
            if prev is not None and secs < prev - 5:
                day += 1  # wrapped past midnight
            prev = secs
            cur_t = day * 86400 + secs
        sm = _STEP.search(line)
        if sm and cur_t is not None:
            steps_t.append((int(sm.group(1)), cur_t))

    print("-" * 60)
    if len(steps_t) >= 2:
        rates = []  # sec/step over consecutive logged steps
        for (s0, t0), (s1, t1) in zip(steps_t, steps_t[1:]):
            ds, dt = s1 - s0, t1 - t0
            if ds > 0 and dt > 0:
                rates.append(dt / ds)
        if rates:
            rates.sort()
            med = rates[len(rates) // 2]
            elapsed = steps_t[-1][1] - steps_t[0][1]
            nsteps = steps_t[-1][0] - steps_t[0][0]
            avg = (elapsed / nsteps) if nsteps > 0 else None
            line2 = f"交叉核对   : {med:.2f} s/step (median, {len(rates)} 个步差)"
            # avg only when sane: avg < median/2 means a step got paired with a stale
            # timestamp (deflated elapsed) — suppress rather than print a bogus number.
            if avg and avg >= med * 0.5:
                line2 += f" | {avg:.2f} s/step (avg over {_hms(elapsed)}, 含卡顿)"
            print(line2)
        else:
            print("交叉核对   : 时间戳有,但步差不足以估算")
    else:
        print("交叉核对   : 日志无 [HH:MM:SS] 时间戳 → 以上面 tqdm 进度条为准")
    print("=" * 60)


if __name__ == "__main__":
    main()
