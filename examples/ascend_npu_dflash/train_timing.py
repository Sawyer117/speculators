#!/usr/bin/env python3
"""Per-step time (s/step) + total/remaining wall-clock for a DFlash run.

Built for the nohup launch (output redirected to a file → non-TTY), where the
tqdm.rich progress bar does NOT render. Two timing sources, in priority order:

  1) TensorBoard events  (--logger tensorboard writes events.out.tfevents* with a
     per-scalar wall_time; sub-second, the accurate source). Default log dir is
     <repo>/logs/<run_name>/ — NOT $OUTPUT_DIR/logs (that holds the text .log).
  2) the text log's RichHandler "[HH:MM:SS]" stamps + global_step  (1-second
     granularity; fallback when no tfevents are found / tensorboard is unreadable).

Usage:
    python train_timing.py logs/                         # search for newest tfevents
    python train_timing.py logs/2026-06-26T.../events.out.tfevents.123
    python train_timing.py outputs/.../logs/train_4b_*.log   # text-log fallback
    python train_timing.py logs/ --total-steps 59000     # exact ETA (else estimated)
"""
# SPDX-License-Identifier: Apache-2.0
import argparse
import glob
import os
import re
import sys

_TS = re.compile(r"\[(\d{2}):(\d{2}):(\d{2})\]")
_STEP = re.compile(r"\bglobal_step=(\d+)")
_LR = re.compile(r"\blr=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def _hms(seconds: float) -> str:
    s = int(seconds)
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{sec:02d}s"


# ---------- source discovery ----------
def _find_tfevents(path: str):
    if os.path.isfile(path) and "tfevents" in os.path.basename(path):
        return path
    if os.path.isdir(path):
        cands = glob.glob(os.path.join(path, "**", "events.out.tfevents*"), recursive=True)
        if cands:
            return max(cands, key=os.path.getmtime)
    return None


def _read_tfevents(evfile: str):
    """Return (step_time_pairs, lr_pairs, tag, rundir) or None on any failure."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (  # noqa: PLC0415
            EventAccumulator,
        )
    except Exception:  # noqa: BLE001
        return None
    rundir = os.path.dirname(evfile) or "."
    try:
        ea = EventAccumulator(rundir, size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        if not tags:
            return None
        tag = "train/loss" if "train/loss" in tags else tags[0]
        pts = [(int(e.step), float(e.wall_time)) for e in ea.Scalars(tag)]
        lr_tag = next((t for t in ("train/lr", "lr", "train/learning_rate") if t in tags), None)
        lrs = [(int(e.step), float(e.value)) for e in ea.Scalars(lr_tag)] if lr_tag else []
        return pts, lrs, tag, rundir
    except Exception as e:  # noqa: BLE001
        print(f"[warn] tensorboard read failed ({type(e).__name__}); falling back to text log")
        return None


def _read_textlog(path: str):
    """Parse RichHandler [HH:MM:SS] stamps + global_step/lr. Returns (pairs, lr_pairs)."""
    pts, lrs = [], []
    cur_t = None
    prev = None
    day = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            tm = _TS.search(line)
            if tm:
                h, mi, s = (int(x) for x in tm.groups())
                secs = h * 3600 + mi * 60 + s
                if prev is not None and secs < prev - 5:
                    day += 1  # midnight wrap
                prev = secs
                cur_t = day * 86400 + secs
            sm = _STEP.search(line)
            if sm and cur_t is not None:
                st = int(sm.group(1))
                pts.append((st, float(cur_t)))
                lm = _LR.search(line)
                if lm:
                    lrs.append((st, float(lm.group(1))))
    return pts, lrs


# ---------- timing math ----------
def _estimate_total_steps(lrs):
    """Linear/cosine warmup peaks LR at end of warmup (default warmup=total//100),
    so total ≈ 100 × peak-LR step — valid only once PAST warmup. Returns int or None."""
    if not lrs:
        return None
    peak_step = max(lrs, key=lambda x: x[1])[0]
    last_step = lrs[-1][0]
    if 0 < peak_step < last_step:
        return peak_step * 100
    return None


def _report(pts, lrs, total_arg, source):
    pts = sorted(set(pts))
    if len(pts) < 2:
        print(f"[{source}] 还不够算时间(只有 {len(pts)} 个步点);等多跑几步再看")
        return
    rates = []  # s/step over consecutive logged points
    for (s0, t0), (s1, t1) in zip(pts, pts[1:]):
        ds, dt = s1 - s0, t1 - t0
        if ds > 0 and dt >= 0:
            rates.append(dt / ds)
    elapsed = pts[-1][1] - pts[0][1]
    nsteps = pts[-1][0] - pts[0][0]
    avg = (elapsed / nsteps) if nsteps > 0 else None
    rates.sort()
    med = rates[len(rates) // 2] if rates else avg

    print(f"来源       : {source}")
    print(f"进度       : global_step {pts[0][0]} → {pts[-1][0]}  ({nsteps} 步已记录)")
    line = f"单步时间   : {med:.2f} s/step (median)"
    if avg:
        line += f" | {avg:.2f} s/step (avg over {_hms(elapsed)}, 含卡顿/抽HS等待)"
    print(line)
    if avg:
        print(f"吞吐       : ~{3600/avg:.0f} steps/h")
    print(f"已用       : {_hms(elapsed)}")

    total = total_arg or _estimate_total_steps(lrs)
    src = "--total-steps" if total_arg else "LR峰值×100 粗估±10%"
    step_t = avg or med
    if total and step_t:
        done = pts[-1][0]
        remaining = max(0, total - done)
        print(f"总步数     : ~{total}  ({src})")
        print(f"剩余       : ~{_hms(remaining * step_t)}   ({remaining} 步)")
        print(f"预计总时长 : ~{_hms(total * step_t)}   (本 epoch)")
    else:
        print("总步数/ETA : 未知(LR 还没过 warmup 峰值)→ 用 steps_per_epoch.py 精确算出"
              "总步数再加 --total-steps N(见 examples/ascend_npu_dflash/steps_per_epoch.py)")


def main():
    ap = argparse.ArgumentParser(description="DFlash 训练单步时间 / 总用时读数。")
    ap.add_argument("path", nargs="+", help="tfevents 文件 / 含 tfevents 的目录 / 文本 .log")
    ap.add_argument("--total-steps", type=int, default=None, help="每 epoch 总步数(精确 ETA)")
    ap.add_argument("--text", action="store_true", help="强制走文本 log,不读 tensorboard")
    args = ap.parse_args()

    print("=" * 60)
    print("DFlash 训练用时")
    print("=" * 60)

    # 1) try tensorboard unless --text or the arg is clearly a .log
    if not args.text:
        for p in args.path:
            ev = _find_tfevents(p)
            if ev:
                got = _read_tfevents(ev)
                if got:
                    pts, lrs, tag, rundir = got
                    _report(pts, lrs, args.total_steps,
                             f"tensorboard wall_time [{tag}] @ {rundir}")
                    print("=" * 60)
                    return

    # 2) text-log fallback (newest .log among the args)
    logs = []
    for p in args.path:
        if os.path.isdir(p):
            logs += glob.glob(os.path.join(p, "**", "*.log"), recursive=True)
        else:
            logs += glob.glob(p) if glob.glob(p) else ([p] if os.path.exists(p) else [])
    logs = [x for x in sorted(set(logs)) if x.endswith(".log")]
    if not logs:
        sys.exit("没找到 tfevents,也没找到 .log 文本日志。给我 logs/ 目录或 train_4b_*.log")
    newest = max(logs, key=os.path.getmtime)
    pts, lrs = _read_textlog(newest)
    _report(pts, lrs, args.total_steps, f"文本时间戳 [HH:MM:SS] @ {newest}")
    print("=" * 60)


if __name__ == "__main__":
    main()
