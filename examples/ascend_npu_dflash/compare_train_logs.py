#!/usr/bin/env python3
"""Compare DFlash training runs across configs and print a total-time table.

Built to answer "how much faster is 2-serve/6-train (DP2) than 1-serve/7-train
(DP1)?" from the logs already on disk. For EVERY train log it computes the real
throughput (avg it/s INCLUDING stalls, plus tokens/s for a fair cross-config
comparison); when the run was produced by the profiling build it also reports the
per-component breakdown (data_wait / h2d / forward / backward / optim) and the
dominant bottleneck.

Why it works on OLD logs too: it/s and tokens/s only need the `[HH:MM:SS]`
timestamps + `global_step=` that every run logs, so a pre-instrumentation DP1 log
still slots into the comparison — you do NOT need to re-run DP1 just to compare
total time. (You'd only re-run DP1 on the profiling branch if you specifically
want its data_wait breakdown, which old logs don't contain.)

Runs are grouped by train-card count `nproc`, read from the ">>> RUN train ..."
header the nohup script writes as the log's first line. serve cards =
--total-cards - nproc, so nproc=7 -> 1 serve ("DP1"), nproc=6 -> 2 serve ("DP2").

tokens/s = it/s * train_cards * --seq-len (each multipack step feeds ~seq_len
tokens per replica). Since the dataset is fixed, the tokens/s RATIO equals the
per-epoch wall-clock SPEEDUP. For absolute hours/epoch pass --packed-per-epoch C,
where C = steps_per_epoch * train_cards is dataset-constant (open_perfectblend
full ~= 241962, i.e. 34566 steps * 7 cards).

Usage:
    python compare_train_logs.py                          # scan ./outputs recursively
    python compare_train_logs.py outputs/**/logs/train_*.log
    python compare_train_logs.py dp1.log dp2.log --packed-per-epoch 241962
    python compare_train_logs.py outputs --csv compare.csv
"""
# SPDX-License-Identifier: Apache-2.0
import argparse
import csv
import glob
import os
import re
import sys

# numeric key=value (ints, floats, scientific); key may contain "/" (e.g. train/loss)
_KV = re.compile(r"([A-Za-z][\w/]*)=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
_TS = re.compile(r"\[(\d{2}):(\d{2}):(\d{2})\]")
_BOTTLE = re.compile(r"bottleneck='?([A-Za-z_]+)'?")
_RUN = re.compile(r">>>\s*RUN\s+train")
_HDR_NPROC = re.compile(r"\bnproc=(\d+)")
_HDR_EPOCHS = re.compile(r"\bepochs=(\d+)")
_HDR_LOSS = re.compile(r"\bloss=([A-Za-z_]+)")
_TIMING_KEYS = ("data_wait", "h2d", "forward", "backward", "optim")


def _hms(seconds):
    if seconds is None:
        return "n/a"
    s = int(seconds)
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{sec:02d}s"


def _resolve_logs(args_log):
    """Expand files/globs/dirs into a de-duped list of plausible train logs."""
    cands = []
    for a in args_log:
        if os.path.isdir(a):
            cands += glob.glob(os.path.join(a, "**", "*.log"), recursive=True)
        else:
            g = glob.glob(a, recursive=True)
            cands += g if g else ([a] if os.path.exists(a) else [])
    cands = [c for c in set(cands) if "train" in os.path.basename(c).lower()]
    return sorted(cands)


def parse_log(path):
    """Parse one log; merge each step's loss-block and timing-block by global_step."""
    by_step = {}
    cur = {}
    prev_secs = None
    day = 0
    cur_t = None
    nproc = epochs = loss_fn = None

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if nproc is None and _RUN.search(line):
                mn, me, ml = (
                    _HDR_NPROC.search(line),
                    _HDR_EPOCHS.search(line),
                    _HDR_LOSS.search(line),
                )
                nproc = int(mn.group(1)) if mn else None
                epochs = int(me.group(1)) if me else None
                loss_fn = ml.group(1) if ml else None
                continue

            m = _TS.search(line)
            if m:
                h, mi, s = (int(x) for x in m.groups())
                secs = h * 3600 + mi * 60 + s
                if prev_secs is not None and secs < prev_secs - 5:
                    day += 1  # wrapped past midnight
                prev_secs = secs
                cur_t = day * 86400 + secs

            bm = _BOTTLE.search(line)
            if bm:
                cur["bottleneck"] = bm.group(1)

            for key, val in _KV.findall(line):
                if key == "global_step":
                    step = int(float(val))
                    rec = by_step.setdefault(step, {})
                    if cur_t is not None and "t" not in rec:
                        rec["t"] = cur_t
                    rec.update(cur)
                    rec["global_step"] = step
                    cur = {}
                elif key == "epoch":
                    cur["epoch"] = int(float(val))
                elif key == "train/loss":
                    cur["loss"] = float(val)
                elif key == "lr":
                    cur["lr"] = float(val)
                elif key.startswith("timing_s_per_step/"):
                    sub = key.split("/", 1)[1]
                    if sub in _TIMING_KEYS:
                        cur[sub] = float(val)
                elif key == "step_s":
                    cur["step_s"] = float(val)

    recs = [by_step[k] for k in sorted(by_step)]
    return {
        "path": path,
        "nproc": nproc,
        "epochs": epochs,
        "loss_fn": loss_fn,
        "records": recs,
    }


def analyze(parsed, seq_len, total_cards):
    recs = parsed["records"]
    nproc = parsed["nproc"]
    out = {
        "path": parsed["path"],
        "nproc": nproc,
        "serve": (total_cards - nproc) if nproc else None,
        "epochs": parsed["epochs"],
        "loss_fn": parsed["loss_fn"],
        "n_records": len(recs),
        "step_first": recs[0]["global_step"] if recs else None,
        "step_last": recs[-1]["global_step"] if recs else None,
        "elapsed": None,
        "it_s": None,
        "tok_s": None,
        "breakdown": None,
        "datawait_pct": None,
        "bottleneck": None,
        "loss_first": None,
        "loss_last": None,
    }
    if not recs:
        return out

    # throughput from timestamps: total steps / total wall-clock (stall-inclusive)
    tp = [(r["global_step"], r["t"]) for r in recs if r.get("t") is not None]
    if len(tp) >= 2:
        elapsed = tp[-1][1] - tp[0][1]
        dsteps = tp[-1][0] - tp[0][0]
        out["elapsed"] = elapsed
        if elapsed > 0 and dsteps > 0:
            out["it_s"] = dsteps / elapsed
            if nproc:
                out["tok_s"] = out["it_s"] * nproc * seq_len

    # per-component breakdown (profiling logs only); drop the cold first step
    timing = [r for r in recs if any(k in r for k in _TIMING_KEYS)]
    if timing:
        warm = timing[1:] if len(timing) > 1 else timing
        sums = dict.fromkeys(_TIMING_KEYS, 0.0)
        n = 0
        for r in warm:
            if all(k in r for k in _TIMING_KEYS):
                for k in _TIMING_KEYS:
                    sums[k] += r[k]
                n += 1
        if n:
            bd = {k: sums[k] / n for k in _TIMING_KEYS}
            out["breakdown"] = bd
            tot = sum(bd.values()) or 1e-9
            out["datawait_pct"] = 100 * bd["data_wait"] / tot
        votes = [r["bottleneck"] for r in warm if "bottleneck" in r]
        if votes:
            out["bottleneck"] = max(set(votes), key=votes.count)

    out["loss_first"] = next((r["loss"] for r in recs if "loss" in r), None)
    out["loss_last"] = next((r["loss"] for r in reversed(recs) if "loss" in r), None)
    return out


def _label(a):
    if a["serve"] is not None and a["nproc"] is not None:
        return f"{a['serve']}s/{a['nproc']}t"
    return f"?s/{a['nproc'] or '?'}t"


def main():
    ap = argparse.ArgumentParser(
        description="Compare DFlash train runs (DP1 vs DP2) total time / throughput."
    )
    ap.add_argument("log", nargs="*", default=["outputs"],
                    help="log files/globs/dirs (default: scan ./outputs recursively)")
    ap.add_argument("--seq-len", type=int, default=3072,
                    help="tokens per replica per step (total-seq-len; default 3072)")
    ap.add_argument("--total-cards", type=int, default=8,
                    help="NPU cards on the box; serve = total - nproc (default 8)")
    ap.add_argument("--packed-per-epoch", type=int, default=None,
                    help="steps_per_epoch * train_cards (dataset-constant) to print "
                         "absolute h/epoch; open_perfectblend full ~= 241962")
    ap.add_argument("--min-steps", type=int, default=3,
                    help="skip logs with fewer than N step records (default 3)")
    ap.add_argument("--csv", default=None, help="also write a comparison CSV here")
    args = ap.parse_args()

    paths = _resolve_logs(args.log)
    if not paths:
        sys.exit(f"No train logs found in: {' '.join(args.log)}")

    runs = []
    for p in paths:
        a = analyze(parse_log(p), args.seq_len, args.total_cards)
        if a["n_records"] >= args.min_steps and a["it_s"]:
            runs.append(a)
        else:
            print(f"[skip] {p}  ({a['n_records']} steps, no usable timing)")
    if not runs:
        sys.exit("No logs had enough steps/timestamps to analyze.")

    def hpe(a):  # hours per epoch, if packed-per-epoch known
        if not (args.packed_per_epoch and a["nproc"] and a["it_s"]):
            return None
        spe = args.packed_per_epoch / a["nproc"]
        return (spe / a["it_s"]) / 3600.0

    # ---- per-log detail ----
    runs.sort(key=lambda a: (-(a["nproc"] or 0), a["path"]))
    print("=" * 100)
    print("PER-LOG DETAIL  (it/s & tok/s are stall-inclusive averages over the log)")
    print("=" * 100)
    hdr = f"{'config':<8} {'steps':>14} {'elapsed':>8} {'it/s':>6} {'tok/s':>9} {'bott':>9} {'dw%':>5}"
    if args.packed_per_epoch:
        hdr += f" {'h/epoch':>8}"
    hdr += "  log"
    print(hdr)
    print("-" * 100)
    for a in runs:
        steprange = f"{a['step_first']}-{a['step_last']}({a['n_records']})"
        tok = f"{a['tok_s']:,.0f}" if a["tok_s"] else "n/a"
        dw = f"{a['datawait_pct']:.0f}" if a["datawait_pct"] is not None else "-"
        bott = a["bottleneck"] or "-"
        row = (
            f"{_label(a):<8} {steprange:>14} {_hms(a['elapsed']):>8} "
            f"{a['it_s']:>6.2f} {tok:>9} {bott:>9} {dw:>5}"
        )
        if args.packed_per_epoch:
            h = hpe(a)
            hstr = f"{h:.1f}h" if h else "-"
            row += f" {hstr:>8}"
        row += f"  {a['path']}"
        print(row)

    # ---- per-config summary (representative = run with the most steps) ----
    groups = {}
    for a in runs:
        groups.setdefault(_label(a), []).append(a)
    reps = {}
    for label, gs in groups.items():
        reps[label] = max(gs, key=lambda a: a["n_records"])

    print("\n" + "=" * 100)
    print("CONFIG COMPARISON  (representative run = most steps; tok/s ratio = per-epoch speedup)")
    print("=" * 100)
    ordered = sorted(reps.values(), key=lambda a: (a["tok_s"] or 0))
    base = ordered[0]
    for a in ordered:
        spd = (a["tok_s"] / base["tok_s"]) if (a["tok_s"] and base["tok_s"]) else None
        tok = f"{a['tok_s']:,.0f}" if a["tok_s"] else "n/a"
        line = (
            f"{_label(a):<8} | {len(groups[_label(a)])} log(s) | "
            f"{a['it_s']:.2f} it/s | {tok} tok/s"
        )
        if a["bottleneck"]:
            line += f" | bottleneck={a['bottleneck']}"
        if a["datawait_pct"] is not None:
            line += f" | data_wait={a['datawait_pct']:.0f}%"
        if spd is not None:
            line += f" | {spd:.2f}x vs {_label(base)}"
        h = hpe(a)
        if h:
            line += f" | ~{h:.1f}h/epoch"
            if a["epochs"]:
                line += f" (~{h * a['epochs']:.1f}h total @ {a['epochs']}ep)"
        print(line)

    if len(ordered) >= 2 and ordered[-1]["tok_s"] and base["tok_s"]:
        top = ordered[-1]
        ratio = top["tok_s"] / base["tok_s"]
        print("-" * 100)
        print(f">>> {_label(top)} is ~{ratio:.2f}x the throughput of {_label(base)} "
              f"=> finishes one epoch in ~1/{ratio:.2f} the wall-clock (same dataset).")
        if base["bottleneck"] is None:
            print(f"    ({_label(base)} has no timing breakdown — it's a pre-profiling "
                  f"log; the it/s gap still quantifies the win.)")

    # ---- CSV ----
    if args.csv:
        cols = ["config", "nproc", "serve", "loss_fn", "epochs", "n_records",
                "step_first", "step_last", "elapsed_s", "it_s", "tok_s",
                "bottleneck", "datawait_pct", "h_per_epoch", "loss_first",
                "loss_last", "path"]
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for a in runs:
                w.writerow([
                    _label(a), a["nproc"], a["serve"], a["loss_fn"], a["epochs"],
                    a["n_records"], a["step_first"], a["step_last"],
                    f"{a['elapsed']:.0f}" if a["elapsed"] else "",
                    f"{a['it_s']:.4f}" if a["it_s"] else "",
                    f"{a['tok_s']:.0f}" if a["tok_s"] else "",
                    a["bottleneck"] or "",
                    f"{a['datawait_pct']:.1f}" if a["datawait_pct"] is not None else "",
                    f"{hpe(a):.2f}" if hpe(a) else "",
                    a["loss_first"], a["loss_last"], a["path"],
                ])
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
