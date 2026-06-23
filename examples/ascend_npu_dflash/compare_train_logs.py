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
# fallback when the RUN header is missing (older logs): the distributed-init line
# "Started distributed with local_rank=.., world_size=N, rank=.." gives nproc on
# a single node. (Assumes single node; multi-node world_size != per-node nproc.)
_WSIZE = re.compile(r"\bworld_size=(\d+)")
_TIMING_KEYS = ("data_wait", "h2d", "forward", "backward", "optim")
_SPIKE_FACTOR = 3.0  # a step is a "spike" (recompile/stall) if step_s > F * median


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
    nproc = epochs = loss_fn = world_size = None

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if world_size is None:
                wm = _WSIZE.search(line)
                if wm:
                    world_size = int(wm.group(1))
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
        "nproc": nproc if nproc is not None else world_size,  # header, else init-line fallback
        "nproc_from_header": nproc is not None,
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
        "nproc_hdr": parsed["nproc_from_header"],
        "elapsed": None,
        "it_s": None,          # avg rate over the window (stall/warmup-inclusive)
        "tok_s": None,         # avg-rate tokens/s
        "steady_step_s": None,  # median step_s = typical spike-free step
        "steady_it_s": None,
        "steady_tok_s": None,
        "n_spikes": 0,
        "spike_time": 0.0,
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

    # steady rate: median of the instrumented step_s, robust to recompile spikes.
    # This is the right basis for FULL-epoch ETA (front-loaded recompiles amortize
    # to ~nothing over 34k+ steps); the avg rate above is dragged down on short runs.
    step_s = [r["step_s"] for r in recs if "step_s" in r]
    if step_s:
        ss = sorted(step_s)
        med = ss[len(ss) // 2]
        out["steady_step_s"] = med
        if med > 0:
            out["steady_it_s"] = 1.0 / med
            if nproc:
                out["steady_tok_s"] = out["steady_it_s"] * nproc * seq_len
            for v in step_s:
                if v > _SPIKE_FACTOR * med:
                    out["n_spikes"] += 1
                    out["spike_time"] += v - med

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

    def steady_it(a):
        return a["steady_it_s"] or a["it_s"]

    def steady_tok(a):
        return a["steady_tok_s"] or a["tok_s"]

    def hpe(a):  # hours per FULL epoch from the STEADY rate (warmup amortizes over 34k+ steps)
        rate = steady_it(a)
        if not (args.packed_per_epoch and a["nproc"] and rate):
            return None
        return (args.packed_per_epoch / a["nproc"] / rate) / 3600.0

    def marker(a):  # flag nproc recovered from world_size (no RUN header)
        return "" if (a["nproc_hdr"] or a["nproc"] is None) else "  [nproc<-world_size]"

    # ---- per-log detail ----
    runs.sort(key=lambda a: (-(a["nproc"] or 0), a["path"]))
    print("=" * 108)
    print("PER-LOG DETAIL")
    print("  avg  = whole-window it/s (INCLUDES warmup + recompile spikes; drops on short runs)")
    print("  stdy = median-step it/s (spike-free typical speed = the full-epoch basis)")
    print("  tok/s & h/ep use the STEADY rate;  spk = #steps with step_s > 3x median (recompile/stall)")
    print("=" * 108)
    hdr = (f"{'config':<8} {'steps':>13} {'elapsed':>7} {'avg':>5} {'stdy':>5} "
           f"{'tok/s':>8} {'bott':>9} {'dw%':>4} {'spk':>4}")
    if args.packed_per_epoch:
        hdr += f" {'h/ep':>6}"
    hdr += "  log"
    print(hdr)
    print("-" * 108)
    for a in runs:
        steprange = f"{a['step_first']}-{a['step_last']}({a['n_records']})"
        stdy = f"{a['steady_it_s']:.2f}" if a["steady_it_s"] else "-"
        ft = steady_tok(a)
        tok = f"{ft:,.0f}" if ft else "n/a"
        dw = f"{a['datawait_pct']:.0f}" if a["datawait_pct"] is not None else "-"
        bott = a["bottleneck"] or "-"
        spk = str(a["n_spikes"]) if a["steady_step_s"] else "-"
        row = (
            f"{_label(a):<8} {steprange:>13} {_hms(a['elapsed']):>7} "
            f"{a['it_s']:>5.2f} {stdy:>5} {tok:>8} {bott:>9} {dw:>4} {spk:>4}"
        )
        if args.packed_per_epoch:
            h = hpe(a)
            hstr = f"{h:.1f}h" if h else "-"
            row += f" {hstr:>6}"
        row += f"  {a['path']}{marker(a)}"
        print(row)

    # ---- component breakdown (profiling logs) ----
    bd_runs = [a for a in runs if a["breakdown"]]
    if bd_runs:
        print("\n" + "=" * 108)
        print("COMPONENT BREAKDOWN  (mean seconds/step; MEAN includes recompile spikes, "
              "MEDIAN step does not)")
        print("  if forward(mean) >> backward but bott=backward, the recompile spikes live in "
              "forward; mean_step vs med_step shows their cost")
        print("=" * 108)
        bh = (f"{'config':<8} {'data_wait':>9} {'h2d':>7} {'forward':>8} {'backward':>9} "
              f"{'optim':>7} {'mean_step':>10} {'med_step':>9} {'spikes(lost)':>16}")
        print(bh)
        print("-" * 108)
        for a in bd_runs:
            bd = a["breakdown"]
            mean_step = sum(bd.values())
            med = a["steady_step_s"] or 0.0
            spikes = f"{a['n_spikes']} ({_hms(a['spike_time'])})"
            print(
                f"{_label(a):<8} {bd['data_wait']:>9.3f} {bd['h2d']:>7.3f} "
                f"{bd['forward']:>8.3f} {bd['backward']:>9.3f} {bd['optim']:>7.3f} "
                f"{mean_step:>10.3f} {med:>9.3f} {spikes:>16}"
            )

    # ---- per-config summary (representative = run with the most steps) ----
    groups = {}
    for a in runs:
        groups.setdefault(_label(a), []).append(a)
    # representative: prefer a run that HAS steady (timing) data, then most steps —
    # so a long pre-profiling log can't drag a config's steady number to its avg.
    reps = {
        label: max(gs, key=lambda a: (1 if a["steady_it_s"] else 0, a["n_records"]))
        for label, gs in groups.items()
    }

    print("\n" + "=" * 108)
    print("CONFIG COMPARISON  (representative = run with timing & most steps; STEADY tok/s "
          "ratio = per-epoch speedup, same dataset)")
    print("=" * 108)
    ordered = sorted(reps.values(), key=lambda a: (steady_tok(a) or 0))
    base = ordered[0]
    for a in ordered:
        ft, fi = steady_tok(a), steady_it(a)
        spd = (ft / steady_tok(base)) if (ft and steady_tok(base)) else None
        tok = f"{ft:,.0f}" if ft else "n/a"
        line = (
            f"{_label(a):<8} | {len(groups[_label(a)])} log(s) | "
            f"{fi:.2f} it/s | {tok} tok/s (steady)"
        )
        if a["bottleneck"]:
            line += f" | bott={a['bottleneck']}"
        if a["datawait_pct"] is not None:
            line += f" | data_wait={a['datawait_pct']:.0f}%"
        if spd is not None:
            line += f" | {spd:.2f}x vs {_label(base)}"
        h = hpe(a)
        if h:
            line += f" | ~{h:.1f}h/epoch"
        print(line)

    if len(ordered) >= 2 and steady_tok(ordered[-1]) and steady_tok(base):
        top = ordered[-1]
        ratio = steady_tok(top) / steady_tok(base)
        print("-" * 108)
        print(f">>> {_label(top)} ~{ratio:.2f}x the steady throughput of {_label(base)} "
              f"=> ~1/{ratio:.2f} the wall-clock per epoch (same dataset).")
        print("    NOTE: 'steady' excludes the front-loaded NPU recompile spikes (see spk/spikes "
              "columns); those are a separate, shared cost worth fixing on BOTH configs.")

    # ---- CSV ----
    if args.csv:
        cols = ["config", "nproc", "nproc_from_header", "serve", "loss_fn", "epochs",
                "n_records", "step_first", "step_last", "elapsed_s",
                "avg_it_s", "steady_it_s", "steady_tok_s", "n_spikes", "spike_time_s",
                "bottleneck", "datawait_pct",
                "fwd_s", "bwd_s", "h2d_s", "datawait_s", "optim_s",
                "h_per_epoch_steady", "loss_first", "loss_last", "path"]
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for a in runs:
                bd = a["breakdown"] or {}
                w.writerow([
                    _label(a), a["nproc"], a["nproc_hdr"], a["serve"], a["loss_fn"],
                    a["epochs"], a["n_records"], a["step_first"], a["step_last"],
                    f"{a['elapsed']:.0f}" if a["elapsed"] else "",
                    f"{a['it_s']:.4f}" if a["it_s"] else "",
                    f"{a['steady_it_s']:.4f}" if a["steady_it_s"] else "",
                    f"{a['steady_tok_s']:.0f}" if a["steady_tok_s"] else "",
                    a["n_spikes"], f"{a['spike_time']:.0f}",
                    a["bottleneck"] or "",
                    f"{a['datawait_pct']:.1f}" if a["datawait_pct"] is not None else "",
                    f"{bd.get('forward'):.3f}" if bd.get("forward") is not None else "",
                    f"{bd.get('backward'):.3f}" if bd.get("backward") is not None else "",
                    f"{bd.get('h2d'):.3f}" if bd.get("h2d") is not None else "",
                    f"{bd.get('data_wait'):.4f}" if bd.get("data_wait") is not None else "",
                    f"{bd.get('optim'):.3f}" if bd.get("optim") is not None else "",
                    f"{hpe(a):.2f}" if hpe(a) else "",
                    a["loss_first"], a["loss_last"], a["path"],
                ])
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
