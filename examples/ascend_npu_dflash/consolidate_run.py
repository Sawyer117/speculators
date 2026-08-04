#!/usr/bin/env python3
"""Consolidate a scattered DSV4-DSpark training LINE into one self-contained directory.

A training line is spread across many resume SEGMENTS: logs ``faithful_ep_<TS>.log`` and
checkpoints under whatever ``--save-path`` each segment used (``<save_path>/<N>/``). This gathers
ONE line into a single, recipe-named folder so it's browsable + reproducible in one place:

  <out-base>/<name>/
    train_full.log                          stitched, global_step-ordered (reuses stitch_train_logs)
    recipe.txt                              newest train_command.txt (argparse) + resolved DSPARK_* env
    MANIFEST.txt                            segments, step ranges, gaps, and the ckpt map
    ckpt/e<E>_s<S>_gs<G>  ->  <abs ckpt dir>  SYMLINKS to the REAL trainer ckpts (no copy)

Each ``ckpt/…`` symlink points at a real trainer checkpoint dir (model.safetensors + optimizer +
training_state.json), named by that ckpt's ``training_state.json`` {epoch, local_step, global_step}
— so you can convert / resume straight from the link, and the whole line's checkpoints line up by
global_step in one directory.

The segment's checkpoint dir is read from the LOG itself (the trainer logs the save-path in its
"checkpoint found in '<path>'" / "Found checkpoint at <path>/<N>" lines) — NOT guessed from the TS,
so copy-resume (a fresh save-path per resume) is handled correctly.

Usage:
  consolidate_run.py [--name <line>] [--out-base .] [--run-dir <dir>] SEG1.log SEG2.log ...
  # --run-dir defaults to the first log's directory; --name auto-derives from the recipe if omitted.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stitch_train_logs import parse_records  # noqa: E402  (same-dir reuse; identical parsing)

# save-path as logged by the trainer (banner echoes are NOT in the log file):
#   "No previous training checkpoint found in '<save_path>'."   (fresh)
#   "Found checkpoint at <save_path>/<N>."                       (resume)
#   "Saving checkpoint to <save_path>/<N>"                       (each save)
_SAVE_FRESH = re.compile(r"checkpoint found in\s*'([^']+)'")
_SAVE_RESUME = re.compile(r"Found checkpoint at\s*'?([^'\s]+)")
_SAVE_WRITE = re.compile(r"(?:Saving checkpoint(?: to)?|Checkpoint saved(?: to)?)\s*'?([^'\s]+)")


_STRIP = "/.,;:'\" \t"


def save_paths_of(log: str) -> set[str]:
    """ALL ckpt-dir candidates this segment touches: its fresh/resume save-path AND any dir it
    RESUMED FROM or SAVED TO (copy-resume can differ; earlier dirs hold earlier ckpts of the same
    line — linking them is desirable). De-dup downstream is by symlink target."""
    head = open(log, errors="ignore").read(512 * 1024)
    paths: set[str] = set()
    for m in _SAVE_FRESH.finditer(head):
        paths.add(m.group(1).rstrip(_STRIP))  # rstrip ONLY — keep the leading '/' of an abs path
    for rx in (_SAVE_RESUME, _SAVE_WRITE):
        for m in rx.finditer(head):
            p = re.sub(r"/\d+$", "", m.group(1).rstrip(_STRIP))  # strip trailing /<N> epoch dir
            paths.add(p)
    return {p for p in paths if p}


def _first_gs(path: str) -> int | None:
    m = re.search(r"global_step=(\d+)", open(path, errors="ignore").read(256 * 1024))
    return int(m.group(1)) if m else None


def _last_gs(path: str) -> int | None:
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        fh.seek(max(0, size - 256 * 1024))
        tail = fh.read().decode("utf-8", "ignore")
    ms = re.findall(r"global_step=(\d+)", tail)
    return int(ms[-1]) if ms else None


def discover_chain(run_dir: str, pattern: str, exclude: set[str]) -> list[str]:
    """Reconstruct ONE resume line by walking global_step backward from the deepest-reaching log.

    Seed = the log with the highest max global_step. Predecessor = the (unused) log whose max
    global_step is CLOSEST to the current chain's min (i.e. the segment it resumed from), tolerating
    the small resume overlap and bridging gaps. Stops at a from-scratch segment (min≈0) or when no
    predecessor is left. Returns paths oldest→newest. Robust to the fact that several from-scratch
    lines share global_step 0..N — only the ONE line that reaches the top is followed."""
    OVERLAP = 3000  # resume boundary can re-log a few thousand steps
    cands = []
    for p in glob.glob(os.path.join(run_dir, pattern)):
        ap = os.path.abspath(p)
        if ap in exclude:
            continue
        lo, hi = _first_gs(p), _last_gs(p)
        if lo is None or hi is None:
            continue
        cands.append({"path": ap, "lo": lo, "hi": hi})
    if not cands:
        return []
    cur = max(cands, key=lambda c: c["hi"])           # deepest endpoint = seed
    chain, used = [cur], {cur["path"]}
    while cur["lo"] > OVERLAP:
        preds = [c for c in cands if c["path"] not in used and c["hi"] <= cur["lo"] + OVERLAP]
        if not preds:
            break
        cur = min(preds, key=lambda c: abs(c["hi"] - cur["lo"]))  # ends nearest the current start
        chain.insert(0, cur)
        used.add(cur["path"])
    return [c["path"] for c in chain]


def ckpt_epoch_dirs(save_path: str) -> list[tuple[int, str]]:
    """(numeric_index, abspath) for each real integer ckpt dir under save_path (skip symlinks)."""
    out = []
    if not save_path or not os.path.isdir(save_path):
        return out
    for name in os.listdir(save_path):
        full = os.path.join(save_path, name)
        if name.isdigit() and os.path.isdir(full) and not os.path.islink(full):
            out.append((int(name), os.path.abspath(full)))
    return out


def state_of(ckpt_dir: str) -> dict:
    """{epoch, local_step, global_step} from training_state.json (empty if absent)."""
    p = os.path.join(ckpt_dir, "training_state.json")
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return {}


def link_name(state: dict, fallback_idx: int, save_path: str) -> str:
    """e<epoch>_s<local_step>_gs<global_step> from training_state; degrade gracefully."""
    e, s, g = state.get("epoch"), state.get("local_step"), state.get("global_step")
    if g is not None:
        return f"e{e}_s{s}_gs{int(g):08d}"
    return f"{os.path.basename(save_path)}_e{fallback_idx}"


def derive_name(newest_cmd: str, env: dict, first_ts: str) -> str:
    """Best-effort recipe tag for the folder name (used when --name omitted)."""
    bits = [f"line_{first_ts}"] if first_ts else ["line"]
    m = re.search(r"--lr\s+(\S+)", newest_cmd)
    if m:
        bits.append(f"lr{m.group(1)}")
    inits = [t.split("--init-")[1] for t in re.findall(r"--init-[a-z-]+", newest_cmd)]
    if inits:
        bits.append("+".join(x.replace("from-target", "ft").replace("moe-no-router", "norouter") for x in inits))
    if env.get("DSPARK_MOE_BALANCE") == "1":
        bits.append(f"bal{env.get('DSPARK_MOE_BALANCE_RATE', '?')}")
    if env.get("DSPARK_TEACHER_DOUBLE_NORM") == "1":
        bits.append("dnorm")
    m = re.search(r"--data-path\s+\S*/([^/\s]+)", newest_cmd)
    if m:
        bits.append(m.group(1).replace("arrow_", "").replace("open_perfectblend.dsv4_rollout", "").strip("._"))
    return re.sub(r"[^A-Za-z0-9_.+-]", "-", "_".join(b for b in bits if b))


def collect_env(logs: list[str]) -> dict:
    """Resolved DSPARK_* env from the logs' patch_getenv lines (train_command.txt lacks these)."""
    env = {}
    rx = re.compile(r"get env (DSPARK_[A-Z_]+|BF16_EXPERTS)\s*=\s*(\S+)")
    for lg in logs:
        for k, v in rx.findall(open(lg, errors="ignore").read(512 * 1024)):
            env.setdefault(k, v)  # first-seen (startup) value
    return env


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Consolidate a scattered training line into one recipe-named dir (stitched log + ckpt symlinks).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage")[-1],
    )
    ap.add_argument("logs", nargs="*", help="the line's segment log files (any order; sorted by mtime)")
    ap.add_argument("--auto-chain", action="store_true",
                    help="AUTO-discover the line: walk global_step back from the deepest-reaching log "
                         "(no need to list segments). Excludes the newest log (your live run) unless --include-newest.")
    ap.add_argument("--glob", default="faithful_ep_*.log", help="--auto-chain scan glob (default: faithful_ep_*.log)")
    ap.add_argument("--include-newest", action="store_true",
                    help="with --auto-chain, do NOT exclude the newest log (use if the live run is not running)")
    ap.add_argument("--name", help="output folder name (default: auto-derived from the recipe)")
    ap.add_argument("--out-base", default=".", help="parent dir for the consolidated folder (default: cwd)")
    ap.add_argument("--run-dir", default=None, help="dir holding the logs + ckpt_*/ (default: cwd / first log's dir)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing consolidated folder's links/log")
    args = ap.parse_args()

    run_dir = args.run_dir or (os.path.dirname(os.path.abspath(args.logs[0])) if args.logs else ".")
    if args.auto_chain:
        exclude = set()
        if not args.include_newest:
            alllogs = glob.glob(os.path.join(run_dir, args.glob))
            if alllogs:
                exclude.add(os.path.abspath(max(alllogs, key=os.path.getmtime)))  # the live run
        chain = discover_chain(run_dir, args.glob, exclude)
        if not chain:
            ap.error(f"--auto-chain found no metric logs under {run_dir}/{args.glob}")
        print(">>> auto-chain discovered (oldest→newest):", file=sys.stderr)
        for p in chain:
            print(f"     {os.path.basename(p)}  gs {_first_gs(p)}..{_last_gs(p)}", file=sys.stderr)
        logs = chain
    elif args.logs:
        logs = sorted({os.path.abspath(p) for p in args.logs}, key=os.path.getmtime)
    else:
        ap.error("pass segment logs, or --auto-chain to discover them")

    # ---- stitch the logs (global_step-ordered, later segment wins on overlap) ----
    merged: dict[int, dict] = {}
    dropped = 0
    per_seg = []
    for p in logs:
        recs = parse_records(p)
        if not recs:
            per_seg.append((os.path.basename(p), 0, None, None, None))
            continue
        steps = [gs for gs, _ in recs]
        per_seg.append((os.path.basename(p), len(recs), min(steps), max(steps), sorted(save_paths_of(p))))
        for gs, rec in recs:
            if gs in merged:
                dropped += 1
            merged[gs] = rec
    if not merged:
        ap.error("no global_step records in any log")
    order = sorted(merged)
    gaps = [(a + 1, b - 1) for a, b in zip(order, order[1:]) if b - a > 1]

    # ---- recipe: newest train_command.txt across the line's ckpts + resolved env ----
    env = collect_env(logs)
    ckpt_rows = []  # (epoch, local_step, global_step, linkname, target)
    newest_cmd, newest_cmd_mtime = "", -1.0
    for p in logs:
        for sp in save_paths_of(p):
            for idx, cdir in ckpt_epoch_dirs(sp):
                st = state_of(cdir)
                ckpt_rows.append((st.get("epoch"), st.get("local_step"), st.get("global_step"),
                                  link_name(st, idx, sp), cdir))
                tc = os.path.join(cdir, "train_command.txt")
                if os.path.isfile(tc) and os.path.getmtime(tc) > newest_cmd_mtime:
                    newest_cmd, newest_cmd_mtime = open(tc, errors="ignore").read(), os.path.getmtime(tc)
    # de-dup ckpt links by target (copy-resume can list the same dir twice)
    seen, ckpt_rows_u = set(), []
    for row in sorted(ckpt_rows, key=lambda r: (r[2] is None, r[2] or 0)):
        if row[4] in seen:
            continue
        seen.add(row[4])
        ckpt_rows_u.append(row)

    first_ts_m = re.search(r"faithful_ep_(\d{8}_\d{6})", os.path.basename(logs[0]))
    name = args.name or derive_name(newest_cmd, env, first_ts_m.group(1) if first_ts_m else "")
    out = os.path.join(os.path.abspath(args.out_base), name)
    os.makedirs(os.path.join(out, "ckpt"), exist_ok=True)

    # ---- write train_full.log ----
    with open(os.path.join(out, "train_full.log"), "w") as fh:
        for gs in order:
            rec = merged[gs]
            fh.write(" ".join(f"{k}={v}" for k, v in rec.items() if k != "global_step") + f" global_step={gs}\n")

    # ---- symlink the real ckpts ----
    n_links = 0
    for e, s, g, lname, target in ckpt_rows_u:
        link = os.path.join(out, "ckpt", lname)
        if os.path.islink(link) or os.path.exists(link):
            if not args.force:
                continue
            os.remove(link)
        os.symlink(target, link)
        n_links += 1

    # ---- recipe.txt ----
    with open(os.path.join(out, "recipe.txt"), "w") as fh:
        fh.write("# Resolved DSPARK_* / BF16_EXPERTS env (from the logs' patch_getenv; NOT in train_command.txt):\n")
        for k in sorted(env):
            fh.write(f"#   {k}={env[k]}\n")
        fh.write("\n# Newest train_command.txt in the line (argparse):\n")
        fh.write(newest_cmd or "#   (none found)\n")

    # ---- MANIFEST.txt ----
    with open(os.path.join(out, "MANIFEST.txt"), "w") as fh:
        fh.write(f"consolidated line: {name}\n")
        fh.write(f"steps: {len(order)} unique  (global_step {order[0]}..{order[-1]})\n\n")
        fh.write("segments (mtime order):\n")
        for nm, n, lo, hi, sp in per_seg:
            rng = f"[{lo}..{hi}]" if lo is not None else "(no metrics)"
            sp_s = ",".join(sp) if isinstance(sp, list) else (sp or "?")
            fh.write(f"  {nm}: {n} steps {rng}  save_path={sp_s}\n")
        fh.write(f"\noverlaps de-duped (later segment won): {dropped}\n")
        if gaps:
            fh.write("GAPS (missing global_step spans = where it broke):\n")
            for a, b in gaps:
                fh.write(f"  {a}..{b}  ({b - a + 1} missing)\n")
        else:
            fh.write("no gaps — continuous global_step coverage\n")
        fh.write("\nckpt symlinks (ckpt/<name> -> real dir):\n")
        for e, s, g, lname, target in ckpt_rows_u:
            fh.write(f"  {lname}  ->  {target}\n")

    # ---- console summary ----
    print(f"✓ consolidated → {out}")
    print(f"   train_full.log: {len(order)} steps (gs {order[0]}..{order[-1]})"
          f"{'  ⚠ ' + str(len(gaps)) + ' gap(s)' if gaps else '  (no gaps)'}")
    print(f"   ckpt symlinks:  {n_links}")
    print(f"   recipe.txt + MANIFEST.txt written")
    print(f"   → analyze: analyze_train_run.py NEW.log --baseline {os.path.join(out, 'train_full.log')}")


if __name__ == "__main__":
    main()
