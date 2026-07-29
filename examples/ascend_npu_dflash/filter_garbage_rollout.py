#!/usr/bin/env python3
"""Filter high-confidence garbage rows out of the DSV4 rollout jsonl.

Self-contained: re-detects garbage directly from the jsonl output field using the
4 rules that were exhaustively audited (0 false positives on the 77W set), so it
does NOT depend on garbage_output_indices.md and has no line-number-drift risk.

Rules (applied to the ASSISTANT output only — prompts are left untouched):
  1. stray_token_坨      : output contains the char 坨 (a stray-token artifact;
                           has no legitimate place in these English rollouts).
  2. encoding_artifact   : output contains U+FFFD "�" (mojibake). NB this catches
                           ~110 rows vs the md's 105 — the extra ~5 are genuine
                           output-side `�` the md's narrower pattern missed.
  3. python_code_fence_loop : finish_reason == "length" AND the output emitted
                           >= --fence-min "```python" fences (degenerate loop).
  4. empty_failed_output : output is empty AND finish_reason is null/empty.

Normal length-truncations (finish=length, no loop) are KEPT — the content is
valid, only the ending is missing.

Non-destructive: writes a NEW <out> jsonl and a <dropped> audit jsonl; never
touches the input. Fast: only json-parses candidate lines (~3% of the file).

Usage:
  python filter_garbage_rollout.py \
    --in  /share/canada_group_folder/dataset/open_perfectblend.dsv4_rollout/rollout_all.clean.jsonl \
    --out /share/canada_group_folder/dataset/open_perfectblend.dsv4_rollout/rollout_all.dedup.jsonl
"""
import argparse
import json
import os
import sys
from collections import Counter


def get_turn(conv, roles):
    for t in conv:
        if t.get("from") in roles:
            return t.get("value", "") or ""
    return ""


def classify(gpt, finish, fence_min):
    """Return the list of garbage reasons for this output (empty list => keep)."""
    reasons = []
    if "坨" in gpt:
        reasons.append("tuo")
    if "�" in gpt:  # U+FFFD replacement char
        reasons.append("encoding")
    if (not gpt.strip()) and (finish in (None, "", "null")):
        reasons.append("empty")
    if finish == "length" and gpt.count("```python") >= fence_min:
        reasons.append("fence")
    return reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="input jsonl")
    ap.add_argument("--out", dest="out", default=None,
                    help="filtered jsonl (default: <in dir>/rollout_all.dedup.jsonl)")
    ap.add_argument("--dropped", default=None,
                    help="audit jsonl of dropped rows (default: <in dir>/rollout_all.dropped.jsonl)")
    ap.add_argument("--fence-min", type=int, default=30,
                    help="min ```python fences (with finish=length) to call a loop. "
                         "30 reproduces the exhaustively-audited 21-row set; lower "
                         "risks catching legit multi-code-block answers.")
    _co = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "dsv4_rollout_garbage_consolidated.tsv")
    ap.add_argument("--drop-list", default=(_co if os.path.exists(_co) else None),
                    help="TSV/MD with a jsonl_line column. DEFAULTS to the co-located, "
                         "audited dsv4_rollout_garbage_consolidated.tsv (3,275 rows) if "
                         "present — so you don't need to pass it. Its rows are dropped IN "
                         "ADDITION to the 4 self-detected rules; this is how the fuzzy "
                         "tail-loop rows (| | | |, Navigation Menu, S. S. S., repeated "
                         "n-grams) get removed (they can't be safely self-detected without "
                         "re-catching prompt-requested repetition). Pass '' to disable.")
    args = ap.parse_args()

    d = os.path.dirname(os.path.abspath(args.inp))
    out = args.out or os.path.join(d, "rollout_all.dedup.jsonl")
    dropped = args.dropped or os.path.join(d, "rollout_all.dropped.jsonl")
    if os.path.abspath(out) == os.path.abspath(args.inp):
        sys.exit("refusing to overwrite the input file")

    # optional audited drop-list (jsonl_line -> reason), from a .tsv or .md table.
    drop_lines = {}
    if args.drop_list and not os.path.exists(args.drop_list):
        print(f"⚠ --drop-list {args.drop_list} not found — skipping it (self-detect rules only)")
        args.drop_list = None
    if args.drop_list:
        with open(args.drop_list, encoding="utf-8") as f:
            for row in f:
                if "\t" in row:
                    parts = row.rstrip("\n").split("\t")
                elif row.startswith("|"):
                    parts = [p.strip() for p in row.split("|")][1:]
                else:
                    continue
                if parts and parts[0].strip().isdigit():
                    ln = int(parts[0].strip())
                    reason = parts[5].strip() if len(parts) > 5 else "listed"
                    drop_lines[ln] = reason
        print(f"loaded drop-list: {len(drop_lines)} jsonl_lines from {args.drop_list}")

    # substrings that mark a line as a garbage *candidate* worth json-parsing.
    # (avoids parsing the ~97% clean rows -> fast). A clean row is exactly a
    # finish=="stop" row with no 坨/�; everything else must be parsed:
    #   - length rows  -> need the fence check
    #   - empty rows   -> their metadata/finish_reason key is ABSENT (not "null"),
    #                     so we can't match a "null" substring; instead we treat any
    #                     row that is NOT a plain "stop" row as a candidate.
    def is_candidate(line):
        if "坨" in line or "�" in line:
            return True
        return not ('"finish_reason": "stop"' in line or '"finish_reason":"stop"' in line)

    def tail_repetitive(gpt):
        toks = gpt[-400:].split()
        return len(toks) >= 20 and len(set(toks[-40:])) <= 5

    n = kept = 0
    cat = Counter()
    listed_seen = 0          # drop-list rows actually hit
    listed_no_signal = []    # listed rows that show NO garbage signal at all -> drift?
    with open(args.inp, encoding="utf-8") as fin, \
         open(out, "w", encoding="utf-8") as fout, \
         open(dropped, "w", encoding="utf-8") as fdrop:
        for i, line in enumerate(fin, 1):
            n += 1
            listed = i in drop_lines
            if not listed and not is_candidate(line):
                fout.write(line); kept += 1; continue
            try:
                obj = json.loads(line)
            except Exception:
                fout.write(line); kept += 1; continue
            conv = obj.get("conversations", []) or []
            md = obj.get("metadata", {}) or {}
            gpt = get_turn(conv, ("gpt", "assistant"))
            finish = md.get("finish_reason")
            reasons = classify(gpt, finish, args.fence_min)
            if listed:
                listed_seen += 1
                reasons = reasons + [f"listed:{drop_lines[i]}"]
                # drift sanity: a listed row should carry SOME signal (a 4-rule hit or a
                # repetitive tail). If none, the jsonl may have drifted vs the list.
                if not classify(gpt, finish, args.fence_min) and not tail_repetitive(gpt):
                    listed_no_signal.append((i, obj.get("id")))
            if not reasons:
                fout.write(line); kept += 1; continue
            for r in reasons:
                cat[r] += 1
            fdrop.write(json.dumps(
                {"jsonl_line": i, "id": obj.get("id"),
                 "idx": md.get("idx"), "finish_reason": finish,
                 "reasons": reasons}, ensure_ascii=False) + "\n")

    ndrop = n - kept
    print("=" * 64)
    print(f"input rows          : {n}")
    print(f"kept (clean)        : {kept}")
    print(f"dropped (garbage)   : {ndrop}   ({100*ndrop/n:.3f}%)")
    print(f"  by reason (a row may carry >1): {dict(cat)}")
    if drop_lines:
        print(f"  drop-list rows hit  : {listed_seen}/{len(drop_lines)}")
        if listed_no_signal:
            print(f"  ⚠ {len(listed_no_signal)} listed rows show NO garbage signal "
                  f"(possible line-number drift vs a different jsonl): {listed_no_signal[:10]}")
        else:
            print("  ✓ every listed row re-confirmed a garbage signal (no drift)")
    print(f"output              : {out}")
    print(f"dropped audit       : {dropped}")
    print("=" * 64)
    print("self-detected rules: tuo~2995, encoding~110, empty~28, fence~21")
    print("+ drop-list adds the audited tail-loop rows (~130): | | | |, Navigation Menu,")
    print("  S. S. S., repeated n-grams. Total with consolidated list ~3.28k / 0.42%.")


if __name__ == "__main__":
    main()
