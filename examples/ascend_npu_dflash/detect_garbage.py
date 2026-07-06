#!/usr/bin/env python3
"""Scan a rollout JSONL for GARBAGE and REPETITION — the two failure modes the DSV4
bf16 KV-overflow bug produces:

  * garbage    — incoherent short outputs (``No`` / ``Hateful`` / stray fragments);
  * repetition — degenerate loops (``的实际的实际…`` / ``X as X as X…``) that usually
                 run all the way to ``max_tokens``.

A rollout has NO ground truth (unlike gsm8k), so this uses text-shape heuristics on the
generated (gpt) turn. A row is FLAGGED if any signal trips; it can hit several:

  EMPTY      error row / no response text
  TOO_SHORT  response < --min-len chars      → the "No" garbage
  LOW_ALPHA  head of response is mostly non-letters → gibberish start
  REPEAT     highly repetitive — zlib compresses it below --rep-comp of its byte size,
             OR its distinct 4-gram ratio < --rep-ngram  (degenerate loop)

Compression ratio is the workhorse: normal prose/code ≈ 0.30–0.45; pathological
repetition ≈ < 0.10. The default --rep-comp 0.12 only trips on genuine loops, so
coherent-but-truncated answers (finish_reason=length, ~1.7% expected) are NOT flagged
— only truncated *loops* are.

Usage:
  python detect_garbage.py rollout_00.jsonl                 # summary
  python detect_garbage.py rollout_00.jsonl --show 8        # + worst examples per flag
  python detect_garbage.py rollout_00.jsonl --dump bad.jsonl
"""
import argparse
import json
import zlib
from collections import Counter


def response_text(row):
    """The generated (gpt/assistant) turn's text, or None for error/malformed rows."""
    for turn in reversed(row.get("conversations") or []):
        if isinstance(turn, dict) and (turn.get("from") or turn.get("role")) in ("gpt", "assistant"):
            return turn.get("value") or turn.get("content")
    return None


def comp_ratio(text):
    """zlib compressed size / raw size. Low = repetitive. 1.0 if too short to judge."""
    b = text.encode("utf-8", "ignore")
    if len(b) < 60:
        return 1.0
    return len(zlib.compress(b, 6)) / len(b)


def distinct_ngram(text, n=4):
    """Fraction of DISTINCT word n-grams. Low = degenerate loop. 1.0 if too short."""
    w = text.split()
    if len(w) < n + 8:
        return 1.0
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return len(set(grams)) / len(grams)


def alpha_head(text, k=40):
    head = text.strip()[:k]
    return sum(c.isalpha() for c in head) / len(head) if head else 0.0


def classify(row, args):
    """Return (set_of_flags, response_text, comp_ratio)."""
    meta = row.get("metadata") or {}
    text = response_text(row)
    if text is None or "error" in meta:
        return {"EMPTY"}, text, 1.0
    t = text.strip()
    flags = set()
    if len(t) < args.min_len:
        flags.add("TOO_SHORT")
    if alpha_head(t) < args.min_alpha:
        flags.add("LOW_ALPHA")
    cr = comp_ratio(t)
    if cr < args.rep_comp or distinct_ngram(t) < args.rep_ngram:
        flags.add("REPEAT")
    return flags, text, cr


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infile", help="rollout JSONL (ShareGPT conversations format)")
    ap.add_argument("--min-len", type=int, default=12, help="flag responses shorter than this")
    ap.add_argument("--min-alpha", type=float, default=0.30, help="min letter-fraction of first 40 chars")
    ap.add_argument("--rep-comp", type=float, default=0.12, help="flag if zlib ratio below this (repetition)")
    ap.add_argument("--rep-ngram", type=float, default=0.25, help="flag if distinct-4gram ratio below this")
    ap.add_argument("--show", type=int, default=0, help="print N worst examples per flag")
    ap.add_argument("--dump", help="write all flagged rows {idx,flags,finish_reason,response} to this jsonl")
    args = ap.parse_args()

    counts = Counter()
    fr_flagged = Counter()          # finish_reason breakdown of flagged rows
    examples = {k: [] for k in ("REPEAT", "TOO_SHORT", "LOW_ALPHA", "EMPTY")}
    total = 0
    dumpf = open(args.dump, "w", encoding="utf-8") if args.dump else None

    with open(args.infile, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                counts["BADJSON"] += 1
                continue
            flags, text, cr = classify(row, args)
            if not flags:
                continue
            counts["FLAGGED"] += 1
            fr = (row.get("metadata") or {}).get("finish_reason")
            fr_flagged[fr] += 1
            idx = (row.get("metadata") or {}).get("idx")
            for fl in flags:
                counts[fl] += 1
                bucket = examples.get(fl)
                if bucket is not None and len(bucket) < 500:
                    # score: REPEAT sorts by comp ratio (asc); others by length (asc)
                    score = cr if fl == "REPEAT" else len((text or "").strip())
                    bucket.append((score, idx, (text or "")[:160]))
            if dumpf:
                dumpf.write(json.dumps({"idx": idx, "flags": sorted(flags),
                                        "finish_reason": fr, "response": text},
                                       ensure_ascii=False) + "\n")
    if dumpf:
        dumpf.close()

    flagged = counts["FLAGGED"]
    print(f"\n===== rollout 质量检测: {args.infile} =====")
    print(f"总行数            {total}")
    print(f"疑似问题(合计)   {flagged}  ({100 * flagged / max(total, 1):.2f}%)")
    for k, label in (("REPEAT", "重复/退化循环"), ("TOO_SHORT", "过短(乱码)"),
                     ("LOW_ALPHA", "非文本开头"), ("EMPTY", "空/报错"), ("BADJSON", "坏JSON")):
        if counts[k]:
            print(f"  {label:<14} {counts[k]:>7}  ({100 * counts[k] / max(total, 1):.2f}%)")
    if flagged and fr_flagged:
        fr_str = ", ".join(f"{k}={v}" for k, v in fr_flagged.most_common())
        print(f"疑似行的 finish_reason 分布: {fr_str}")

    if args.show:
        for k in ("REPEAT", "TOO_SHORT", "LOW_ALPHA", "EMPTY"):
            rows = sorted(examples[k])[:args.show]
            if rows:
                print(f"\n--- {k} 最差 {len(rows)} 例 ---")
                for score, idx, snip in rows:
                    tag = f"comp={score:.3f}" if k == "REPEAT" else f"len={score}"
                    print(f"  [idx {idx} {tag}] {snip!r}")
    if args.dump:
        print(f"\n全部疑似行 → {args.dump}  (grep/看一眼确认是真乱码还是误报)")


if __name__ == "__main__":
    main()
