#!/usr/bin/env python3
"""Sample SINGLE-TURN prompts from the TRAINING rollout jsonl → a file Evaluator can eval.

Purpose: the train-vs-serve mismatch test. Training `accept_len` (~3.57) is on the
rollout distribution; `--dataset all` is on gsm8k/... (a different, HARDER
distribution). The clean test = SERVE accept_len on the SAME distribution. This
builds a same-distribution sample of the training prompts, so:
  * serve ≈ train on it -> the metric transfers (no serve mismatch).
  * serve << train on it -> the train metric overestimates real serve (exposure
    bias / mismatch).

★ SINGLE-TURN ONLY by default (`--single-turn`, on): keep conversations with
exactly ONE user turn + one assistant turn, so the fresh serve generation is a
fair single-prompt match to training (multi-turn would test the first turn out of
context = artificially hard). Pass `--include-multi-turn` to disable. Prints
`single-turn eligible / total` = the turn-distribution check.

Usage:
  python build_train_eval_sample.py \
    --in  /home/canada_group_folder/dataset/open_perfectblend.dsv4_rollout/out_bf16_clean/rollout_all.clean.jsonl \
    --out ~/train_sample_500.jsonl --n 500
Then (serve up on :7000):
  TOKENIZER=$TOK SAMPLE_FILE=~/train_sample_500.jsonl CONCURRENCY=48 \
    python eval_trainsample.py | tee ~/eval_trainsample.txt
"""
import argparse
import json
import random

_HUMAN = ("human", "user")
_ASSISTANT = ("gpt", "assistant")


def first_turn(conv, roles):
    for t in conv:
        if t.get("from") in roles:
            return t.get("value", "") or ""
    return ""


def is_single_turn(conv):
    n_h = sum(1 for t in conv if t.get("from") in _HUMAN)
    n_a = sum(1 for t in conv if t.get("from") in _ASSISTANT)
    return n_h == 1 and n_a >= 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="training rollout jsonl")
    ap.add_argument("--out", required=True, help="output prompts jsonl for Evaluator")
    ap.add_argument("--n", type=int, default=500, help="sample size (default 500)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-multi-turn", dest="single_turn", action="store_false",
                    help="also sample multi-turn conversations (default: single-turn only)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    kept: list[dict] = []  # reservoir of extracted items
    eligible = 0  # single-turn (or all, if --include-multi-turn) that passed
    total = 0
    multi = 0
    with open(args.inp, encoding="utf-8") as f:
        for line in f:
            total += 1
            try:
                obj = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            conv = obj.get("conversations", []) or []
            if not is_single_turn(conv):
                multi += 1
                if args.single_turn:
                    continue
            prompt = first_turn(conv, _HUMAN)
            if not prompt.strip():
                continue
            item = {"prompt": prompt, "rollout_answer": first_turn(conv, _ASSISTANT)}
            eligible += 1
            if len(kept) < args.n:
                kept.append(item)
            else:  # reservoir replace
                j = rng.randint(0, eligible - 1)
                if j < args.n:
                    kept[j] = item

    with open(args.out, "w", encoding="utf-8") as fo:
        for item in kept:
            fo.write(json.dumps(item, ensure_ascii=False) + "\n")

    mode = "single-turn only" if args.single_turn else "all turns"
    print(
        f"[{mode}] {total} rows scanned | {multi} multi-turn ({100*multi/max(total,1):.1f}%) | "
        f"{eligible} eligible -> sampled {len(kept)} -> {args.out}"
    )


if __name__ == "__main__":
    main()
