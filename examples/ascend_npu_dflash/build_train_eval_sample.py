#!/usr/bin/env python3
"""Sample prompts from the TRAINING rollout jsonl → a file Evaluator.py can eval.

Purpose: the train-vs-serve mismatch test. The training `accept_len` metric is
measured on the rollout distribution (open_perfectblend, greedy temp0); the
`--dataset all` eval is on gsm8k/math500/... (a HARDER, different distribution).
So `train ≈ eval` doesn't prove "no serve mismatch" — it's confounded by data
difficulty. This script builds a SAME-DISTRIBUTION sample: random prompts drawn
from the exact training rollout. Serve-eval accept_len on THIS sample vs the
training accept_len (~3.57) then disentangles the two:
  * serve ≈ train on this sample  -> NO serve mismatch; the eval gap is data
    difficulty (gsm8k harder than rollout) -> lever = data/recipe.
  * serve << train on this sample  -> serve mismatch confirmed -> hunt the bug.

Reservoir-samples N conversations without loading the whole (77W-row) file, then
writes one JSON object per line: {"prompt": <first user turn>, "rollout_answer":
<recorded assistant turn, for reference>}. Evaluator loads it via
`load_dataset("json", data_files=<out>)` under the gated `trainsample` dataset.

Usage:
  python build_train_eval_sample.py \
    --in  /home/canada_group_folder/dataset/open_perfectblend.dsv4_rollout/out_bf16_clean/rollout_all.clean.jsonl \
    --out ~/train_sample_500.jsonl --n 500
Then eval it (serve up on :7000):
  TRAIN_SAMPLE_FILE=~/train_sample_500.jsonl DATASET=trainsample CONCURRENCY=48 \
    bash run_dspark_eval.sh | tee ~/eval_trainsample.txt
"""
import argparse
import json
import random


def get_turn(conv, roles):
    for t in conv:
        if t.get("from") in roles:
            return t.get("value", "") or ""
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="training rollout jsonl")
    ap.add_argument("--out", required=True, help="output prompts jsonl for Evaluator")
    ap.add_argument("--n", type=int, default=500, help="sample size (default 500)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    kept: list[str] = []  # reservoir of raw lines
    total = 0
    with open(args.inp, encoding="utf-8") as f:
        for i, line in enumerate(f):
            total = i + 1
            if len(kept) < args.n:
                kept.append(line)
            else:
                j = rng.randint(0, i)
                if j < args.n:
                    kept[j] = line

    written = 0
    with open(args.out, "w", encoding="utf-8") as fo:
        for line in kept:
            try:
                obj = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            conv = obj.get("conversations", []) or []
            prompt = get_turn(conv, ("human", "user"))
            if not prompt.strip():
                continue
            fo.write(
                json.dumps(
                    {"prompt": prompt, "rollout_answer": get_turn(conv, ("gpt", "assistant"))},
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

    print(f"sampled {written} training prompts from {total} rows -> {args.out}")


if __name__ == "__main__":
    main()
