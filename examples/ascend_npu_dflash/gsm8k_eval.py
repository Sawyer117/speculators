#!/usr/bin/env python3
"""Quick GSM8K accuracy eval against an OpenAI-compatible /v1/chat/completions endpoint.

0-shot CoT: send each GSM8K *test* question, extract the LAST number in the model's
answer (the standard flexible-extract rule), compare to the ground truth (the number
after '####'). Reports accuracy. Concurrent via aiohttp.

This is a sanity check that the served model is not just coherent but CORRECT — the
AtomGit two-node bf16 reference reports gsm8k ~97.27.

Usage:
  python gsm8k_eval.py --endpoint http://127.0.0.1:7000/v1/chat/completions \
      --model dsv4 --limit 300 --concurrency 64
"""
import argparse
import asyncio
import json
import re

import aiohttp
from datasets import load_dataset
from tqdm import tqdm

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def last_number(text):
    nums = NUM_RE.findall(text or "")
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def extract_answer(text):
    """DeepSeek answers in \\boxed{...}. Prefer the LAST boxed value (survives any
    post-answer verification text); else fall back to the last number overall."""
    if not text:
        return None
    boxes = BOXED_RE.findall(text)
    if boxes:
        return last_number(boxes[-1]) or boxes[-1].strip()
    return last_number(text)


def gt_answer(ans):
    m = re.search(r"####\s*(-?[\d,]+)", ans or "")
    return m.group(1).replace(",", "") if m else None


async def ask(session, endpoint, model, ex, max_tokens, sem):
    prompt = (ex["question"]
              + "\nPlease reason step by step, and put your final numeric answer at the end.")
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "temperature": 0, "max_tokens": max_tokens}
    async with sem:
        try:
            async with session.post(endpoint, json=payload) as r:
                d = await r.json()
                return d["choices"][0]["message"]["content"], gt_answer(ex["answer"]), ex["question"]
        except Exception:  # noqa: BLE001
            return None, gt_answer(ex["answer"]), ex["question"]


def is_correct(resp, gt):
    pred = extract_answer(resp)
    if pred is None or gt is None:
        return False
    try:
        return abs(float(pred) - float(gt)) < 1e-4
    except ValueError:
        return str(pred).strip() == str(gt).strip()


async def main_async(args):
    if args.local_file:
        # local jsonl/parquet with {question, answer} — no HF hub access needed.
        fmt = "parquet" if args.local_file.endswith(".parquet") else "json"
        ds = load_dataset(fmt, data_files=args.local_file, split="train")
    else:
        ds = load_dataset("openai/gsm8k", "main", split="test")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    sem = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=None)
    correct = total = errors = 0
    dumpf = open(args.dump, "w", encoding="utf-8") if args.dump else None
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [asyncio.create_task(ask(session, args.endpoint, args.model, ex, args.max_tokens, sem))
                 for ex in ds]
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="GSM8K", unit="q"):
            resp, gt, q = await fut
            total += 1
            ok = False
            if resp is None:
                errors += 1
            else:
                ok = is_correct(resp, gt)
                if ok:
                    correct += 1
            if dumpf:
                dumpf.write(json.dumps({"correct": ok, "pred": extract_answer(resp), "gt": gt,
                                        "question": q, "response": resp}, ensure_ascii=False) + "\n")
    if dumpf:
        dumpf.close()
    acc = 100 * correct / total if total else 0
    print(f"\n===== GSM8K =====\n{correct}/{total} correct = {acc:.2f}%  (errors: {errors})")
    if args.dump:
        print(f"per-question dump → {args.dump}  (grep '\"correct\": false' to see misses)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", default="dsv4")
    ap.add_argument("--limit", type=int, default=300, help="0 = full 1319 test set")
    ap.add_argument("--local-file", help="local gsm8k jsonl/parquet ({question,answer}) — "
                    "skips the HF download (for offline/proxied machines)")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--dump", help="write per-question {correct,pred,gt,question,response} jsonl")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
