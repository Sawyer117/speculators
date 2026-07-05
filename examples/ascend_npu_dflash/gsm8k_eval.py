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
import re

import aiohttp
from datasets import load_dataset
from tqdm import tqdm

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def last_number(text):
    nums = NUM_RE.findall(text or "")
    return nums[-1].replace(",", "").rstrip(".") if nums else None


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
                return d["choices"][0]["message"]["content"], gt_answer(ex["answer"])
        except Exception as e:  # noqa: BLE001
            return None, gt_answer(ex["answer"])


def is_correct(resp, gt):
    pred = last_number(resp)
    if pred is None or gt is None:
        return False
    try:
        return abs(float(pred) - float(gt)) < 1e-4
    except ValueError:
        return False


async def main_async(args):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    sem = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=None)
    correct = total = errors = 0
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [asyncio.create_task(ask(session, args.endpoint, args.model, ex, args.max_tokens, sem))
                 for ex in ds]
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="GSM8K", unit="q"):
            resp, gt = await fut
            total += 1
            if resp is None:
                errors += 1
            elif is_correct(resp, gt):
                correct += 1
    acc = 100 * correct / total if total else 0
    print(f"\n===== GSM8K =====\n{correct}/{total} correct = {acc:.2f}%  (errors: {errors})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", default="dsv4")
    ap.add_argument("--limit", type=int, default=300, help="0 = full 1319 test set")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
