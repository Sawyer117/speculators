#!/usr/bin/env python3
"""Smoke-check a running vLLM serve and measure conc-1 decode throughput + accept length.

Exists because the same three numbers get taken repeatedly -- once without speculation to
establish the denominator, then once per speculative configuration -- and a hand-typed curl
each time is how those numbers stop being comparable.

WHAT IT DOES
  1. CORRECTNESS FIRST. Counting to 40 is the cheapest possible corruption detector: any
     repeat, gap, or drift is visible at a glance, where a fluent-looking paragraph is not.
     A server that starts is not a server that computes -- an over-batched bf16 DSV4 once
     served confident garbage for hours.
  2. Conc-1 decode throughput, with ignore_eos so the token count is exactly what we asked
     for and the rate is not an artefact of where the model chose to stop. Prefill is
     excluded by timing against the completion tokens only, and a warm-up request is thrown
     away first so compilation and allocator growth do not land in the measurement.
  3. Accept length from /metrics, read as a BEFORE/AFTER difference rather than as a
     cumulative average: those counters are cumulative and reset when the engine restarts,
     so a raw read mixes this measurement with everything the server did earlier.

USAGE
  python3 examples/ascend_npu_dflash/quick_serve_check.py                     # all of it
  python3 examples/ascend_npu_dflash/quick_serve_check.py --label AR          # tag the run
  python3 examples/ascend_npu_dflash/quick_serve_check.py --tokens 512 --reps 3
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request

COUNT_PROMPT = "Count from 1 to 40, separated by single spaces. Output only the numbers.\n"
REASON_PROMPT = (
    "A tank has two inlet pipes. Pipe A alone fills it in 6 hours, pipe B alone in 4 hours. "
    "How long to fill it with both open? Show your reasoning."
)


# ⚠ NEVER go through a proxy. These boxes keep an http_proxy exported for outbound access,
# urllib honours it, and the request to 127.0.0.1 then leaves the machine and comes back as
# `HTTP Error 504: Gateway Time-out` -- which reads like the server is wedged when it is
# serving perfectly. An empty ProxyHandler disables proxying for this opener only.
# (curl needs --noproxy '*' for the same reason.)
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(url: str, payload: dict, timeout: int = 600):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with _OPENER.open(req, timeout=timeout) as r:
        return json.load(r)


def get_text(base: str, path: str, timeout: int = 30) -> str:
    with _OPENER.open(base + path, timeout=timeout) as r:
        return r.read().decode(errors="ignore")


def metrics_counters(base: str) -> dict:
    """The two spec-decode counters, or {} when speculation is off (they are simply absent)."""
    try:
        raw = get_text(base, "/metrics")
    except Exception:
        return {}
    out = {}
    for name in ("vllm:spec_decode_num_draft_tokens_total",
                 "vllm:spec_decode_num_accepted_tokens_total",
                 "vllm:spec_decode_num_drafts_total"):
        total = 0.0
        found = False
        for m in re.finditer(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)$", raw, re.M):
            total += float(m.group(1))
            found = True
        if found:
            out[name] = total
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8900", help="server base URL")
    ap.add_argument("--model", default="dsv4")
    ap.add_argument("--label", default="", help="tag for the printed result line")
    ap.add_argument("--tokens", type=int, default=256, help="decode tokens per timed request")
    ap.add_argument("--reps", type=int, default=3, help="timed requests (median is reported)")
    ap.add_argument("--skip-gen", action="store_true", help="only measure, skip the two text checks")
    args = ap.parse_args()

    comp = f"{args.base}/v1/completions"
    chat = f"{args.base}/v1/chat/completions"
    tag = f"[{args.label}] " if args.label else ""
    print("=" * 72)
    print(f" {tag}QUICK CHECK  {args.base}  model={args.model}")
    print("=" * 72)

    if not args.skip_gen:
        print("\n-- 1. counting to 40 (corruption detector) --")
        txt = post(comp, {"model": args.model, "prompt": COUNT_PROMPT,
                          "max_tokens": 160, "temperature": 0})["choices"][0]["text"]
        nums = [int(n) for n in re.findall(r"\d+", txt)]
        ok = nums[:40] == list(range(1, 41))
        print(f"  {txt.strip()[:200]}")
        print(f"  ⟹ {'✓ exactly 1..40' if ok else '✗ NOT 1..40 — do NOT trust any throughput number from this server'}")
        if not ok and nums:
            print(f"     got {len(nums)} numbers, first 10: {nums[:10]}")

        print("\n-- 2. a short reasoning task (coherence) --")
        msg = post(chat, {"model": args.model,
                          "messages": [{"role": "user", "content": REASON_PROMPT}],
                          "max_tokens": 512, "temperature": 0})["choices"][0]["message"]
        body = (msg.get("content") or "").strip()
        print(f"  {body[:400]}")
        print(f"  ⟹ the answer should be 2.4 hours (12/5).")

    print(f"\n-- 3. conc-1 decode throughput ({args.reps} reps x {args.tokens} tokens) --")
    warm = {"model": args.model, "prompt": "Hello.", "max_tokens": 16,
            "temperature": 0, "ignore_eos": True}
    post(comp, warm)   # discard: compilation + allocator growth must not land in the numbers

    before = metrics_counters(args.base)
    rates = []
    for i in range(args.reps):
        payload = {"model": args.model, "prompt": REASON_PROMPT, "max_tokens": args.tokens,
                   "temperature": 0, "ignore_eos": True}
        t0 = time.perf_counter()
        resp = post(comp, payload)
        dt = time.perf_counter() - t0
        n = (resp.get("usage") or {}).get("completion_tokens") or args.tokens
        rates.append(n / dt)
        print(f"  rep {i+1}: {n} tok in {dt:.2f}s  ->  {n/dt:.1f} tok/s")
    after = metrics_counters(args.base)

    rates.sort()
    med = rates[len(rates) // 2]
    print(f"\n  ★ {tag}conc-1 decode: {med:.1f} tok/s (median of {args.reps})")

    d = {k: after.get(k, 0) - before.get(k, 0) for k in set(before) | set(after)}
    drafts = d.get("vllm:spec_decode_num_drafts_total", 0)
    acc = d.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    draft_toks = d.get("vllm:spec_decode_num_draft_tokens_total", 0)
    if drafts > 0:
        print(f"  ★ accept length: {1 + acc/drafts:.3f}   "
              f"(accepted {acc:.0f} of {draft_toks:.0f} drafted over {drafts:.0f} drafts, "
              f"acceptance {acc/draft_toks:.1%})" if draft_toks else "")
        print("     accept length = 1 + accepted/drafts: the free token always counts.")
    else:
        print("  accept length: n/a — no spec-decode counters moved (autoregressive run).")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
