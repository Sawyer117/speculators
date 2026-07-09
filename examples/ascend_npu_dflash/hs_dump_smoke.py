#!/usr/bin/env python3
"""Plan B HS-dump smoke test — verify the DSpark HS dumper writes correct files.

Run on the HEAD node, in the same env as the serve (torch + safetensors available),
AFTER bringing up the serve with HS_DUMP=1 on both nodes:

    HS_DUMP=1 bash serve_dsv4_bf16_dualnode.sh head      # (worker: same, role=worker)
    # once ">>> READY":
    python hs_dump_smoke.py                                # defaults: localhost:7000, dsv4

What it does (prefill-only, teacher-forced):
  1. POST a couple of raw-text prompts to /v1/completions with max_tokens=1 (the prefill
     computes every prompt token's hidden; the runner hook dumps them).
  2. Poll DSPARK_HS_DIR for the new hs_*.safetensors.
  3. Load each and assert the standardized 4-key layout + shapes:
        hidden_states               [seq, num_layers * hidden_size]   (aux [40,41,42])
        verifier_last_hidden_states [seq, hidden_size]                (post-norm final)
        input_ids                   [seq]  (long)
        loss_mask                   [seq]  (bool)
     seq must equal the prompt's token count.

Exit code 0 = all captured files well-formed; non-zero otherwise.
"""

import argparse
import glob
import json
import os
import sys
import time
import urllib.request

DEFAULT_PROMPTS = [
    "The quick brown fox jumps over the lazy dog, and then it keeps running.",
    "深度学习模型通过反向传播来更新参数，从而不断降低损失函数的值。",
]


def _post_completion(base_url, model, prompt, req_id, timeout):
    """Send one prefill-only completion; return (ok, prompt_tokens or err)."""
    body = json.dumps(
        {"model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0}
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json", "X-Request-Id": req_id},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out = json.loads(resp.read())
        # usage.prompt_tokens = how many tokens the prefill saw = expected seq len
        return True, int(out.get("usage", {}).get("prompt_tokens", -1))
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def _load_and_check(path, hidden_size, num_layers, expect_seq):
    import torch  # local import so --help works without torch
    from safetensors.torch import load_file

    d = load_file(path)
    keys = set(d.keys())
    want = {"hidden_states", "verifier_last_hidden_states", "input_ids", "loss_mask"}
    problems = []
    if not want.issubset(keys):
        problems.append(f"missing keys: {sorted(want - keys)} (have {sorted(keys)})")
        return problems  # can't check shapes without the keys

    seq = d["input_ids"].shape[0]
    hs = tuple(d["hidden_states"].shape)
    vl = tuple(d["verifier_last_hidden_states"].shape)
    lm = tuple(d["loss_mask"].shape)

    if hs != (seq, num_layers * hidden_size):
        problems.append(f"hidden_states {hs} != ({seq}, {num_layers}*{hidden_size})")
    if vl != (seq, hidden_size):
        problems.append(f"verifier_last_hidden_states {vl} != ({seq}, {hidden_size})")
    if lm != (seq,):
        problems.append(f"loss_mask {lm} != ({seq},)")
    if d["input_ids"].dtype != torch.long:
        problems.append(f"input_ids dtype {d['input_ids'].dtype} != long")
    if expect_seq is not None and expect_seq >= 0 and seq != expect_seq:
        problems.append(f"seq {seq} != prompt_tokens {expect_seq}")
    if not problems:
        print(
            f"  OK {os.path.basename(path)}: seq={seq}  "
            f"hidden_states={hs}  verifier_last={vl}  "
            f"dtypes={d['hidden_states'].dtype}/{d['verifier_last_hidden_states'].dtype}"
        )
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("HS_SMOKE_HOST", "localhost"))
    ap.add_argument("--port", default=os.environ.get("API_PORT", "7000"))
    ap.add_argument("--model", default="dsv4")
    ap.add_argument("--hs-dir", default=os.environ.get("DSPARK_HS_DIR", ""))
    ap.add_argument("--hidden-size", type=int, default=int(os.environ.get("HS_HIDDEN_SIZE", "4096")))
    ap.add_argument("--num-layers", type=int, default=int(os.environ.get("HS_NUM_LAYERS", "3")),
                    help="len(dspark_target_layer_ids), default 3 for [40,41,42]")
    ap.add_argument("--n", type=int, default=len(DEFAULT_PROMPTS), help="number of smoke requests")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--settle", type=float, default=8.0, help="seconds to wait for files to land")
    args = ap.parse_args()

    if not args.hs_dir:
        print("!! --hs-dir (or DSPARK_HS_DIR) is required — the serve's dump directory")
        return 2
    base_url = f"http://{args.host}:{args.port}"
    prompts = (DEFAULT_PROMPTS * ((args.n // len(DEFAULT_PROMPTS)) + 1))[: args.n]

    before = set(glob.glob(os.path.join(args.hs_dir, "*.safetensors")))
    print(f">>> serve={base_url} model={args.model}  hs_dir={args.hs_dir}")
    print(f">>> sending {len(prompts)} prefill-only request(s) (max_tokens=1)…")

    expected = {}  # req_id -> prompt_tokens
    for i, p in enumerate(prompts):
        rid = f"hs_{i}"
        ok, info = _post_completion(base_url, args.model, p, rid, args.timeout)
        if ok:
            expected[rid] = info
            print(f"  req {rid}: prompt_tokens={info}")
        else:
            print(f"  req {rid}: FAILED — {info}")

    if not expected:
        print("!! no requests succeeded — is the serve up (>>> READY) and HS_DUMP=1 set?")
        return 3

    # files are written during the prefill; give the tmp+rename a moment to land.
    print(f">>> waiting up to {args.settle}s for hs_*.safetensors to appear…")
    deadline = time.time() + args.settle
    new = set()
    while time.time() < deadline:
        new = set(glob.glob(os.path.join(args.hs_dir, "*.safetensors"))) - before
        if len(new) >= len(expected):
            break
        time.sleep(1.0)

    if not new:
        print(f"!! NO new hs_*.safetensors in {args.hs_dir}.")
        print("!! checklist: HS_DUMP=1 on the serve? config has dspark_target_layer_ids "
              "(--hf-overrides)? writing rank = TP-rank-0 of this DP replica? DSPARK_HS_DIR shared?")
        return 4

    print(f">>> {len(new)} new file(s): {sorted(os.path.basename(x) for x in new)}")
    # map each new file back to its request's expected seq (by matching hs_<i> stem)
    all_problems = []
    for path in sorted(new):
        stem = os.path.basename(path).replace(".safetensors", "")
        exp = expected.get(stem, None)  # None if the serve renamed the id
        probs = _load_and_check(path, args.hidden_size, args.num_layers, exp)
        if probs:
            print(f"  BAD {os.path.basename(path)}:")
            for pr in probs:
                print(f"      - {pr}")
            all_problems.extend(probs)

    if all_problems:
        print(f"\n=== FAIL: {len(all_problems)} problem(s) across the captured files ===")
        return 5
    print("\n=== PASS: all captured hs_*.safetensors are well-formed (Plan B P2 smoke OK) ===")
    print(">>> next: load one through speculators ArrowDataset to confirm the trainer reads it, "
          "then drive the full rollout (P3).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
