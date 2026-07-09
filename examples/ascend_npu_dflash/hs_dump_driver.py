#!/usr/bin/env python3
"""Plan B P3 — drive the full HS dump over a prepped Arrow dataset.

Reads the preprocessed rollout Arrow dataset (the SAME one the trainer's ArrowDataset
loads, with per-row ``input_ids``) and, for each row, sends its ``input_ids`` to the
HS_DUMP serve as a token-id prompt with ``max_tokens=1`` (prefill-only). The serve-side
dumper writes ``hs_<index>.safetensors`` (extract-connector format) to DSPARK_HS_DIR.

Why token-id prompts (not text): the trainer's ``check_hidden_states`` asserts the file's
``token_ids`` equal the dataset row's ``input_ids``. Sending the row's ``input_ids`` as the
prompt makes vLLM prefill exactly those tokens (no re-tokenize / BOS), so they match by
construction — and ``loss_mask`` needs NO producer work (it stays in the Arrow dataset).

Resume is by FILE EXISTENCE: a row is skipped iff ``hs_<index>.safetensors`` already
exists (a failed request writes nothing, so it is naturally retried) — no error-row
pollution. Shard across machines with --start/--end.

Run on a node that can reach the serve endpoint AND the shared DSPARK_HS_DIR:

    python hs_dump_driver.py \
        --datapath /share/.../open_perfectblend.dsv4_rollout/arrow \
        --hs-dir   /share/.../dsv4_hs_dump \
        --endpoint http://127.0.0.1:7000/v1 --model dsv4 --concurrency 32
"""

import argparse
import asyncio
import os
import sys
import time


def _load_input_ids(datapath):
    from datasets import load_from_disk

    data = load_from_disk(datapath)
    return data


def _row_ids(data, i):
    ids = data[i]["input_ids"]
    return ids.tolist() if hasattr(ids, "tolist") else list(ids)


async def _amain(args):
    from openai import AsyncOpenAI
    from tqdm import tqdm

    data = _load_input_ids(args.datapath)
    n = len(data)
    start = args.start
    end = args.end if args.end is not None else n
    end = min(end, n)
    if start >= end:
        print(f"!! empty range [{start}, {end}) over {n} rows")
        return 2
    rows = range(start, end)
    span = len(rows)

    def hs_path(i):
        return os.path.join(args.hs_dir, f"hs_{i}.safetensors")

    done0 = sum(1 for i in rows if os.path.exists(hs_path(i)))
    print(
        f">>> dataset={args.datapath}  rows={n}  range=[{start},{end})  "
        f"already done={done0}  to do={span - done0}"
    )
    print(f">>> serve={args.endpoint}  model={args.model}  hs_dir={args.hs_dir}  conc={args.concurrency}")

    client = AsyncOpenAI(base_url=args.endpoint, api_key="EMPTY", max_retries=0)
    it = iter(rows)
    it_lock = asyncio.Lock()
    stats = {"ok": 0, "skip": 0, "err": 0}
    progress = tqdm(total=span, initial=done0, desc="Dumping HS", unit="row", dynamic_ncols=True)

    async def worker():
        while True:
            async with it_lock:
                i = next(it, None)
            if i is None:
                return
            if args.resume and os.path.exists(hs_path(i)):
                stats["skip"] += 1  # already counted in `initial`
                continue
            try:
                await client.completions.create(
                    model=args.model,
                    prompt=_row_ids(data, i),
                    max_tokens=1,
                    extra_headers={"X-Request-Id": f"hs_{i}"},
                    extra_body={"return_token_ids": True},
                    timeout=args.timeout,
                )
                stats["ok"] += 1
                progress.update(1)
            except Exception as e:  # noqa: BLE001
                stats["err"] += 1
                if stats["err"] <= 5:
                    progress.write(f"  row {i} FAILED: {e!r}")
            progress.set_postfix(ok=stats["ok"], err=stats["err"], refresh=False)

    t0 = time.time()
    await asyncio.gather(*[worker() for _ in range(args.concurrency)])
    progress.close()
    dt = max(time.time() - t0, 1e-9)
    print(
        f"\n>>> done: {stats['ok']} written, {stats['skip']} already-had, {stats['err']} errors "
        f"in {dt:.1f}s ({stats['ok'] / dt:.2f} row/s)"
    )
    if stats["err"]:
        print(f"!! {stats['err']} rows errored (wrote nothing) — re-run to retry them "
              f"(resume skips the ones that landed). If errors persist, lower --concurrency.")
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datapath", required=True, help="preprocessed Arrow dataset dir (load_from_disk)")
    ap.add_argument("--hs-dir", default=os.environ.get("DSPARK_HS_DIR", ""),
                    help="serve's DSPARK_HS_DIR (for resume-by-existence + naming target)")
    ap.add_argument("--endpoint", default="http://127.0.0.1:7000/v1", help="OpenAI base URL")
    ap.add_argument("--model", default="dsv4")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--start", type=int, default=0, help="first row index (for sharding)")
    ap.add_argument("--end", type=int, default=None, help="end row index exclusive (default: all)")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="do NOT skip rows whose hs_<i>.safetensors already exists")
    args = ap.parse_args()

    if not args.hs_dir:
        print("!! --hs-dir (or DSPARK_HS_DIR) required")
        return 2
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
