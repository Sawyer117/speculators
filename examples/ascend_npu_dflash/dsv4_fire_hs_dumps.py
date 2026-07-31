#!/usr/bin/env python3
"""Fire faithful HS-dump requests at the live DSpark HS-dump serve, at a chosen
CONCURRENCY, then poll+collect the written ``hs_<idx>.safetensors`` into an out dir.

This reproduces EXACTLY what the trainer's ``ArrowDataset._dump_generate_hs`` does
(``data.py:382-389``): a prefill-only ``completions.create`` with the same token-id
prompt (read straight from the training Arrow), ``max_tokens=1``,
``X-Request-Id=hs_<idx>``, ``return_token_ids=True``. The serve-side dumper writes
``hs_<idx>.safetensors`` to its ``DSPARK_HS_DIR``; nobody deletes it here (deletion is
trainer-side), so on a shared-FS serve the file persists and we copy it out.

Purpose: the HS train/serve-consistency test. Run once at ``--concurrency 1`` (clean
baseline) and once at the training over-subscription level (``NPROC*NUM_WORKERS``, e.g.
96) to the SAME rows, then diff with ``dsv4_hs_integrity_check.py``. A conc-1-clean /
conc-96-corrupt split == bf16 over-subscription garbage confirmed.

Usage (env or flags):
  ENDPOINT=http://<serve>:7000/v1 ARROW=<arrow_dir> HS_DIR=<serve DSPARK_HS_DIR> \
    python dsv4_fire_hs_dumps.py --out ~/hs_conc1 --n 8 --concurrency 1 --id-base 800000
"""
import argparse
import concurrent.futures as cf
import os
import shutil
import time


def _load_rows(arrow: str):
    from datasets import load_from_disk  # noqa: PLC0415 — box dep, imported lazily

    ds = load_from_disk(arrow)
    # arrow_0720_77w is a saved Dataset; guard against a DatasetDict just in case.
    if hasattr(ds, "keys") and not hasattr(ds, "num_rows"):
        ds = ds[next(iter(ds.keys()))]
    return ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=os.environ.get("ENDPOINT"), help="vLLM /v1 base url")
    ap.add_argument("--arrow", default=os.environ.get("ARROW"), help="training Arrow dataset dir")
    ap.add_argument("--hs-dir", default=os.environ.get("HS_DIR"),
                    help="the serve's DSPARK_HS_DIR (where hs_<idx>.safetensors land)")
    ap.add_argument("--out", required=True, help="copy the collected dumps here")
    ap.add_argument("--n", type=int, default=8, help="number of sequences to dump")
    ap.add_argument("--concurrency", type=int, default=1, help="simultaneous in-flight requests")
    ap.add_argument("--id-base", type=int, default=800000, help="hs_<id-base+i> tag (keep conc levels distinct)")
    ap.add_argument("--start-row", type=int, default=0, help="first Arrow row (use the SAME for each conc level)")
    ap.add_argument("--timeout", type=float, default=600.0, help="per-request + poll timeout (s)")
    args = ap.parse_args()

    for req in ("endpoint", "arrow", "hs_dir"):
        if not getattr(args, req):
            raise SystemExit(f"--{req.replace('_', '-')} (or env {req.upper()}) is required")

    import openai  # noqa: PLC0415 — box dep

    ds = _load_rows(args.arrow)
    client = openai.OpenAI(base_url=args.endpoint, api_key="EMPTY", max_retries=0)
    model = client.models.list().data[0].id
    print(f"serve model={model} | firing n={args.n} @ conc={args.concurrency} | id-base={args.id_base} "
          f"| rows [{args.start_row}, {args.start_row + args.n})")

    def fire(i: int):
        row = args.start_row + i
        client.completions.create(
            model=model,
            prompt=list(ds[row]["input_ids"]),
            max_tokens=1,
            extra_headers={"X-Request-Id": f"hs_{args.id_base + i}"},
            extra_body={"return_token_ids": True},
            timeout=args.timeout,
        )

    errs = 0
    t0 = time.monotonic()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(fire, i) for i in range(args.n)]
        for f in cf.as_completed(futs):
            try:
                f.result()
            except Exception as e:  # noqa: BLE001 — over-batch may 500; count, keep going
                errs += 1
                if errs <= 3:
                    print(f"  fire error: {e}")
    fired_s = time.monotonic() - t0

    os.makedirs(args.out, exist_ok=True)
    got = 0
    deadline = time.monotonic() + args.timeout
    for i in range(args.n):
        idx = args.id_base + i
        src = os.path.join(args.hs_dir, f"hs_{idx}.safetensors")
        while time.monotonic() < deadline and not os.path.exists(src):
            time.sleep(0.2)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, f"hs_{idx}.safetensors"))
            got += 1
    print(f"fired {args.n}@conc{args.concurrency} in {fired_s:.1f}s (errors={errs}); "
          f"collected {got}/{args.n} -> {args.out}")
    if got == 0:
        print("  ⚠ no files collected — check HS_DIR is the serve's DSPARK_HS_DIR (shared-FS) and the "
              "serve has DSPARK_HS_DUMP=1. In remote/sidecar mode the file is deleted after streaming.")


if __name__ == "__main__":
    main()
