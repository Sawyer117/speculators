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
    # Force python (not torch) format so a row's input_ids is a plain list[int];
    # a persisted torch format makes ds[row]["input_ids"] a tensor -> list() on a
    # 0-d slice raises "iteration over a 0-d tensor".
    try:
        ds = ds.with_format(None)
    except Exception:  # noqa: BLE001
        pass
    return ds


def _dump_dir_candidates(hs_dir: str) -> list:
    """Where the serve might ACTUALLY dump — the DSPARK_HS_DIR often has a 'dataset/' layer
    or a different root than assumed. Try the given dir first, then common variants."""
    hs_dir = hs_dir.rstrip("/")
    base = os.path.basename(hs_dir)
    parent = os.path.dirname(hs_dir)
    cands = [hs_dir, os.path.join(parent, "dataset", base)]
    for root in ("/share/canada_group_folder", "/mnt/nfs/canada_group_folder", os.path.expanduser("~")):
        cands += [os.path.join(root, "dataset", base), os.path.join(root, base)]
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _locate_dump_dir(hs_dir: str, first_name: str, timeout: float):
    """Poll the candidate dirs for the first expected dump file (it exists once the request
    returns). Returns the dir that has it, or None on timeout."""
    cands = _dump_dir_candidates(hs_dir)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for d in cands:
            if os.path.exists(os.path.join(d, first_name)):
                return d
        time.sleep(0.3)
    return None


def _row_ids(ds, row: int, col: str) -> list:
    v = ds[row][col]
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, (int, float)):
        raise SystemExit(
            f"row {row} col '{col}' is a scalar {v!r} — wrong column? available: "
            f"{getattr(ds, 'column_names', '?')} (pass --col)"
        )
    return list(v)


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
    ap.add_argument("--col", default="input_ids", help="Arrow column holding the token ids")
    ap.add_argument("--timeout", type=float, default=600.0, help="per-request timeout (s)")
    ap.add_argument("--collect-timeout", type=float, default=90.0,
                    help="how long to wait for dumped files to appear before failing LOUD (s)")
    args = ap.parse_args()

    for req in ("endpoint", "arrow", "hs_dir"):
        if not getattr(args, req):
            raise SystemExit(f"--{req.replace('_', '-')} (or env {req.upper()}) is required")

    import openai  # noqa: PLC0415 — box dep

    ds = _load_rows(args.arrow)
    client = openai.OpenAI(base_url=args.endpoint, api_key="EMPTY", max_retries=0)
    model = client.models.list().data[0].id
    _probe = ds[args.start_row][args.col]
    print(f"serve model={model} | firing n={args.n} @ conc={args.concurrency} | id-base={args.id_base} "
          f"| rows [{args.start_row}, {args.start_row + args.n}) "
          f"| cols={getattr(ds, 'column_names', '?')} | {args.col} type={type(_probe).__name__} "
          f"shape={getattr(_probe, 'shape', None)}")

    def fire(i: int):
        row = args.start_row + i
        client.completions.create(
            model=model,
            prompt=_row_ids(ds, row, args.col),
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
    names = [f"hs_{args.id_base + i}.safetensors" for i in range(args.n)]
    # The request returns AFTER the serve dumps, so the files exist by now. Auto-locate the
    # real dump dir (candidates cover the common 'dataset/' layer / root mismatch), fail LOUD
    # instead of hanging for the whole request timeout on a wrong --hs-dir.
    src_dir = _locate_dump_dir(args.hs_dir, names[0], args.collect_timeout)
    if src_dir is None:
        raise SystemExit(
            f"\n⚠ 0/{args.n} dumps found within {args.collect_timeout}s. The serve dumps to its OWN "
            f"DSPARK_HS_DIR, which is NOT --hs-dir={args.hs_dir!r}.\n   Locate it, then pass --hs-dir "
            f"(or HS_DIR=):\n     find /share /home /mnt/nfs -name '{names[0]}' 2>/dev/null | head")
    if os.path.abspath(src_dir) != os.path.abspath(args.hs_dir):
        print(f"  ⚠ serve dumped to {src_dir} (NOT --hs-dir={args.hs_dir}); collecting from there. "
              f"Use HS_DIR={src_dir} next time.")
    got = 0
    deadline = time.monotonic() + args.collect_timeout
    for name in names:
        src = os.path.join(src_dir, name)
        while time.monotonic() < deadline and not os.path.exists(src):
            time.sleep(0.2)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, name))
            got += 1
    print(f"fired {args.n}@conc{args.concurrency} in {fired_s:.1f}s (errors={errs}); "
          f"collected {got}/{args.n} -> {args.out}  (dump dir: {src_dir})")


if __name__ == "__main__":
    main()
