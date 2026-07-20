#!/usr/bin/env python3
"""Benchmark the HS-sidecar transfer bandwidth between the SERVE box and the TRAINER box, in
isolation from training. Sweeps client concurrency to answer:
  * what's the MAX aggregate MB/s this HTTP(S) link + sidecar can push?
  * at what concurrency does it SATURATE (i.e. is --workers 8 / N client streams enough, or does
    more help)?  single-stream (concurrency 1) is your ~30 MB/s floor.

It fetches ONE persisted test file repeatedly through the real sidecar path, so the number is the
same transport the trainer's HS_FETCH_BASE uses.

── SERVER side (on the SERVE box, e.g. 182) — a SPARE port so it never touches the training
   sidecar on 9009, a ~100 MB file (≈ one seq-3072 HS), and --no-delete so it can be re-fetched:
     mkdir -p /tmp/hs_bench && head -c 100000000 /dev/urandom > /tmp/hs_bench/hs_0.safetensors
     HS_SIDECAR_TOKEN=bench python hs_sidecar.py \
         --root /tmp/hs_bench --port 9010 --workers 8 --no-delete &
   (to find the best server --workers, re-run this with 4 / 8 / 16 / 32 and repeat the client sweep.)

── CLIENT side (on the TRAINER box, e.g. 176) — bypass the proxy for the serve IP:
     no_proxy=<serve-ip> HS_SIDECAR_TOKEN=bench python hs_bench.py \
         --url http://<serve-ip>:9010 --path /tmp/hs_bench/hs_0.safetensors
   (HTTPS: --url https://... plus --insecure or REQUESTS_CA_BUNDLE=<cert>.)

Read the table: aggregate MB/s should climb with concurrency then FLATTEN — the flat value is the
link ceiling, and the knee is the concurrency worth running. If it's still climbing at the last
level, raise --levels.
"""
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import statistics
import threading
import time


def _worker(url, path, headers, verify, per, out, idx):
    import requests  # noqa: PLC0415

    s = requests.Session()
    got, lat = 0, []
    for _ in range(per):
        t0 = time.monotonic()
        r = s.get(url + "/hs", params={"path": path}, headers=headers,
                  verify=verify, stream=True, timeout=600)
        if r.status_code != 200:
            out[idx] = (0, [], f"HTTP {r.status_code}: {r.text[:80]}")
            return
        n = 0
        for chunk in r.iter_content(1 << 20):   # 1 MiB, real over-the-wire pull
            n += len(chunk)
        got += n
        lat.append(time.monotonic() - t0)
    out[idx] = (got, lat, None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="sidecar base, e.g. http://182:9010")
    ap.add_argument("--path", required=True, help="the served file key (= its abs path under --root)")
    ap.add_argument("--token", default=os.environ.get("HS_SIDECAR_TOKEN"))
    ap.add_argument("--insecure", action="store_true", help="skip TLS verify (else $REQUESTS_CA_BUNDLE)")
    ap.add_argument("--levels", default="1,2,4,8,16,32,64",
                    help="comma-separated concurrency levels to sweep (default 1..64)")
    ap.add_argument("--per", type=int, default=3,
                    help="fetches PER connection at each level (data moved = level*per*filesize)")
    args = ap.parse_args()

    url = args.url.rstrip("/")
    headers = {"X-HS-Token": args.token} if args.token else {}
    verify = not args.insecure
    levels = [int(x) for x in args.levels.split(",") if x.strip()]

    # one warm-up fetch to size the file + prime the path
    import requests  # noqa: PLC0415
    w = requests.get(url + "/hs", params={"path": args.path}, headers=headers,
                     verify=verify, stream=True, timeout=600)
    if w.status_code != 200:
        print(f"!! warm-up failed: HTTP {w.status_code}: {w.text[:200]}")
        print("   (server up on that port? token match? --insecure/CA for https? file present?)")
        return 2
    fsize = sum(len(c) for c in w.iter_content(1 << 20))
    print(f">>> {url}   file={args.path}   size={fsize/1e6:.1f} MB   token={'on' if args.token else 'off'}"
          f"   verify={'off' if args.insecure else 'on'}   per-conn={args.per}")
    print(f">>> single-stream floor is the concurrency=1 row; ceiling is where aggregate flattens.\n")
    print(f"{'concurrency':>11} | {'aggregate MB/s':>15} | {'per-conn MB/s':>14} | "
          f"{'p50 lat s':>10} | {'data GB':>8}")
    print("-" * 72)

    for c in levels:
        out = [None] * c
        ths = [threading.Thread(target=_worker,
                                args=(url, args.path, headers, verify, args.per, out, i))
               for i in range(c)]
        t0 = time.monotonic()
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        dt = time.monotonic() - t0
        errs = [o[2] for o in out if o and o[2]]
        if errs:
            print(f"{c:>11} | ERROR: {errs[0]}")
            continue
        total = sum(o[0] for o in out)
        lats = [x for o in out for x in o[1]]
        agg = total / 1e6 / dt
        p50 = statistics.median(lats) if lats else 0.0
        print(f"{c:>11} | {agg:>15.1f} | {agg / c:>14.1f} | {p50:>10.2f} | {total/1e9:>8.2f}")

    print("\n>>> knee (aggregate stops climbing) = the concurrency worth running; the flat value = "
          "the link ceiling. If HS ~100 MB/file, needed rows/s ≈ your step's batch size / step_ms.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
