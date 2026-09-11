#!/usr/bin/env python3
"""Pre-download the five eval datasets into the local HF cache, THROUGH a corporate proxy.

WHY THIS EXISTS. ``run_dspark_eval.sh`` runs with ``OFFLINE=1`` by design: the eval is a
multi-hour job and must not depend on the network once started. That only works if the
datasets are already cached. On a fresh box they are not, and getting them there behind a
corporate MITM proxy took three separate fixes -- each of which fails with an error that
points somewhere else. All three are baked in here.

⚠️ THE THREE TRAPS, in the order they fire (2026-09-12, a fresh A3):

1. ``HF_ENDPOINT`` set to the Huawei mirror -> ``HTTP Error 504`` on every HEAD.
   The mirror does not serve the datasets namespace. Direct + proxy is the working route.
   ⚠️⚠️ And do NOT "clear" it with ``export HF_ENDPOINT=''`` -- an EMPTY endpoint makes
   httpx raise ``UnsupportedProtocol: Request URL is missing an 'http://' or 'https://'
   protocol``, which reads like a bug in datasets and is not. ``unset`` it.

2. hf-xet. ``huggingface_hub`` defaults to the Xet/CAS chunked backend, which fetches from
   ``us.aws.cdn.hf.co`` -- a DIFFERENT host from ``huggingface.co``. The metadata request
   succeeds (so you get a signed URL) and the chunk fetch dies:
   ``File reconstruction error: CAS Client Error``. Symptom to recognise: the progress bar
   prints ``downloading bytes: 0.00B`` then ``reconstructing file: 0%``.
   Fix = ``HF_HUB_DISABLE_XET=1`` -> classic single-file download.

3. The proxy's self-signed certificate -> ``[SSL: CERTIFICATE_VERIFY_FAILED] self-signed
   certificate in certificate chain``. ``huggingface_hub`` >= 1.x uses **httpx**, so the
   old ``requests``-era knobs do nothing. httpx honours ``SSL_CERT_FILE`` (prefer that, see
   ``--ca-bundle``); ``--insecure`` forces ``verify=False`` on httpx's clients instead.

⚠️ A dataset that is ALREADY cached still returns OK while the network is broken --
``datasets`` silently falls back to the cache and prints "Using the latest cached version".
Do not read one OK line as proof the network works; judge by the whole table.

Usage
-----
    # preferred: point at the corporate CA, verification stays ON
    python fetch_eval_datasets.py --ca-bundle /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem

    # no CA available: skip verification for THIS process only
    python fetch_eval_datasets.py --insecure

Then run the eval with its default ``OFFLINE=1`` -- it reads the cache and never touches
the network again.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

# (repo_id args, load_kwargs, expected n) -- mirrors Evaluator.py's DATASETS registry.
JOBS = [
    (("openai/gsm8k", "main"), {"split": "test"}, 1319),
    (("HuggingFaceH4/MATH-500",), {"split": "test"}, 500),
    (("openai/openai_humaneval",), {"split": "test"}, 164),
    (("google-research-datasets/mbpp", "sanitized"), {"split": "test"}, 257),
    (("HuggingFaceH4/mt_bench_prompts",), {"split": "train"}, 80),
]


def _disable_httpx_verify() -> None:
    """Force verify=False on every httpx client created in THIS process."""
    import httpx

    for cls in (httpx.Client, httpx.AsyncClient):
        original = cls.__init__

        def make(orig):
            def patched(self, *args, **kwargs):
                kwargs["verify"] = False
                return orig(self, *args, **kwargs)

            return patched

        cls.__init__ = make(original)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ca-bundle", help="PEM with the corporate root CA; keeps verification ON (preferred)")
    ap.add_argument("--insecure", action="store_true", help="skip TLS verification for this process only")
    ap.add_argument("--keep-xet", action="store_true", help="do NOT set HF_HUB_DISABLE_XET (trap 2 -- rarely right)")
    args = ap.parse_args()

    # Trap 1: an empty HF_ENDPOINT is WORSE than an unset one.
    if os.environ.get("HF_ENDPOINT") == "":
        del os.environ["HF_ENDPOINT"]
    if os.environ.get("HF_ENDPOINT"):
        print(f"⚠  HF_ENDPOINT={os.environ['HF_ENDPOINT']} -- mirrors often 504 on /datasets; unset it if this fails")

    # This tool exists to POPULATE the cache, so offline mode must be off.
    for var in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.pop(var, None)

    # Trap 2.
    if not args.keep_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    # Trap 3.
    if args.ca_bundle:
        os.environ["SSL_CERT_FILE"] = args.ca_bundle
        os.environ["REQUESTS_CA_BUNDLE"] = args.ca_bundle
        print(f">>> TLS: verifying against {args.ca_bundle}")
    elif args.insecure:
        warnings.filterwarnings("ignore")
        _disable_httpx_verify()
        print(">>> TLS: verification DISABLED for this process (--insecure)")
    else:
        print(">>> TLS: default verification. Behind a MITM proxy expect CERTIFICATE_VERIFY_FAILED")
        print("    -> re-run with --ca-bundle <pem>  (preferred), or --insecure")

    proxy = os.environ.get("https_proxy") or os.environ.get("http_proxy")
    print(f">>> proxy: {'set' if proxy else 'NOT SET — source your proxy script first'}")

    from datasets import load_dataset  # noqa: PLC0415  (after the env is arranged)

    failures = 0
    for load_args, load_kwargs, expected in JOBS:
        name = load_args[0]
        try:
            ds = load_dataset(*load_args, **load_kwargs)
            mark = "OK  " if len(ds) == expected else "WARN"
            if len(ds) != expected:
                failures += 1
            print(f"{mark} {name:42} n={len(ds)} (expected {expected})")
        except Exception as exc:  # noqa: BLE001 -- report every dataset, never stop at the first
            failures += 1
            print(f"FAIL {name:42} {type(exc).__name__}: {str(exc)[:160]}")

    if failures:
        print(f"\n⚠  {failures} dataset(s) not ready — the eval will die at OFFLINE=1. See the traps in this file's docstring.")
        return 1
    print("\n✅ all five cached — run the eval with its default OFFLINE=1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
