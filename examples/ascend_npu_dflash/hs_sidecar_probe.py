#!/usr/bin/env python3
"""HS no-shared-FS round-trip probe — the ISOLATION test for the 182→176 HS transport.

This exercises EXACTLY what the trainer's ``HS_FETCH_BASE`` path (``speculators/train/data.py``)
does, but with NO model build and NO training — so a failure localises cleanly to the transport,
not the trainer. Run it AFTER (a) the serve is up with ``HS_DUMP=1`` on the SERVE box (182) and
(b) ``hs_sidecar.py`` is running there. Run the probe FROM THE TRAINER box (176), as the trainer
user, so it proves the real cross-box, cross-uid path end to end.

What it does, in one shot:
  1. FIRE a prefill-only ``/v1/completions`` at the serve (``--endpoint``, plain HTTP :7000) tagged
     ``X-Request-Id=hs_<tag>`` — identical to how ArrowDataset triggers a dump. The serve-side
     DsparkHSDumper writes ``hs_<tag>.safetensors`` into its ``DSPARK_HS_DIR``.
  2. FETCH ``GET {--fetch-base}/hs?path={--hs-dir}/hs_<tag>.safetensors`` from the sidecar
     (``--fetch-base``, HTTPS :9009), sending ``X-HS-Token`` — identical to ``_fetch_hs_remote``.
     TLS trust comes from ``$REQUESTS_CA_BUNDLE`` (set it to the sidecar cert), or pass ``--insecure``.
  3. VALIDATE the bytes: ``hidden_states`` is ``[seq, num_aux+1, H]`` (aux [40,41,42] + verifier-last),
     no NaN, ``token_ids`` is long and length == the prefill's prompt_tokens. H is REPORTED, not
     asserted (DSV4's hidden size is whatever it is — the probe tells you).
  4. RE-FETCH the same path → expect **404** = the sidecar deleted-after-send (the rolling buffer the
     trainer relies on). With a ``--no-delete`` sidecar, pass ``--expect-persist`` to accept 200.

Exit 0 = the whole no-shared-FS path works; wire the trainer. Non-zero + a reason otherwise.

    # on 182 (serve box): serve up with HS_DUMP=1, then start the sidecar (HTTPS):
    #   openssl req -x509 -newkey rsa:2048 -nodes -keyout hs.key -out hs.crt -days 365 -subj "/CN=182"
    #   HS_SIDECAR_TOKEN=s3cret python hs_sidecar.py --root "$DSPARK_HS_DIR" --port 9009 \
    #       --certfile hs.crt --keyfile hs.key
    # on 176 (trainer box), trainer user, bypass the box proxy for the two internal IPs:
    #   scp 182:.../hs.crt /tmp/hs.crt      # so REQUESTS_CA_BUNDLE can verify it
    #   no_proxy="$IP182" NO_PROXY="$IP182" REQUESTS_CA_BUNDLE=/tmp/hs.crt HS_SIDECAR_TOKEN=s3cret \
    #     python examples/ascend_npu_dflash/hs_sidecar_probe.py \
    #       --endpoint http://$IP182:7000 --fetch-base https://$IP182:9009 \
    #       --hs-dir /home/canada_group_folder/dataset/dsv4_hs_dump
"""
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import sys


def _ok(msg):
    print(f"  OK  {msg}")


def _fail(msg):
    print(f"  !!  {msg}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--endpoint", required=True,
                    help="serve base, e.g. http://182:7000 (or .../v1) — where the prefill is fired")
    ap.add_argument("--fetch-base", required=True,
                    help="sidecar base, e.g. https://182:9009 — where the HS bytes are fetched")
    ap.add_argument("--hs-dir", required=True,
                    help="the serve's DSPARK_HS_DIR string (opaque key; MUST match the serve exactly)")
    ap.add_argument("--model", default="dsv4")
    ap.add_argument("--tag", default="90909090",
                    help="numeric id → X-Request-Id=hs_<tag> → hs_<tag>.safetensors (pick one no real "
                         "training index will collide with)")
    ap.add_argument("--seq", type=int, default=32, help="prompt length in token ids (prefill size)")
    ap.add_argument("--num-aux", type=int, default=3, help="len(dspark_target_layer_ids); [40,41,42]→3")
    ap.add_argument("--token", default=os.environ.get("HS_SIDECAR_TOKEN"),
                    help="sidecar shared secret → X-HS-Token (default $HS_SIDECAR_TOKEN)")
    ap.add_argument("--insecure", action="store_true", help="skip TLS verify (else $REQUESTS_CA_BUNDLE)")
    ap.add_argument("--expect-persist", action="store_true",
                    help="sidecar was started --no-delete → 2nd fetch should be 200, not 404")
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    import requests  # noqa: PLC0415

    ep = args.endpoint.rstrip("/")
    completions = ep + "/completions" if ep.endswith("/v1") else ep + "/v1/completions"
    fetch_url = args.fetch_base.rstrip("/") + "/hs"
    file_path = f"{args.hs_dir.rstrip('/')}/hs_{args.tag}.safetensors"
    req_id = f"hs_{args.tag}"
    hdr_token = {"X-HS-Token": args.token} if args.token else {}
    verify = False if args.insecure else True  # requests honours $REQUESTS_CA_BUNDLE when verify=True

    print(f">>> endpoint   {completions}")
    print(f">>> fetch-base {fetch_url}   token={'on' if args.token else 'off'}   "
          f"verify={'off (insecure)' if args.insecure else os.environ.get('REQUESTS_CA_BUNDLE', 'system')}")
    print(f">>> file key   {file_path}  (X-Request-Id={req_id})")

    # ---- 1. fire the prefill-only dump (token-id prompt so seq is exact + tokenizer-free) ----
    prompt_ids = list(range(1, args.seq + 1))
    try:
        r = requests.post(
            completions,
            json={"model": args.model, "prompt": prompt_ids, "max_tokens": 1, "temperature": 0},
            headers={"Content-Type": "application/json", "X-Request-Id": req_id},
            timeout=args.timeout,
        )
        r.raise_for_status()
        prompt_tokens = int(r.json().get("usage", {}).get("prompt_tokens", -1))
    except Exception as e:  # noqa: BLE001
        _fail(f"prefill POST to serve failed: {e}")
        print("      → serve unreachable / not in HS_DUMP mode / proxy not bypassed (set no_proxy).")
        return 2
    _ok(f"prefill fired, serve saw prompt_tokens={prompt_tokens}")

    # ---- 2. fetch the dumped file from the sidecar over HTTPS (mirrors _fetch_hs_remote) ----
    try:
        g = requests.get(fetch_url, params={"path": file_path}, headers=hdr_token,
                         timeout=args.timeout, verify=verify)
    except Exception as e:  # noqa: BLE001
        _fail(f"sidecar GET failed (connection/TLS): {e}")
        print("      → cert mismatch (set REQUESTS_CA_BUNDLE=<sidecar cert> or --insecure), "
              "sidecar down, or wrong --fetch-base.")
        return 3
    if g.status_code == 403:
        _fail("sidecar 403 — bad/missing X-HS-Token, or path outside --root."); return 3
    if g.status_code == 404:
        _fail("sidecar 404 — file never appeared. Dumper off, or --hs-dir != serve DSPARK_HS_DIR, "
              "or wait-ms too short."); return 3
    if g.status_code != 200:
        _fail(f"sidecar HTTP {g.status_code}: {g.text[:200]}"); return 3
    payload = g.content
    _ok(f"fetched {len(payload):,} bytes over {args.fetch_base.split(':')[0]}")

    # ---- 3. validate the tensor payload (standalone: safetensors only, no speculators import) ----
    try:
        import torch  # noqa: PLC0415
        from safetensors.torch import load as st_load  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        _fail(f"cannot import torch/safetensors to validate: {e}"); return 4
    try:
        d = st_load(payload)
    except Exception as e:  # noqa: BLE001
        _fail(f"payload is not valid safetensors: {e}"); return 4

    problems = []
    if not {"hidden_states", "token_ids"}.issubset(d):
        _fail(f"missing keys; have {sorted(d)}"); return 4
    hs, ids = d["hidden_states"], d["token_ids"]
    seq = ids.shape[0]
    if hs.dim() != 3:
        problems.append(f"hidden_states dim {hs.dim()} != 3 {tuple(hs.shape)}")
    else:
        if hs.shape[0] != seq:
            problems.append(f"hidden_states seq {hs.shape[0]} != token_ids {seq}")
        if hs.shape[1] != args.num_aux + 1:
            problems.append(f"hidden_states layers {hs.shape[1]} != num_aux+1 {args.num_aux + 1}")
    if ids.dtype != torch.long:
        problems.append(f"token_ids dtype {ids.dtype} != long")
    if hs.isnan().any():
        problems.append("hidden_states contains NaN")
    if prompt_tokens >= 0 and seq != prompt_tokens:
        problems.append(f"seq {seq} != prompt_tokens {prompt_tokens}")
    if problems:
        for p in problems:
            _fail(p)
        return 4
    H = hs.shape[2]
    _ok(f"payload valid: seq={seq}  hidden_states=[{seq}, {args.num_aux}+1, {H}]  "
        f"dtype={hs.dtype}  (hidden_size H={H})")

    # ---- 4. re-fetch → the sidecar should have deleted it (rolling buffer the trainer relies on) ----
    try:
        g2 = requests.get(fetch_url, params={"path": file_path}, headers=hdr_token,
                         timeout=args.timeout, verify=verify)
    except Exception as e:  # noqa: BLE001
        _fail(f"second GET failed: {e}"); return 5
    if args.expect_persist:
        if g2.status_code == 200:
            _ok("re-fetch 200 — sidecar started --no-delete, file kept (as requested)")
        else:
            _fail(f"expected persist (200) but got {g2.status_code}"); return 5
    else:
        if g2.status_code == 404:
            _ok("re-fetch 404 — sidecar deleted-after-send ✔ (rolling buffer works)")
        else:
            _fail(f"expected 404 after delete-on-send, got {g2.status_code} — file NOT deleted; "
                  "disk will grow. Check sidecar delete path / permissions.")
            return 5

    print("\n>>> PASS — the no-shared-FS HS path (182 dump → sidecar → 176 fetch → delete) works. "
          "Wire the trainer: ENDPOINT=$endpoint  HS_FETCH_BASE=$fetch_base  "
          "HS_DIR/hidden_states_path=--hs-dir  (+ REQUESTS_CA_BUNDLE / HS_SIDECAR_TOKEN).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
