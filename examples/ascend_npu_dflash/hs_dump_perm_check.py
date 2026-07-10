#!/usr/bin/env python3
"""Minimal DSPARK_HS_DUMP read/permission debug — NO training, NO model build.

Fires ONE prefill-only dump request at the serve (exactly like ArrowDataset's
_dump_generate_hs), waits for the HS file, then reports who owns it, its mode,
and whether THIS user/process can actually read it. Distinguishes the three
hypotheses cleanly:

  * file owned by a DIFFERENT user + mode 0600  -> cross-user PERMISSION problem
    (serve runs as user A, training as user B; A's umask writes 0600).
  * os.path.exists True but load fails "No such file" then succeeds on retry
    -> cross-node NFS dirent-vs-data visibility lag.
  * file NEVER appears -> DSPARK_HS_DIR mismatch / wrong mount / dumper off.

Run on the TRAINING node, as the TRAINING user (bypass the box proxy):

  no_proxy=127.0.0.1,localhost,80.5.5.115,80.5.5.116 \
  python examples/ascend_npu_dflash/hs_dump_perm_check.py \
    --endpoint http://80.5.5.115:7000/v1 --model dsv4 \
    --hs-dir /share/canada_group_folder/dataset/dsv4_hs_dump
"""
from __future__ import annotations

import argparse
import getpass
import grp
import os
import pwd
import time
import traceback


def _name(fn, gid_or_uid, default="?"):
    try:
        rec = fn(gid_or_uid)
        return rec.pw_name if hasattr(rec, "pw_name") else rec.gr_name
    except Exception:  # noqa: BLE001
        return default


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", default="dsv4")
    ap.add_argument("--hs-dir", required=True)
    # MUST be digits: the serve dumper names the file from the request id via the
    # regex hs_\d+, so a non-numeric tag (e.g. "permcheck") is never written.
    ap.add_argument("--tag", default="77777777")
    ap.add_argument("--wait", type=float, default=60.0)
    args = ap.parse_args()

    import openai  # noqa: PLC0415

    print(f">>> me: uid={os.getuid()} euid={os.geteuid()} user={getpass.getuser()}")

    # also stat the DIRECTORY itself (need x to stat files, r to list)
    try:
        dst = os.stat(args.hs_dir)
        print(
            f">>> hs-dir: {args.hs_dir}  uid={dst.st_uid}({_name(pwd.getpwuid, dst.st_uid)}) "
            f"gid={dst.st_gid}({_name(grp.getgrgid, dst.st_gid)}) mode={oct(dst.st_mode & 0o777)} "
            f"| me R_OK={os.access(args.hs_dir, os.R_OK)} X_OK={os.access(args.hs_dir, os.X_OK)} "
            f"W_OK={os.access(args.hs_dir, os.W_OK)}"
        )
    except Exception as e:  # noqa: BLE001
        print(f"!! cannot stat hs-dir: {e}")

    req_id = f"hs_{args.tag}"
    path = os.path.join(args.hs_dir, f"{req_id}.safetensors")
    print(f">>> firing dump request X-Request-Id={req_id} -> expect {path}")

    client = openai.OpenAI(base_url=args.endpoint, api_key="EMPTY", max_retries=0)
    client.completions.create(
        model=args.model,
        prompt=[1, 2, 3, 4, 5],
        max_tokens=1,
        extra_headers={"X-Request-Id": req_id},
        extra_body={"return_token_ids": True},
        timeout=120,
    )
    print(">>> request returned; polling os.path.exists ...")

    t0 = time.monotonic()
    seen = False
    while time.monotonic() - t0 < args.wait:
        if os.path.exists(path):
            seen = True
            break
        time.sleep(0.2)
    dt = time.monotonic() - t0

    if not seen:
        print(
            f"!! file NEVER appeared after {dt:.1f}s (os.path.exists False). "
            "=> NOT a permission issue. Check the serve's DSPARK_HS_DIR == --hs-dir, "
            "the mount on this node, and that the serve has HS_DUMP=1."
        )
        return

    print(f">>> os.path.exists True after {dt:.2f}s")
    st = os.stat(path)
    print(
        f">>> file: uid={st.st_uid}({_name(pwd.getpwuid, st.st_uid)}) "
        f"gid={st.st_gid}({_name(grp.getgrgid, st.st_gid)}) "
        f"mode={oct(st.st_mode & 0o777)} size={st.st_size}"
    )
    print(f">>> os.access(path, R_OK) = {os.access(path, os.R_OK)}")

    try:
        with open(path, "rb") as f:
            f.read(16)
        print(">>> raw open('rb').read(16): OK")
    except Exception as e:  # noqa: BLE001
        print(
            f"!! raw open FAILED: {type(e).__name__}: {e} "
            f"(errno={getattr(e, 'errno', None)})  <== if EACCES/13 => PERMISSION"
        )

    try:
        from safetensors.torch import load_file  # noqa: PLC0415

        d = load_file(path)
        print(f">>> safetensors load_file OK: { {k: tuple(v.shape) for k, v in d.items()} }")
    except Exception as e:  # noqa: BLE001
        print(f"!! safetensors load_file FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()

    print(
        "\n>>> VERDICT: if the file's uid/user differs from mine AND mode is 0600 "
        "(or R_OK=False / open errno=13), it's a CROSS-USER PERMISSION problem: "
        "the serve writes as its own user with a private umask. Fixes: (A) serve "
        "umask 022 so dumps are 0644; (B) run hs_sidecar.py AS THE SERVE USER "
        "(it owns the files) and set HS_FETCH_BASE on the trainer; (C) shared group + g+r."
    )


if __name__ == "__main__":
    main()
