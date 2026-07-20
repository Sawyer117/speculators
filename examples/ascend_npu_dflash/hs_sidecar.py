#!/usr/bin/env python3
"""HS sidecar — serve the DSpark HS-dump files to a REMOTE trainer over HTTP(S) (NO shared FS).

Pairs with the client in ``speculators/train/data.py`` (the ``HS_FETCH_BASE`` path). When the serve
box and the trainer box do NOT share a filesystem (our A3 182-infer / 176-train split), the trainer
cannot read the ``hs_<idx>.safetensors`` files the serve dumps. This sidecar bridges that gap:

  * SERVE box (182): the ``DSPARK_HS_DUMP`` serve writes ``hs_<idx>.safetensors`` into ``DSPARK_HS_DIR``
    (local disk); THIS sidecar exposes that dir over HTTP:  ``GET /hs?path=<abs_path>`` → raw
    safetensors bytes, then DELETES the file (the on_generate=delete rolling buffer, done server-side
    → peak disk ≈ in-flight files).
  * TRAINER box (176): set ``HS_FETCH_BASE=http(s)://182:<port>``. The trainer drives the prefill via
    ``--vllm-endpoint`` (→ serve dumps the file), then GETs its bytes straight into memory — nothing
    lands on the trainer's disk. ``hidden_states_path`` on the trainer MUST equal the serve's
    ``DSPARK_HS_DIR`` string (it is passed through as an opaque key so this sidecar can resolve it).

SECURITY: ``path`` is an opaque key from the client; we ONLY serve files that (a) resolve UNDER
``--root`` and (b) match ``hs_<digits>.safetensors`` — never arbitrary paths. Optional shared secret
(``--token`` / ``$HS_SIDECAR_TOKEN``, checked vs the client's ``X-HS-Token`` header). Optional TLS.

    # HTTP (trusted internal link — simplest; matches the client's pooled http:// adapter):
    python hs_sidecar.py --root "$DSPARK_HS_DIR" --port 9009

    # HTTPS (self-signed): make a cert, run with TLS, and on the TRAINER set
    #   REQUESTS_CA_BUNDLE=<cert.pem>   (requests then verifies it — NO client code change needed)
    openssl req -x509 -newkey rsa:2048 -nodes -keyout hs.key -out hs.crt -days 365 -subj "/CN=<serve-ip>"
    python hs_sidecar.py --root "$DSPARK_HS_DIR" --port 9009 --certfile hs.crt --keyfile hs.key
"""
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import re
import signal
import ssl
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

_HS_NAME = re.compile(r"^hs_\d+\.safetensors$")


def make_handler(root: Path, token, delete: bool, wait_ms: int):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"          # keep-alive (client pools connections)

        def _reply(self, code, body=b"", ctype="text/plain"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def log_message(self, *a):             # quiet by default
            pass

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/health":
                return self._reply(200, b"ok")
            if u.path != "/hs":
                return self._reply(404, b"not found")
            if token and self.headers.get("X-HS-Token") != token:
                return self._reply(403, b"bad token")
            q = parse_qs(u.query).get("path", [])
            if not q:
                return self._reply(400, b"missing path")
            try:
                p = Path(q[0]).resolve()
            except (OSError, ValueError):
                return self._reply(400, b"bad path")
            # confine to root + only hs_<n>.safetensors (never arbitrary file read)
            if not _HS_NAME.match(p.name) or root not in p.parents:
                return self._reply(403, b"path not allowed")
            # brief poll for a just-triggered dump to flush (prefill+dump is usually done already)
            deadline = time.monotonic() + wait_ms / 1000.0
            while not p.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            if not p.exists():
                return self._reply(404, b"not ready")
            try:
                size = p.stat().st_size
            except OSError as e:
                return self._reply(500, str(e).encode())
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(size))
            self.end_headers()
            # STREAM in chunks — do NOT read the whole ~100 MB into memory: with many concurrent
            # fetches that is GBs of allocation churn per process. If a write raises (client hung
            # up) we keep the file (return before the unlink).
            try:
                with open(p, "rb") as fh:
                    while True:
                        chunk = fh.read(1 << 20)   # 1 MiB
                        if not chunk:
                            break
                        self.wfile.write(chunk)
            except OSError:
                return
            if delete:
                try:
                    p.unlink()
                except OSError:
                    pass

    return H


def _serve(httpd, pidfile=None, child_pids=()):
    """serve_forever with graceful SIGTERM/SIGINT shutdown. The parent forwards the signal to its
    prefork children then reaps them; a child just stops itself. ``pkill -f hs_sidecar.py`` signals
    every process directly; ``kill $(cat pidfile)`` hits the parent, which fans out to the children."""
    def _graceful(signum, _frame):
        for cp in child_pids:
            try:
                os.kill(cp, signal.SIGTERM)
            except OSError:
                pass
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _graceful)
    signal.signal(signal.SIGINT, _graceful)
    if pidfile:
        with open(pidfile, "w") as f:
            f.write(str(os.getpid()))
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
        for cp in child_pids:
            try:
                os.waitpid(cp, 0)
            except OSError:
                pass
        if pidfile:
            try:
                os.remove(pidfile)
            except OSError:
                pass
            print(">>> hs_sidecar: stopped", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="dir the serve dumps into (= DSPARK_HS_DIR); ONLY files under it are served")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9009)
    ap.add_argument("--token", default=os.environ.get("HS_SIDECAR_TOKEN"),
                    help="shared secret; client sends it as X-HS-Token (default $HS_SIDECAR_TOKEN)")
    ap.add_argument("--certfile", help="TLS cert (enables HTTPS)")
    ap.add_argument("--keyfile", help="TLS key (with --certfile)")
    ap.add_argument("--no-delete", action="store_true",
                    help="keep files after sending (default: delete after send = rolling buffer)")
    ap.add_argument("--wait-ms", type=int, default=3000,
                    help="poll this long for a not-yet-flushed file before returning 404")
    ap.add_argument("--pidfile", help="write my PID here on start, remove on clean exit "
                    "(for `kill $(cat pidfile)` graceful shutdown)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("HS_SIDECAR_WORKERS", "1")),
                    help="prefork this many processes sharing the listen socket (default 1 / "
                    "$HS_SIDECAR_WORKERS). >1 lets many concurrent ~100 MB transfers run in TRUE "
                    "parallel (one Python process can't — GIL/memory), reclaiming the multi-stream "
                    "bandwidth a single-stream sidecar leaves on the table. Use 8 for the A3 remote link.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        sys.exit(f"!! --root is not a directory: {root}")
    if bool(args.certfile) ^ bool(args.keyfile):
        sys.exit("!! --certfile and --keyfile must be given together")

    httpd = ThreadingHTTPServer((args.host, args.port),
                                make_handler(root, args.token, not args.no_delete, args.wait_ms))
    scheme = "http"
    if args.certfile:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(args.certfile, args.keyfile)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        scheme = "https"

    workers = max(1, args.workers)
    # PREFORK: the socket is bound ONCE (above); fork workers-1 children that all serve_forever()
    # on the SHARED listening socket (the OS load-balances accept() across them). One Python
    # process can't push many concurrent ~100 MB transfers in parallel (GIL + memory churn); N
    # processes reclaim the multi-stream bandwidth a single-stream sidecar leaves unused.
    child_pids = []
    for _ in range(workers - 1):
        pid = os.fork()
        if pid == 0:
            _serve(httpd)                    # child: no pidfile, no children of its own
            os._exit(0)
        child_pids.append(pid)

    print(f">>> hs_sidecar {scheme}://{args.host}:{args.port}/hs   root={root}   "
          f"token={'on' if args.token else 'off'}   delete={'off' if args.no_delete else 'on'}   "
          f"workers={workers}   pid={os.getpid()}", flush=True)
    _serve(httpd, pidfile=args.pidfile, child_pids=child_pids)


if __name__ == "__main__":
    main()
