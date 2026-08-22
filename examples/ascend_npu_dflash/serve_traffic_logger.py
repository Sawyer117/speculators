#!/usr/bin/env python3
"""Transparent logging proxy in front of a vLLM serve: every request, every response, JSONL.

Sits on its own port and forwards everything to the engine untouched, so nothing about the
server changes and the recipe under test stays byte-identical. Point clients at the proxy
port instead of the engine port and the whole session is on disk afterwards.

WHY A PROXY AND NOT THE ENGINE'S OWN LOGGING
vLLM's request logging writes prose into the engine log, mixed with everything else, and does
not record what came back. What we actually want to answer later -- "what was asked, what did
it answer, how long did it take, how much of that was drafted" -- wants one structured record
per exchange.

⚠ THE HONEST LIMIT ON SPECULATION NUMBERS. vLLM's spec-decode counters are ENGINE-GLOBAL
cumulative totals on /metrics; the OpenAI response carries no per-request accept length. So a
/metrics delta around one request is that request's only when nothing else was in flight. The
proxy tracks concurrency and records `spec.attributable: false` when it was not alone, rather
than reporting a precise-looking number that is actually several requests blended. At
concurrency 1 -- how we benchmark -- every record is exact.

STREAMING is forwarded chunk by chunk, never buffered, so time-to-first-token stays real; the
text is reassembled for the log as it passes. Token counts come from the response's `usage`
when present -- for streaming the client must send stream_options {"include_usage": true},
otherwise the record says tokens are unknown instead of guessing from chunk counts.

USAGE
  python3 examples/ascend_npu_dflash/serve_traffic_logger.py            # :8901 -> :8900
  python3 examples/ascend_npu_dflash/serve_traffic_logger.py --listen 9000 --upstream 8900 \
      --log /home/a00652497/2026/dspark/logs/traffic_dspark5.jsonl
  # then point clients at :8901 instead of :8900

READING IT BACK
  jq -r 'select(.path|endswith("completions")) | [.ts,.latency_s,.usage.completion_tokens,
         .spec.accept_len] | @tsv' traffic.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Never proxy our own traffic through the box's outbound http_proxy: the request would leave
# the machine and come back as a 504 (already cost us a debugging round on this fleet).
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

_SPEC_KEYS = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)

_state = threading.Lock()
_inflight = 0
_logfh = None
_upstream = "http://127.0.0.1:8900"
_want_spec = True


def _log(rec: dict) -> None:
    line = json.dumps(rec, ensure_ascii=False)
    with _state:
        _logfh.write(line + "\n")
        _logfh.flush()          # flush per record: a crashed run must still leave its history


def _metrics() -> dict:
    if not _want_spec:
        return {}
    try:
        with _OPENER.open(_upstream + "/metrics", timeout=5) as r:
            raw = r.read().decode(errors="ignore")
    except Exception:
        return {}
    out = {}
    for k in _SPEC_KEYS:
        tot, seen = 0.0, False
        for m in re.finditer(rf"^{re.escape(k)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)$", raw, re.M):
            tot += float(m.group(1)); seen = True
        if seen:
            out[k] = tot
    return out


def _spec_record(before: dict, after: dict, alone: bool) -> dict:
    if not before or not after:
        return {"attributable": False, "reason": "no spec-decode counters (autoregressive?)"}
    d = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in _SPEC_KEYS}
    drafts = d.get("vllm:spec_decode_num_drafts_total", 0.0)
    dtok = d.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
    acc = d.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    rec = {"drafts": drafts, "draft_tokens": dtok, "accepted_tokens": acc,
           "attributable": bool(alone)}
    if not alone:
        rec["reason"] = "another request was in flight; the delta blends them"
    if drafts > 0:
        # accept_len counts the always-free bonus token, matching how the ledger reports it.
        rec["accept_len"] = 1.0 + acc / drafts
        if dtok:
            rec["acceptance_rate"] = acc / dtok
    return rec


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "vllm-traffic-logger"

    def log_message(self, *a):        # the JSONL is the log; keep stderr for real problems
        pass

    def _relay(self, method: str) -> None:
        global _inflight
        body = b""
        n = int(self.headers.get("Content-Length") or 0)
        if n:
            body = self.rfile.read(n)

        with _state:
            _inflight += 1
            alone = _inflight == 1
        peak = 1 if alone else 2
        before = _metrics() if alone else {}

        req_json = None
        try:
            req_json = json.loads(body) if body else None
        except Exception:
            pass

        t0 = time.perf_counter()
        ttft = None
        chunks: list[str] = []
        status = 0
        err = None
        try:
            fwd = urllib.request.Request(
                _upstream + self.path, data=body or None, method=method,
                headers={k: v for k, v in self.headers.items()
                         if k.lower() not in ("host", "connection", "content-length")},
            )
            if body:
                fwd.add_header("Content-Length", str(len(body)))
            with _OPENER.open(fwd, timeout=3600) as up:
                status = up.status
                self.send_response(status)
                streaming = "text/event-stream" in (up.headers.get("Content-Type") or "")
                for k, v in up.headers.items():
                    if k.lower() in ("transfer-encoding", "content-length", "connection"):
                        continue
                    self.send_header(k, v)
                # Forward chunked so streaming stays streaming: buffering to compute a
                # Content-Length would silently destroy time-to-first-token.
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                while True:
                    # read1, NOT read: read(n) on a buffered stream blocks until it has n
                    # bytes or the body ends, so SSE events pile up and get forwarded in one
                    # go at the end -- the client still sees every token, but time-to-first-
                    # token becomes total latency and the stream is no longer a stream. This
                    # was caught only because the proxy records ttft: it read exactly equal
                    # to latency. read1 returns as soon as any data is available.
                    buf = up.read1(8192)
                    if not buf:
                        break
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    chunks.append(buf.decode(errors="replace"))
                    self.wfile.write(f"{len(buf):X}\r\n".encode() + buf + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
        except urllib.error.HTTPError as e:                       # noqa: PERF203
            status, err = e.code, e.read().decode(errors="replace")
            self.send_response(status); self.send_header("Content-Length", str(len(err)))
            self.end_headers(); self.wfile.write(err.encode())
        except Exception as e:                                    # noqa: BLE001
            status, err = 502, f"{type(e).__name__}: {e}"
            self.send_response(502); self.send_header("Content-Length", str(len(err)))
            self.end_headers(); self.wfile.write(err.encode())

        dt = time.perf_counter() - t0
        after = _metrics() if alone else {}
        with _state:
            _inflight -= 1

        raw = "".join(chunks)
        resp_json, usage, text = None, None, None
        try:
            resp_json = json.loads(raw)
        except Exception:
            pass
        if isinstance(resp_json, dict):
            usage = resp_json.get("usage")
            ch = (resp_json.get("choices") or [{}])[0]
            text = ch.get("text") or (ch.get("message") or {}).get("content")
        elif raw.startswith("data:"):
            # SSE: stitch the deltas back together and pick up usage if the client asked for it
            parts = []
            for line in raw.splitlines():
                if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                    continue
                try:
                    ev = json.loads(line[6:])
                except Exception:
                    continue
                if ev.get("usage"):
                    usage = ev["usage"]
                c = (ev.get("choices") or [{}])[0]
                parts.append(c.get("text") or (c.get("delta") or {}).get("content") or "")
            text = "".join(parts)

        _log({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "path": self.path,
            "status": status,
            "latency_s": round(dt, 4),
            "ttft_s": round(ttft, 4) if ttft is not None else None,
            "streaming": raw.startswith("data:"),
            "peak_concurrency": peak,
            "request": req_json,
            "response_text": text,
            "usage": usage,
            "spec": _spec_record(before, after, alone),
            "error": err,
        })

    def do_POST(self): self._relay("POST")
    def do_GET(self): self._relay("GET")
    def do_DELETE(self): self._relay("DELETE")


def main() -> int:
    global _logfh, _upstream, _want_spec
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen", type=int, default=8901, help="port clients talk to")
    ap.add_argument("--upstream", type=int, default=8900, help="the vLLM serve port")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--log", default=None, help="JSONL path (default ./traffic_<ts>.jsonl)")
    ap.add_argument("--no-spec", action="store_true",
                    help="skip the /metrics reads (one extra local request per exchange)")
    args = ap.parse_args()

    _upstream = f"http://127.0.0.1:{args.upstream}"
    _want_spec = not args.no_spec
    path = args.log or os.path.join(os.getcwd(), f"traffic_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    _logfh = open(path, "a", buffering=1, encoding="utf-8")

    print("=" * 72)
    print(f" traffic logger  {args.host}:{args.listen}  ->  {_upstream}")
    print(f" 📋 {path}")
    print(f" spec attribution: {'on (exact only while a request is alone)' if _want_spec else 'off'}")
    print(" point clients at the LISTEN port; the engine port keeps working unlogged.")
    print("=" * 72)
    ThreadingHTTPServer((args.host, args.listen), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
