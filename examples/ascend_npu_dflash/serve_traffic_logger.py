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

BUILT FOR CODING HARNESSES (Claude Code, Codex, dsh) pointed at this server, which is a very
different shape of traffic from a chat box:
  * Claude Code speaks the ANTHROPIC protocol on /v1/messages and authenticates with an
    `x-api-key` header, not `Authorization: Bearer`. Accepting only Bearer rejects it outright.
  * Message content is a LIST OF BLOCKS ({type:text}, {type:tool_use}, {type:tool_result}),
    not a string, so naive text extraction yields nothing.
  * The system prompt and tool definitions are resent in full on EVERY turn, and one agent
    loop is dozens of turns. Logged verbatim that is gigabytes a day of the same bytes. The
    system prompt is therefore stored ONCE per distinct digest and referenced afterwards, and
    long fields are capped -- a log too big to open answers no questions.

WHO SENT IT. Three kinds of identity, and the record keeps them apart because they are not
equally trustworthy:
  * VERIFIED -- an API key the caller had to be given. With --keys the proxy becomes the auth
    point: each person gets their own token, the proxy checks it, labels the record with their
    name, and forwards using the upstream key (if any). vLLM itself needs no change. Only the
    label and a short digest are logged, never the token.
  * SELF-DECLARED -- the OpenAI `user` body field, X-User / X-Client headers. Free when clients
    bother to set them, and trivially spoofable. Recorded, never trusted.
  * OBSERVED -- client IP and User-Agent. Useful corroboration, but NAT collapses a whole team
    onto one address and one person moves between several.
Conversations are grouped too: every turn of a chat resends the history, so hashing the leading
system+first-user message gives a stable id across the turns of one conversation. Two people
opening with the exact same first message would collide; that is the price of not needing the
client to cooperate.

STREAMING is forwarded chunk by chunk, never buffered, so time-to-first-token stays real; the
text is reassembled for the log as it passes. Token counts come from the response's `usage`
when present -- for streaming the client must send stream_options {"include_usage": true},
otherwise the record says tokens are unknown instead of guessing from chunk counts.

USAGE
  python3 examples/ascend_npu_dflash/serve_traffic_logger.py            # :8901 -> :8900
  # per-person keys: one "token name" per line, blank lines and # comments ignored
  python3 examples/ascend_npu_dflash/serve_traffic_logger.py --keys /path/keys.txt
  # harnesses on other machines need a reachable bind:
  python3 ... --host 0.0.0.0 --keys /path/keys.txt --max-field 2000

HARNESS SETUP (each person uses their own key, so every record is attributable)
  Claude Code : ANTHROPIC_BASE_URL=http://<host>:8901  ANTHROPIC_AUTH_TOKEN=sk-alice-...
  Codex / SDK : OPENAI_BASE_URL=http://<host>:8901/v1  OPENAI_API_KEY=sk-alice-...
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
import hashlib
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
_keys: dict[str, str] = {}        # token -> person; empty = open access
_upstream_key = ""
_max_field = 4000                 # chars kept per logged text field
_seen_systems: set[str] = set()   # system-prompt digests already written out in full


def _short(s: str) -> str:
    """A stable 8-hex handle for a value we must never write down in full."""
    return hashlib.sha256(s.encode()).hexdigest()[:8]


def _clip(s, limit=None):
    """Keep a field readable without keeping all of it."""
    if not isinstance(s, str):
        return s
    lim = limit or _max_field
    return s if len(s) <= lim else s[:lim] + f"…[truncated, {len(s)} chars total]"


def _blocks_text(content) -> str:
    """Anthropic content is a list of typed blocks; OpenAI content is a string. Handle both,
    and note the non-text blocks rather than dropping them silently -- in an agent loop the
    tool traffic IS most of the conversation."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    out = []
    for b in content:
        if not isinstance(b, dict):
            continue
        kind = b.get("type")
        if kind == "text":
            out.append(b.get("text") or "")
        elif kind == "tool_use":
            out.append(f"[tool_use {b.get('name')}]")
        elif kind == "tool_result":
            out.append("[tool_result]")
    return "".join(out)


def _system_text(req) -> str:
    """/v1/messages carries `system` at the top level (string or block list); the OpenAI shape
    puts it in messages[0]."""
    if not isinstance(req, dict):
        return ""
    s = req.get("system")
    if s:
        return _blocks_text(s)
    msgs = req.get("messages") or []
    if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
        return _blocks_text(msgs[0].get("content"))
    return ""


def _summarise_request(req):
    """What is worth keeping from a harness request: the last turn in full-ish, the shape of
    everything before it, and a reference to the system prompt instead of another copy."""
    if not isinstance(req, dict):
        return None, None
    sys_txt = _system_text(req)
    sys_dig = _short(sys_txt) if sys_txt else None
    msgs = [m for m in (req.get("messages") or []) if isinstance(m, dict)]
    last = _clip(_blocks_text(msgs[-1].get("content"))) if msgs else None
    tools = req.get("tools") or []
    summary = {
        "model": req.get("model"),
        "turns": len(msgs),
        "last_message": last,
        "roles": [m.get("role") for m in msgs][-6:],
        "system_digest": sys_dig,
        "system_chars": len(sys_txt) or None,
        "n_tools": len(tools) if isinstance(tools, list) else None,
        "stream": bool(req.get("stream")),
        "max_tokens": req.get("max_tokens"),
    }
    # The system prompt is identical across every turn of a session; write it once and refer
    # to it by digest from then on.
    first_sight = None
    if sys_dig and sys_dig not in _seen_systems:
        _seen_systems.add(sys_dig)
        first_sight = {"digest": sys_dig, "text": _clip(sys_txt, 20000),
                       "tool_names": [t.get("name") for t in tools if isinstance(t, dict)][:64]}
    return summary, first_sight


def _conversation_id(req) -> str | None:
    """Stable across the turns of one chat: every turn resends the history, so the leading
    system + first user message is the part that does not change."""
    if not isinstance(req, dict):
        return None
    msgs = req.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return None
    seed = [_system_text(req)[:4000]]
    for m in msgs:
        if not isinstance(m, dict):
            continue
        seed.append(f"{m.get('role')}:{_blocks_text(m.get('content'))[:4000]}")
        if m.get("role") == "user":
            break                  # stop at the FIRST user turn; later turns vary
    return _short("|".join(seed))


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

        # --- identity -----------------------------------------------------------------
        # Bearer for OpenAI-shaped clients, x-api-key for Anthropic-shaped ones (Claude Code
        # sends only the latter), and a query parameter as the last resort for tools that
        # cannot set headers at all.
        auth = self.headers.get("Authorization") or ""
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        token = token or (self.headers.get("x-api-key") or "").strip()
        if not token and "api_key=" in self.path:
            token = re.search(r"api_key=([^&]+)", self.path).group(1)
        who = _keys.get(token) if _keys else None
        if _keys and who is None:
            msg = b'{"error":{"message":"unknown or missing API key","type":"invalid_request_error"}}'
            self.send_response(401); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg))); self.end_headers()
            self.wfile.write(msg)
            _log({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "path": self.path, "status": 401,
                  "identity": {"verified": None, "key_digest": _short(token) if token else None,
                               "client_ip": self.client_address[0],
                               "user_agent": self.headers.get("User-Agent")},
                  "error": "rejected: unknown API key"})
            return

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
            if _upstream_key:
                fwd.add_header("Authorization", f"Bearer {_upstream_key}")
                fwd.add_header("x-api-key", _upstream_key)
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

        _summary, _first_system = _summarise_request(req_json)
        raw = "".join(chunks)
        resp_json, usage, text = None, None, None
        try:
            resp_json = json.loads(raw)
        except Exception:
            pass
        if isinstance(resp_json, dict):
            usage = resp_json.get("usage")
            if "content" in resp_json and "choices" not in resp_json:
                text = _blocks_text(resp_json.get("content"))      # Anthropic /v1/messages
            else:
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
                # Anthropic streams content_block_delta {delta:{type:text_delta,text:...}}
                # and reports usage inside message_start / message_delta.
                if ev.get("type", "").startswith(("content_block", "message")):
                    d = ev.get("delta") or {}
                    parts.append(d.get("text") or "")
                    for holder in (ev.get("message") or {}, ev):
                        if isinstance(holder, dict) and holder.get("usage"):
                            usage = holder["usage"]
                    continue
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
            "identity": {
                "verified": who,                                   # from the API key; trustworthy
                "key_digest": _short(token) if token else None,    # never the token itself
                "declared_user": (req_json or {}).get("user")
                                 if isinstance(req_json, dict) else None,
                "declared_header": self.headers.get("X-User") or self.headers.get("X-Client"),
                "client_ip": self.client_address[0],
                "user_agent": self.headers.get("User-Agent"),
            },
            "conversation_id": _conversation_id(req_json),
            # Written once per distinct system prompt, then referenced by digest. Without this
            # an agent session logs the same 30 KB preamble on every one of its turns.
            "system_prompt_first_seen": _first_system,
            "request": _summary,
            "response_text": _clip(text),
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
    ap.add_argument("--keys", default=None, metavar="FILE",
                    help="'token name' per line. Enables auth: unknown tokens get 401, and every "
                         "record carries the verified person. Without it access is open and only "
                         "self-declared/observed identity is available.")
    ap.add_argument("--upstream-key", default="", metavar="KEY",
                    help="key to present to vLLM, if the engine itself was started with --api-key")
    ap.add_argument("--max-field", type=int, default=4000, metavar="N",
                    help="chars kept per logged text field (default 4000). Harness turns are "
                         "long; a log too big to open answers no questions.")
    ap.add_argument("--no-spec", action="store_true",
                    help="skip the /metrics reads (one extra local request per exchange)")
    args = ap.parse_args()

    global _keys, _upstream_key, _max_field
    _upstream_key = args.upstream_key
    _max_field = args.max_field
    if args.keys:
        with open(args.keys) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                tok, _, name = line.partition(" ")
                if tok:
                    _keys[tok] = name.strip() or "unnamed"
    _upstream = f"http://127.0.0.1:{args.upstream}"
    _want_spec = not args.no_spec
    path = args.log or os.path.join(os.getcwd(), f"traffic_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    _logfh = open(path, "a", buffering=1, encoding="utf-8")

    print("=" * 72)
    print(f" traffic logger  {args.host}:{args.listen}  ->  {_upstream}")
    print(f" 📋 {path}")
    print(f" spec attribution: {'on (exact only while a request is alone)' if _want_spec else 'off'}")
    print(f" auth: {f'{len(_keys)} key(s) -> ' + ', '.join(sorted(set(_keys.values()))) if _keys else 'OPEN (identity is self-declared/observed only)'}")
    print(f" field cap: {_max_field} chars   system prompts: stored once per digest")
    print(" harness env:  ANTHROPIC_BASE_URL / OPENAI_BASE_URL -> this port, key per person")
    print(" point clients at the LISTEN port; the engine port keeps working unlogged.")
    print("=" * 72)
    ThreadingHTTPServer((args.host, args.listen), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
