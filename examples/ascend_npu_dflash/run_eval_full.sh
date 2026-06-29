#!/usr/bin/env bash
# One-command DFlash eval, single terminal: backgrounds run_server.sh, waits until
# it's ready, runs run_eval.sh, then stops the serve. No two-terminal flow.
# FORK-ONLY team eval — not upstream.
#
# Export DRAFT (and optionally TARGET_MODEL / EVAL_MAX_MODEL_LEN / DATASET / ...) first,
# source CANN, then:
#   bash examples/ascend_npu_dflash/run_eval_full.sh
# Keep the serve running afterwards (to re-eval quickly) with KEEP_SERVE=1.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/config_qwen3_4b.sh"

EVAL_PORT="${EVAL_PORT:-30000}"
SERVE_LOG="${SERVE_LOG:-/tmp/eval_serve.log}"
READY_TRIES="${READY_TRIES:-180}"          # 180 * 5s = 15 min max wait for serve
KEEP_SERVE="${KEEP_SERVE:-0}"

echo ">>> serve (background) -> $SERVE_LOG"
bash "$SCRIPT_DIR/run_server.sh" > "$SERVE_LOG" 2>&1 &
SERVE_PID=$!

echo ">>> waiting for serve on :$EVAL_PORT (up to $((READY_TRIES*5))s) ..."
ready=0
for _ in $(seq 1 "$READY_TRIES"); do
  if curl -sf --noproxy '*' "http://localhost:$EVAL_PORT/v1/models" >/dev/null 2>&1; then
    ready=1; break
  fi
  # if the serve launcher already died, stop waiting and show why
  kill -0 "$SERVE_PID" 2>/dev/null || { echo ">>> serve exited early — tail $SERVE_LOG:"; tail -20 "$SERVE_LOG"; exit 1; }
  sleep 5
done
[ "$ready" = 1 ] || { echo ">>> serve not ready after $((READY_TRIES*5))s — tail $SERVE_LOG:"; tail -20 "$SERVE_LOG"; exit 1; }

echo ">>> serve ready:"
curl -s --noproxy '*' "http://localhost:$EVAL_PORT/v1/models" | head -c 400; echo

echo ">>> benchmark"
bash "$SCRIPT_DIR/run_eval.sh"
rc=$?

if [ "$KEEP_SERVE" = 1 ]; then
  echo ">>> KEEP_SERVE=1 — serve left running; stop later: pkill -f 'vllm serve|EngineCore'"
else
  echo ">>> stopping serve"
  pkill -9 -f "vllm serve|EngineCore|APIServer|Worker_TP" 2>/dev/null || true
fi
exit $rc
