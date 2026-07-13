#!/usr/bin/env bash
# Terminal 1 (background) — vLLM serves the Qwen3-4B target + extracts hidden states
# for online DFlash training on Ascend NPU. Launched with nohup so it survives SSH
# disconnect (must outlive the train job, which depends on this endpoint).
#
# Graph mode by default; fall back to eager:  ENFORCE_EAGER=1 bash serve_qwen3_4b_nohup.sh
# Run with bash, do NOT source.
[ "${BASH_SOURCE[0]}" != "$0" ] && { echo "Run with 'bash $0', do not source."; return 1 2>/dev/null; }
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/config_qwen3_4b.sh"

EAGER_FLAG=""
[ "${ENFORCE_EAGER:-0}" = "1" ] && EAGER_FLAG="--enforce-eager"

# Clears any stale hidden states before serving (same as foreground serve script).
rm -rf "$HS_DIR" && mkdir -p "$HS_DIR"

export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0

LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S 2>/dev/null || echo "$$")"
LOG_FILE="$LOG_DIR/serve_4b_${STAMP}.log"
PID_FILE="$LOG_DIR/serve_4b.pid"

echo ">>> nohup serve Qwen3-4B on NPU $SERVE_CARDS (TP=$TP DP=$VLLM_DP port=$PORT max-model-len=$MAX_MODEL_LEN gpu-mem=$GPU_MEM_UTIL eager=${ENFORCE_EAGER:-0})"
echo ">>> HS_DIR=$HS_DIR"
echo ">>> log -> $LOG_FILE"

# Record the effective run config as the log's FIRST line (self-documenting).
echo ">>> RUN serve Qwen3-4B | card=$SERVE_CARDS TP=$TP DP=$VLLM_DP port=$PORT max-model-len=$MAX_MODEL_LEN gpu-mem=$GPU_MEM_UTIL eager=${ENFORCE_EAGER:-0} | HS_DIR=$HS_DIR" > "$LOG_FILE"

nohup env ASCEND_RT_VISIBLE_DEVICES="$SERVE_CARDS" python "$REPO_ROOT/scripts/launch_vllm.py" \
  "$TARGET_MODEL" \
  --target-layer-ids 1 9 17 25 33 \
  --hidden-states-path "$HS_DIR" \
  -- --tensor-parallel-size "$TP" --data-parallel-size "$VLLM_DP" --port "$PORT" \
     --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM_UTIL" $EAGER_FLAG \
  >> "$LOG_FILE" 2>&1 &

SERVE_PID=$!
echo "$SERVE_PID" > "$PID_FILE"
echo ">>> started, pid=$SERVE_PID (saved to $PID_FILE)"
echo ">>> wait for ready:  curl -s --noproxy '*' http://localhost:$PORT/v1/models | head"
echo ">>> follow:  tail -f $LOG_FILE"
echo ">>> stop:    pkill -f 'launch_vllm|EngineCore|APIServer'   # vLLM forks/retitles; kill the whole family (saved PID alone may miss EngineCore)"
