#!/usr/bin/env bash
# Terminal 1 — vLLM serves the Qwen3-4B target and extracts hidden states for
# online DFlash training on Ascend NPU (single machine, TP=1 by default).
# Run with bash, do NOT source.
#
# Graph mode by default; fall back to eager:  ENFORCE_EAGER=1 bash serve_qwen3_4b.sh
[ "${BASH_SOURCE[0]}" != "$0" ] && { echo "Run with 'bash $0', do not source."; return 1 2>/dev/null; }
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/config_qwen3_4b.sh"

EAGER_FLAG=""
[ "${ENFORCE_EAGER:-0}" = "1" ] && EAGER_FLAG="--enforce-eager"

rm -rf "$HS_DIR" && mkdir -p "$HS_DIR"

export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0

echo ">>> serve Qwen3-4B on NPU $SERVE_CARDS (TP=$TP port=$PORT max-model-len=$MAX_MODEL_LEN gpu-mem=$GPU_MEM_UTIL eager=${ENFORCE_EAGER:-0}); HS_DIR=$HS_DIR"
ASCEND_RT_VISIBLE_DEVICES="$SERVE_CARDS" python "$REPO_ROOT/scripts/launch_vllm.py" \
  "$TARGET_MODEL" \
  --target-layer-ids 1 9 17 25 33 \
  --hidden-states-path "$HS_DIR" \
  -- --tensor-parallel-size "$TP" --port "$PORT" \
     --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM_UTIL" $EAGER_FLAG
