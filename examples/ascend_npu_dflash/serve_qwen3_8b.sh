#!/usr/bin/env bash
# Terminal 1 — vLLM serves TARGET_MODEL and extracts hidden states for online
# DFlash training on Ascend NPU (single machine). See
# docs/deployment/ascend-npu-dflash-training.md (§5). Run with bash, do NOT source.
#
# Graph mode by default (faster; op-level fused kernels still apply). If vLLM errors
# during NPU graph capture, fall back to eager:  ENFORCE_EAGER=1 bash serve_qwen3_8b.sh
[ "${BASH_SOURCE[0]}" != "$0" ] && { echo "Run with 'bash $0', do not source."; return 1 2>/dev/null; }
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/config.sh"

EAGER_FLAG=""
[ "${ENFORCE_EAGER:-0}" = "1" ] && EAGER_FLAG="--enforce-eager"

rm -rf "$HS_DIR" && mkdir -p "$HS_DIR"

export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# TASK_QUEUE_ENABLE=1 (NOT 2): 2 conflicts with NPU graph capture (§3)
export TASK_QUEUE_ENABLE=1 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0

echo ">>> serve $TARGET_MODEL on NPU $SERVE_CARDS (TP=$TP port=$PORT eager=${ENFORCE_EAGER:-0}); HS_DIR=$HS_DIR"
ASCEND_RT_VISIBLE_DEVICES="$SERVE_CARDS" python "$REPO_ROOT/scripts/launch_vllm.py" \
  "$TARGET_MODEL" \
  --target-layer-ids 1 9 17 25 33 \
  --hidden-states-path "$HS_DIR" \
  -- --tensor-parallel-size "$TP" --port "$PORT" $EAGER_FLAG
