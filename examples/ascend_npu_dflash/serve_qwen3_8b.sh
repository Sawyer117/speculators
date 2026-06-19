#!/usr/bin/env bash
# Terminal 1 — vLLM serves the Qwen3-8B target and extracts hidden states for
# online DFlash training on Ascend NPU (single machine). See
# docs/deployment/ascend-npu-dflash-training.md (§5).
#
# Run it, do NOT source it:   bash serve_qwen3_8b.sh
# Override any path/knob via env, e.g.:  MODEL=/path/to/Qwen3-8B PORT=8000 bash serve_qwen3_8b.sh
#
# Graph mode is the default (faster; keeps op-level fused kernels). If vLLM errors
# during NPU graph capture, fall back to eager:  ENFORCE_EAGER=1 bash serve_qwen3_8b.sh

# safety: if accidentally sourced, bail without killing the shell
(return 0 2>/dev/null) && { echo "Run with 'bash $0', do not source."; return 1 2>/dev/null; }
set -euo pipefail

MODEL="${MODEL:-/share/canada_group_folder/ckpt/Qwen3-8B}"
WORKDIR="${WORKDIR:-/home/a00652497/2026/dflash-vllm}"
HS_DIR="${HS_DIR:-$WORKDIR/tmp/hs_qwen3_dflash}"
SERVE_CARDS="${SERVE_CARDS:-0,1}"
TP="${TP:-2}"
PORT="${PORT:-8000}"

# graph mode by default; set ENFORCE_EAGER=1 to skip NPU graph capture
EAGER_FLAG=""
[ "${ENFORCE_EAGER:-0}" = "1" ] && EAGER_FLAG="--enforce-eager"

cd "$WORKDIR"
rm -rf "$HS_DIR" && mkdir -p "$HS_DIR"

export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# TASK_QUEUE_ENABLE=1 (NOT 2): 2 conflicts with NPU graph capture (§3)
export TASK_QUEUE_ENABLE=1 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0

echo ">>> serving $MODEL on NPU $SERVE_CARDS (TP=$TP, port=$PORT, eager=${ENFORCE_EAGER:-0}); HS_DIR=$HS_DIR"
ASCEND_RT_VISIBLE_DEVICES="$SERVE_CARDS" python speculators/scripts/launch_vllm.py \
  "$MODEL" \
  --target-layer-ids 1 9 17 25 33 \
  --hidden-states-path "$HS_DIR" \
  -- --tensor-parallel-size "$TP" --port "$PORT" $EAGER_FLAG
