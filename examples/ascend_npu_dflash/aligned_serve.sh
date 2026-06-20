#!/usr/bin/env bash
# WINDOW 1 — vLLM serve for Qwen3-4B DFlash hidden-state extraction.
# Colleague-aligned: graph mode (no --enforce-eager), DP=1, --gpu-memory-utilization
# 0.85, --max-model-len 3328, port 8000. Keep this running; train in window 2.
#   bash examples/ascend_npu_dflash/aligned_serve.sh
set -eo pipefail
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO_ROOT"

MODEL="${MODEL:-/share/canada_group_folder/ckpt/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
VLLM_PORT="${VLLM_PORT:-8000}"
HS_DIR="${HS_DIR:-$REPO_ROOT/output/dflash_aligned_qwen3_4b/hidden_states}"   # window-2 train MUST use the same HS_DIR
VLLM_NPUS="${VLLM_NPUS:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-3328}"
TARGET_LAYER_IDS="1 9 17 25 33"

export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
rm -rf "$HS_DIR"; mkdir -p "$HS_DIR"

echo ">>> serve Qwen3-4B on NPU $VLLM_NPUS (DP=1, graph mode, port $VLLM_PORT); HS_DIR=$HS_DIR"
TASK_QUEUE_ENABLE=1 ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --hidden-states-path "$HS_DIR" \
    -- --data-parallel-size 1 --port "$VLLM_PORT" \
       --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization 0.85
