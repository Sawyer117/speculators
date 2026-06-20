#!/usr/bin/env bash
# WINDOW 2 — DFlash trainer for Qwen3-4B on UNMODIFIED main code, colleague-aligned
# config. Connects to the serve from window 1 (same VLLM_PORT + HS_DIR). NPUs 1-7.
# Restart this freely without touching the window-1 serve.
#   bash examples/ascend_npu_dflash/aligned_train.sh 2>&1 | tee logs/aligned_train.log
#
# Aligned to the colleague: --log-freq 10, --scheduler-type cosine, --noise-std 0.0,
# --request-timeout 180 --max-retries 8, TORCH_COMPILE_DISABLE/TORCHDYNAMO_DISABLE
# (his "force eager"), no --use-off-policy-tokens (open_perfectblend isn't regen),
# epochs 1. Only deviation: --draft-attn-impl sdpa (main gates flex behind #589).
set -eo pipefail
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO_ROOT"

MODEL="${MODEL:-/share/canada_group_folder/ckpt/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
# already-prepared Arrow dataset (no prepare). Default = colleague's prepared data.
DATA_DIR="${DATA_DIR:-/share/canada_group_folder/dataset/dflash_separate_qwen3_4b}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/output/dflash_aligned_qwen3_4b}"
HS_DIR="${HS_DIR:-$OUTPUT_DIR/hidden_states}"   # MUST match the window-1 serve's HS_DIR
VLLM_PORT="${VLLM_PORT:-8000}"
TRAIN_NPUS="${TRAIN_NPUS:-1,2,3,4,5,6,7}"
NUM_TRAIN_NPUS="${NUM_TRAIN_NPUS:-7}"
TOTAL_SEQ_LEN="${TOTAL_SEQ_LEN:-3072}"

export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1   # his "force eager"

echo "waiting for the window-1 serve at http://localhost:${VLLM_PORT}/health ..."
until curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; do sleep 5; done
echo "serve ready."

echo ">>> train Qwen3-4B DFlash on NPU $TRAIN_NPUS (main code, his config); data=$DATA_DIR"
TASK_QUEUE_ENABLE=2 ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$DATA_DIR" \
    --hidden-states-path "$HS_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --epochs 1 --lr 6e-4 --total-seq-len "$TOTAL_SEQ_LEN" \
    --speculator-type dflash --block-size 16 --max-anchors 512 \
    --num-layers 5 --target-layer-ids 1 9 17 25 33 \
    --draft-arch qwen3 --draft-hidden-act silu --mask-token-id 151669 \
    --draft-attn-impl sdpa \
    --noise-std 0.0 --scheduler-type cosine \
    --logger tensorboard --run-name dflash_aligned_qwen3_4b --log-dir ./logs/aligned_qwen3_4b \
    --on-missing generate --on-generate delete \
    --request-timeout 180 --max-retries 8 \
    --log-freq 10 \
    --no-resume-from-checkpoint --seed 42
