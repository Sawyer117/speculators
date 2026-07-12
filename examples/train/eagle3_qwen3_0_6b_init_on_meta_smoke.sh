#!/bin/bash
# E2E smoke test for --init-on-meta: a REAL 2-GPU FSDP training run.
#
# Trains a tiny Eagle3 draft on Qwen3-0.6B for 1 epoch over a handful of ShareGPT
# samples, WITH --init-on-meta and NUM_TRAIN_GPUS=2. The goal is not model quality
# but to exercise the real training loop (forward -> loss -> backward ->
# optimizer.step) under FSDP2 with meta-init on the non-rank0 rank.
#
# Why 2 training GPUs matter: the failure mode is a cross-rank inconsistency --
# if the frozen verifier weights (lm_head / verifier_lm_head / verifier_norm) end
# up requires_grad=True on non-rank0 (because the meta path skipped the freeze)
# while rank0 has them frozen, FSDP2's post_backward reduce_scatter collects a
# different param set per rank and the FIRST backward HANGS. A single GPU never
# shards across ranks, so it cannot reproduce this. A clean run to completion here
# validates that --init-on-meta keeps the trainable-param set rank-consistent.
#
# Requirements: >=3 GPUs (1 for vLLM, 2 for training), vLLM installed, HF access.
# Usage: bash examples/train/eagle3_qwen3_0_6b_init_on_meta_smoke.sh
set -euo pipefail

# ============ Configuration ============
MODEL="Qwen/Qwen3-0.6B"
DATASET="sharegpt"
OUTPUT_DIR="./output/init_on_meta_smoke"
VLLM_PORT=8137
MAX_SAMPLES=64          # tiny: we only need a few real training steps
SEQ_LENGTH=1024
EPOCHS=1
LR=1e-4
DRAFT_VOCAB_SIZE=32000

VLLM_GPUS="0"
TRAIN_GPUS="1,2"        # 2 GPUs -> multi-rank FSDP2 (REQUIRED to trigger the hang)
NUM_TRAIN_GPUS=2
# =======================================

echo "=== Step 1: prepare data ==="
python scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH"

echo "=== Step 2: launch vLLM server (on-the-fly hidden states) ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py "$MODEL" \
    -- --port "$VLLM_PORT" &
VLLM_PID=$!
cleanup() {
    echo "Stopping vLLM server..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT
echo "Waiting for vLLM server..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do sleep 2; done
echo "vLLM server ready."

echo "=== Step 3: TRAIN with --init-on-meta on ${NUM_TRAIN_GPUS} GPUs ==="
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --init-on-meta \
    --on-missing generate \
    --on-generate delete

echo
echo "=== PASS: training completed WITHOUT hanging."
echo "    --init-on-meta backward is rank-consistent (requires_grad freeze applied on the meta path). ==="
