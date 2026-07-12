#!/bin/bash
# Real 2-GPU --init-on-meta training smoke test for dflash / dspark.
#
# eagle3 has its own smoke (eagle3_qwen3_0_6b_init_on_meta_smoke.sh); peagle inherits
# eagle3's load_verifier_weights, so the eagle3 run already covers its code path.
#
# Usage: bash examples/train/dflash_dspark_qwen3_0_6b_init_on_meta_smoke.sh <dflash|dspark>
#
# Trains the draft for 1 epoch on a few ShareGPT samples with --init-on-meta on 2 GPUs.
# A clean run to completion (no hang) validates that --init-on-meta is rank-consistent
# for this speculator type -- dflash (sliding-window attn) and dspark (Markov +
# confidence heads) build differently from eagle3, so they are worth a real run.
#
# Requirements: >=3 GPUs (1 vLLM, 2 training), vLLM installed, HF access.
set -euo pipefail
TYPE="${1:?usage: $0 <dflash|dspark>}"

# ============ Configuration ============
MODEL="Qwen/Qwen3-0.6B"
OUTPUT_DIR="./output/init_on_meta_smoke_${TYPE}"
VLLM_PORT=8137
MAX_SAMPLES=64          # tiny: we only need a few real training steps
SEQ_LENGTH=1024
DRAFT_VOCAB_SIZE=32000
BLOCK_SIZE=8
MAX_ANCHORS=3072
NUM_LAYERS=3
TARGET_LAYER_IDS="2 14 25"   # must match the vLLM aux layers below

VLLM_GPUS="0"
TRAIN_GPUS="1,2"        # 2 GPUs -> multi-rank FSDP2 (required to trigger any hang)
NUM_TRAIN_GPUS=2
# =======================================

case "$TYPE" in
  dflash)
    TRAIN_EXTRA=(--speculator-type dflash) ;;
  dspark)
    TRAIN_EXTRA=(--speculator-type dspark
                 --markov-rank 256 --markov-head-type vanilla
                 --enable-confidence-head --confidence-head-with-markov
                 --loss-fn '{"ce": 0.1, "tv": 0.9}' --confidence-head-alpha 1.0) ;;
  *) echo "unknown type: $TYPE (use dflash or dspark)"; exit 1 ;;
esac

echo "=== [$TYPE] Step 1: prepare data ==="
python scripts/prepare_data.py \
    --model "$MODEL" \
    --data sharegpt \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH"

echo "=== [$TYPE] Step 2: launch vLLM server (aux layers: $TARGET_LAYER_IDS) ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
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

echo "=== [$TYPE] Step 3: TRAIN with --init-on-meta on ${NUM_TRAIN_GPUS} GPUs ==="
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --epochs 1 \
    --lr 1e-4 \
    --total-seq-len "$SEQ_LENGTH" \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --num-layers "$NUM_LAYERS" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --init-on-meta \
    "${TRAIN_EXTRA[@]}" \
    --on-missing generate \
    --on-generate delete

echo
echo "=== [$TYPE] PASS: training completed WITHOUT hanging (--init-on-meta rank-consistent) ==="
