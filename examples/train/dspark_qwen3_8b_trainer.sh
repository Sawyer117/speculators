#!/bin/bash
# Online DSpark Training Script for Qwen3-8B on Ascend NPU
#
# Runs the full online DSpark training pipeline on Ascend: data preparation,
# vLLM server launch, and training with hidden states generated on-the-fly.
# DSpark extends DFlash with a sequential correction and confidence head.
#
# Usage: Copy this script, modify the configuration variables below, then run:
#   bash examples/train/dspark_qwen3_8b_sharegpt_online_ascend.sh
#
# Note: This assumes your environment has torch_npu and an Ascend-compatible
# vLLM installation that supports hidden-state extraction.

set -euo pipefail
export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0
export NO_PROXY=localhost,127.0.0.1,80.5.5.45,80.5.5.44,80.5.5.54 no_proxy=localhost,127.0.0.1,80.5.5.45,80.5.5.44,80.5.5.54
# ============ Configuration ============
MODEL="/mnt/pipeline-data/beta_lab/weights/Qwen3-8B"
DATASET="sharegpt"                # sharegpt, ultrachat, or path to custom data
OUTPUT_DIR="./output/dspark_qwen3_8b_sharegpt_ascend"
VLLM_PORT=8000
MAX_SAMPLES=5000
SEQ_LENGTH=8192
EPOCHS=5
LR=6e-4
LOGGER="tensorboard"

# DSpark-specific parameters
SPECULATOR_TYPE="dspark"
BLOCK_SIZE=8
MAX_ANCHORS=512
NUM_LAYERS=5
DRAFT_VOCAB_SIZE=32000
TARGET_LAYER_IDS="1 9 17 25 33"  # Must match vLLM's eagle_aux_hidden_state_layer_ids
DRAFT_ATTN_IMPL="sdpa"     # Use eager/sdpa on hardware without flex attention.

# Correction + confidence head settings. Use an empty CORRECTION_HEAD_ARGS array
# to fall back to the legacy Markov head.
MARKOV_RANK=256
MARKOV_HEAD_TYPE="vanilla"   # vanilla | gated | rnn
CORRECTION_HEAD_ARGS=(--enable-correction-head)
CORRECTION_HIDDEN_SIZE=512
CORRECTION_RANK=256
CORRECTION_NUM_LAYERS=1
CORRECTION_NUM_HEADS=8
CORRECTION_GATE_BIAS=0.0
# Teacher forcing is the default. Add --correction-generated-token-training to
# CORRECTION_HEAD_ARGS for a generated-token self-feedback experiment.
CORRECTION_ROLLOUT_METRICS_ARGS=(--correction-rollout-metrics)
# Enable only when validation-only base change/gain diagnostics are worth an
# additional LM-head projection. Correction itself never consumes base logits.
CORRECTION_BASE_DIAGNOSTICS_ARGS=(--no-correction-base-diagnostics)
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
CONFIDENCE_HEAD_ALPHA=1.0
CONFIDENCE_LENGTH_ALPHA=0.0
CONFIDENCE_LOSS_WEIGHTING="uniform"   # uniform | match-draft
# Use --confidence-detach-features to keep confidence fully auxiliary.
CONFIDENCE_DETACH_FEATURES_ARGS=(--no-confidence-detach-features)
FIRST_ERROR_FOCAL_ALPHA=0.0
ADAPTIVE_LOSS="none"                  # none | cat | ssal
# Set SSAL_CURRICULUM_ARGS=(--ssal-curriculum) to enable decay→SSAL mix.
SSAL_CURRICULUM_ARGS=(--no-ssal-curriculum)
SSAL_CURRICULUM_START=0.1
SSAL_CURRICULUM_END=0.6

# Ascend NPU assignments (online training needs separate devices for vLLM/training)
VLLM_NPUS="0,1,2,3"
TRAIN_NPUS="4,5,6,7"
NUM_TRAIN_NPUS=4

# Extra vLLM arguments for Ascend. Remove --enforce-eager if your stack supports
# graph mode for this path.
#VLLM_EXTRA_ARGS=(--enforce-eager --data-parallel-size 4)
# =======================================

# Step 1: Prepare data
echo "=== Step 1: Preparing data ==="
# python scripts/prepare_data.py \
#     --model "$MODEL" \
#     --data "$DATASET" \
#     --output "$OUTPUT_DIR" \
#     --max-samples "$MAX_SAMPLES" \
#     --seq-length "$SEQ_LENGTH"

# Step 3: Train DSpark against the live vLLM server
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="$LOG_DIR/train.pid"

echo "=== Step 3: Training on Ascend NPU(s): $TRAIN_NPUS ==="
nohup env ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "/mnt/pipeline-data/beta_lab/datasets/perfectblend-regenerated/processed_data" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --logger "$LOGGER" \
    --total-seq-len "$SEQ_LENGTH" \
    --speculator-type "$SPECULATOR_TYPE" \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --num-layers "$NUM_LAYERS" \
    --draft-attn-impl "$DRAFT_ATTN_IMPL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --markov-rank "$MARKOV_RANK" \
    --markov-head-type "$MARKOV_HEAD_TYPE" \
    "${CORRECTION_HEAD_ARGS[@]}" \
    --correction-hidden-size "$CORRECTION_HIDDEN_SIZE" \
    --correction-rank "$CORRECTION_RANK" \
    --correction-num-layers "$CORRECTION_NUM_LAYERS" \
    --correction-num-heads "$CORRECTION_NUM_HEADS" \
    --correction-gate-bias "$CORRECTION_GATE_BIAS" \
    "${CORRECTION_ROLLOUT_METRICS_ARGS[@]}" \
    "${CORRECTION_BASE_DIAGNOSTICS_ARGS[@]}" \
    --enable-confidence-head \
    --confidence-head-with-markov \
    --loss-fn "$LOSS_FN" \
    --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA" \
    --confidence-length-alpha "$CONFIDENCE_LENGTH_ALPHA" \
    --confidence-loss-weighting "$CONFIDENCE_LOSS_WEIGHTING" \
    "${CONFIDENCE_DETACH_FEATURES_ARGS[@]}" \
    --first-error-focal-alpha "$FIRST_ERROR_FOCAL_ALPHA" \
    --adaptive-loss "$ADAPTIVE_LOSS" \
    "${SSAL_CURRICULUM_ARGS[@]}" \
    --ssal-curriculum-start "$SSAL_CURRICULUM_START" \
    --ssal-curriculum-end "$SSAL_CURRICULUM_END" \
    --on-missing generate \
    --on-generate delete \
    > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "Log file: $LOG_FILE"
echo "TensorBoard: tensorboard --logdir ./logs --host 0.0.0.0 --port 6006"
echo "View log with: tail -f $LOG_FILE"
echo "Stop with: kill \$(cat $PID_FILE)"
