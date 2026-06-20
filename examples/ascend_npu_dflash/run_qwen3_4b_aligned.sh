#!/usr/bin/env bash
# ============================================================================
# A/B TEST: run UNMODIFIED main training code with the COLLEAGUE'S config, to
# see whether the rich fork-deadlock reproduces and what the speed is.
#
# This branch (test/colleague-aligned-4b) only ADDS this script; the training
# code (trainer.py / logger.py / data.py / dflash) is exactly `main` — i.e. it
# still has tqdm.rich + RichHandler and the original torch.nonzero select_anchors.
#
# Config mirrors the colleague's working dflash_qwen3_4b.sh:
#   --log-freq 10, --scheduler-type cosine, --noise-std 0.0,
#   TORCH_COMPILE_DISABLE=1 / TORCHDYNAMO_DISABLE=1 (his "force eager"),
#   vLLM graph mode (no --enforce-eager), DP=1, --gpu-memory-utilization 0.85,
#   data source = open_perfectblend_full.jsonl.
#
# ONE necessary deviation: --draft-attn-impl sdpa. The colleague is on an OLDER
# speculators (0.5.0.dev0) that runs flex; main (dev167) gates flex behind #589,
# so flex_attention would crash on NPU. sdpa is the NPU path and does NOT affect
# the hang/speed question.
#
# Run on a CLEAN main checkout (this branch):
#   bash examples/ascend_npu_dflash/run_qwen3_4b_aligned.sh 2>&1 | tee logs/aligned.log
# ============================================================================
set -eo pipefail
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ---- our paths; colleague's hyperparameters ----
MODEL="${MODEL:-/share/canada_group_folder/ckpt/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/dflash_aligned_qwen3_4b}"   # local: checkpoints / HS / vllm log
# Reuse an ALREADY-PREPARED Arrow dataset (no re-prepare). Default = colleague's
# prepared data (copied to /share). Other option: your half50 set
#   DATA_DIR=/share/canada_group_folder/dataset/perfectblend_train_regen.half50.qwen3.seq3072
DATA_DIR="${DATA_DIR:-/share/canada_group_folder/dataset/dflash_separate_qwen3_4b}"
# fallback only (used if DATA_DIR has no *.arrow):
DATASET="${DATASET:-/share/canada_group_folder/dataset/open_perfectblend_full.jsonl}"
MAX_SAMPLES="${MAX_SAMPLES:-200000}"
PREP_SEQ_LEN=3072
TOTAL_SEQ_LEN=3072
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MAX_MODEL_LEN=$((PREP_SEQ_LEN + 256))
HS_DIR="${HS_DIR:-$OUTPUT_DIR/hidden_states}"   # serve writes / train reads; with SKIP_SERVE set this to the running serve's --hidden-states-path
EPOCHS=1
LR=6e-4
SEED=42
BLOCK_SIZE=16
MAX_ANCHORS=512
NUM_LAYERS=5
TARGET_LAYER_IDS="1 9 17 25 33"
VLLM_NPUS="${VLLM_NPUS:-0}"
TRAIN_NPUS="${TRAIN_NPUS:-1,2,3,4,5,6,7}"
NUM_TRAIN_NPUS="${NUM_TRAIN_NPUS:-7}"

# NPU runtime env that makes serve/train work on our box (his env has equivalents)
export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

echo "=== Step 1: data ==="
if ls "$DATA_DIR"/*.arrow >/dev/null 2>&1; then
    echo "reusing already-prepared dataset: $DATA_DIR (skip prepare)"
else
    echo "no *.arrow in $DATA_DIR -> preparing $MAX_SAMPLES samples from $(basename "$DATASET")"
    python scripts/prepare_data.py \
        --model "$MODEL" --data "$DATASET" --output "$DATA_DIR" \
        --max-samples "$MAX_SAMPLES" --seq-length "$PREP_SEQ_LEN" --overwrite --trust-remote-code
    rm -f "$DATA_DIR"/d2t.npy "$DATA_DIR"/t2d.npy
fi

if [ "${SKIP_SERVE:-0}" = "1" ]; then
    echo "=== Step 2: SKIP launch — reuse the serve already running at port $VLLM_PORT ==="
    curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1 \
        || { echo "ERROR: SKIP_SERVE=1 but nothing healthy at http://localhost:${VLLM_PORT}/health"; exit 1; }
    echo "Existing serve healthy at $VLLM_PORT. (make sure HS_DIR=$HS_DIR matches its --hidden-states-path)"
else
    echo "=== Step 2: launch vLLM (NPU $VLLM_NPUS, DP=1, GRAPH mode like his) ==="
    mkdir -p "$OUTPUT_DIR"; rm -rf "$HS_DIR"; mkdir -p "$HS_DIR"
    VLLM_LOG="$OUTPUT_DIR/vllm_server.log"
    TASK_QUEUE_ENABLE=1 ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" python scripts/launch_vllm.py "$MODEL" \
        --target-layer-ids $TARGET_LAYER_IDS \
        --hidden-states-path "$HS_DIR" \
        -- --data-parallel-size 1 --port "$VLLM_PORT" \
           --max-model-len "$VLLM_MAX_MODEL_LEN" --gpu-memory-utilization 0.85 2>&1 | tee "$VLLM_LOG" &
    VLLM_PROCS="launch_vllm.py|vllm.entrypoints|EngineCore|APIServer|vllm serve|from_engine_args"
    cleanup() { echo "Stopping vLLM..."; pkill -f "$VLLM_PROCS" 2>/dev/null || true; }
    trap cleanup EXIT

    echo "Waiting for server health (1800s cap)..."
    WAITED=0; BOOT_TIMEOUT=1800; GRACE=120
    until curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; do
        if [ "$WAITED" -ge "$GRACE" ] && ! pgrep -f "$VLLM_PROCS" >/dev/null 2>&1; then
            echo "ERROR: no vLLM process and port down after ${WAITED}s. Last 40 lines:"; tail -n 40 "$VLLM_LOG" || true; exit 1; fi
        if [ "$WAITED" -ge "$BOOT_TIMEOUT" ]; then echo "ERROR: vLLM not healthy after ${BOOT_TIMEOUT}s"; tail -n 40 "$VLLM_LOG" || true; exit 1; fi
        sleep 5; WAITED=$((WAITED+5)); [ $((WAITED % 30)) -eq 0 ] && echo "  ...waiting ${WAITED}s" || true
    done
    echo "Server ready after ${WAITED}s."
fi

# his "force eager": disable torch.compile / dynamo on the trainer
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1

echo "=== Step 3: train — UNMODIFIED main code, his config (NPU $TRAIN_NPUS) ==="
TASK_QUEUE_ENABLE=2 ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$DATA_DIR" \
    --hidden-states-path "$HS_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --epochs "$EPOCHS" --lr "$LR" --total-seq-len "$TOTAL_SEQ_LEN" \
    --speculator-type dflash --block-size "$BLOCK_SIZE" --max-anchors "$MAX_ANCHORS" \
    --num-layers "$NUM_LAYERS" --target-layer-ids $TARGET_LAYER_IDS \
    --draft-arch qwen3 --draft-hidden-act silu --mask-token-id 151669 \
    --draft-attn-impl sdpa \
    --noise-std 0.0 --scheduler-type cosine \
    --logger tensorboard --run-name dflash_aligned_qwen3_4b --log-dir ./logs/aligned_qwen3_4b \
    --on-missing generate --on-generate delete \
    --request-timeout 180 --max-retries 8 \
    --log-freq 10 \
    --no-resume-from-checkpoint --seed "$SEED"

echo "Done."
