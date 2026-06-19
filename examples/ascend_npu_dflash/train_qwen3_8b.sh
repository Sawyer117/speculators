#!/usr/bin/env bash
# Terminal 2 — DFlash FSDP training for Qwen3-8B on Ascend NPU (single machine,
# 6 cards). Reads hidden states written by serve_qwen3_8b.sh. See
# docs/deployment/ascend-npu-dflash-training.md (§6). Run with bash, do NOT source.
#
# Prereqs: prepare_qwen3_8b.sh done (DATA_DIR exists) + serve_qwen3_8b.sh up
# ("Application startup complete"). HS_DIR/DATA_DIR come from config.sh (shared).
[ "${BASH_SOURCE[0]}" != "$0" ] && { echo "Run with 'bash $0', do not source."; return 1 2>/dev/null; }
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/config.sh"

export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# eager FSDP does no graph capture, so TASK_QUEUE_ENABLE=2 is fine (and faster) here
export TASK_QUEUE_ENABLE=2 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
# NOTE: TORCHDYNAMO_DISABLE not needed — #600 makes DFlash skip torch.compile on NPU.

echo ">>> train Qwen3-8B DFlash on NPU $TRAIN_CARDS (nproc=$NPROC); data=$DATA_DIR save=$SAVE_DIR"
ASCEND_RT_VISIBLE_DEVICES="$TRAIN_CARDS" torchrun \
  --nproc_per_node "$NPROC" --nnodes 1 --node_rank 0 \
  --master_addr 127.0.0.1 --master_port "$MASTER_PORT" \
  "$REPO_ROOT/scripts/train.py" \
  --verifier-name-or-path "$TARGET_MODEL" \
  --data-path "$DATA_DIR" \
  --vllm-endpoint "http://localhost:$PORT/v1" \
  --hidden-states-path "$HS_DIR" \
  --save-path "$SAVE_DIR" \
  --speculator-type dflash \
  --draft-arch qwen3 \
  --num-layers 5 \
  --block-size 16 \
  --max-anchors 512 \
  --draft-attn-impl sdpa \
  --target-layer-ids 1 9 17 25 33 \
  --mask-token-id 151669 \
  --draft-hidden-act silu \
  --epochs 6 \
  --lr 6e-4 \
  --total-seq-len "$SEQ_LEN" \
  --use-off-policy-tokens \
  --logger tensorboard \
  --on-missing generate \
  --on-generate delete \
  --trust-remote-code
