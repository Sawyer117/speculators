#!/usr/bin/env bash
# Terminal 2 — DFlash FSDP training for Qwen3-8B on Ascend NPU (single machine,
# 6 cards). Online: reads hidden states written by serve_qwen3_8b.sh (terminal 1).
# See docs/deployment/ascend-npu-dflash-training.md (§6).
#
# Prereqs: (1) data prepared with scripts/prepare_data.py into $DATA (§4),
#          (2) serve_qwen3_8b.sh up and showing "Application startup complete" (§5),
#          (3) $HS_DIR is the SAME absolute path as the serve script.
#
# Run it, do NOT source it:   bash train_qwen3_8b.sh

# safety: if accidentally sourced, bail without killing the shell
(return 0 2>/dev/null) && { echo "Run with 'bash $0', do not source."; return 1 2>/dev/null; }
set -euo pipefail

MODEL="${MODEL:-/share/canada_group_folder/ckpt/Qwen3-8B}"
WORKDIR="${WORKDIR:-/home/a00652497/2026/dflash-vllm}"
HS_DIR="${HS_DIR:-$WORKDIR/tmp/hs_qwen3_dflash}"
DATA="${DATA:-$WORKDIR/train_data}"
SAVE="${SAVE:-$WORKDIR/output/qwen3-8b-dflash-npu/checkpoints}"
TRAIN_CARDS="${TRAIN_CARDS:-2,3,4,5,6,7}"
NPROC="${NPROC:-6}"
PORT="${PORT:-8000}"
MASTER_PORT="${MASTER_PORT:-29533}"

cd "$WORKDIR"

export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# eager FSDP does no graph capture, so TASK_QUEUE_ENABLE=2 is fine (and faster) here
export TASK_QUEUE_ENABLE=2 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
# NOTE: TORCHDYNAMO_DISABLE not needed — #600 makes DFlash skip torch.compile on NPU.

echo ">>> training Qwen3-8B DFlash on NPU $TRAIN_CARDS (nproc=$NPROC); data=$DATA save=$SAVE"
ASCEND_RT_VISIBLE_DEVICES="$TRAIN_CARDS" torchrun \
  --nproc_per_node "$NPROC" --nnodes 1 --node_rank 0 \
  --master_addr 127.0.0.1 --master_port "$MASTER_PORT" \
  speculators/scripts/train.py \
  --verifier-name-or-path "$MODEL" \
  --data-path "$DATA" \
  --vllm-endpoint "http://localhost:$PORT/v1" \
  --hidden-states-path "$HS_DIR" \
  --save-path "$SAVE" \
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
  --total-seq-len 3072 \
  --use-off-policy-tokens \
  --logger tensorboard \
  --on-missing generate \
  --on-generate delete \
  --trust-remote-code
