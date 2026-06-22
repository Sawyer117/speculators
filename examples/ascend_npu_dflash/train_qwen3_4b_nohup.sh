#!/usr/bin/env bash
# Terminal 2 (background) — DFlash FSDP training for Qwen3-4B on Ascend NPU,
# launched with nohup so it survives SSH disconnect and logs to a file.
#
# Bonus: redirecting stdout to a file makes it non-TTY, which also sidesteps the
# rich-fork DataLoader deadlock (that bug only fires on a real TTY). So this is the
# safe way to run long jobs regardless of the worker_init_fn fix branch.
#
# Run with bash, do NOT source. Same prereqs/flags as train_qwen3_4b.sh:
# DATA_DIR (shared tokenized Arrow) exists + serve_qwen3_4b.sh up.
[ "${BASH_SOURCE[0]}" != "$0" ] && { echo "Run with 'bash $0', do not source."; return 1 2>/dev/null; }
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/config_qwen3_4b.sh"

export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
# TORCHDYNAMO_DISABLE not needed — #600 makes DFlash skip torch.compile on NPU.

# off-policy tokens: REQUIRED for regenerated data; OMIT for original/non-regen
# datasets (run with USE_OFF_POLICY=0). See config_qwen3_4b.sh.
OFF_POLICY_FLAG=""
[ "${USE_OFF_POLICY:-1}" = "1" ] && OFF_POLICY_FLAG="--use-off-policy-tokens"

LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"
mkdir -p "$LOG_DIR"
# timestamp from `date`; falls back to PID if `date` is unavailable
STAMP="$(date +%Y%m%d_%H%M%S 2>/dev/null || echo "$$")"
LOG_FILE="$LOG_DIR/train_4b_${STAMP}.log"
PID_FILE="$LOG_DIR/train_4b.pid"

echo ">>> nohup train Qwen3-4B DFlash on NPU $TRAIN_CARDS (nproc=$NPROC epochs=$EPOCHS loss=$LOSS_FN off_policy=${USE_OFF_POLICY:-1})"
echo ">>> data=$DATA_DIR save=$SAVE_DIR"
echo ">>> log -> $LOG_FILE"

nohup env ASCEND_RT_VISIBLE_DEVICES="$TRAIN_CARDS" torchrun \
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
  --epochs "$EPOCHS" \
  --lr 6e-4 \
  --loss-fn "$LOSS_FN" \
  --total-seq-len "$SEQ_LEN" \
  $OFF_POLICY_FLAG \
  --logger tensorboard \
  --on-missing generate \
  --on-generate delete \
  --trust-remote-code \
  > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
echo ">>> started, pid=$TRAIN_PID (saved to $PID_FILE)"
echo ">>> follow:  tail -f $LOG_FILE"
echo ">>> stop:    kill \$(cat $PID_FILE)   # or: pkill -f scripts/train.py"
