#!/usr/bin/env bash
# One-command DSV4-DSpark training launcher (twin of serve_dsv4_bf16_dualnode.sh).
# Backgrounds a torchrun training run, logs (timestamped) into $RUN, prints the tail cmd.
#
# USAGE:
#   bash examples/ascend_npu_dflash/train_dsv4_dspark.sh [reduced|faithful]
#     reduced  (default) = 1 layer x 32 experts, 1 card, no --init-on-meta  (Gate-2 / A-B smoke)
#     faithful           = 3 layers x 256 experts, 8 cards, --init-on-meta + HCCL port ranges
#
# TOGGLE fused grouped-GEMM MoE (NPU; kills the per-shape MoE recompile spikes):
#   DSPARK_GROUPED_MOE=1 bash ... reduced      # A/B: run once without, once with, compare
#   (grouped-GEMM needs experts LOCAL -> reduced/single-card is fine; the faithful per-expert
#    FSDP run must NOT enable it yet -- needs EP/gather; the script warns if you try.)
#
# OVERRIDES (env, validated defaults baked in):
#   RUN VERIFIER DATA HS_DIR ENDPOINT LR MAX_ANCHORS NPROC CKPT_FREQ CANN_ENV
set -euo pipefail

MODE="${1:-reduced}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"          # the speculators checkout

# ---- paths (A2 shared /share; override via env) ----
VERIFIER="${VERIFIER:-/share/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16}"
DATA="${DATA:-/share/canada_group_folder/dataset/open_perfectblend.dsv4_rollout/arrow}"
HS_DIR="${HS_DIR:-/share/canada_group_folder/dataset/dsv4_hs_dump}"
ENDPOINT="${ENDPOINT:-http://80.5.5.115:7000/v1}"
RUN="${RUN:-/home/a00652497/dspark_austin/run}"
CANN_ENV="${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
LR="${LR:-2e-4}"
MAX_ANCHORS="${MAX_ANCHORS:-64}"
GROUPED="${DSPARK_GROUPED_MOE:-0}"

# ---- per-mode config ----
if [ "$MODE" = "faithful" ]; then
  NPROC="${NPROC:-8}"; LAYERS=3; EXPERTS=256
  EXTRA="--init-on-meta --checkpoint-freq ${CKPT_FREQ:-0.1}"
  PORTS="HCCL_NPU_SOCKET_PORT_RANGE=61000-62000 HCCL_HOST_SOCKET_PORT_RANGE=60000-61000"
  if [ "$GROUPED" = "1" ]; then
    echo "!! faithful uses per-expert FSDP; grouped-GEMM needs experts LOCAL -> it will get"
    echo "   sharded weights and be WRONG. Use reduced/single-card for grouped, or add EP first."
    echo "   Refusing to enable grouped-GEMM on faithful. Unset DSPARK_GROUPED_MOE."; exit 2
  fi
elif [ "$MODE" = "reduced" ]; then
  NPROC="${NPROC:-1}"; LAYERS=1; EXPERTS=32; EXTRA=""; PORTS=""
else
  echo "usage: $0 [reduced|faithful]"; exit 1
fi

# ---- preflight ----
[ -f "$CANN_ENV" ] && source "$CANN_ENV" || echo "WARN: CANN set_env not at $CANN_ENV"
mkdir -p "$RUN"
rm -f "$HS_DIR"/hs_*.safetensors*                     # stale dumps collide with data row idx
export no_proxy=127.0.0.1,localhost,80.5.5.115,80.5.5.116 NO_PROXY=127.0.0.1,localhost,80.5.5.115,80.5.5.116
if ! curl -sf --noproxy '*' "$ENDPOINT/models" >/dev/null 2>&1; then
  echo "WARN: serve not reachable at $ENDPOINT — start 115/116 first (or set ENDPOINT=)."
fi

TS="$(date +%Y%m%d_%H%M%S)"
TAG="${MODE}$([ "$GROUPED" = 1 ] && echo _grouped)"
LOG="$RUN/${TAG}_${TS}.log"

echo "==================================================================="
echo " DSV4-DSpark TRAIN  mode=$MODE  nproc=$NPROC  ${LAYERS}L x ${EXPERTS}E  lr=$LR  grouped_moe=$GROUPED"
echo " verifier=$VERIFIER"
echo " data=$DATA"
echo " 📋 log -> $LOG   (rank0 mirror also in $RUN/train_*.log)"
echo "==================================================================="

nohup env \
  DSPARK_HS_DUMP=1 DSPARK_GROUPED_MOE="$GROUPED" \
  HCCL_CONNECT_TIMEOUT=1800 HCCL_EXEC_TIMEOUT=1800 $PORTS \
  torchrun --nproc_per_node "$NPROC" "$REPO_ROOT/scripts/train.py" \
    --speculator-type dsv4_dspark --served-model-name dsv4 \
    --num-layers "$LAYERS" --n-routed-experts "$EXPERTS" \
    --block-size 5 --target-layer-ids 40 41 42 --max-anchors "$MAX_ANCHORS" \
    --draft-attn-impl sdpa --loss-fn '{"ce":0.1,"tv":0.9}' \
    --optimizer adamw --lr "$LR" $EXTRA \
    --on-missing generate --on-generate delete \
    --hidden-states-path "$HS_DIR" --vllm-endpoint "$ENDPOINT" \
    --verifier-name-or-path "$VERIFIER" --data-path "$DATA" \
    --log-dir "$RUN" \
  > "$LOG" 2>&1 &

echo ">>> started PID $!  |  tail -f $LOG"
echo ">>> watch: profile/grad_norm (blow-up before NaN), profile/fwd_ms (grouped => spikes gone),"
echo ">>>        train/loss (non-zero, decreasing), NaN => kill immediately."
