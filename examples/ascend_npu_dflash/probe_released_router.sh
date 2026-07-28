#!/usr/bin/env bash
# Router-load PROBE for a RELEASED (or standalone-converted) DSV4-DSpark draft.
# Loads the released draft, runs it FORWARD-ONLY over real HS batches from the live
# HS-dump serve (NO backward / optimizer / checkpoint -> the weights NEVER drift), and
# reports each MoE router's GLOBAL per-expert selection histogram: used/dead experts,
# top-16 mass, normalized entropy, and the hottest expert IDs. Answers "is the released
# draft's router collapsed to a few experts, or spread out?" -- the dynamic counterpart
# to the static gate.bias inspection.
#
# SINGLE CARD, PLAIN PYTHON (no torchrun) on purpose: that takes the trainer's
# non-distributed path (.to(bf16).to(npu:0)) -- NO FSDP, NO fp32-master upcast, NO rank0
# state-dict copy -- so the ~43GB bf16 faithful draft (3L x 256E) fits one 64GB A2 with no
# transient spike. Experts stay LOCAL so DSPARK_GROUPED_MOE=1 uses the fused grouped-GEMM.
#
# USAGE:
#   RELEASED=/share/canada_group_folder/ckpt/released_draft_bf16_standalone \
#     bash examples/ascend_npu_dflash/probe_released_router.sh
#
# OVERRIDES (env): RELEASED (required) STEPS MAX_ANCHORS VERIFIER DATA HS_DIR ENDPOINT
#   RUN CANN_ENV NUM_WORKERS NONCAUSAL. Point ENDPOINT at the live HS-dump serve.
# NB: no `set -u` -- CANN's 900env / conda activate reference unbound vars (matches serve).
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---- the draft to probe (REQUIRED) ----
RELEASED="${RELEASED:-${1:-}}"
if [ -z "$RELEASED" ]; then
  echo "!! set RELEASED=<released_draft_bf16_standalone dir>  (or pass as arg 1)"; exit 2
fi
if [ ! -e "$RELEASED/config.json" ]; then
  echo "!! RELEASED=$RELEASED has no config.json -- is it a draft dir?"; exit 2
fi

# ---- paths (A2 shared /share; override via env) ----
VERIFIER="${VERIFIER:-/share/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16}"
DATA="${DATA:-/share/canada_group_folder/dataset/open_perfectblend.dsv4_rollout/arrow}"
HS_DIR="${HS_DIR:-/share/canada_group_folder/dataset/dsv4_hs_dump}"
ENDPOINT="${ENDPOINT:-http://80.5.5.115:7000/v1}"
RUN="${RUN:-/home/a00652497/dspark_austin/run}"
CANN_ENV="${CANN_ENV:-/home/a00652497/900env_npu.sh}"

STEPS="${STEPS:-200}"                 # forward passes before the aggregate report (0 = whole epoch)
MAX_ANCHORS="${MAX_ANCHORS:-512}"     # anchors/forward; more = more routing samples/forward
SEQLEN="${SEQLEN:-3072}"
BLOCK="${BLOCK:-5}"
MASK_TOKEN="${MASK_TOKEN:-128799}"
SWA_WINDOW="${SWA_WINDOW:-128}"
NONCAUSAL="${NONCAUSAL:-1}"           # match the trained/served non-causal block attention
LOSS_FN="${LOSS_FN:-{\"ce\":0.1,\"tv\":1.8}}"   # forward computes it but --forward-only ignores it
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"

NONCAUSAL_FLAG=""
[ "$NONCAUSAL" = "1" ] && NONCAUSAL_FLAG="--sliding-window-non-causal"

# ---- preflight: CANN env, no-proxy for the serve, serve reachable ----
if [ -f "$CANN_ENV" ]; then
  set +e; source "$CANN_ENV"; set -e
else
  echo "WARN: CANN env not found at $CANN_ENV -- pass CANN_ENV=/path/to/900env_npu.sh"
fi
mkdir -p "$RUN" 2>/dev/null || { echo "!! cannot create RUN=$RUN -- override e.g. RUN=\$HOME/dspark_austin/run"; exit 1; }
_ep_host="$(printf '%s' "$ENDPOINT" | sed -E 's#^https?://([^:/]+).*#\1#')"
export no_proxy="127.0.0.1,localhost,${_ep_host}"; export NO_PROXY="$no_proxy"
if ! curl -sf --noproxy '*' "$ENDPOINT/models" >/dev/null 2>&1; then
  echo "WARN: serve not reachable at $ENDPOINT -- start the HS-dump serve (115/116) first."
fi
rm -f "$HS_DIR"/hs_*.safetensors*      # stale dumps collide with the data row idx

TS="$(date +%Y%m%d_%H%M%S)"
LOG="$RUN/probe_router_${TS}.log"
SAVE_PATH="$RUN/_probe_scratch_${TS}"  # never written (forward-only saves nothing); satisfies the CLI

echo "==================================================================="
echo " DSV4-DSpark ROUTER PROBE  (forward-only, single card, no drift)"
echo " draft    = $RELEASED"
echo " verifier = $VERIFIER"
echo " data=$DATA  endpoint=$ENDPOINT"
echo " steps=$STEPS  max_anchors=$MAX_ANCHORS  block=$BLOCK  noncausal=$NONCAUSAL"
echo " 📋 log -> $LOG"
echo "==================================================================="

# PLAIN python (NOT torchrun) => trainer non-distributed path => memory-safe single card.
env \
  DSPARK_HS_DUMP=1 DSPARK_GROUPED_MOE=1 DSPARK_EP=0 \
  DSPARK_LOG_EXPERT_LOAD=1 \
  PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}" \
  python "$REPO_ROOT/scripts/train.py" \
    --speculator-type dsv4_dspark --served-model-name dsv4 \
    --num-layers 3 --n-routed-experts 256 \
    --block-size "$BLOCK" --target-layer-ids 40 41 42 --max-anchors "$MAX_ANCHORS" \
    --sliding-window "$SWA_WINDOW" $NONCAUSAL_FLAG \
    --total-seq-len "$SEQLEN" --mask-token-id "$MASK_TOKEN" --noise-std 0 \
    --draft-attn-impl sdpa --loss-fn "$LOSS_FN" \
    --from-released "$RELEASED" --forward-only --forward-only-steps "$STEPS" \
    --on-missing generate --on-generate delete --no-validation \
    --no-resume-from-checkpoint \
    --num-workers "$NUM_WORKERS" --prefetch-factor "$PREFETCH_FACTOR" \
    --hidden-states-path "$HS_DIR" --vllm-endpoint "$ENDPOINT" \
    --verifier-name-or-path "$VERIFIER" --data-path "$DATA" \
    --save-path "$SAVE_PATH" --log-dir "$RUN" \
  2>&1 | tee "$LOG"

echo
echo ">>> DONE. The verdict lines:  grep '\[PROBE-AGG' $LOG"
echo ">>>   entropy ~1.0 = balanced across experts;  low + stable hot IDs = collapsed."
echo ">>>   per-forward snapshots also logged as [MOE-LOAD L*] (one batch each)."
