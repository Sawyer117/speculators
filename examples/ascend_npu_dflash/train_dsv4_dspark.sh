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
#   RECOMPUTE=1 (activation checkpointing: recompute draft layers in bwd -> raise MAX_ANCHORS past OOM)
# NB: no `set -u` — CANN's 900env / conda activate reference unbound vars (matches serve).
set -eo pipefail

MODE="${1:-reduced}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"          # the speculators checkout

# ---- paths (A2 shared /share; override via env) ----
VERIFIER="${VERIFIER:-/share/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16}"
DATA="${DATA:-/share/canada_group_folder/dataset/open_perfectblend.dsv4_rollout/arrow}"
HS_DIR="${HS_DIR:-/share/canada_group_folder/dataset/dsv4_hs_dump}"
ENDPOINT="${ENDPOINT:-http://80.5.5.115:7000/v1}"
RUN="${RUN:-/home/a00652497/dspark_austin/run}"
CANN_ENV="${CANN_ENV:-/home/a00652497/900env_npu.sh}"
LR="${LR:-6e-4}"
EPOCHS="${EPOCHS:-5}"                   # number of training epochs (--epochs). Override e.g. EPOCHS=20.
MAX_ANCHORS="${MAX_ANCHORS:-64}"
SEQLEN="${SEQLEN:-3072}"                # --total-seq-len (default 3072; was 8192). Shorter cuts draft-forward
                                       # activation memory (room for more anchors) + shortens the
                                       # HS prefill; anchor utilization = MAX_ANCHORS/SEQLEN.
MASK_TOKEN="${MASK_TOKEN:-128799}"     # DSpark noise token (config.py noise_token_id). Draft's masked
                                       # positions embed as embed_tokens[MASK_TOKEN]; MUST match serve.
                                       # Without it, resolve_mask_token_id falls back to pad_token_id=1
                                       # (wrong: collides with real pad + mismatches the official 128799).
BLOCK="${BLOCK:-6}"                     # ★ BLOCK WIDTH = anchor(slot 0) + gamma draft masks. The trainer
                                       # drafts BLOCK-1 tokens (slot 0 is the GIVEN anchor, loss-masked;
                                       # same convention as DFlash block16 -> 15 drafted). DSV4 DSpark
                                       # gamma=5 (released num_spec=5; the released draft has 5 per-position
                                       # accepts) => BLOCK=6. Passing 5 (the released config's dspark_block_size,
                                       # which is GAMMA, not width) drafts only 4 -> logs show position_1..4,
                                       # accept_len ceiling 5. DO NOT set 5. (Serve still gets dspark_block_size
                                       # = BLOCK-1 = 5; verify at save/convert.) NB: BLOCK scales draft-forward
                                       # tokens = MAX_ANCHORS*BLOCK -> raising it costs memory; drop MAX_ANCHORS
                                       # to keep MAX_ANCHORS*BLOCK ~constant (256*5=1280 -> ~213*6).
GROUPED="${DSPARK_GROUPED_MOE:-0}"
EP="${DSPARK_EP:-0}"
RECOMPUTE="${RECOMPUTE:-0}"             # activation checkpointing: recompute each draft layer in backward
                                       # -> frees activation so MAX_ANCHORS scales past the memory wall
                                       # (the lever to slow an HS-bound step to the serve's HS rate).
COMPILE="${COMPILE:-0}"                # ★ SEED TECH: torch.compile'd experts (kills the ~42% grouped-GEMM
                                       # recompile, ~1.74x). REQUIRES the torch-2.12 stack (torch 2.12+cpu /
                                       # torch_npu 2.12rc1 / inductor_npu_ext / triton-ascend) — do NOT set
                                       # on the 2.10 main stack (desyncs train vs the 2.10 serve).
NOVAL="${NO_VAL:-0}"                    # NO_VAL=1 -> cancel the per-epoch validation pass. Val does SERIAL
                                       # online HS generation for the 10% held-out split (num_workers=0),
                                       # which dominates the epoch; --no-validation trains on the FULL data.
INITMOE="${INIT_MOE:-0}"               # INIT_MOE=1 -> warm-start the draft MoE (experts+router+shared) from
                                       # the verifier's target layers (draft n <- target_layer_ids[n]).
                                       # Trainable init; faithful (256x2048) only. A/B vs from-scratch.

# ---- per-mode config ----
if [ "$MODE" = "faithful" ]; then
  NPROC="${NPROC:-8}"; LAYERS=3; EXPERTS=256
  PORTS="HCCL_NPU_SOCKET_PORT_RANGE=61000-62000 HCCL_HOST_SOCKET_PORT_RANGE=60000-61000"
  if [ "$EP" = "1" ]; then
    # EP: routed experts are Shard(0) DTensors (256/NPROC per card); grouped-GEMM runs on
    # the local slice. NO --init-on-meta (each rank builds only its 1/EP slice per-rank).
    # Checkpoints work under EP (DCP gathers the Shard(0) expert DTensors). DEFAULT = once per
    # epoch (CKPT_FREQ=1.0): EP save is DTensor-native (fast, no 768 per-expert gather), and
    # per-epoch keeps overhead minimal on the multi-day run. While DEBUGGING save/resume, set a
    # small CKPT_FREQ (e.g. 0.1 => every round(N*freq) steps) to hit the save/resume path fast.
    EXTRA="--checkpoint-freq ${CKPT_FREQ:-1.0}"
  else
    EXTRA="--init-on-meta --checkpoint-freq ${CKPT_FREQ:-1.0}"
    if [ "$GROUPED" = "1" ]; then
      echo "!! faithful uses per-expert FSDP; grouped-GEMM needs experts LOCAL. Turn on EP"
      echo "   instead:  DSPARK_EP=1 bash $0 faithful  (partitions experts + all-to-all)."
      echo "   Refusing grouped-GEMM on per-expert-FSDP faithful."; exit 2
    fi
  fi
elif [ "$MODE" = "reduced" ]; then
  NPROC="${NPROC:-1}"; LAYERS=1; EXPERTS=32; EXTRA=""; PORTS=""
  [ "$EP" = "1" ] && PORTS="HCCL_NPU_SOCKET_PORT_RANGE=61000-62000 HCCL_HOST_SOCKET_PORT_RANGE=60000-61000"
else
  echo "usage: $0 [reduced|faithful]"; exit 1
fi

# EP preconditions: >=2 cards and experts divisible by cards.
if [ "$EP" = "1" ]; then
  [ "$NPROC" -ge 2 ] || { echo "!! DSPARK_EP=1 needs NPROC>=2 (got $NPROC). e.g. DSPARK_EP=1 NPROC=2 bash $0 reduced"; exit 2; }
  [ $((EXPERTS % NPROC)) -eq 0 ] || { echo "!! EXPERTS=$EXPERTS not divisible by NPROC=$NPROC"; exit 2; }
fi

# NO_VAL=1 -> append --no-validation (skip the slow per-epoch val pass, train on full data).
if [ "$NOVAL" = "1" ]; then EXTRA="$EXTRA --no-validation"; fi
# INIT_MOE=1 -> append --init-moe-from-target (warm-start draft MoE from verifier layers).
if [ "$INITMOE" = "1" ]; then EXTRA="$EXTRA --init-moe-from-target"; fi

# ---- preflight ----
if [ -f "$CANN_ENV" ]; then
  set +e; source "$CANN_ENV"; set -e         # env scripts return nonzero benignly
else
  echo "WARN: CANN env not found at $CANN_ENV — pass CANN_ENV=/path/to/900env_npu.sh"
fi
mkdir -p "$RUN" 2>/dev/null || { echo "!! cannot create RUN=$RUN (different box user?) — override e.g. RUN=\$HOME/dspark_austin/run"; exit 1; }
rm -f "$HS_DIR"/hs_*.safetensors*                     # stale dumps collide with data row idx
export no_proxy=127.0.0.1,localhost,80.5.5.115,80.5.5.116 NO_PROXY=127.0.0.1,localhost,80.5.5.115,80.5.5.116
if ! curl -sf --noproxy '*' "$ENDPOINT/models" >/dev/null 2>&1; then
  echo "WARN: serve not reachable at $ENDPOINT — start 115/116 first (or set ENDPOINT=)."
fi

TS="$(date +%Y%m%d_%H%M%S)"
# NB: build TAG with if-blocks, NOT `TAG="...$([ ] && echo ...)"` — under set -e a
# command substitution whose test fails returns nonzero and silently kills the script.
TAG="$MODE"
if [ "$EP" = "1" ]; then TAG="${TAG}_ep"; fi
if [ "$GROUPED" = "1" ]; then TAG="${TAG}_grouped"; fi
LOG="$RUN/${TAG}_${TS}.log"
# Fresh per-run save-path so we DON'T auto-resume a stale (possibly non-EP) checkpoint
# from ./output. Override SAVE_PATH=<dir> to resume a specific run.
SAVE_PATH="${SAVE_PATH:-$RUN/ckpt_${TAG}_${TS}}"

echo "==================================================================="
echo " DSV4-DSpark TRAIN  mode=$MODE  nproc=$NPROC  ${LAYERS}L x ${EXPERTS}E  lr=$LR  epochs=$EPOCHS  ep=$EP  grouped_moe=$GROUPED  recompute=$RECOMPUTE  compile=$COMPILE  noval=$NOVAL  init_moe=$INITMOE"
echo " block=$BLOCK (drafts $((BLOCK-1)) tokens = gamma; slot 0 anchor)  seqlen=$SEQLEN  max_anchors=$MAX_ANCHORS"
echo " draft-forward tokens = max_anchors*block = $((MAX_ANCHORS*BLOCK))  (anchor util = $MAX_ANCHORS/$SEQLEN)"
echo " verifier=$VERIFIER"
echo " data=$DATA"
echo " 📋 log -> $LOG   (rank0 mirror also in $RUN/train_*.log)"
echo " 💾 save -> $SAVE_PATH"
echo "==================================================================="

nohup env \
  DSPARK_HS_DUMP=1 DSPARK_GROUPED_MOE="$GROUPED" DSPARK_EP="$EP" DSPARK_RECOMPUTE="$RECOMPUTE" DSPARK_COMPILE="$COMPILE" \
  PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}" \
  HCCL_CONNECT_TIMEOUT=1800 HCCL_EXEC_TIMEOUT=1800 $PORTS \
  torchrun --nproc_per_node "$NPROC" "$REPO_ROOT/scripts/train.py" \
    --speculator-type dsv4_dspark --served-model-name dsv4 \
    --num-layers "$LAYERS" --n-routed-experts "$EXPERTS" \
    --block-size "$BLOCK" --target-layer-ids 40 41 42 --max-anchors "$MAX_ANCHORS" \
    --total-seq-len "$SEQLEN" --mask-token-id "$MASK_TOKEN" \
    --draft-attn-impl sdpa --loss-fn '{"ce":0.1,"tv":0.9}' \
    --optimizer adamw --lr "$LR" --epochs "$EPOCHS" $EXTRA \
    --on-missing generate --on-generate delete \
    --hidden-states-path "$HS_DIR" --vllm-endpoint "$ENDPOINT" \
    --verifier-name-or-path "$VERIFIER" --data-path "$DATA" \
    --save-path "$SAVE_PATH" --log-dir "$RUN" \
  > "$LOG" 2>&1 &

echo ">>> started PID $!  |  tail -f $LOG"
echo ">>> watch: profile/grad_norm (blow-up before NaN), profile/fwd_ms (grouped => spikes gone),"
echo ">>>        train/loss (non-zero, decreasing), NaN => kill immediately."
