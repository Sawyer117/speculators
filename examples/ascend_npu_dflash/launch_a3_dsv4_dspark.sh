#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# PINNED launcher for the A3 (183 train + 182 serve, shared NFS) DSV4-DSpark
# faithful full-alignment run. Thin wrapper over train_dsv4_dspark.sh that bakes
# the CONFIRMED box paths + the A3 standard env as DEFAULTS, so nobody has to
# reconstruct the env line by hand (that's how the DATA path got mis-typed once
# → load_from_disk FileNotFoundError).
#
# Usage:
#   bash launch_a3_dsv4_dspark.sh                # defaults: LR 3e-4, EP16 option-A fp32, 77W, 10 ep
#   LR=2.5e-4 bash launch_a3_dsv4_dspark.sh      # tweak ANY knob via env (all are ${VAR:-default})
#   DATA=/mnt/nfs/.../other_arrow bash launch_a3_dsv4_dspark.sh
#
# The underlying runner backgrounds torchrun (nohup) and prints the tail cmd.
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── CONFIRMED A3 183/182 shared-NFS paths (2026-07-21) ──────────────────────
export ENDPOINT="${ENDPOINT:-http://80.48.17.182:7000/v1}"
export HS_DIR="${HS_DIR:-/mnt/nfs/canada_group_folder/dsv4_hs_dump}"                 # MUST == 182 serve DSPARK_HS_DIR
export DATA="${DATA:-/mnt/nfs/canada_group_folder/dataset/open_perfectblend.dsv4_rollout/arrow_0720_77w}"
export VERIFIER="${VERIFIER:-/mnt/nfs/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16}"
export RUN="${RUN:-$HOME/dsv4_run}"                                                  # $HOME is LOCAL disk on 183 (14T)
export CANN_ENV="${CANN_ENV:-/home/a00652497/900env_npu.sh}"

# ── A3 standard training knobs ──────────────────────────────────────────────
# option-A fp32 experts = EP=1 + A3 128G AND do NOT set BF16_EXPERTS (runner default "auto" = option A).
export DSPARK_EP="${DSPARK_EP:-1}"
export NPROC="${NPROC:-16}"           # A3 = 8 chips x 2 = 16 logical devices = world_size 16 (FSDP16 + EP16)
export MAX_ANCHORS="${MAX_ANCHORS:-512}"
export RECOMPUTE="${RECOMPUTE:-1}"
export COMPILE="${COMPILE:-1}"
export NO_VAL="${NO_VAL:-1}"
export INIT_LAYER="${INIT_LAYER:-1}"  # warm-start whole draft layer from verifier [40,41,42]
export EPOCHS="${EPOCHS:-10}"         # 10 (not A2's 5): 2x DP16 halves steps/epoch, so 10 ep = same total updates
export CKPT_FREQ="${CKPT_FREQ:-0.5}"  # half-epoch saves → earlier eval point
export LR="${LR:-3e-4}"               # sqrt batch-scaling from A2's 2e-4 (A3 = 2x DP). 6e-4 NaN'd → keep <=3e-4.

cat <<CFG
── PINNED A3 DSV4-DSpark launch ────────────────────────────────────────────
  DATA      = $DATA
  VERIFIER  = $VERIFIER
  HS_DIR    = $HS_DIR
  ENDPOINT  = $ENDPOINT
  RUN       = $RUN
  NPROC=$NPROC  EP=$DSPARK_EP  COMPILE=$COMPILE  MAX_ANCHORS=$MAX_ANCHORS  EPOCHS=$EPOCHS  CKPT_FREQ=$CKPT_FREQ  LR=$LR
─────────────────────────────────────────────────────────────────────────────
  * HS_DIR must equal the 182 serve's DSPARK_HS_DIR, and that serve must be UP.
  * COMPILE=1: if you just pulled, verify moe_compile.py is the compiled (117724b)
    version — the branch tip may carry the broken fully-eager one:
      git checkout 117724b -- src/speculators/models/dsv4_dspark/backbone/moe_compile.py
  * First ~100 steps: watch grad_norm/loss for NaN (LR 3e-4 + tv 1.8 is a new combo);
    if it NaNs early, drop LR (e.g. 2.5e-4) or lengthen WARMUP_RATIO=0.06.
─────────────────────────────────────────────────────────────────────────────
CFG

exec bash "$SCRIPT_DIR/train_dsv4_dspark.sh" faithful
