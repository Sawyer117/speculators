#!/usr/bin/env bash
# One-shot HS TRAIN/SERVE-consistency test (MODE 1) — traceable, no inline snippets.
#
# Fires faithful HS-dump requests (from the training Arrow input_ids) at the live
# DSpark HS-dump serve at TWO concurrency levels — conc=1 (clean baseline) and
# conc=$OVERSUB (the training over-subscription level NPROC*NUM_WORKERS) to the SAME
# rows — then runs dsv4_hs_integrity_check.py on each. The check compares
# argmax(lm_head(dumped final HS)) vs the rollout token; the tail-decile mismatch is
# the signal.
#
# VERDICT:
#   conc1 tail ~0% & conc$OVERSUB tail ~0%   -> dump clean at every load; the gap is
#                                               NOT over-subscription -> escalate to MODE 2
#                                               (dsv4_hs_integrity_check.py --hf-model).
#   conc1 tail ~0% & conc$OVERSUB tail ~11%  -> bf16 OVER-SUBSCRIPTION GARBAGE confirmed
#                                               (== the slot0 train/serve gap 0.819-0.704)
#                                               -> fix: throttle dump concurrency <= serve
#                                               cap + regenerate HS + retrain.
#   conc1 tail already ~11%                   -> the dump build itself emits wrong HS (not
#                                               load) -> capture/build/precision, go MODE 2.
#
# Prereqs: the HS-dump serve is UP (separate process; killing training doesn't kill it),
# env has `datasets` + `openai` + `safetensors` + torch (the training env, e.g. austin).
#
# Usage (A2 defaults shown; override any via env):
#   ENDPOINT=http://80.5.5.115:7000/v1 \
#     bash examples/ascend_npu_dflash/run_hs_consistency_check.sh
# HS_DIR defaults to the A2 serve dump dir; the fire step also auto-locates it (handles the
# 'dataset/' layer) and fails loud if it truly can't find the files, so a wrong HS_DIR won't hang.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

ENDPOINT="${ENDPOINT:?set ENDPOINT=http://<HS-dump-serve-ip>:7000/v1 (the austin dump serve, NOT the eval serve)}"
HS_DIR="${HS_DIR:-/share/canada_group_folder/dataset/dsv4_hs_dump}"   # confirmed A2 serve DSPARK_HS_DIR (note the dataset/ layer)
ARROW="${ARROW:-/share/canada_group_folder/dataset/open_perfectblend.dsv4_rollout/arrow_0720_77w}"
MODEL="${MODEL:-/share/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16}"
N="${N:-8}"                 # sequences for the clean baseline
OVERSUB="${OVERSUB:-96}"    # over-subscription level = your training NPROC*NUM_WORKERS
OUT1="${OUT1:-$HOME/hs_conc1}"
OUT2="${OUT2:-$HOME/hs_conc${OVERSUB}}"
FIRE="$HERE/dsv4_fire_hs_dumps.py"
CHECK="$HERE/dsv4_hs_integrity_check.py"

echo "############## conc=1  (CLEAN baseline) ##############"
ENDPOINT="$ENDPOINT" ARROW="$ARROW" HS_DIR="$HS_DIR" \
  python "$FIRE" --out "$OUT1" --n "$N" --concurrency 1 --id-base 800000 --start-row 0
echo "---- inspect (confirm dump format: hidden_states [T,4,H] + token_ids [T]) ----"
python "$CHECK" --hs-dir "$OUT1" --inspect
echo "---- MODE 1 self-consistency @ conc=1 ----"
python "$CHECK" --hs-dir "$OUT1" --model-dir "$MODEL"

echo
echo "############## conc=$OVERSUB  (OVER-SUBSCRIPTION, same rows) ##############"
ENDPOINT="$ENDPOINT" ARROW="$ARROW" HS_DIR="$HS_DIR" \
  python "$FIRE" --out "$OUT2" --n "$OVERSUB" --concurrency "$OVERSUB" --id-base 900000 --start-row 0
echo "---- MODE 1 self-consistency @ conc=$OVERSUB ----"
python "$CHECK" --hs-dir "$OUT2" --model-dir "$MODEL"

echo
echo "############## READ THE TWO 'AGGREGATE ... mismatch' + tail-decile lines ##############"
echo "conc1 clean & conc$OVERSUB ~11%  => over-subscription garbage = root cause."
echo "conc1 already ~11%               => dump build itself wrong (not load) => MODE 2 (--hf-model $MODEL)."
echo "both ~0%                         => not the dumped HS => MODE 2 / dumper-writer branch."
