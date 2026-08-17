#!/usr/bin/env bash
# DSV4-DSpark + Correction head — TYS5537/dspark_next feature set on OUR model.
#
#   bash examples/ascend_npu_dflash/train_dsv4_dspark_correction.sh faithful
#   DRY_RUN=1 bash examples/ascend_npu_dflash/train_dsv4_dspark_correction.sh faithful
#
# This is a THIN WRAPPER: it only assembles EXTRA_ARGS and hands off to the proven
# `train_dsv4_dspark.sh`, which is left untouched. Every dimension, path, optimizer and
# schedule therefore stays exactly as in the run that produced the 5-epoch deliverable.
#
# ---------------------------------------------------------------------------------------
# PROVENANCE. The flag values below are transcribed from the collaborator's
# `dspark_qwen3_8b_sharegpt_online_ascend.sh` ("his best config"), which was tuned on a
# DENSE Qwen3-4B. What was taken and what was NOT:
#
#   TAKEN (feature switches — the thing under test):
#     the whole --correction-*, --dflash-* and --confidence-* block, verbatim.
#
#   NOT TAKEN (our model / our validated recipe wins — agreed explicitly):
#     --lr 6e-4          -> 2e-4. 6e-4 is the DeepSpec reference and it DIVERGED TO NaN on
#                           this stack around step 931 once warmup finished
#                           (train_dsv4_dspark.sh:32). His dense 4B tolerates it; our
#                           256-expert MoE does not.
#     --loss-fn tv 0.9   -> tv 1.8. NOT a disagreement: speculators' `tv_loss` computes TVD,
#                           which is half of L1, so the DSpark paper's l1_alpha=0.9 is 1.8
#                           here. His 0.9 is the paper coefficient applied to a TVD loss,
#                           i.e. half strength.
#     --block-size 7     -> 5. Kept for a clean A/B against `ep5p0-ropefix`: this run tests
#                           the Correction head, not the block width. (The block width is a
#                           separate, independently-motivated change — see the num_spec=7
#                           section of the eval ledger.)
#     --num-layers 5     -> 3, --target-layer-ids "1 9 17 25 33" -> "40 41 42": the released
#                           DSV4 draft geometry.
#     --draft-vocab-size 32000 -> NOT PASSED AT ALL. DSV4 trains on the full 129,280 vocab;
#                           passing this builds a head for the wrong vocabulary.
#     --speculator-type dspark -> dsv4_dspark (MLA + per-head sink + 256-expert MoE + mHC).
#     his MODEL / data path / vLLM-on-same-box NPU split -> our A2 layout.
#
# ⚠ DEPARTURE FROM OUR OWN BASELINE, deliberate and requested:
#     --no-confidence-detach-features. This fork defaults `confidence_detach_features` to
#     True (dsv4_dspark/core.py) precisely so that a merge from a sibling fork cannot flip
#     it as a side effect — every checkpoint in the eval ledger was trained with the
#     confidence inputs detached. His config turns it off, so this run does too, but it
#     means the confidence head now backpropagates into the draft. If this run regresses,
#     that is one of the two prime suspects (the other being the Correction head itself).
#
# ★ The three --dflash-* backbone flags below were NO-OPS on this model until the companion
#   patch to `dsv4_dspark/core.py::_backbone_forward` routed it through the inherited
#   `_fuse_target_hidden` / `_condition_noise_embedding` helpers. They now do something.
#   All of them are zero-initialised, so they start as exact identities.
# ---------------------------------------------------------------------------------------
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CORRECTION_ARGS=(
  # ---- correction core -------------------------------------------------------
  --enable-correction-head
  --correction-output-mode logits          # feed previous logits + a Markov-like vocab bias
  --correction-hidden-size 512
  --correction-rank 256
  --no-correction-lm-head-fusion           # hidden-mode-only optimisation; inert in logits mode
  --correction-num-layers 1
  --correction-num-heads 8
  --correction-gate-bias 0.0

  # ---- correction MoE: OFF (his baseline), values kept so enabling is one edit ----
  --no-correction-moe
  --correction-moe-shared-rank 128
  --correction-moe-expert-rank 64
  --correction-moe-num-experts 4
  --correction-moe-load-balance-weight 0.01
  --no-correction-moe-logit-routing

  # ---- representation supervision + recurrent feedback: ON -------------------
  --correction-hidden-aux-loss
  --correction-hidden-aux-weight 0.1
  --correction-hidden-feedback

  # ---- cross-block memory OFF; output composition ON -------------------------
  --no-correction-cross-block-memory
  --correction-memory-gate-bias -2.0
  --correction-project-corrected-hidden    # LMHead(h_DFlash + delta_hidden) + delta_logits

  # ---- gated Correction + Markov collaboration: ON ---------------------------
  --correction-with-markov
  --correction-markov-gate-bias -2.0
  --markov-head-type vanilla

  # ---- feedback curriculum: teacher forcing (ratio 0 = baseline) -------------
  --correction-generated-token-ratio 0.0
  --correction-generated-token-warmup 0.2
  --correction-generated-token-ramp 0.4
  --no-correction-rollout-metrics          # validation-only extra rollout pass
  --no-correction-base-diagnostics

  # ---- DFlash backbone experiments (live only with the _backbone_forward patch) ----
  --dflash-context-residual
  --no-dflash-verifier-final-residual      # ⚠ its tensor identification is unvalidated here
  --dflash-block-position-embedding
  --dflash-gated-layer-fusion
  --no-dflash-dfly-layer-residual          # requires gated-layer-fusion; his baseline is off
  --no-dflash-heterogeneous-kv-projections

  # ---- confidence head -------------------------------------------------------
  --enable-confidence-head
  --confidence-head-with-markov
  --confidence-head-alpha 1.0
  --confidence-length-alpha 0.0
  --confidence-loss-weighting match-draft  # DSpark paper default
  --no-confidence-detach-features          # ⚠ see the departure note above
  --first-error-focal-alpha 0.0
  --adaptive-loss none
  --no-ssal-curriculum
  --ssal-curriculum-start 0.1
  --ssal-curriculum-end 0.6
)

# `--dry-run` builds the model and the first batch, then exits — the only way to find out
# whether the Correction head constructs against MLA + 256-expert MoE + mHC without
# spending days of NPU time. It has NEVER been run on this geometry. Do it first.
[ "${DRY_RUN:-0}" = "1" ] && CORRECTION_ARGS+=(--dry-run)

export EXTRA_ARGS="${CORRECTION_ARGS[*]} ${EXTRA_ARGS:-}"

echo ">>> DSV4-DSpark + Correction (TYS5537/dspark_next feature set)"
echo ">>> dimensions/LR/loss = OURS; feature switches = HIS"
[ "${DRY_RUN:-0}" = "1" ] && echo ">>> DRY RUN — builds the model and one batch, then exits"

exec bash "$SCRIPT_DIR/train_dsv4_dspark.sh" "$@"
