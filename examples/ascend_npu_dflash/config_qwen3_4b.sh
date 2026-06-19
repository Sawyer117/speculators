# Shared config for the Ascend-NPU Qwen3-4B DFlash run scripts.
# Edit values here, or override via the environment. Safe to source.
#
# 4B differs from 8B only by: verifier weights (hidden_size 2560, auto-derived),
# and a TP=1 device split (4B fits one card -> more cards for training).
# The tokenized dataset is tokenizer-only, so it is the SAME Arrow as 8B
# (the whole Qwen3 family shares the tokenizer) -> DATA_DIR points at the shared set.

export TARGET_MODEL="${TARGET_MODEL:-/share/canada_group_folder/ckpt/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
export DATA_DIR="${DATA_DIR:-/share/canada_group_folder/dataset/perfectblend_train_regen.half50.qwen3.seq3072}"
export OUTPUT_DIR="${OUTPUT_DIR:-./outputs/qwen3-4b-dflash-npu}"

# resolve OUTPUT_DIR to an absolute path so serve & train agree on HS_DIR
mkdir -p "$OUTPUT_DIR"
export OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

export HS_DIR="${HS_DIR:-$OUTPUT_DIR/hidden_states}"   # 4B hidden states ([tokens,6,2560]); separate from 8B
export SAVE_DIR="${SAVE_DIR:-$OUTPUT_DIR/checkpoints}"

export PORT="${PORT:-8001}"            # distinct from 8B (8000) in case both ever run
export SEQ_LEN="${SEQ_LEN:-3072}"
export SERVE_CARDS="${SERVE_CARDS:-0}"           # 4B fits TP=1 on one card
export TRAIN_CARDS="${TRAIN_CARDS:-1,2,3,4,5,6,7}"   # 7-way FSDP
export TP="${TP:-1}"
export NPROC="${NPROC:-7}"
export MASTER_PORT="${MASTER_PORT:-29534}"
