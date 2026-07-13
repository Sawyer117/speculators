# Shared config for the Ascend-NPU Qwen3-8B DFlash run scripts.
# Edit values here, or override any of them via the environment. Safe to source.
#
# Data flow:  TRAIN_DATA (raw jsonl) --prepare--> DATA_DIR (Arrow) --train-->
#             checkpoints in SAVE_DIR;  serve writes hidden states to HS_DIR.

export TARGET_MODEL="${TARGET_MODEL:-/share/canada_group_folder/ckpt/Qwen3-8B}"
export TRAIN_DATA="${TRAIN_DATA:-/share/canada_group_folder/dataset/perfectblend_train_10ksubset.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-./outputs/qwen3-8b-dflash-npu}"

# resolve OUTPUT_DIR to an absolute path so serve & train agree on HS_DIR
mkdir -p "$OUTPUT_DIR"
export OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

export DATA_DIR="${DATA_DIR:-$OUTPUT_DIR/train_data}"      # Arrow dataset (prepare output = train input)
export HS_DIR="${HS_DIR:-$OUTPUT_DIR/hidden_states}"       # online hidden-state exchange (serve writes / train reads)
export SAVE_DIR="${SAVE_DIR:-$OUTPUT_DIR/checkpoints}"

export PORT="${PORT:-8000}"
export SEQ_LEN="${SEQ_LEN:-3072}"
export SERVE_CARDS="${SERVE_CARDS:-0,1}"
export TRAIN_CARDS="${TRAIN_CARDS:-2,3,4,5,6,7}"
export TP="${TP:-2}"
export NPROC="${NPROC:-6}"
export MASTER_PORT="${MASTER_PORT:-29533}"
