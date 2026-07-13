#!/usr/bin/env bash
# Step 0 (CPU): tokenize TRAIN_DATA (jsonl) into the Arrow dataset at DATA_DIR.
# Run once before training.   bash prepare_qwen3_8b.sh   (do NOT source)
#
# Speed: NUM_WORKERS = CPU procs for dataset.map (default 8; raise it on big boxes).
# Optional random subset (shuffles by SEED, then keeps N — see preprocessing.py):
#   MAX_SAMPLES=750000 SEED=42 bash prepare_qwen3_8b.sh
[ "${BASH_SOURCE[0]}" != "$0" ] && { echo "Run with 'bash $0', do not source."; return 1 2>/dev/null; }
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/config.sh"

NUM_WORKERS="${NUM_WORKERS:-8}"
EXTRA=()
[ -n "${MAX_SAMPLES:-}" ] && EXTRA+=(--max-samples "$MAX_SAMPLES")
[ -n "${SEED:-}" ] && EXTRA+=(--seed "$SEED")

echo ">>> prepare: $TRAIN_DATA  ->  $DATA_DIR"
echo "    model=$TARGET_MODEL seq-len=$SEQ_LEN workers=$NUM_WORKERS ${MAX_SAMPLES:+max-samples=$MAX_SAMPLES seed=${SEED:-0}}"
python "$REPO_ROOT/scripts/prepare_data.py" \
  --model "$TARGET_MODEL" \
  --data "$TRAIN_DATA" \
  --output "$DATA_DIR" \
  --seq-length "$SEQ_LEN" \
  --num-preprocessing-workers "$NUM_WORKERS" \
  --overwrite \
  --trust-remote-code \
  ${EXTRA[@]+"${EXTRA[@]}"}
