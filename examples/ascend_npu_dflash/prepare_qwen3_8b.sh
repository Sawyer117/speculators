#!/usr/bin/env bash
# Step 0 (CPU): tokenize TRAIN_DATA (jsonl) into the Arrow dataset at DATA_DIR.
# Run once before training.   bash prepare_qwen3_8b.sh   (do NOT source)
[ "${BASH_SOURCE[0]}" != "$0" ] && { echo "Run with 'bash $0', do not source."; return 1 2>/dev/null; }
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/config.sh"

echo ">>> prepare: $TRAIN_DATA  ->  $DATA_DIR   (model=$TARGET_MODEL, seq-len=$SEQ_LEN)"
python "$REPO_ROOT/scripts/prepare_data.py" \
  --model "$TARGET_MODEL" \
  --data "$TRAIN_DATA" \
  --output "$DATA_DIR" \
  --seq-length "$SEQ_LEN" \
  --overwrite \
  --trust-remote-code
