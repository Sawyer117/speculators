#!/usr/bin/env bash
# Response ROLLOUT for Qwen3-4B DFlash on-policy training data, with the aligned
# defaults baked in (matches SpecForge / the colleague's setup):
#   - open-perfectblend prompts (the confirmed seed dataset)
#   - greedy (--temperature 0), no top_p (irrelevant at temp 0)
#   - --no-thinking (Qwen3 thinking off, so responses fit the seq-len budget)
#   - --max-tokens 3072 (= training seq-len; longer is wasted, truncated in prepare)
#
# PREREQ: a vLLM serve of the target is up on $PORT
#   (start it with rollout_serve_qwen3_4b.sh). The generated jsonl feeds
#   scripts/prepare_data.py, then train (DFlash ignores USE_OFF_POLICY — EAGLE-only).
#
# Env knobs (all overridable):
#   LIMIT (empty=full), CONCURRENCY, OUTFILE, DATASET_PATH, PORT, MAX_TOKENS, TEMPERATURE
#
# Run with bash, do NOT source.
[ "${BASH_SOURCE[0]}" != "$0" ] && { echo "Run with 'bash $0', do not source."; return 1 2>/dev/null; }
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PORT="${PORT:-8000}"
DATASET_PATH="${DATASET_PATH:-/share/canada_group_folder/dataset/open_perfectblend_full.jsonl}"
OUTFILE="${OUTFILE:-/share/canada_group_folder/dataset/open_perfectblend.qwen3-4b-rollout.jsonl}"
CONCURRENCY="${CONCURRENCY:-256}"     # plain serve: 256; DFlash spec-decode serve: 32
MAX_TOKENS="${MAX_TOKENS:-3072}"
TEMPERATURE="${TEMPERATURE:-0}"       # greedy = aligned default
LIMIT="${LIMIT:-}"                    # empty = full dataset; set e.g. 500 for a quick test

export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1

LIMIT_FLAG=""
[ -n "$LIMIT" ] && LIMIT_FLAG="--limit $LIMIT"

echo ">>> rollout | port=$PORT temp=$TEMPERATURE max_tokens=$MAX_TOKENS concurrency=$CONCURRENCY limit=${LIMIT:-full}"
echo ">>> seed=$DATASET_PATH"
echo ">>> out=$OUTFILE"

# shellcheck disable=SC2086
python "$REPO_ROOT/scripts/response_regeneration/script.py" \
  --endpoint "http://127.0.0.1:$PORT/v1/chat/completions" \
  --dataset open_perfectblend \
  --dataset-path "$DATASET_PATH" \
  --no-thinking \
  --temperature "$TEMPERATURE" \
  --max-tokens "$MAX_TOKENS" \
  --concurrency "$CONCURRENCY" \
  $LIMIT_FLAG \
  --outfile "$OUTFILE"

echo ">>> done -> $OUTFILE"
echo ">>> next: prepare_data.py --data $OUTFILE --output <Arrow> --seq-length 3072 ; then train (DFlash ignores USE_OFF_POLICY, EAGLE-only)"
