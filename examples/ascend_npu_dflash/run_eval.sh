#!/usr/bin/env bash
# DFlash eval CLIENT — team-standard acceptance/throughput benchmark against a
# running run_server.sh. FORK-ONLY (team-internal alignment) — never upstream.
#
# One command, team-aligned defaults so `bash run_eval.sh` just works. The
# determinism knobs that MUST match across the team are pinned here + in
# Evaluator.py (temperature=0, top_p=1, top_k=1, and a fixed random.seed(42)
# for sample shuffling). Override any var via the environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/config_qwen3_4b.sh"          # TARGET_MODEL

TARGET="${TARGET:-$TARGET_MODEL}"                          # = --model passed to vllm serve
EVAL_PORT="${EVAL_PORT:-30000}"
BASE_URL="${BASE_URL:-http://localhost:$EVAL_PORT}"
DATASET="${DATASET:-all}"                                  # gsm8k/math500/humaneval/mbpp/mt-bench
CONCURRENCY="${CONCURRENCY:-8}"
WARMUP="${WARMUP:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0}"                            # 0 = greedy (deterministic, headline)

# corp proxy (netentsec) intercepts localhost -> /metrics & /v1 return 504; bypass it
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"

python "$SCRIPT_DIR/Evaluator.py" \
    --base-url "$BASE_URL" \
    --model "$TARGET" \
    --dataset "$DATASET" \
    --concurrency "$CONCURRENCY" \
    --warmup-steps "$WARMUP" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE"
