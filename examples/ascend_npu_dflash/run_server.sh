#!/usr/bin/env bash
# DFlash Qwen3-4B — eval/inference server (vLLM V1 spec-decode) on Ascend NPU.
#
# Serves the Qwen3-4B *verifier* and points --speculative-config at the trained
# DFlash *draft*. This is the team-internal eval-alignment recipe (so everyone's
# acceptance numbers are comparable). FORK-ONLY — never submitted upstream.
#
# Defaults are derived from config_qwen3_4b.sh + the training output, so a plain
#   bash examples/ascend_npu_dflash/run_server.sh
# just works after a training run. Override any var via the environment.
#
# Pair with run_eval.sh (the benchmark client). Source CANN first (or set
# CANN_ENV/NNAL_ENV below) so torch_npu / vllm-ascend load.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/config_qwen3_4b.sh"          # TARGET_MODEL, SAVE_DIR, ...

# ---------------- eval-specific knobs (env-overridable) ----------------
TARGET="${TARGET:-$TARGET_MODEL}"                          # verifier weights
DRAFT="${DRAFT:-$SAVE_DIR/checkpoint_best}"                # trained DFlash draft ckpt
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-15}"                   # = training block_size(16) - 1
EVAL_PORT="${EVAL_PORT:-30000}"
EVAL_CARD="${EVAL_CARD:-0}"                                # one free NPU card
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-2048}"
# NPU spec-decode gotcha: the scheduler reserves max_num_seqs*(1+num_spec_tokens)
# token slots/step. If vLLM refuses to start with a NEGATIVE max_num_scheduled_tokens,
# set EXTRA_FLAGS="--max-num-batched-tokens 8192" (and tell the team so all stay aligned).
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

# optional CANN / NNAL sourcing (override per box; default = standard install path)
CANN_ENV="${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
NNAL_ENV="${NNAL_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}"
# shellcheck disable=SC1090
[ -f "$CANN_ENV" ] && source "$CANN_ENV"
# shellcheck disable=SC1090
[ -f "$NNAL_ENV" ] && source "$NNAL_ENV"
# Running vLLM/vllm-ascend from source trees instead of the wheel? Prepend them:
#   export PYTHONPATH=/path/to/vllm:/path/to/vllm-ascend:$PYTHONPATH

# clean any stale vLLM (it forks / retitles its workers)
pkill -9 -f "vllm serve|EngineCore|multiproc_executor|Worker_TP" 2>/dev/null || true
sleep 3

echo ">>> eval serve | verifier=$TARGET"
echo ">>> draft=$DRAFT | num_speculative_tokens=$NUM_SPEC_TOKENS | port=$EVAL_PORT card=$EVAL_CARD"
echo ">>> ready check:  curl -s --noproxy '*' http://localhost:$EVAL_PORT/v1/models | head"

ASCEND_RT_VISIBLE_DEVICES="$EVAL_CARD" VLLM_USE_V1=1 \
vllm serve "$TARGET" \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-model-len "$EVAL_MAX_MODEL_LEN" \
    --speculative-config "{\"model\":\"$DRAFT\",\"num_speculative_tokens\":$NUM_SPEC_TOKENS,\"draft_tensor_parallel_size\":1}" \
    --host 0.0.0.0 \
    --port "$EVAL_PORT" \
    $EXTRA_FLAGS
