#!/usr/bin/env bash
# Plain vLLM serve of the Qwen3-4B target for RESPONSE ROLLOUT
# (scripts/response_regeneration) on Ascend NPU.
#
#   DEFAULT          -> plain decode (no draft). Max throughput for bulk rollout.
#   DRAFT=<ckpt>     -> serve that DFlash draft instead = speculative decoding.
#   (or pass --draft <ckpt>)
#
# Rollout is a high-throughput batch job, so plain usually wins; DFlash spec-decode
# needs --max-num-seqs 32 (spec scheduling) which caps concurrency. But it's cheap to
# A/B: run a few hundred prompts each way and compare the regen script's it/s.
#
# The generated responses are IDENTICAL either way (spec-decode is lossless) — this
# toggle only affects SPEED, not the data.
#
# Run with bash, do NOT source.
[ "${BASH_SOURCE[0]}" != "$0" ] && { echo "Run with 'bash $0', do not source."; return 1 2>/dev/null; }
set -euo pipefail

# --- args / env ---
DRAFT="${DRAFT:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --draft) DRAFT="$2"; shift 2 ;;
    *) echo "unknown arg: $1 (only --draft <ckpt> supported)"; exit 1 ;;
  esac
done

# Target verifier (auto-resolve the HF-cache snapshot; override with MODEL=...).
MODEL="${MODEL:-$(ls -d /share/canada_group_folder/ckpt/models--Qwen--Qwen3-4B/snapshots/*/ 2>/dev/null | head -1)}"
MODEL="${MODEL%/}"
[ -z "$MODEL" ] && { echo "MODEL not found — set MODEL=/path/to/Qwen3-4B (snapshot dir)"; exit 1; }

PORT="${PORT:-8000}"
SERVE_CARDS="${SERVE_CARDS:-0,1,2,3,4,5,6,7}"
VLLM_DP="${VLLM_DP:-8}"
TP="${TP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-3072}"      # match training seq-len; longer is wasted (truncated in prepare)
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"          # only used in DFlash mode (spec scheduling needs it)

# NPU env
export OMP_NUM_THREADS=1 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1

# --- build serve command ---
EXTRA=()
if [ -n "$DRAFT" ]; then
  MODE="DFlash spec-decode"
  SERVE_TARGET="${DRAFT%/}"                 # serving a DFlash ckpt resolves verifier + draft
  EXTRA=(--max-num-batched-tokens 8192 --max-num-seqs "$MAX_NUM_SEQS")
else
  MODE="plain (no draft)"
  SERVE_TARGET="$MODEL"
fi

LOG="/tmp/rollout_serve_${PORT}.log"
PIDF="/tmp/rollout_serve_${PORT}.pid"
echo ">>> rollout serve | mode=$MODE"
echo ">>> serve=$SERVE_TARGET"
echo ">>> cards=$SERVE_CARDS DP=$VLLM_DP TP=$TP port=$PORT max-model-len=$MAX_MODEL_LEN gpu-mem=$GPU_MEM_UTIL${DRAFT:+ max-num-seqs=$MAX_NUM_SEQS}"
echo ">>> log -> $LOG"

nohup env ASCEND_RT_VISIBLE_DEVICES="$SERVE_CARDS" vllm serve "$SERVE_TARGET" \
  --host 127.0.0.1 --port "$PORT" --api-key "" \
  --data-parallel-size "$VLLM_DP" --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM_UTIL" \
  "${EXTRA[@]}" \
  > "$LOG" 2>&1 &

SERVE_PID=$!
echo "$SERVE_PID" > "$PIDF"
echo ">>> started, pid=$SERVE_PID (saved to $PIDF)"
echo ">>> wait ready: until curl -sf --noproxy '*' http://localhost:$PORT/v1/models >/dev/null 2>&1; do sleep 5; done; echo READY"
echo ">>> stop:       pkill -f 'vllm serve|EngineCore|APIServer'"
