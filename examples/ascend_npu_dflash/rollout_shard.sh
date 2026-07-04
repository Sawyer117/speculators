#!/usr/bin/env bash
# One-command-per-machine DSV4-Flash w8a8 rollout of a pre-split shard.
#
# Each machine runs an INDEPENDENT single-node serve (8 NPUs, DP1/TP8) and rolls
# out its own shard — embarrassingly parallel, zero cross-machine comms. This is
# NOT a distributed serve (w8a8 fits on 8 cards).
#
# Prereq: the dataset was pre-split into shard_00.jsonl .. shard_(N-1).jsonl via
#   split -n r/N -d -a 2 --additional-suffix=.jsonl full.jsonl $SHARDDIR/shard_
# (round-robin r/N => balanced shards, no shuffle needed).
#
# Usage (run under nohup/tmux — it takes hours to days):
#   nohup bash examples/ascend_npu_dflash/rollout_shard.sh 0 > ~/shard0.log 2>&1 &
#   ...machine 1: ... 1 ; machine 2: ... 2 ; up to N-1
#
# Override any of these via env, e.g. `CONC=64 PORT=7000 bash rollout_shard.sh 3`:
#   MODEL CANN_ENV CONDA_ENV SHARDDIR OUTDIR PORT CONC MAXTOK
#
# Config choices baked in (measured, see docs/deployment/ascend-npu-dsv4-rollout-benchmark.md):
#   - AR (NO speculative-config): MTP/spec is a net throughput LOSS at batch.
#   - max-num-seqs 64 + multistream_overlap true: throughput saturates ~64 on 284B MoE.
#   - --resume: crash-safe; rerun the same command to skip already-generated rows.
set -uo pipefail

SHARD_ID="${1:?usage: bash rollout_shard.sh <SHARD_ID 0..N-1>}"
SID=$(printf "%02d" "$SHARD_ID")

MODEL="${MODEL:-/share/canada_group_folder/ckpt/DeepSeek-V4-Flash-w8a8-mtp}"
CANN_ENV="${CANN_ENV:-/home/a00652497/900env_npu.sh}"
CONDA_ENV="${CONDA_ENV:-dspark-dsv4-base}"
BASE="${BASE:-/share/canada_group_folder/dataset/open_perfectblend.dsv4_rollout}"
SHARDDIR="${SHARDDIR:-$BASE/shards}"
OUTDIR="${OUTDIR:-$BASE/out}"
PORT="${PORT:-7000}"
CONC="${CONC:-64}"
MAXTOK="${MAXTOK:-3072}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHARD="$SHARDDIR/shard_$SID.jsonl"
OUT="$OUTDIR/rollout_$SID.jsonl"
SERVE_LOG="$OUTDIR/serve_$SID.log"
mkdir -p "$OUTDIR"
[ -f "$SHARD" ] || { echo "!! shard not found: $SHARD"; exit 2; }

# --- env ---
# shellcheck disable=SC1090
source "$CANN_ENV"
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "$CONDA_ENV"
export OMP_PROC_BIND=false OMP_NUM_THREADS=8 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACL_OP_INIT_MODE=1 VLLM_ASCEND_ENABLE_FLASHCOMM1=1 USE_MULTI_GROUPS_KV_CACHE=1
export TASK_QUEUE_ENABLE=1 HCCL_OP_EXPANSION_MODE="AIV" HCCL_BUFFSIZE=512 USE_MULTI_BLOCK_POOL=1

# --- (re)start this machine's serve ---
pkill -9 -u "$USER" -f vllm 2>/dev/null; sleep 15
echo ">>> [shard $SID] starting AR serve on :$PORT (log $SERVE_LOG)"
nohup vllm serve "$MODEL" --served-model-name dsv4 \
  --data-parallel-size 1 --tensor-parallel-size 8 --enable-expert-parallel \
  --quantization ascend \
  --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --enable-auto-tool-choice \
  --max-model-len 135168 --max-num-seqs "$CONC" --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.92 --block-size 128 \
  --safetensors-load-strategy prefetch --no-enable-prefix-caching --async-scheduling \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":16}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[8,16,24,32,48,64]}' \
  --additional-config '{"enable_cpu_binding":true,"multistream_overlap_shared_expert":true}' \
  --port "$PORT" > "$SERVE_LOG" 2>&1 &
echo ">>> serve PID $!"

# --- wait until ready (fail fast on startup error) ---
until curl -s --noproxy '*' "http://localhost:$PORT/v1/models" >/dev/null 2>&1; do
  if grep -qiE "error|traceback|out of memory|failed to start" "$SERVE_LOG"; then
    echo "!! serve failed to start — tail of $SERVE_LOG:"; tail -25 "$SERVE_LOG"; exit 3
  fi
  sleep 10
done
echo ">>> serve READY"

# --- rollout this shard (resume-safe) ---
echo ">>> [shard $SID] rollout $SHARD (conc=$CONC max_tokens=$MAXTOK) -> $OUT"
python "$REPO_ROOT/scripts/response_regeneration/script.py" \
  --endpoint "http://127.0.0.1:$PORT/v1/chat/completions" \
  --dataset open_perfectblend --dataset-path "$SHARD" \
  --temperature 0 --max-tokens "$MAXTOK" --concurrency "$CONC" --resume \
  --outfile "$OUT"

echo ">>> [shard $SID] DONE: $(wc -l < "$OUT") lines in $OUT"
echo ">>> (serve still running on :$PORT — pkill -9 -u \$USER -f vllm to free the cards)"
