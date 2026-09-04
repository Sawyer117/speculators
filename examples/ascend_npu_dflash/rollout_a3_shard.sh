#!/usr/bin/env bash
# One-command-per-A3-node DSV4-Flash **bf16** rollout of a pre-split shard.
#
# A3 twin of rollout_shard.sh (which is w8a8 / DP1·TP8). Here each A3 node runs an
# INDEPENDENT single-node **bf16** serve (16 logical devices, DP2 / TP8 / **EP16**,
# graph FULL_DECODE_ONLY) and rolls out its own shard — embarrassingly parallel,
# zero cross-node comms. We default bf16 (not w8a8) so the rollout precision matches
# the training data end-to-end (see docs/.../ascend-npu-dsv4-a3-singlenode-benchmark.md:
# measured 1.15 rows/s, 657 gen tok/s, clean).
#
# Prereq: pre-split the dataset into shard_00.jsonl .. shard_(N-1).jsonl once:
#   split -n r/N -d -a 2 --additional-suffix=.jsonl full.jsonl $SHARDDIR/shard_
# (round-robin r/N => balanced shards, no shuffle needed).
#
# Usage (under nohup/tmux — hours to days):
#   nohup bash examples/ascend_npu_dflash/rollout_a3_shard.sh 0 > ~/a3_shard0.log 2>&1 &
#   ...node 1: ... 1 ; node 2: ... 2 ; up to N-1  (one A3 box per shard)
#
# Override via env, e.g. `CONC=64 PORT=7000 bash rollout_a3_shard.sh 3`:
#   MODEL CANN_ENV CONDA_ENV BASE SHARDDIR OUTDIR PORT CONC MAXTOK EAGER
# NB: no `set -u` — sourcing CANN/conda references unbound vars ($ZSH_VERSION).
set -o pipefail

SHARD_ID="${1:?usage: bash rollout_a3_shard.sh <SHARD_ID 0..N-1>}"
SID=$(printf "%02d" "$((10#$SHARD_ID))")

MODEL="${MODEL:-/home/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16}"   # A3 convention: /home/canada_group_folder (symlink-faked per box)
CANN_ENV="${CANN_ENV:-/home/a00652497/900env_npu.sh}"     # CANN 9.0.0
CONDA_ENV="${CONDA_ENV:-dspark-dsv4-base}"
BASE="${BASE:-/home/canada_group_folder/dataset/open_perfectblend.dsv4_rollout}"
SHARDDIR="${SHARDDIR:-$BASE/shards}"
OUTDIR="${OUTDIR:-$BASE/out}"
PORT="${PORT:-7000}"
CONC="${CONC:-64}"          # 64 = 32 seqs/DP-replica = validated-clean ceiling (KV-overflow rule)
MAXTOK="${MAXTOK:-3072}"
EAGER="${EAGER:-0}"         # 0 = graph FULL_DECODE_ONLY (peak, ~3.5x eager). 1 = --enforce-eager (bring-up)
# ★ QUALITY GATE — gsm8k-score THIS serve before trusting the rollout. errors=0 is NOT a quality
# signal (KV-overflow / bad-parallel path returns HTTP 200 garbage). bf16 ref (AtomGit 2-node) ~97.27%.
GATE="${GATE:-1}"          # 1 = run gsm8k gate before rollout; 0 = skip (only if already validated)
GATE_LIMIT="${GATE_LIMIT:-0}"     # gsm8k questions for the gate; 0 = FULL 1319 test set (no subsample noise, ~15min)
GATE_MIN="${GATE_MIN:-90}"        # abort rollout if acc < this (garbage tanks to <50; bf16 ref ~97.27, measured 96.59)
# Box has NO HF access (proxy can't reach the LFS CDN) → gate reads gsm8k test from a LOCAL parquet.
# curl -kL https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet -o <this>
GATE_GSM8K="${GATE_GSM8K:-/home/canada_group_folder/dataset/gsm8k/main/test-00000-of-00001.parquet}"

TP="${TP:-8}"; DP="${DP:-2}"
MAXLEN="${MAXLEN:-8192}"; MAXBATCHTOK="${MAXBATCHTOK:-8192}"
GPUUTIL="${GPUUTIL:-0.9}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHARD="$SHARDDIR/shard_$SID.jsonl"
OUT="$OUTDIR/rollout_$SID.jsonl"
SERVE_LOG="$OUTDIR/serve_a3_$SID.log"
mkdir -p "$OUTDIR"
[ -f "$SHARD" ] || { echo "!! shard not found: $SHARD"; exit 2; }

# --- env (single node; no cross-node socket/HCCL_IF_IP — 16 cards talk over HCCS) ---
# shellcheck disable=SC1090
source "$CANN_ENV"
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "$CONDA_ENV"
export OMP_PROC_BIND=false OMP_NUM_THREADS=10 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACL_OP_INIT_MODE=1 TASK_QUEUE_ENABLE=1 HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=1024
export USE_MULTI_BLOCK_POOL=1 USE_MULTI_GROUPS_KV_CACHE=1 VLLM_ASCEND_BALANCE_SCHEDULING=1
# ★ A3-SPECIFIC (official vllm-ascend A3 recipe) — enable flag + fused-MC2 MoE fast path.
export ASCEND_A3_ENABLE="${ASCEND_A3_ENABLE:-1}"
export VLLM_ASCEND_ENABLE_FUSED_MC2="${VLLM_ASCEND_ENABLE_FUSED_MC2:-1}"
export VLLM_ASCEND_ENABLE_FLASHCOMM1="${VLLM_ASCEND_ENABLE_FLASHCOMM1:-1}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"

EAGER_FLAG=""; GRAPH_ARGS=()
if [ "$EAGER" = "1" ]; then EAGER_FLAG="--enforce-eager"
else GRAPH_ARGS=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'); fi

# --- (re)start this node's bf16 A3 serve ---
pkill -9 -u "$USER" -f vllm 2>/dev/null; sleep 15
echo ">>> [A3 shard $SID] starting bf16 serve on :$PORT  DP$DP/TP$TP/EP16 eager=$EAGER (log $SERVE_LOG)"
nohup vllm serve "$MODEL" --served-model-name dsv4 --port "$PORT" \
  --data-parallel-size "$DP" --data-parallel-size-local "$DP" \
  --tensor-parallel-size "$TP" --enable-expert-parallel \
  --tokenizer-mode deepseek_v4 \
  --max-model-len "$MAXLEN" --max-num-seqs "$CONC" --block-size 128 \
  --max-num-batched-tokens "$MAXBATCHTOK" \
  --gpu-memory-utilization "$GPUUTIL" --no-enable-prefix-caching --async-scheduling \
  --additional-config '{"enable_cpu_binding":true,"multistream_overlap_shared_expert":true}' \
  --safetensors-load-strategy prefetch \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":16}' \
  $EAGER_FLAG "${GRAPH_ARGS[@]}" > "$SERVE_LOG" 2>&1 &
echo ">>> serve PID $!"

# --- wait until ready (fail fast on startup error) ---
until curl -s --noproxy '*' "http://localhost:$PORT/v1/models" >/dev/null 2>&1; do
  if grep -qiE "error|traceback|out of memory|failed to start" "$SERVE_LOG"; then
    echo "!! serve failed to start — tail of $SERVE_LOG:"; tail -25 "$SERVE_LOG"; exit 3
  fi
  sleep 10
done
echo ">>> serve READY"

# --- QUALITY GATE: gsm8k-score this exact serve before committing the rollout ---
if [ "$GATE" = "1" ]; then
  GATE_LOCAL=(); [ -f "$GATE_GSM8K" ] && GATE_LOCAL=(--local-file "$GATE_GSM8K")
  echo ">>> [gate] gsm8k quality check ($GATE_LIMIT q, temp0) — bf16 ref ~97.27%, min $GATE_MIN%  local=${GATE_GSM8K:-<HF>}"
  GATE_OUT=$(python "$REPO_ROOT/examples/ascend_npu_dflash/gsm8k_eval.py" \
    --endpoint "http://127.0.0.1:$PORT/v1/chat/completions" --model dsv4 \
    --limit "$GATE_LIMIT" --concurrency "$CONC" "${GATE_LOCAL[@]}" --dump "$OUTDIR/gsm8k_a3_$SID.jsonl")
  echo "$GATE_OUT"
  ACC=$(echo "$GATE_OUT" | grep -oE '= [0-9.]+%' | tr -d '= %')
  if ! awk "BEGIN{exit !(${ACC:-0} >= $GATE_MIN)}"; then
    echo "!! [gate] gsm8k ${ACC:-?}% < $GATE_MIN% — serve config SUSPECT (garbage/KV-overflow?)."
    echo "!! ABORTING before rollout. Inspect $OUTDIR/gsm8k_a3_$SID.jsonl + $SERVE_LOG. Serve left up on :$PORT."
    exit 4
  fi
  echo ">>> [gate] PASS: gsm8k ${ACC}% >= $GATE_MIN% — proceeding to rollout."
fi

# --- rollout this shard (resume-safe) ---
echo ">>> [A3 shard $SID] rollout $SHARD (conc=$CONC max_tokens=$MAXTOK) -> $OUT"
python "$REPO_ROOT/scripts/response_regeneration/script.py" \
  --endpoint "http://127.0.0.1:$PORT/v1/chat/completions" \
  --dataset open-perfectblend --dataset-path "$SHARD" \
  --temperature 0 --max-tokens "$MAXTOK" --concurrency "$CONC" --resume \
  --outfile "$OUT"

echo ">>> [A3 shard $SID] DONE: $(wc -l < "$OUT") lines in $OUT"
echo ">>> (serve still on :$PORT — pkill -9 -u \$USER -f vllm to free the cards)"
