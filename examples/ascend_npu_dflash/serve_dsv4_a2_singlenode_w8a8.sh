#!/usr/bin/env bash
# DeepSeek-V4-Flash-0731-**w8a8** single-node serve WITH DSpark speculative decoding, on A2.
#
# Transcribed from the "A2 series with dspark" recipe in vllm-ascend
# docs/source/tutorials/models/DeepSeek-V4-Flash.md (main). ⚠ That is a DIFFERENT recipe from
# the single-node command in DeepSeek-V4-Flash-DSpark.md, which is written for A3 (16 devices,
# DP4 x TP4). Adapting the A3 one to 8 cards by hand -- which is what the first version of this
# script did -- gets the parallelism wrong AND misses two settings that matter more than the
# parallelism does. What the A2 recipe actually says:
#
#   --tensor-parallel-size 8 --data-parallel-size 1
#       No DP. DP replicates every non-expert weight per replica and only pays off under high
#       concurrency. Measured here at DP4 x TP2: 46.0 GB resident per card against a 293 GB
#       checkpoint = 368 GB, ~75 GB of pure duplication, taken straight out of the KV budget.
#
#   --no-disable-hybrid-kv-cache-manager
#       ★ The one that unlocks long context. DSV4 is a HYBRID attention model (Compress-4 and
#       Compress-128); the hybrid manager sizes KV per layer type instead of assuming the
#       worst case for every layer. Without it the engine refused to start: 4.84 GiB available
#       against 17.85 GiB "needed", and it suggested a max length of 8794.
#
#   no --additional-config at all
#       So no enable_flashcomm1 -- and FC1 IS vllm-ascend's sequence parallelism for non-VL
#       quantized models (their docs call it "an enhanced version of Sequence Parallelism";
#       the pass-based SP does not support quantization). With SP off, the cudagraph
#       constraint behind issue #14260 -- "shapes that are both a multiple of
#       num_speculative_tokens+1 and of tensor_parallel_size" -- simply does not arise, which
#       is why the official A2 recipe can run TP=8 at all.
#
#   --block-size 128, --max-num-batched-tokens 8192, --max-num-seqs 32, --max-model-len 800000
#
# ⚠ num_speculative_tokens: the upstream A2 recipe says 7, but it serves a different weight
# (DeepSeek-V4-Flash-DSpark-w4a8-test). The 0731 released draft is trained at block_size 5 and
# emits 5, so NUM_SPEC defaults to 5 here. Do not raise it without checking the draft's own
# block width -- asking for more tokens than it produces is not a tuning knob.
#
# ⚠ MACHINE. This occupies a whole node. Do NOT run it on a box that is dumping hidden states
# for a live training run -- the trainer's --vllm-endpoint dies with it.
#
# USAGE:  MODEL=/data/ckpt/DeepSeek-V4-Flash-0731-w8a8 bash examples/ascend_npu_dflash/serve_dsv4_a2_singlenode_w8a8.sh
# ENV:
#   MODEL       w8a8 weights dir (required)
#   NPUS        device count (default 8)      TP  tensor parallel (default = NPUS, i.e. DP=1)
#   NUM_SPEC    speculative tokens (default 5 = the released draft's block width)
#   MAX_LEN     default 800000                BLOCK_SIZE     default 128
#   MAX_BATCHED default 8192                  MAX_SEQS       default 32
#   DRAFT       our own converted draft dir; unset = the mtp.* head inside MODEL
#   SPEC_METHOD "dspark" (default) or "mtp" -- NOT a rename, it changes vLLM kernel behaviour
#               (parallel drafting, dspark_draft_topk). See worklog section 11.3.
#   EXTRA_CFG   raw JSON for --additional-config. OFF by default, matching the A2 recipe.
#               Setting it re-enables whatever you put in, including flashcomm1 -- and with
#               that, the #14260 constraint comes back.
#   CANN_HOME   default /home/a00652497/CANN/9.1.0.0627
#   PORT        default 8900        LOG  default ./serve_dsv4_w8a8_<ts>.log
set -euo pipefail

MODEL="${MODEL:?set MODEL=/path/to/DeepSeek-V4-Flash-0731-w8a8}"
NPUS="${NPUS:-8}"
NUM_SPEC="${NUM_SPEC:-5}"
SPEC_METHOD="${SPEC_METHOD:-dspark}"
PORT="${PORT:-8900}"
MAX_LEN="${MAX_LEN:-800000}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
MAX_BATCHED="${MAX_BATCHED:-8192}"
MAX_SEQS="${MAX_SEQS:-32}"
EXTRA_CFG="${EXTRA_CFG:-}"
LOG="${LOG:-$PWD/serve_dsv4_w8a8_$(date +%Y%m%d_%H%M%S).log}"

# ---- parallelism ----------------------------------------------------------------------
# Default TP = NPUS (so DP = 1), matching the official A2 recipe. The spec-window divisibility
# rule from #14260 only binds while sequence parallelism is on, i.e. only if you re-enable
# flashcomm1 through EXTRA_CFG -- so warn about it there and nowhere else.
TP="${TP:-$NPUS}"
[ $((NPUS % TP)) -eq 0 ] || { echo "!! TP=$TP does not divide NPUS=$NPUS"; exit 1; }
DP=$((NPUS / TP))
WINDOW=$((NUM_SPEC + 1))
if [ -n "$EXTRA_CFG" ] && [ $((WINDOW % TP)) -ne 0 ]; then
  echo "⚠ EXTRA_CFG is set and (num_spec+1)=$WINDOW is not a multiple of TP=$TP."
  echo "  If it turns flashcomm1 on, expect issue #14260:"
  echo "    \"Can't determine cudagraph shapes that are both a multiple of $WINDOW ...\""
fi

SPEC_CFG="{\"method\":\"$SPEC_METHOD\",\"num_speculative_tokens\":$NUM_SPEC,\"enforce_eager\":true"
[ -n "${DRAFT:-}" ] && SPEC_CFG="$SPEC_CFG,\"model\":\"$DRAFT\""
SPEC_CFG="$SPEC_CFG}"

echo "==================================================================="
echo " DSV4-Flash W8A8 + DSpark   NPUS=$NPUS  ->  DP=$DP x TP=$TP   (EP on)"
echo " num_spec=$NUM_SPEC  max_model_len=$MAX_LEN  block_size=$BLOCK_SIZE  seqs=$MAX_SEQS"
echo " hybrid KV cache manager: ON   additional-config: ${EXTRA_CFG:-<none, per the A2 recipe>}"
echo " method=$SPEC_METHOD   draft=${DRAFT:-<mtp.* inside the checkpoint>}"
echo " model=$MODEL"
echo " 📋 log -> $LOG"
echo "==================================================================="

# CANN + the nnal/atb set_env. Serving needs BOTH; sourcing only ascend-toolkit surfaces later
# as `libatb.so: cannot open shared object` or a `Mki::Dl` error deep in the engine, which reads
# like a model bug and is not one. On these boxes CANN is NOT under /usr/local -- it sits in the
# shared account, so point CANN_HOME at it.
CANN_HOME="${CANN_HOME:-/home/a00652497/CANN/9.0.0.0430}"
# ⚠ set +u around the source. CANN's own set_env.sh scripts are NOT written for `nounset`
# -- nnal/atb/set_env.sh reads $ZSH_VERSION to detect the shell, which under `set -u` is a
# FATAL "unbound variable" and kills us before the server ever starts. Turn nounset off for
# the source alone, then restore it.
for e in "$CANN_HOME/ascend-toolkit/set_env.sh" "$CANN_HOME/nnal/atb/set_env.sh"; do
  if [ -f "$e" ]; then
    set +u; . "$e"; set -u
    echo "sourced $e"
  else
    echo "⚠ missing $e — expect libatb.so / Mki::Dl errors deep in the engine"
  fi
done

# ⚠ conda-forge envs need their OWN libstdc++ to win over the system one. conda-forge's
# libsqlite is built with the ICU extension, so `import sqlite3` pulls libicui18n.so.78, which
# needs CXXABI_1.3.15 -- newer than /usr/lib64/libstdc++.so.6. The env already ships
# libstdcxx-16.1.0; it just loses the search order, because CANN's set_env puts system paths
# ahead of it. Prepending $CONDA_PREFIX/lib fixes it. Symptom without this, at the END of a
# long pip log and easy to miss under the dependency-conflict noise:
#   ImportError: /usr/lib64/libstdc++.so.6: version `CXXABI_1.3.15' not found
#   RuntimeError: Failed to load the backend extension: torch_npu
[ -n "${CONDA_PREFIX:-}" ] && export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Official env block, verbatim.
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# jemalloc: RESOLVE it, do not hardcode. The official command's path is the Debian/Ubuntu
# multiarch layout (/usr/lib/aarch64-linux-gnu); on a RHEL-family box the library lives in
# /usr/lib64 or is absent, and ld.so then prints six copies of
#   ERROR: ld.so: object '.../libjemalloc.so.2' from LD_PRELOAD cannot be preloaded: ignored
# -- alarming, non-fatal, and it silently drops the allocator the recipe wanted.
JEMALLOC="${JEMALLOC:-$(ldconfig -p 2>/dev/null | awk '/libjemalloc\.so\.2/{print $NF; exit}')}"
if [ -n "$JEMALLOC" ] && [ -f "$JEMALLOC" ]; then
  export LD_PRELOAD="$JEMALLOC:${LD_PRELOAD:-}"
  echo "jemalloc: $JEMALLOC"
else
  echo "note: libjemalloc.so.2 not found — running with the system allocator."
  echo "      (fine to start; install jemalloc if long runs fragment memory)"
fi
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096

nohup vllm serve "$MODEL" \
    --max-model-len "$MAX_LEN" \
    --max-num-batched-tokens "$MAX_BATCHED" \
    --served-model-name dsv4 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs "$MAX_SEQS" \
    --data-parallel-size "$DP" \
    --tensor-parallel-size "$TP" \
    --enable-expert-parallel \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --no-disable-hybrid-kv-cache-manager \
    --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
    --quantization ascend \
    --port "$PORT" \
    --block-size "$BLOCK_SIZE" \
    --speculative-config "$SPEC_CFG" \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    ${EXTRA_CFG:+--additional-config "$EXTRA_CFG"} \
    > "$LOG" 2>&1 &

PID=$!
echo ">>> started PID $PID  |  tail -f $LOG"
echo ">>> waiting for /v1/models (weight load on a 1M-ctx w8a8 model takes a while)…"
for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo ">>> UP after ~$((i*15))s"; exit 0
  fi
  kill -0 "$PID" 2>/dev/null || { echo "!! serve died — full log:"; tail -40 "$LOG"; exit 1; }
  sleep 15
done
echo "!! not up after 60 min — read the FULL log ($LOG), not just the tail."
exit 1
