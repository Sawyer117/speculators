#!/usr/bin/env bash
# DeepSeek-V4-Flash-0731-**w8a8** single-node serve WITH DSpark speculative decoding.
#
# Transcribed from the official single-node command in vllm-ascend
# docs/source/tutorials/models/DeepSeek-V4-Flash-DSpark.md (branch releases/v0.25.1rc),
# with ONE substantive change: the parallelism split. See "THE TP CONSTRAINT" below.
#
# ⚠ THE OFFICIAL COMMAND DOES NOT RUN AS WRITTEN. Issue #14260 (OPEN since 2026-08-14):
#
#   RuntimeError: Can't determine cudagraph shapes that are both a multiple of 6
#   (num_speculative_tokens + 1) required by spec-decode and 4 (tensor_parallel_size)
#   required by sequence parallelism
#
# The reporter's fix was num_speculative_tokens 5 -> 7, because 7+1=8 is a multiple of 4.
# ⚠ WE CANNOT USE THAT. Our draft is trained at block_size 5 and emits exactly 5 tokens;
# asking for 7 asks it for something it does not produce. (It is fine for the RELEASED draft
# only if that draft's own block size allows it -- do not assume.)
#
# THE TP CONSTRAINT. With sequence parallelism on, vLLM needs a cudagraph batch size that is
# a common multiple of (num_spec + 1) and tensor_parallel_size. Holding num_spec = 5 fixed:
#
#     (5 + 1) % TP == 0   =>   TP divides 6   =>   TP in {1, 2, 3, 6}
#     TP must also divide the device count    =>   on any 2^n node, TP in {1, 2}
#
# So on 8- or 16-card nodes, **TP is capped at 2** as long as num_spec is 5 and SP is on.
# This script picks the largest TP satisfying both and derives DP = NPUS / TP. The official
# A3 command is DP4 x TP4 = 16 ranks; ours is DP(NPUS/2) x TP2, which keeps EP the same width.
#
# ⚠ The third option -- keep TP=4 and DISABLE sequence parallelism -- is not wired here
# because I have not confirmed which switch turns SP on in this config (enable_flashcomm1 and
# the compilation config are both suspects). If you want TP=4, find that first; do not guess.
#
# ⚠ MACHINE. This occupies a whole node. Do NOT run it on a box that is dumping hidden states
# for a live training run -- the trainer's --vllm-endpoint dies with it.
#
# USAGE:  bash examples/ascend_npu_dflash/serve_dsv4_a2_singlenode_w8a8.sh
# ENV:
#   MODEL      w8a8 weights dir (required)
#   NPUS       device count (default 8)
#   NUM_SPEC   speculative tokens (default 5 = our draft's block width)
#   DRAFT      our own converted draft dir; unset = use the mtp.* head inside MODEL
#   SPEC_METHOD "dspark" (default) or "mtp" -- NOT a rename, it changes vLLM kernel behaviour
#              (parallel drafting, dspark_draft_topk validation). See worklog section 11.3.
#   PORT       default 8900
#   LOG        default $PWD/serve_dsv4_w8a8_<ts>.log
set -euo pipefail

MODEL="${MODEL:?set MODEL=/path/to/DeepSeek-V4-Flash-0731-w8a8}"
NPUS="${NPUS:-8}"
NUM_SPEC="${NUM_SPEC:-5}"
SPEC_METHOD="${SPEC_METHOD:-dspark}"
PORT="${PORT:-8900}"
LOG="${LOG:-$PWD/serve_dsv4_w8a8_$(date +%Y%m%d_%H%M%S).log}"

# ---- pick TP: largest divisor of NPUS that also divides (NUM_SPEC+1) --------------------
WINDOW=$((NUM_SPEC + 1))
TP=""
for c in $(seq "$NPUS" -1 1); do
  if [ $((NPUS % c)) -eq 0 ] && [ $((WINDOW % c)) -eq 0 ]; then TP="$c"; break; fi
done
[ -n "$TP" ] || { echo "!! no TP divides both NPUS=$NPUS and num_spec+1=$WINDOW"; exit 1; }
DP=$((NPUS / TP))

# Fail HERE with the reason, not 20 minutes later with the cudagraph message.
[ $((WINDOW % TP)) -eq 0 ] || { echo "!! (num_spec+1)=$WINDOW not a multiple of TP=$TP — see #14260"; exit 1; }

SPEC_CFG="{\"method\":\"$SPEC_METHOD\",\"num_speculative_tokens\":$NUM_SPEC,\"enforce_eager\":true"
[ -n "${DRAFT:-}" ] && SPEC_CFG="$SPEC_CFG,\"model\":\"$DRAFT\""
SPEC_CFG="$SPEC_CFG}"

echo "==================================================================="
echo " DSV4-Flash W8A8 + DSpark   NPUS=$NPUS  ->  DP=$DP x TP=$TP   (EP on)"
echo " num_spec=$NUM_SPEC  window=$WINDOW  ($WINDOW % $TP == 0, so #14260 does not fire)"
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
    --max-model-len 1048576 \
    --max-num-batched-tokens 10240 \
    --served-model-name dsv4 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 64 \
    --data-parallel-size "$DP" \
    --tensor-parallel-size "$TP" \
    --enable-expert-parallel \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
    --quantization ascend \
    --port "$PORT" \
    --block-size 32 \
    --speculative-config "$SPEC_CFG" \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{
        "ascend_compilation_config": {
            "enable_npugraph_ex": true,
            "enable_static_kernel": false
        },
        "enable_cpu_binding": true,
        "enable_dsa_cp": true,
        "enable_flashcomm1": true,
        "multistream_overlap_shared_expert": true
    }' > "$LOG" 2>&1 &

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
