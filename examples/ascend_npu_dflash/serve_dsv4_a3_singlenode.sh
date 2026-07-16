#!/usr/bin/env bash
# DeepSeek-V4-Flash **bf16** SINGLE-NODE serve on ONE Atlas 800 **A3** (16 cards in one box).
#
# WHY this differs from the A2 dual-node script:
#   - A2: bf16 needs 2 nodes (8 cards each). Cross-node EP16 dispatch is UNSUPPORTED
#     (aclnnMoeDistributeDispatchV4 → 561000) and the cross-node HcclAllGather DEADLOCKS,
#     so on A2 we ran DP2/TP8 with **EP OFF** (experts TP-sharded, allgather path).
#   - A3: all 16 cards live in ONE node on the HCCS fabric, so the official DP2 / TP8 /
#     **EP16** recipe runs with `--enable-expert-parallel` ON — the EP dispatch/allgather
#     stays intra-node, no cross-node hang. This is the faster (native EP dispatch) path.
#   A3 = 128G×8 cards = 16 × 64G logical devices (1 card holds 2 dies). Experts EP-sharded across
#   all 16 (ep_world=16) ≈ 38GB/device either layout — that part fits regardless.
#   Layout for BF16 = **DP2 × TP8 / EP16** (default) — matches the A2-proven per-device fit
#   (~37GB weights, ~15GB KV). The official vllm-ascend A3 recipe uses DP4×TP4, but that's for
#   **w8a8** (half the size, 2× headroom); on bf16 TP4 shards the dense weights less (dense/4 vs
#   dense/8) → smaller per-device KV → more KV-overflow-garbage risk, so we keep TP8/DP2 for bf16.
#   Env aligned to the official A3 recipe: ASCEND_A3_ENABLE=1, VLLM_ASCEND_ENABLE_FUSED_MC2=1,
#   HCCL_BUFFSIZE=1024.
#
#   PRECISION: official A3 recipe is **w8a8** (--quantization ascend, faster, and DP4/TP4 fits it
#   comfortably). We DEFAULT to bf16 to match the 115/116 rollout (consistent training data).
#   For w8a8 instead: QUANT=ascend MODEL=<…-w8a8-mtp> TP=4 DP=4.
#
# Runs `vllm serve` in the FOREGROUND (this script's stdout IS the full engine log — no
# wrapper/poll, so nothing to Ctrl+C by accident). Launch it under nohup:
#   nohup bash serve_dsv4_a3_singlenode.sh > ~/dsv4_a3.log 2>&1 &
#   tail -f ~/dsv4_a3.log
# When you see "Application startup complete", smoke-test:
#   curl -s --noproxy '*' http://localhost:7000/v1/chat/completions -H 'Content-Type: application/json' \
#     -d '{"model":"dsv4","messages":[{"role":"user","content":"从1数到40，用空格分隔"}],"temperature":0,"max_tokens":256}' \
#     | python -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'])"
# NB: no `set -u` — sourcing CANN/conda references unbound vars ($ZSH_VERSION).
set -o pipefail

MODEL="${MODEL:-/home/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16}"   # A3 convention: /home/canada_group_folder (symlink-faked per box)
CANN_ENV="${CANN_ENV:-/home/a00652497/900env_npu.sh}"     # CANN 9.0.0 (same across the fleet)
CONDA_ENV="${CONDA_ENV:-dspark-dsv4-base}"
API_PORT="${API_PORT:-7000}"
TP="${TP:-8}"; DP="${DP:-2}"       # DP2×TP8/EP16 for BF16 = matches the A2-proven fit (~37GB weights,
                                   # ~15GB KV/device). The official A3 recipe's DP4/TP4 shards the dense
                                   # weights LESS (dense/4 vs dense/8) → smaller per-device KV — fine for
                                   # w8a8 (2× headroom), but tighter for bf16 (more KV-overflow risk).
                                   # For w8a8 you can use TP=4 DP=4 (the official layout).
MAXLEN="${MAXLEN:-8192}"; MAXBATCHTOK="${MAXBATCHTOK:-8192}"; MAXSEQS="${MAXSEQS:-64}"
GPUUTIL="${GPUUTIL:-0.9}"
EAGER="${EAGER:-1}"                # 1 = --enforce-eager (reliable FIRST bring-up); 0 = graph (peak)
QUANT="${QUANT:-}"                 # empty = bf16; QUANT=ascend + MODEL=<w8a8 ckpt> to serve w8a8
ENABLE_EP="${ENABLE_EP:-1}"        # ★ ON by default on A3 (intra-node EP works). Set ENABLE_EP=0 to TP-shard.
PREFETCH="${PREFETCH:-1}"; LOAD_THREADS="${LOAD_THREADS:-16}"
DRAFT="${DRAFT:-}"                 # set DRAFT=<dspark mtp dir> → spec-decode with a DSpark draft (else plain serve)
NUM_SPEC="${NUM_SPEC:-5}"          # = dspark_block_size (released DSV4 draft = 5). draft shards under the engine's TP/EP.

# shellcheck disable=SC1090
source "$CANN_ENV"
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "$CONDA_ENV"

# --- single-node env (NO cross-node socket / HCCL_IF_IP / port-range stuff — this is ONE box;
#     no HCCL_INTRA_PCIE_ENABLE either — A3's 16 cards talk over HCCS, let HCCL pick it) ---
export OMP_PROC_BIND=false OMP_NUM_THREADS=10 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACL_OP_INIT_MODE=1 TASK_QUEUE_ENABLE=1 HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=1024
export USE_MULTI_BLOCK_POOL=1 USE_MULTI_GROUPS_KV_CACHE=1 VLLM_ASCEND_BALANCE_SCHEDULING=1
# ★ A3-SPECIFIC (from the official vllm-ascend DeepSeek-V4-Flash A3 recipe) — the A3 enable flag
# and the fused-MC2 MoE dispatch/combine fast path. WITHOUT these the A3 EP path is wrong or slow.
export ASCEND_A3_ENABLE="${ASCEND_A3_ENABLE:-1}"
export VLLM_ASCEND_ENABLE_FUSED_MC2="${VLLM_ASCEND_ENABLE_FUSED_MC2:-1}"
# EP is ON here, so Flash Comm v1 (which asserts enable_expert_parallel=True) is allowed. If the
# first bring-up misbehaves, try VLLM_ASCEND_ENABLE_FLASHCOMM1=0.
export VLLM_ASCEND_ENABLE_FLASHCOMM1="${VLLM_ASCEND_ENABLE_FLASHCOMM1:-1}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"

EAGER_FLAG=""; GRAPH_ARGS=()
if [ "$EAGER" = "1" ]; then EAGER_FLAG="--enforce-eager"
else GRAPH_ARGS=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'); fi
QUANT_ARGS=(); [ -n "$QUANT" ] && QUANT_ARGS=(--quantization "$QUANT")
EP_ARGS=(); [ "$ENABLE_EP" = "1" ] && EP_ARGS=(--enable-expert-parallel)
LOAD_ARGS=(); [ "$PREFETCH" = "1" ] && LOAD_ARGS=(--safetensors-load-strategy prefetch)
LOAD_ARGS+=(--model-loader-extra-config "{\"enable_multithread_load\":true,\"num_threads\":$LOAD_THREADS}")
# DSpark spec-decode: point the draft at the converted mtp.* dir (method mtp). Draft shards under the
# engine's TP/EP; num_speculative_tokens = dspark_block_size (5). Unset DRAFT → plain target serve.
# ★ CRITICAL flag: STANDARD_DSA=1 routes the draft attention through the PA_ND (paged) op path. The
# default (=0) uses the custom TND wrapper which NaNs at KV>128 (our KV=window128+block5=133). PA_ND is
# the correct, op-validated path (kernel session: bit-equal to the triton kernel, meanAbs 1.26e-7).
SPEC_ARGS=()
if [ -n "$DRAFT" ]; then
  export VLLM_ASCEND_DSPARK_USE_STANDARD_DSA="${VLLM_ASCEND_DSPARK_USE_STANDARD_DSA:-1}"
  SPEC_ARGS=(--speculative-config "{\"model\":\"$DRAFT\",\"num_speculative_tokens\":$NUM_SPEC,\"method\":\"mtp\"}")
fi

pkill -9 -u "$USER" -f vllm 2>/dev/null; sleep 10

echo ">>> [A3 single-node] model=$MODEL  DP$DP / TP$TP / EP=$ENABLE_EP  eager=$EAGER  port=$API_PORT"
echo ">>> draft=${DRAFT:-<none, plain serve>}  num_spec=$NUM_SPEC  STANDARD_DSA=${VLLM_ASCEND_DSPARK_USE_STANDARD_DSA:-<unset>}"
echo ">>> full engine log = THIS stdout (you launched under nohup → ~/dsv4_a3.log). No poll to Ctrl+C."
exec vllm serve "$MODEL" --served-model-name dsv4 --port "$API_PORT" \
  --data-parallel-size "$DP" --data-parallel-size-local "$DP" \
  --tensor-parallel-size "$TP" "${EP_ARGS[@]}" "${QUANT_ARGS[@]}" \
  --tokenizer-mode deepseek_v4 \
  --max-model-len "$MAXLEN" --max-num-seqs "$MAXSEQS" --block-size 128 \
  --max-num-batched-tokens "$MAXBATCHTOK" \
  --gpu-memory-utilization "$GPUUTIL" --no-enable-prefix-caching --async-scheduling \
  --additional-config '{"enable_cpu_binding":true,"multistream_overlap_shared_expert":true}' \
  "${LOAD_ARGS[@]}" "${SPEC_ARGS[@]}" \
  $EAGER_FLAG "${GRAPH_ARGS[@]}"
