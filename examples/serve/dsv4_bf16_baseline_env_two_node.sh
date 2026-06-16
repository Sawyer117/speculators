#!/usr/bin/env bash
set -euo pipefail

# Minimal DeepSeek-V4 bf16 baseline serve for an already-prepared environment.
#
# This script intentionally does not:
#   - activate conda
#   - source CANN
#   - modify PYTHONPATH
#
# Run after activating the desired environment on both nodes.
#
# Node 109:
#   NODE_RANK=1 bash examples/serve/dsv4_bf16_baseline_env_two_node.sh
#
# Node 108:
#   NODE_RANK=0 bash examples/serve/dsv4_bf16_baseline_env_two_node.sh

TARGET=${TARGET:-/home/n84449292/m84379596/Huggingface/DeepSeek-V4-Flash-bf16}

MASTER_ADDR=${MASTER_ADDR:-80.5.5.108}
MASTER_PORT=${MASTER_PORT:-29501}
NNODES=${NNODES:-2}
NODE_RANK=${NODE_RANK:?Set NODE_RANK=0 on 80.5.5.108 and NODE_RANK=1 on 80.5.5.109}
NET_PREFIX=${NET_PREFIX:-80.5.5.}

TP_SIZE=${TP_SIZE:-16}
PP_SIZE=${PP_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.80}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1024}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-1024}
BLOCK_SIZE=${BLOCK_SIZE:-128}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-30000}

export OMP_PROC_BIND=${OMP_PROC_BIND:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export VE_OMP_NUM_THREADS=${VE_OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_ASCEND_APPLY_DSV4_PATCH=${VLLM_ASCEND_APPLY_DSV4_PATCH:-1}
export DSV4_VLLM_SERVE_PATCH=${DSV4_VLLM_SERVE_PATCH:-1}
export DFLASH_DISABLE_QLI=${DFLASH_DISABLE_QLI:-0}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1800}

if [[ -n "${VLLM_ASCEND_ENABLE_FLASHCOMM1:-}" ]]; then
  export VLLM_ASCEND_ENABLE_FLASHCOMM1
else
  unset VLLM_ASCEND_ENABLE_FLASHCOMM1
fi

if [[ -z "${GLOO_SOCKET_IFNAME:-}" ]]; then
  GLOO_SOCKET_IFNAME=$(ip -o -4 addr show | awk -v prefix="$NET_PREFIX" 'index($0, prefix) {print $2; exit}')
  if [[ -z "$GLOO_SOCKET_IFNAME" ]]; then
    echo "Could not infer network interface for prefix $NET_PREFIX. Set GLOO_SOCKET_IFNAME/HCCL_SOCKET_IFNAME manually." >&2
    exit 1
  fi
  export GLOO_SOCKET_IFNAME
fi
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-$GLOO_SOCKET_IFNAME}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-$GLOO_SOCKET_IFNAME}
export GLOO_USE_IPV6=${GLOO_USE_IPV6:-0}

if [[ "$NODE_RANK" == "0" ]]; then
  node_args=(--host "$HOST" --port "$PORT")
else
  node_args=(--headless)
fi

echo "Serving target: $TARGET"
echo "node_rank=$NODE_RANK nnodes=$NNODES tp=$TP_SIZE pp=$PP_SIZE master=$MASTER_ADDR:$MASTER_PORT"
echo "DFLASH_DISABLE_QLI=$DFLASH_DISABLE_QLI GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME"

vllm serve "$TARGET" \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size "$TP_SIZE" \
  --pipeline-parallel-size "$PP_SIZE" \
  --nnodes "$NNODES" \
  --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" \
  --master-port "$MASTER_PORT" \
  --enable-expert-parallel \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --block-size "$BLOCK_SIZE" \
  --no-enable-prefix-caching \
  "${node_args[@]}"
