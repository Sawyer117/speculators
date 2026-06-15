#!/usr/bin/env bash
set -euo pipefail

# Baseline DeepSeek-V4 bf16 vLLM serve on two Ascend nodes.
#
# Node 108:
#   NODE_RANK=0 bash examples/serve/dsv4_bf16_baseline_two_node.sh
#
# Node 109:
#   NODE_RANK=1 bash examples/serve/dsv4_bf16_baseline_two_node.sh
#
# Common overrides:
#   TARGET=/path/to/DeepSeek-V4-Flash-bf16
#   MASTER_ADDR=80.5.5.108
#   VLLM_ROOT=/path/to/vLLM_NPU/vllm
#   VLLM_ASCEND_ROOT=/path/to/vLLM_NPU/vllm-ascend

ENV_NAME=${ENV_NAME:-speculator-base}
TARGET=${TARGET:-/home/n84449292/m84379596/Huggingface/DeepSeek-V4-Flash-bf16}

MASTER_ADDR=${MASTER_ADDR:-80.5.5.108}
MASTER_PORT=${MASTER_PORT:-29501}
NNODES=${NNODES:-2}
NODE_RANK=${NODE_RANK:?Set NODE_RANK=0 on 80.5.5.108 and NODE_RANK=1 on 80.5.5.109}

VLLM_ROOT=${VLLM_ROOT:-/home/a00652497/2026/dflash-vllm/installation/vllm}
VLLM_ASCEND_ROOT=${VLLM_ASCEND_ROOT:-}
CANN_HOME=${CANN_HOME:-}
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

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook)"
  if [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]]; then
    conda activate "$ENV_NAME"
  fi
fi

source_cann() {
  local roots=()
  if [[ -n "$CANN_HOME" ]]; then
    roots+=("$CANN_HOME")
  fi
  roots+=("/home/n84449292/m84379596/CANN/CANN9.0.0" "/usr/local/Ascend")

  local root
  for root in "${roots[@]}"; do
    if [[ -f "$root/ascend-toolkit/set_env.sh" ]]; then
      CANN_HOME="$root"
      set +u
      # shellcheck disable=SC1090
      source "$root/ascend-toolkit/set_env.sh"
      if [[ -f "$root/nnal/atb/set_env.sh" ]]; then
        # shellcheck disable=SC1090
        source "$root/nnal/atb/set_env.sh"
      elif [[ -f "$root/nnal/asdsip/set_env.sh" ]]; then
        # shellcheck disable=SC1090
        source "$root/nnal/asdsip/set_env.sh"
      fi
      set -u
      return 0
    fi
  done

  echo "Could not find CANN set_env.sh. Set CANN_HOME=/path/to/CANN9.0.0 or /usr/local/Ascend." >&2
  return 1
}

source_cann

if [[ -d "$VLLM_ROOT" ]]; then
  cd "$VLLM_ROOT"
  export PYTHONPATH="$VLLM_ROOT:${PYTHONPATH:-}"
fi
if [[ -n "$VLLM_ASCEND_ROOT" && -d "$VLLM_ASCEND_ROOT" ]]; then
  export PYTHONPATH="$VLLM_ASCEND_ROOT:${PYTHONPATH:-}"
fi

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VE_OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_ASCEND_APPLY_DSV4_PATCH=1
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1800}

# Keep FlashComm1 off by default. When sequence parallelism is enabled, vLLM
# requires cudagraph capture batch sizes to be multiples of TP size; the small
# smoke/benchmark default MAX_NUM_SEQS=1 intentionally does not satisfy that.
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
export GLOO_USE_IPV6=0

if [[ "$NODE_RANK" == "0" ]]; then
  node_args=(--host "$HOST" --port "$PORT")
else
  node_args=(--headless)
fi

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
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --block-size "$BLOCK_SIZE" \
  --no-enable-prefix-caching \
  "${node_args[@]}"
