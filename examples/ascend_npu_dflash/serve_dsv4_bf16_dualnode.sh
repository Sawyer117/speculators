#!/usr/bin/env bash
# DeepSeek-V4-Flash **bf16** dual-node serve on 2× Atlas 800 A2 (64 GB × 8 each).
#
# WHY 2 nodes: bf16 weights ≈ 568 GB > a single node's 8×64 = 512 GB. w8a8 (284 GB)
# fits on one node; bf16 does NOT — it needs 2 nodes.
#
# PARALLELISM: TP8 / DP2 / EP16.
#   - Each node = 1 DP replica × TP8 = 8 cards. Two nodes → data_parallel_size=2.
#   - EP auto-expands to the whole world: ep_world_size = (DP2 × TP8) / PP1 = 16.
#   - EP16 is exactly what trips vllm-ascend's MC2 fast path (256 routed experts /
#     16 = 16 experts/device ≤ 24 AND ep_world_size ≥ 16). So DP2×TP8 is not just
#     "fits", it's the good config.
#   - NO ray: vLLM native DP rendezvous over --data-parallel-address / -rpc-port.
#
# ┌─ RUN ORDER (three commands total) ────────────────────────────────────────┐
# │ 0. ONCE per node, with sudo (firewalld blocks HCCL otherwise → deadlock):  │
# │       sudo bash serve_dsv4_bf16_dualnode.sh firewall                       │
# │ 1. HEAD node (rank 0, hosts the API):   bash ... head                      │
# │ 2. WORKER node (rank 1, headless):      bash ... worker                    │
# │    Start head first; the worker connects to the head's rpc port and waits. │
# └────────────────────────────────────────────────────────────────────────────┘
#
# PREREQS (BOTH nodes, identical):
#   - conda env `dspark-dsv4-base`, CANN 9.0.0 (source 900env), vllm-ascend built
#     (patch installed! see docs §4), on the **6cdb99e** checkout (clean bf16 stack,
#     no o_proj fix — bf16 wo_a is 3D so it doesn't need it).
#   - bf16 ckpt on shared /share, same path on both nodes.
#
# Everything below is env-overridable, e.g.:
#   HEAD_IP=80.5.5.115 WORKER_IP=80.5.5.116 NIC=enp189s0f0 bash ... head
set -uo pipefail

ROLE="${1:?usage: bash serve_dsv4_bf16_dualnode.sh <firewall|head|worker>}"

# ---- config (override via env) ----
MODEL="${MODEL:-/share/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16}"
HEAD_IP="${HEAD_IP:-80.5.5.115}"          # rank-0 / API node
WORKER_IP="${WORKER_IP:-80.5.5.116}"      # rank-1 / headless node
NIC="${NIC:-enp189s0f0}"                  # the IP-network NIC (HCCL uses its own fabric)
API_PORT="${API_PORT:-7000}"              # OpenAI API on the head
DP_RPC_PORT="${DP_RPC_PORT:-13389}"       # vLLM DP rendezvous port (head binds it)
CANN_ENV="${CANN_ENV:-/home/a00652497/900env_npu.sh}"
CONDA_ENV="${CONDA_ENV:-dspark-dsv4-base}"
TP="${TP:-8}"
DP="${DP:-2}"
MAXLEN="${MAXLEN:-8192}"
MAXSEQS="${MAXSEQS:-16}"
GPUUTIL="${GPUUTIL:-0.9}"
EAGER="${EAGER:-1}"                       # 1 = --enforce-eager (reliable first bring-up); 0 = graph mode (faster)

# ---- firewall subcommand: whitelist BOTH peer IPs in firewalld's trusted zone ----
# Ascend HCCL opens many ephemeral ports between ranks; the default zone REJECTs them
# → ranks hang forever at comm init. Trusting the peer /32 lets them through.
if [ "$ROLE" = "firewall" ]; then
  echo ">>> whitelisting $HEAD_IP/32 and $WORKER_IP/32 in firewalld trusted zone (needs sudo)"
  firewall-cmd --zone=trusted --add-source="$HEAD_IP/32"   --permanent
  firewall-cmd --zone=trusted --add-source="$WORKER_IP/32" --permanent
  firewall-cmd --reload
  echo ">>> trusted zone now:"; firewall-cmd --zone=trusted --list-sources
  echo ">>> done. run this ONCE on EACH node."
  exit 0
fi

[ "$ROLE" = "head" ] || [ "$ROLE" = "worker" ] || { echo "!! role must be firewall|head|worker"; exit 2; }

# ---- this node's IP on $NIC (for HCCL/GLOO socket bootstrap) ----
THIS_IP="$(ip -4 -o addr show "$NIC" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)"
[ -n "$THIS_IP" ] || { echo "!! could not read an IPv4 on NIC '$NIC' — set NIC=<iface>"; ip -4 -o addr show | awk '{print $2, $4}'; exit 2; }
echo ">>> role=$ROLE  this_ip=$THIS_IP  head=$HEAD_IP  nic=$NIC"

# ---- env ----
# shellcheck disable=SC1090
source "$CANN_ENV"
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "$CONDA_ENV"

export VLLM_HOST_IP="$THIS_IP" HCCL_IF_IP="$THIS_IP"
export GLOO_SOCKET_IFNAME="$NIC" TP_SOCKET_IFNAME="$NIC" HCCL_SOCKET_IFNAME="$NIC"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"   # 30 min: model load is slow, ranks must wait
export OMP_PROC_BIND=false OMP_NUM_THREADS=8 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACL_OP_INIT_MODE=1 TASK_QUEUE_ENABLE=1 HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=512

EAGER_FLAG=""; [ "$EAGER" = "1" ] && EAGER_FLAG="--enforce-eager"

# common serve flags (bf16 = NO --quantization)
COMMON=( "$MODEL"
  --data-parallel-size "$DP" --data-parallel-size-local 1
  --data-parallel-address "$HEAD_IP" --data-parallel-rpc-port "$DP_RPC_PORT"
  --tensor-parallel-size "$TP" --enable-expert-parallel
  --tokenizer-mode deepseek_v4
  --max-model-len "$MAXLEN" --max-num-seqs "$MAXSEQS"
  --gpu-memory-utilization "$GPUUTIL" --no-enable-prefix-caching
  $EAGER_FLAG )

pkill -9 -u "$USER" -f vllm 2>/dev/null; sleep 15

if [ "$ROLE" = "worker" ]; then
  LOG=~/dsv4_bf16_worker.log
  echo ">>> [worker/rank1] starting headless engine → $LOG  (connects to head $HEAD_IP:$DP_RPC_PORT)"
  nohup vllm serve "${COMMON[@]}" --headless --data-parallel-start-rank 1 > "$LOG" 2>&1 &
  echo ">>> worker PID $!  —  tail -f $LOG"
  echo ">>> (worker has no API; it joins the head. Watch for 'init … rank 1' then steady state.)"
  exit 0
fi

# ---- head (rank 0) ----
LOG=~/dsv4_bf16_head.log
echo ">>> [head/rank0] starting API serve on :$API_PORT → $LOG"
nohup vllm serve "${COMMON[@]}" --served-model-name dsv4 --port "$API_PORT" > "$LOG" 2>&1 &
echo ">>> head PID $!"

echo ">>> waiting for cluster ready (both nodes load ~half of 568 GB — minutes)…"
until curl -s --noproxy '*' "http://localhost:$API_PORT/v1/models" >/dev/null 2>&1; do
  if grep -qiE "error|traceback|out of memory|assert|refused|timed? ?out" "$LOG"; then
    echo "=== FAIL — tail of $LOG ==="; tail -50 "$LOG"
    echo "!! if it's a connect/timeout: (1) run the 'firewall' step on BOTH nodes,"
    echo "!!   (2) check the worker is up (tail ~/dsv4_bf16_worker.log), (3) confirm HEAD_IP/NIC."
    exit 3
  fi
  echo "  …still loading (tail: $(tail -1 "$LOG" | cut -c1-100))"; sleep 15
done
echo ">>> READY"

echo ">>> smoke test — 数数:"
curl -s --noproxy '*' "http://localhost:$API_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv4","messages":[{"role":"user","content":"从1数到40，用空格分隔"}],"temperature":0,"max_tokens":256}' \
  | python -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'])"
echo ">>> if you see a coherent 1 2 3 … 40, the bf16 dual-node cluster is live."
echo ">>> (serve stays up on :$API_PORT — pkill -9 -u \$USER -f vllm on BOTH nodes to stop)"
