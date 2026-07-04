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
#     (patch installed! see docs §4).
#   - ⚠️ BOTH nodes MUST be on the **same vllm-ascend commit** — a multi-node serve
#     shards weights and runs collectives across ranks, so mismatched op code =
#     silent-wrong / crash. Verify: `git -C <src> log --oneline -1` matches on both.
#     Either commit works for bf16: the o_proj-fix build (6036507, the one that also
#     serves w8a8) only ADDS a 2-D wo_a branch, so bf16's 3-D path is unchanged; or
#     the pristine 6cdb99e. Just make them EQUAL. (If your node ran the w8a8 数数
#     smoke, it's already on 6036507 — put the other node there too, no rebuild.)
#   - bf16 ckpt on shared /share, same path on both nodes.
#
# Everything below is env-overridable, e.g.:
#   HEAD_IP=80.5.5.115 WORKER_IP=80.5.5.116 NIC=enp189s0f0 bash ... head
# NB: no `set -u` — CANN's set_env.sh / conda activate reference unbound vars
# ($ZSH_VERSION etc.) and would abort the script under nounset.
set -o pipefail

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

export VLLM_HOST_IP="$THIS_IP"
# Do NOT set HCCL_IF_IP to the host (1GbE) IP: HCCL's data plane must run over the
# device RoCE NICs (hccn_tool IPs, e.g. 124.0.9.x), which HCCL auto-selects. Pinning
# HCCL_IF_IP to the 1GbE host NIC misdirects it → the first cross-node collective
# hangs (cards idle, EngineCore stuck in shm_broadcast). Only the host SOCKET ifname
# (out-of-band rendezvous) is needed — matches examples/serve/dsv4_bf16_baseline_two_node.sh.
export GLOO_SOCKET_IFNAME="$NIC" TP_SOCKET_IFNAME="$NIC" HCCL_SOCKET_IFNAME="$NIC" GLOO_USE_IPV6=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"   # 30 min: model load is slow, ranks must wait
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"  # bf16 568GB load ≈16min/node ≫ default 600s → ApiServer else times out mid-load
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
  --safetensors-load-strategy prefetch
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":16}'
  $EAGER_FLAG )
# NB: the ckpt FS reports as "DPC" → vLLM disables auto-prefetch (only NFS/Lustre
# auto-detected), so weight load took ~20 min. --safetensors-load-strategy prefetch
# + multithread_load force it faster (RAM is ~1.4 TB, half-model prefetch fits easily).

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
SERVE_PID=$!
echo ">>> head PID $SERVE_PID"

# Readiness = poll /v1/models. Failure = the serve PROCESS actually died (kill -0),
# NOT a keyword grep — vLLM's startup config dump contains 'asserts'/'error_*'/etc.
# which used to false-trip an over-eager grep. A still-loading engine keeps its PID.
echo ">>> waiting for cluster ready (both nodes load ~half of 568 GB — minutes)…"
WAITED=0; MAX_WAIT="${MAX_WAIT:-1800}"
while true; do
  if curl -s --noproxy '*' "http://localhost:$API_PORT/v1/models" >/dev/null 2>&1; then
    echo ">>> READY"; break
  fi
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then
    echo "=== FAIL: serve process $SERVE_PID exited — tail of $LOG ==="; tail -60 "$LOG"
    echo "!! usual causes: (1) 'firewall' not run on BOTH nodes → HCCL REJECT (a real HCCL"
    echo "!!   timeout traceback shows above), (2) worker not up / wrong HEAD_IP/NIC,"
    echo "!!   (3) bf16 weights incomplete. Fix, pkill -9 -f vllm on BOTH nodes, rerun."
    exit 3
  fi
  if [ "$WAITED" -ge "$MAX_WAIT" ]; then
    echo "=== process alive but not serving after ${MAX_WAIT}s — likely HCCL stuck at comm init ==="
    echo "!! run 'firewall' on BOTH nodes (trust peer /32), confirm the worker joined"
    echo "!! (tail ~/dsv4_bf16_worker.log for 'rank 1'). tail of head log:"; tail -40 "$LOG"
    exit 4
  fi
  sleep 15; WAITED=$((WAITED+15))
  echo "  …still loading (${WAITED}s, tail: $(tail -1 "$LOG" | cut -c1-100))"
done

echo ">>> smoke test — 数数:"
curl -s --noproxy '*' "http://localhost:$API_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv4","messages":[{"role":"user","content":"从1数到40，用空格分隔"}],"temperature":0,"max_tokens":256}' \
  | python -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'])"
echo ">>> if you see a coherent 1 2 3 … 40, the bf16 dual-node cluster is live."
echo ">>> (serve stays up on :$API_PORT — pkill -9 -u \$USER -f vllm on BOTH nodes to stop)"
