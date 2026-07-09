#!/usr/bin/env bash
# DeepSeek-V4-Flash **bf16** dual-node serve on 2× Atlas 800 A2 (64 GB × 8 each).
#
# WHY 2 nodes: bf16 weights ≈ 568 GB > a single node's 8×64 = 512 GB. w8a8 (284 GB)
# fits on one node; bf16 does NOT — it needs 2 nodes.
#
# PARALLELISM: TP8 / DP2, **expert-parallel OFF** (default).
#   - Each node = 1 DP replica × TP8 = 8 cards, holding a FULL copy of the model
#     (w8a8 ≈ 35 GB/card, bf16 more). Two nodes → data_parallel_size=2 = 2 replicas.
#   - ⚠️ DO NOT pass --enable-expert-parallel: on two-node A2 it makes the MoE do a
#     CROSS-node EP16 all-gather (HcclAllGather) that DEADLOCKS at startup (the endless
#     `shm_broadcast: No available block` hang). Proven by the AtomGit A2 two-node
#     V4-Flash report ("删除 --enable-expert-parallel 后不再报错") + our own plog
#     (HcclAllGather stuck at seq_num 1, NPUs idle). Without EP the MoE is TP-sharded
#     INSIDE each node → all collectives stay intra-node → no hang. (ENABLE_EP=1 to opt in.)
#   - NO ray: vLLM native DP rendezvous over --data-parallel-address / -rpc-port.
#
# ┌─ RUN ORDER ───────────────────────────────────────────────────────────────┐
# │ (firewalld is usually not running on these nodes; the 'firewall' step is a  │
# │  no-op then — skip it. Run it only if `firewall-cmd --state` says running.) │
# │ 1. HEAD node (rank 0, hosts the API):   bash ... head                       │
# │ 2. WORKER node (rank 1, headless):      bash ... worker                     │
# │    Start head first; the worker connects to the head's rpc port and waits.  │
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
MAXBATCHTOK="${MAXBATCHTOK:-8192}"        # per-forward token budget. Token SIZE was a RED HERRING for the
                                          # two-node hang — the AtomGit A2 two-node V4-Flash report runs 8192
                                          # fine. The real cause was --enable-expert-parallel (now default-off,
                                          # see ENABLE_EP). 8192 = the report's value.
MAXSEQS="${MAXSEQS:-16}"
GPUUTIL="${GPUUTIL:-0.9}"
EAGER="${EAGER:-1}"                       # 1 = --enforce-eager (reliable first bring-up); 0 = graph mode (faster)
QUANT="${QUANT:-}"                        # empty = bf16; set QUANT=ascend (+ MODEL=<w8a8 ckpt>) to serve w8a8.
ENABLE_EP="${ENABLE_EP:-}"                # ★ empty = NO expert-parallel (DEFAULT). --enable-expert-parallel on
                                          # two-node A2 triggers a cross-node EP16 HcclAllGather deadlock (the
                                          # shm_broadcast hang). Confirmed by the AtomGit A2 two-node V4-Flash
                                          # report ("删除 --enable-expert-parallel 后不再报错") + our plog
                                          # (HcclAllGather stuck at seq_num 1). Without EP, MoE experts are
                                          # TP-sharded intra-node (no cross-node all-gather). Set ENABLE_EP=1
                                          # only after the cross-node HCCL AllGather is actually fixed.

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

export VLLM_HOST_IP="$THIS_IP" HCCL_IF_IP="$THIS_IP"   # host NIC IP — the official recipe DOES set this
export GLOO_SOCKET_IFNAME="$NIC" TP_SOCKET_IFNAME="$NIC" HCCL_SOCKET_IFNAME="$NIC" GLOO_TCP_IFACE="$NIC" GLOO_USE_IPV6=0
# HCCL_INTRA_PCIE_ENABLE=1: intra-node MoE comm over PCIe/SDMA (both AtomGit A2 two-node
#   V4-Flash reports set this). We do NOT set HCCL_INTRA_ROCE_ENABLE=0 — that came from the
#   Ascend-SACT gitcode recipe, which is UNRELIABLE (it also told us to pass
#   --enable-expert-parallel, which DEADLOCKS two-node, see ENABLE_EP). The two AtomGit
#   reports that actually validate two-node V4-Flash (gsm8k 97.27) don't set it.
export HCCL_INTRA_PCIE_ENABLE=1
# Flash Comm v1 ASSERTS enable_expert_parallel=True for MoE — but we run WITHOUT EP
# (cross-node EP dispatch unsupported). If the user's shell exported
# VLLM_ASCEND_ENABLE_FLASHCOMM1=1 it crashes config validation. Force it OFF here.
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export VLLM_ASCEND_BALANCE_SCHEDULING=1 USE_MULTI_BLOCK_POOL=1 TRITON_ALL_BLOCKS_PARALLEL=1
export VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"   # 30 min: model load is slow, ranks must wait
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"  # bf16 568GB load ≈16min/node ≫ default 600s → ApiServer else times out mid-load
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-600000}" VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-6000}"  # from the AtomGit A2 report
export OMP_PROC_BIND=false OMP_NUM_THREADS=10 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACL_OP_INIT_MODE=1 TASK_QUEUE_ENABLE=1 HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=1024
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-60000-61000}" HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-61000-62000}"  # from the AtomGit bf16 A2 report

# EAGER=1 → --enforce-eager (simple/reliable first bring-up). EAGER=0 → graph mode
# (cudagraph FULL_DECODE_ONLY, the AtomGit report's config) for faster decode / higher throughput.
EAGER_FLAG=""; GRAPH_ARGS=()
if [ "$EAGER" = "1" ]; then EAGER_FLAG="--enforce-eager"
else GRAPH_ARGS=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'); fi
QUANT_ARGS=(); [ -n "$QUANT" ] && QUANT_ARGS=(--quantization "$QUANT")
EP_ARGS=(); [ -n "$ENABLE_EP" ] && EP_ARGS=(--enable-expert-parallel)   # default OFF — see ENABLE_EP above
# worker uses lower gpu-mem-util (0.85 vs 0.90) — avoids cudagraph-warmup OOM (per the AtomGit report)
GPUUTIL_EFF="$GPUUTIL"; [ "$ROLE" = "worker" ] && GPUUTIL_EFF="${WORKER_GPUUTIL:-0.85}"

# weight-load strategy. prefetch reads shards into RAM BEFORE committing to NPU — fast on a quick FS,
# but on a SLOW /share it stalls at "0/46" for minutes with zero committed, so the HEAD's API server
# times out and tears the cluster down while the headless WORKER (no such timeout) keeps loading.
# If the head dies at 0/46 but the worker survives → set PREFETCH=0 (incremental load: progress shows
# immediately, engine stays responsive). LOAD_THREADS lowers the multithread fan-out on a slow mount.
PREFETCH="${PREFETCH:-1}"; LOAD_THREADS="${LOAD_THREADS:-16}"
LOAD_ARGS=(); [ "$PREFETCH" = "1" ] && LOAD_ARGS=(--safetensors-load-strategy prefetch)
LOAD_ARGS+=(--model-loader-extra-config "{\"enable_multithread_load\":true,\"num_threads\":$LOAD_THREADS}")

# ---- optional HIDDEN-STATE EXTRACTION (DSpark/DFlash training-data producer) ----
# HS_EXTRACT=1 turns this serve into a HS producer: vLLM's `extract_hidden_states`
# spec method dumps aux hidden states at EAGLE_AUX_LAYERS (target layers + last), and
# the ExampleHiddenStatesConnector (kv_producer) writes them as hs_*.safetensors to
# HS_PATH. Put HS_PATH on SHARED storage (both nodes + the trainer must see it).
# Default OFF => identical to the plain bf16 serve. Enable both HEAD and WORKER.
HS_ARGS=()
if [ "$HS_EXTRACT" = "1" ]; then
  HS_PATH="${HS_PATH:-/share/canada_group_folder/dataset/dsv4_hs_smoketest}"   # A2 = real /share shared storage (NOT A3's faked /home)
  EAGLE_AUX_LAYERS="${EAGLE_AUX_LAYERS:-[40,41,42,43]}"   # target 40/41/42 + last layer 43
  mkdir -p "$HS_PATH"
  # kv_role=kv_producer puts vLLM in PD-DISAGGREGATED mode, which is INCOMPATIBLE with
  # Ascend balance-scheduling (pydantic: "enable_balance_scheduling only supports PD-mixed
  # mode"). Force it OFF when extracting HS.
  export VLLM_ASCEND_BALANCE_SCHEDULING=0
  HS_ARGS=(
    --speculative_config "{\"method\":\"extract_hidden_states\",\"num_speculative_tokens\":1,\"draft_model_config\":{\"hf_config\":{\"eagle_aux_hidden_state_layer_ids\":$EAGLE_AUX_LAYERS}}}"
    --kv_transfer_config "{\"kv_connector\":\"ExampleHiddenStatesConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"shared_storage_path\":\"$HS_PATH\"}}"
    --no-enable-chunked-prefill )
  echo ">>> [HS_EXTRACT] aux hidden layers $EAGLE_AUX_LAYERS -> $HS_PATH (hs_*.safetensors)"
fi

# ---- optional PLAN B HIDDEN-STATE DUMP (DSpark training-data producer, memory-light) ----
# HS_DUMP=1 turns this PLAIN serve into a HS producer WITHOUT extract_hidden_states'
# HiddenStateCacheSpec (which OOMs on DSV4 — see docs/deployment/ascend-npu-dsv4-hs-dumper-planB.md).
# The DSV4 forward already captures the aux target layers into a ~200 MB scratch buffer
# (gated ONLY on config.dspark_target_layer_ids → a plain serve populates it, no draft
# ckpt); our runner hook (vllm-ascend feat/dsv4-hs-dumper) copies that + the post-norm
# final hidden to CPU and writes hs_<id>.safetensors to DSPARK_HS_DIR. No kv_transfer,
# no PD-disagg. Enable on BOTH head and worker; DSPARK_HS_DIR must be SHARED storage
# (each node's TP-rank-0 writes its own DP replica's requests). Drive prefill-only
# (max_tokens=1) — see hs_dump_smoke.py. Mutually exclusive with HS_EXTRACT.
HS_DUMP_ARGS=()
if [ "$HS_DUMP" = "1" ]; then
  [ "$HS_EXTRACT" = "1" ] && { echo "!! set only ONE of HS_EXTRACT / HS_DUMP"; exit 2; }
  export DSPARK_HS_DUMP=1
  export DSPARK_HS_DIR="${DSPARK_HS_DIR:-/share/canada_group_folder/dataset/dsv4_hs_dump}"   # A2 real /share shared storage
  DSPARK_LAYERS="${DSPARK_LAYERS:-[40,41,42]}"   # aux target layers -> hidden_states [seq, 3*hidden_size]
  mkdir -p "$DSPARK_HS_DIR"
  # Ensure the target config carries dspark_target_layer_ids so the model allocates + fills
  # the dspark buffer (spec-method-independent). --hf-overrides merges this key into the config.
  HS_DUMP_ARGS=( --hf-overrides "{\"dspark_target_layer_ids\":$DSPARK_LAYERS}" )
  echo ">>> [HS_DUMP] plan B dumper ON: layers $DSPARK_LAYERS -> $DSPARK_HS_DIR (hs_*.safetensors); drive prefill-only (max_tokens=1)"
fi

# common serve flags (bf16 = NO --quantization; NO --enable-expert-parallel by default)
COMMON=( "$MODEL"
  --data-parallel-size "$DP" --data-parallel-size-local 1
  --data-parallel-address "$HEAD_IP" --data-parallel-rpc-port "$DP_RPC_PORT"
  --tensor-parallel-size "$TP" "${EP_ARGS[@]}" "${QUANT_ARGS[@]}"
  --tokenizer-mode deepseek_v4
  --max-model-len "$MAXLEN" --max-num-seqs "$MAXSEQS" --block-size 128
  --max-num-batched-tokens "$MAXBATCHTOK"
  --gpu-memory-utilization "$GPUUTIL_EFF" --no-enable-prefix-caching
  --additional-config '{"enable_cpu_binding":true,"multistream_overlap_shared_expert":true}'
  "${LOAD_ARGS[@]}" "${HS_ARGS[@]}" "${HS_DUMP_ARGS[@]}"
  $EAGER_FLAG "${GRAPH_ARGS[@]}" )
# NB: the ckpt FS reports as "DPC" → vLLM disables auto-prefetch (only NFS/Lustre
# auto-detected), so weight load took ~20 min. --safetensors-load-strategy prefetch
# + multithread_load force it faster (RAM is ~1.4 TB, half-model prefetch fits easily).

pkill -9 -u "$USER" -f vllm 2>/dev/null; sleep 15

if [ "$ROLE" = "worker" ]; then
  LOG=~/dsv4_bf16_worker.log
  echo ">>> [worker/rank1] starting headless engine → $LOG  (connects to head $HEAD_IP:$DP_RPC_PORT)"
  nohup vllm serve "${COMMON[@]}" --headless --data-parallel-start-rank 1 > "$LOG" 2>&1 &
  echo ">>> worker PID $!"
  echo ">>> 📋 FULL engine log (tail THIS for ALL detail/errors):  tail -f $LOG"
  echo ">>> (worker has no API; it joins the head. Watch for 'init … rank 1' then steady state.)"
  exit 0
fi

# ---- head (rank 0) ----
LOG=~/dsv4_bf16_head.log
echo ">>> [head/rank0] starting API serve on :$API_PORT → $LOG"
nohup vllm serve "${COMMON[@]}" --served-model-name dsv4 --port "$API_PORT" > "$LOG" 2>&1 &
SERVE_PID=$!
echo ">>> head PID $SERVE_PID"
echo ">>> 📋 FULL engine log (tail THIS for ALL detail/errors):  tail -f $LOG"
echo ">>>    (if you ran this via 'nohup … > ~/head_run.log', that file is ONLY this wrapper's"
echo ">>>     summary — the real per-rank detail is in $LOG)"

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
