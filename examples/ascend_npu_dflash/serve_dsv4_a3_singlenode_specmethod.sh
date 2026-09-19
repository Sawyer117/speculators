#!/usr/bin/env bash
#
# ════════════════════════════════════════════════════════════════════════════════════════
# VARIANT of serve_dsv4_a3_singlenode.sh. ONE difference: speculative_config's "method" is
# "dspark" (override with SPEC_METHOD=mtp) instead of the hard-coded "mtp". Everything else
# is byte-identical, so an A/B against the parent script isolates the method string.
#
# WHY. Inside the Ascend proposer the two are the same: get_spec_decode_method() accepts
# BOTH "mtp" and "dspark" and keys the DSV4 branch on the draft config carrying
# dspark_block_size, and AscendDeepSeekV4DSparkProposer -- itself a subclass of
# AscendDsparkProposer -- overwrites self.method = "dflash" in __init__, so every
# method-driven branch in llm_base_proposer runs identically either way.
#
# But vLLM's OWN SpeculativeConfig branches on the string before the proposer ever exists:
#
#   use_eagle()   in ("eagle","eagle3","mtp","dflash","dspark")   -> same for both
#   use_dspark()  == "dspark"                                     -> FALSE for us today
#
#   method="dspark"  -> enables parallel drafting automatically; validates dspark_draft_topk
#   method="mtp"     -> neither; and applies an MTP divisibility check
#                       (num_speculative_tokens % n_predict) that is meaningless here
#
# ★ So today the core does NOT set parallel drafting and the Ascend proposer compensates by
#   assigning self.parallel_drafting = True itself. It works, but we are papering over a
#   config the core would have set. "dspark" is the spelling that matches what we are.
#
# ⚠ THEREFORE THIS IS NOT A NO-OP RENAME. It changes vLLM-core behaviour and must be
#   measured, not assumed. Acceptance: per-position acceptance rates match the current
#   baseline (ep5p0-ropefix, gsm8k accept_len 4.849) within noise, and tok/s does not drop.
#   If either moves, the parent script is the fallback and the delta is the finding.
#
# Historical note: "mtp" was used because the draft weights live under mtp.* in the released
# DeepSeek-V4-Flash checkpoint, and that alias was what worked during bring-up.
#
# ⚠ Upstream gap worth a small PR: vLLM auto-detects method="dspark" from
#   `"Qwen3DSparkModel" in architectures`, which does not cover
#   DSparkDeepseekV4ForCausalLM -- so our draft can never be auto-detected.
# ════════════════════════════════════════════════════════════════════════════════════════
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
#
# HS PRODUCER for training (Plan B) — TWO modes:
#   * shared FS (default): HS_DUMP=1 → verifier-only prefill dumps hs_<id>.safetensors into DSPARK_HS_DIR;
#     the trainer reads that dir DIRECTLY (A2 /share). No sidecar.
#   * NO shared FS (A3 182-serve / 176-train): add HS_SIDECAR=1 → this script ALSO auto-starts
#     hs_sidecar.py (one command, --root = the same DSPARK_HS_DIR) so a remote trainer pulls via HS_FETCH_BASE:
#       HS_DUMP=1 HS_SIDECAR=1 HS_SIDECAR_TOKEN=s3cret DSPARK_HS_DIR=/home/n84449292/dsv4_hs_dump \
#         nohup bash serve_dsv4_a3_singlenode.sh > ~/dsv4_a3_hsdump.log 2>&1 &
# NB: no `set -u` — sourcing CANN/conda references unbound vars ($ZSH_VERSION).
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # to locate the sibling hs_sidecar.py

MODEL="${MODEL:-/home/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16}"   # A3 convention: /home/canada_group_folder (symlink-faked per box)
CANN_ENV="${CANN_ENV:-/home/a00652497/900env_npu.sh}"     # CANN 9.0.0 (same across the fleet)
CONDA_ENV="${CONDA_ENV:-dspark-dsv4-serving}"   # A3 #12006 eval/HS-producer env (setup_dsv4_serve_a3.sh
                                   # creates it). NOT dspark-dsv4-base — that lacks vllm-ascend → the serve
                                   # dies "Failed to infer device type" (NPU platform plugin not installed).
API_PORT="${API_PORT:-7000}"
TP="${TP:-8}"; DP="${DP:-2}"       # DP2×TP8/EP16 for BF16 = matches the A2-proven fit (~37GB weights,
                                   # ~15GB KV/device). The official A3 recipe's DP4/TP4 shards the dense
                                   # weights LESS (dense/4 vs dense/8) → smaller per-device KV — fine for
                                   # w8a8 (2× headroom), but tighter for bf16 (more KV-overflow risk).
                                   # For w8a8 you can use TP=4 DP=4 (the official layout).
MAXLEN="${MAXLEN:-8192}"; MAXBATCHTOK="${MAXBATCHTOK:-8192}"; MAXSEQS="${MAXSEQS:-64}"
GPUUTIL="${GPUUTIL:-0.9}"
EAGER="${EAGER:-0}"                # ★ DEFAULT = graph mode (ACLGraph FULL_DECODE_ONLY, peak decode).
                                   # Manual EAGER=1 → --enforce-eager (safest first bring-up / debug).
QUANT="${QUANT:-}"                 # empty = bf16; QUANT=ascend + MODEL=<w8a8 ckpt> to serve w8a8
ENABLE_EP="${ENABLE_EP:-1}"        # ★ ON by default on A3 (intra-node EP works). Set ENABLE_EP=0 to TP-shard.
# ⚠ 默认从 1 改成 0 —— 见下面 LOAD_ARGS 处的说明:1 会让本脚本必然起不来。
PREFETCH="${PREFETCH:-0}"; LOAD_THREADS="${LOAD_THREADS:-16}"
DRAFT="${DRAFT:-}"                 # set DRAFT=<dspark mtp dir> → spec-decode with a DSpark draft (else plain serve)
NUM_SPEC="${NUM_SPEC:-5}"          # = dspark_block_size (released DSV4 draft = 5). draft shards under the engine's TP/EP.
DSPARK_AUX_LAYERS="${DSPARK_AUX_LAYERS:-[40,41,42]}"  # target aux layers the draft's main_proj consumes (3 → 3*H),
                                  # passed via speculative_config.draft_model_config.hf_config — else 4-layer default → dim mismatch.
HS_DUMP="${HS_DUMP:-}"            # set HS_DUMP=1 → Plan B HS PRODUCER for TRAINING (verifier prefill dumps
                                  # hs_<id>.safetensors, NO draft). Needs the DsparkHSDumper hook in vllm-ascend
                                  # (vllm_ascend/dspark_hs_dumper.py + model_runner). Mutually exclusive with DRAFT.
DSPARK_HS_DIR="${DSPARK_HS_DIR:-/home/canada_group_folder/dataset/dsv4_hs_dump}"  # where the dumps land.
                                  # DEFAULT ASSUMPTION: this dir is on SHARED storage the trainer reads DIRECTLY
                                  # (A2 /share) → NO sidecar. Only if the trainer box does NOT share this FS
                                  # (A3 182-serve / 176-train) do you flip HS_SIDECAR=1 below.
HS_SIDECAR="${HS_SIDECAR:-0}"     # ★ NO-SHARED-FS switch. =1 (only acts with HS_DUMP=1) auto-starts hs_sidecar.py
                                  # HERE with --root=$DSPARK_HS_DIR (same var → CAN'T mismatch), so a REMOTE
                                  # trainer pulls over HTTP(S) via HS_FETCH_BASE. Leave 0 when the dump dir is shared.
HS_SIDECAR_PORT="${HS_SIDECAR_PORT:-9009}"
HS_SIDECAR_CERT="${HS_SIDECAR_CERT:-}"   # set BOTH cert+key → sidecar serves HTTPS; else plain HTTP (fine intra-cluster)
HS_SIDECAR_KEY="${HS_SIDECAR_KEY:-}"
HS_SIDECAR_LOG="${HS_SIDECAR_LOG:-$HOME/hs_sidecar.log}"   # sidecar's own log (token on/off, requests) lands here
HS_SIDECAR_PIDFILE="${HS_SIDECAR_PIDFILE:-$HOME/hs_sidecar.pid}"  # graceful stop: kill $(cat $HS_SIDECAR_PIDFILE)
HS_SIDECAR_WORKERS="${HS_SIDECAR_WORKERS:-8}"  # prefork N processes → concurrent ~100MB HS transfers run in
                                  # TRUE parallel (single Python process is GIL/mem-bound → leaves multi-stream
                                  # bandwidth unused; the A3 remote link needs it). Set 1 for the old behavior.

# shellcheck disable=SC1090
source "$CANN_ENV"
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "$CONDA_ENV"
# ⚠ conda-forge/miniforge 环境要让自己的 libstdc++ 赢过系统那个。conda-forge 的 libsqlite 带
# ICU 扩展,`import sqlite3` 会拉 libicui18n.so.78,它要 CXXABI_1.3.15 —— 比 /usr/lib64 的
# libstdc++.so.6 新。环境里本来就装了 libstdcxx-16.1.0,只是输在查找顺序上。症状是 torch_npu
# 加载失败,报错在一屏 traceback 的最底下,容易被当成 torch_npu 装坏了:
#   ImportError: /usr/lib64/libstdc++.so.6: version `CXXABI_1.3.15\' not found
#   RuntimeError: Failed to load the backend extension: torch_npu
# ⚠ 必须在 conda activate 之后 —— CONDA_PREFIX 那时才指向本 env。
[ -n "${CONDA_PREFIX:-}" ] && export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# --- single-node env (NO cross-node socket / HCCL_IF_IP / port-range stuff — this is ONE box;
#     no HCCL_INTRA_PCIE_ENABLE either — A3's 16 cards talk over HCCS, let HCCL pick it) ---
export OMP_PROC_BIND=false OMP_NUM_THREADS=10 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACL_OP_INIT_MODE=1 TASK_QUEUE_ENABLE=1 HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=1024
export USE_MULTI_BLOCK_POOL=1 USE_MULTI_GROUPS_KV_CACHE=1 VLLM_ASCEND_BALANCE_SCHEDULING=1
# ★ A3-SPECIFIC (from the official vllm-ascend DeepSeek-V4-Flash A3 recipe) — the A3 enable flag
# and the fused-MC2 MoE dispatch/combine fast path. WITHOUT these the A3 EP path is wrong or slow.
export ASCEND_A3_ENABLE="${ASCEND_A3_ENABLE:-1}"
export VLLM_ASCEND_ENABLE_FUSED_MC2="${VLLM_ASCEND_ENABLE_FUSED_MC2:-1}"
# FlashComm v1 (sequence-parallel comm) is a throughput win for PLAIN serve / HS-dump. BUT under graph
# mode it forces cudagraph batch sizes to a multiple of TP, which CONFLICTS with spec-decode's required
# multiple of (num_speculative_tokens+1) → "Can't determine cudagraph shapes ... disable sequence
# parallelism" crash. So AUTO-DEFAULT it OFF when a DRAFT (spec-decode) is set, ON otherwise. Explicit
# VLLM_ASCEND_ENABLE_FLASHCOMM1=... still overrides. (EP is ON here, so FlashComm1 is otherwise allowed.)
export VLLM_ASCEND_ENABLE_FLASHCOMM1="${VLLM_ASCEND_ENABLE_FLASHCOMM1:-$([ -n "$DRAFT" ] && echo 0 || echo 1)}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
# ★ engine-ready timeout: the API frontend waits VLLM_ENGINE_READY_TIMEOUT_S for the engine cores.
# Loading the 543 GB bf16 model + KV alloc + warmup takes ~11-12 min (> the 600s default), so a fresh
# bring-up hits "TimeoutError: Timed out waiting for engine core processes to start" and the ApiServer
# dies (exit 1) even though the engine was ~90s from ready. 1800s gives the big-model load headroom.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"

EAGER_FLAG=""; GRAPH_ARGS=()
if [ "$EAGER" = "1" ]; then EAGER_FLAG="--enforce-eager"
else GRAPH_ARGS=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'); fi
QUANT_ARGS=(); [ -n "$QUANT" ] && QUANT_ARGS=(--quantization "$QUANT")
EP_ARGS=(); [ "$ENABLE_EP" = "1" ] && EP_ARGS=(--enable-expert-parallel)
# ⚠ 0.27.1 起 `--safetensors-load-strategy prefetch` 与 `enable_multithread_load` 互斥
#   (0.23.0 上两者共存)。两个同时传下去,worker 起来时就抛:
#     ValueError: enable_multithread_load does not support safetensors_load_strategy='prefetch'
#   而原来的写法【默认就是同时传】—— 也就是说本脚本按默认值手动起必然失败。一直没暴露,
#   只因为 eval_blk15_drafts.sh 显式传了 PREFETCH=0。所以默认改成 0(= 多线程加载,本项目
#   所有跑成功的 serve 用的都是这一档);真要 prefetch 就自动把多线程关掉,而不是让 vLLM 抛。
if [ "$PREFETCH" = "1" ]; then
  LOAD_ARGS=(--safetensors-load-strategy prefetch)
  echo ">>> PREFETCH=1:用 prefetch 策略,多线程加载自动关闭(0.27.1 上两者互斥)"
else
  LOAD_ARGS=(--model-loader-extra-config "{\"enable_multithread_load\":true,\"num_threads\":$LOAD_THREADS}")
fi
# DSpark spec-decode: point the draft at the converted mtp.* dir (method mtp). Draft shards under the
# engine's TP/EP; num_speculative_tokens = dspark_block_size (5). Unset DRAFT → plain target serve.
# ★ CRITICAL flag: STANDARD_DSA=1 routes the draft attention through the PA_ND (paged) op path. The
# default (=0) uses the custom TND wrapper which NaNs at KV>128 (our KV=window128+block5=133). PA_ND is
# the correct, op-validated path (kernel session: bit-equal to the triton kernel, meanAbs 1.26e-7).
SPEC_ARGS=()
if [ -n "$DRAFT" ]; then
  export VLLM_ASCEND_DSPARK_USE_STANDARD_DSA="${VLLM_ASCEND_DSPARK_USE_STANDARD_DSA:-1}"
  # pin aux layers via BOTH channels (target --hf-overrides + draft eagle_aux) — see dualnode script.
  SPEC_ARGS=(--hf-overrides "{\"dspark_target_layer_ids\":$DSPARK_AUX_LAYERS}"
             --speculative-config "{\"model\":\"$DRAFT\",\"num_speculative_tokens\":$NUM_SPEC,\"method\":\"${SPEC_METHOD:-dspark}\",\"draft_model_config\":{\"hf_config\":{\"eagle_aux_hidden_state_layer_ids\":$DSPARK_AUX_LAYERS}}}")
fi

# Plan B HS PRODUCER (HS_DUMP=1): verifier prefill dumps hs_<id>.safetensors via the DsparkHSDumper runner
# hook (vllm_ascend/dspark_hs_dumper.py). Needs the target config to carry dspark_target_layer_ids so the model
# allocates the aux buffer get_mtp_target_hidden_states() reads (same --hf-overrides the draft path uses). NO
# draft: target-only prefill. The dumper chmods each file 0777 so a cross-uid remote trainer can read+unlink it.
HS_ARGS=()
if [ "$HS_DUMP" = "1" ]; then
  [ -n "$DRAFT" ] && { echo "!! HS_DUMP=1 is mutually exclusive with DRAFT (the HS producer serves the target only)"; exit 2; }
  export DSPARK_HS_DUMP=1 DSPARK_HS_DIR
  mkdir -p "$DSPARK_HS_DIR" 2>/dev/null; chmod 0777 "$DSPARK_HS_DIR" 2>/dev/null || true
  HS_ARGS=(--hf-overrides "{\"dspark_target_layer_ids\":$DSPARK_AUX_LAYERS}")
  echo ">>> [HS_DUMP] Plan B producer → $DSPARK_HS_DIR (hs_<id>.safetensors); aux $DSPARK_AUX_LAYERS; NO draft"
  # NO-SHARED-FS: bind the sidecar INTO this launch so there's ONE command and --root can't drift from
  # the dump dir. Started in the background BEFORE `exec vllm` (survives the exec, keeps serving the dir).
  if [ "$HS_SIDECAR" = "1" ]; then
    pkill -TERM -u "$USER" -f 'hs_sidecar.py' 2>/dev/null; sleep 1   # graceful (SIGTERM → httpd.shutdown)
    pkill -9    -u "$USER" -f 'hs_sidecar.py' 2>/dev/null || true    # backstop if it didn't exit
    SIDECAR_TLS=(); _scheme=http
    if [ -n "$HS_SIDECAR_CERT" ] && [ -n "$HS_SIDECAR_KEY" ]; then
      SIDECAR_TLS=(--certfile "$HS_SIDECAR_CERT" --keyfile "$HS_SIDECAR_KEY"); _scheme=https
    fi
    nohup python "$SCRIPT_DIR/hs_sidecar.py" \
      --root "$DSPARK_HS_DIR" --port "$HS_SIDECAR_PORT" "${SIDECAR_TLS[@]}" \
      --workers "$HS_SIDECAR_WORKERS" --pidfile "$HS_SIDECAR_PIDFILE" \
      > "$HS_SIDECAR_LOG" 2>&1 &
    echo ">>> [HS_SIDECAR] $_scheme://0.0.0.0:$HS_SIDECAR_PORT/hs  root=$DSPARK_HS_DIR  pid=$!  (token from \$HS_SIDECAR_TOKEN)  log=$HS_SIDECAR_LOG"
    echo ">>>   remote trainer → HS_FETCH_BASE=$_scheme://<this-box-ip>:$HS_SIDECAR_PORT  hidden_states_path=$DSPARK_HS_DIR"
    echo ">>>   graceful stop → kill \$(cat $HS_SIDECAR_PIDFILE)   (SIGTERM → drains in-flight, then exits)"
  fi
fi
if [ "$HS_SIDECAR" = "1" ] && [ "$HS_DUMP" != "1" ]; then
  echo ">>> [HS_SIDECAR] ignored — needs HS_DUMP=1 (no producer = nothing to serve). Add HS_DUMP=1."
fi

# ⚠ `pkill -f` 区分大小写,而 vLLM 把子进程改名成 **VLLM::EngineCore / VLLM::Worker**(大写),
# 只有 API server 是小写的 `vllm serve`。小写模式 = 杀掉 server、**留下 engine core 活着攥住
# 每一字节 HBM**,下一次 serve 于是死在:
#   ValueError: Free memory on device (5.61/61.27 GiB) on startup is less than desired
#   GPU memory utilization (0.9, 55.14 GiB)
# 这个坑 eval_all_drafts.sh 早就记过并用 `pkill -if "$PROCPAT"` 绕开了,而这里没有 —— 实测踩过。
# 另外 `sleep 10` 是盲等:进程被 -9 之后 HBM 还要几秒才归还,残留多的时候不够。改成轮询到
# 真的没了为止,再 settle。
# ⚠ 模式里**绝不能**放 serve_dsv4_a3_singlenode —— 本脚本自己就叫这个名字,`pkill -f` 匹配
# 的是完整命令行,加进去等于 kill -9 自己。eval_all_drafts.sh 能用那个模式,是因为它是外部
# 驱动、不叫这个名。这里只打 HBM 的真正持有者。
_PROCPAT="${PROCPAT:-vllm|EngineCore}"
pkill -9 -i -u "$USER" -f "$_PROCPAT" 2>/dev/null || true
for _i in $(seq 1 30); do
  pgrep -u "$USER" -if "$_PROCPAT" >/dev/null 2>&1 || break
  [ "$_i" = 1 ] && echo ">>> 等旧进程退出(HBM 归还)…"
  sleep 2
done
if pgrep -u "$USER" -if "$_PROCPAT" >/dev/null 2>&1; then
  echo "!! 60s 后仍有进程没死,它们攥着 HBM,这次起服务多半 OOM:"
  pgrep -u "$USER" -aif "$_PROCPAT" | head -10 | sed 's/^/     /'
  echo "   手动处理后重来(必要时 sudo -n pkill -9 -if '$_PROCPAT')。"   # 引号别丢,| 是管道
  exit 1
fi
sleep 10   # 进程没了之后,驱动侧归还 HBM 还要几秒

echo ">>> [A3 single-node] model=$MODEL  DP$DP / TP$TP / EP=$ENABLE_EP  eager=$EAGER  port=$API_PORT"
echo ">>> draft=${DRAFT:-<none, plain serve>}  num_spec=$NUM_SPEC  STANDARD_DSA=${VLLM_ASCEND_DSPARK_USE_STANDARD_DSA:-<unset>}"
echo ">>> full engine log = THIS stdout (you launched under nohup → ~/dsv4_a3.log). No poll to Ctrl+C."
# DSA_OVERLAP=0 -> 关掉 multistream_dsv4_dsa_overlap(ascend_config.py 里默认 True)。
# 为什么给这个开关:4ce367a 上做 target-only 的 aux 捕获(HS_DUMP=1,无草稿)时,worker 在
#   attention/dsa_v1.py:1356 _mla_prolog_multistream -> cv_wkv.matmul 里抛 aicore 异常
#   (507015 / EE9999 rtEventQueryStatus, reason=aicore exception),且只炸 DP1 不炸 DP0。
# dsa_v1.py:1517 的分支是 `if self.multistream_dsv4_dsa_overlap:` —— 置 0 就整个绕开那段。
# ⚠ 这是【绕过】不是【修复】:同一个开关在老 pin 386530d12 上也默认 True,而那套
#   (老 pin + HS dumper)在 A3 双机上长期跑通过,所以真正的回归在 386530d12→4ce367a 之间
#   的别处。绕过只是为了先把 HS 产出来,回归要单独查/上报。
DSA_OVERLAP="${DSA_OVERLAP:-1}"
ACFG='{"enable_cpu_binding":true,"multistream_overlap_shared_expert":true}'
if [ "$DSA_OVERLAP" = "0" ]; then
  ACFG='{"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"multistream_dsv4_dsa_overlap":false}'
  echo ">>> DSA_OVERLAP=0:关闭 multistream_dsv4_dsa_overlap(绕开 _mla_prolog_multistream)"
fi

exec vllm serve "$MODEL" --served-model-name dsv4 --port "$API_PORT" \
  --data-parallel-size "$DP" --data-parallel-size-local "$DP" \
  --tensor-parallel-size "$TP" "${EP_ARGS[@]}" "${QUANT_ARGS[@]}" \
  --tokenizer-mode deepseek_v4 \
  --max-model-len "$MAXLEN" --max-num-seqs "$MAXSEQS" --block-size 128 \
  --max-num-batched-tokens "$MAXBATCHTOK" \
  --gpu-memory-utilization "$GPUUTIL" --no-enable-prefix-caching --async-scheduling \
  --additional-config "$ACFG" \
  "${LOAD_ARGS[@]}" "${SPEC_ARGS[@]}" "${HS_ARGS[@]}" \
  $EAGER_FLAG "${GRAPH_ARGS[@]}"
