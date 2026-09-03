#!/usr/bin/env bash
# One-command DSV4-DSpark training launcher (twin of serve_dsv4_bf16_dualnode.sh).
# Backgrounds a torchrun training run, logs (timestamped) into $RUN, prints the tail cmd.
#
# USAGE:
#   bash examples/ascend_npu_dflash/train_dsv4_dspark.sh [reduced|faithful]
#     reduced  (default) = 1 layer x 32 experts, 1 card, no --init-on-meta  (Gate-2 / A-B smoke)
#     faithful           = 3 layers x 256 experts, 8 cards, --init-on-meta + HCCL port ranges
#
# TOGGLE fused grouped-GEMM MoE (NPU; kills the per-shape MoE recompile spikes):
#   DSPARK_GROUPED_MOE=1 bash ... reduced      # A/B: run once without, once with, compare
#   (grouped-GEMM needs experts LOCAL -> reduced/single-card is fine; the faithful per-expert
#    FSDP run must NOT enable it yet -- needs EP/gather; the script warns if you try.)
#
# OVERRIDES (env, validated defaults baked in):
#   RUN VERIFIER DATA HS_DIR ENDPOINT LR MAX_ANCHORS NPROC CKPT_FREQ CANN_ENV
#   TRAIN_PY  entry script torchrun launches (default: $REPO_ROOT/scripts/train.py).
#             Only for read-only probes that must wrap something inside train.py before it
#             runs -- e.g. examples/ascend_npu_dflash/recall_headroom_probe.py. Default
#             behaviour is unchanged when unset.
#   RECOMPUTE=1 (activation checkpointing: recompute draft layers in bwd -> raise MAX_ANCHORS past OOM)
# NB: no `set -u` — CANN's 900env / conda activate reference unbound vars (matches serve).
set -eo pipefail

MODE="${1:-reduced}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"          # the speculators checkout

# ---- paths (A2 shared /share; override via env) ----
VERIFIER="${VERIFIER:-/share/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16}"
DATA="${DATA:-/share/canada_group_folder/dataset/open_perfectblend.dsv4_rollout/arrow_0720_77w}"  # ★ 77W (775,937). The bare `/…/arrow` is the WRONG 17W default (159,785) — never use it.
HS_DIR="${HS_DIR:-/share/canada_group_folder/dataset/dsv4_hs_dump}"
ENDPOINT="${ENDPOINT:-http://80.5.5.115:7000/v1}"
RUN="${RUN:-/home/a00652497/dspark_austin/run}"
CANN_ENV="${CANN_ENV:-/home/a00652497/900env_npu.sh}"
LR="${LR:-2e-4}"                       # 6e-4 (DeepSpec ref) diverged to NaN ~step 931 once warmup reached
EPOCHS="${EPOCHS:-5}"                   # number of training epochs (--epochs). Override e.g. EPOCHS=20.
MAX_ANCHORS="${MAX_ANCHORS:-64}"
SEQLEN="${SEQLEN:-3072}"                # --total-seq-len (default 3072; was 8192). Shorter cuts draft-forward
                                       # activation memory (room for more anchors) + shortens the
                                       # HS prefill; anchor utilization = MAX_ANCHORS/SEQLEN.
MASK_TOKEN="${MASK_TOKEN:-128799}"     # DSpark noise token (config.py noise_token_id). Draft's masked
                                       # positions embed as embed_tokens[MASK_TOKEN]; MUST match serve.
                                       # Without it, resolve_mask_token_id falls back to pad_token_id=1
                                       # (wrong: collides with real pad + mismatches the official 128799).
BLOCK="${BLOCK:-5}"                     # ★ BLOCK = # draft slots per block = gamma = num_spec. With
                                       # sample_from_anchor=True (DSpark default, RESTORED 2026-07-17) ALL
                                       # BLOCK slots are sampled predictions: slot 0 predicts the NEXT token
                                       # from the anchor's own hidden (no target shift, slot 0 IS trained),
                                       # slots 1.. from noise. => BLOCK=5 matches released dspark_block_size=5
                                       # / serve num_spec=5 EXACTLY (5-slot block, all predicted, logs show
                                       # position_0..4). ⚠️ The OLD BLOCK=6 was the sample_from_anchor=False
                                       # workaround (slot 0 = given anchor, drafts BLOCK-1=5) — that trained the
                                       # WRONG off-by-one task with slot 0 untrained and collapsed at serve
                                       # (fixed in dspark core). DO NOT use 6 now. NB: BLOCK scales draft-forward
                                       # tokens = MAX_ANCHORS*BLOCK -> memory (256*5=1280).
MAX_STEPS="${MAX_STEPS:-}"              # 空 = 跑满 EPOCHS。给个数就在这么多优化器步后停,
                                       # 用于只量显存/耗时的短测(30 步足够看 mem/max_reserved_gb
                                       # 和 profile/*_ms 稳下来),不用等一个 epoch。
OPTIM="${OPTIM:-adamw}"                  # adamw | muon.  ⚠ muon 只作用于 2D 权重矩阵,
                                       # 其余(norm/bias/embed/head)仍走 AdamW。它省的是优化器
                                       # 状态:AdamW 两个 buffer(exp_avg+exp_avg_sq),Muon 一个
                                       # (momentum)。⚠ MUON_LR 不设时 schema 默认 10*lr —— 对我们
                                       # 2e-4 就是 2e-3,而 AdamW 在 6e-4 就 NaN 过。做显存/耗时
                                       # 对照时用 MUON_ADJUST=match_rms_adamw 并显式给 MUON_LR。
MUON_LR="${MUON_LR:-}"                  # 空 = 用 schema 默认(10*lr)
MUON_ADJUST="${MUON_ADJUST:-match_rms_adamw}"   # original | match_rms_adamw
DECAY_GAMMA="${DECAY_GAMMA:-4.0}"       # loss per-position decay: weight_k = exp(-k/gamma) over the BLOCK
                                       # slots. Canonical DSpark = 4.0 -> slots [1.0,0.78,0.61,0.47,0.37].
                                       # LARGER gamma = flatter = more gradient to LATER slots (pos3/4/5);
                                       # very large -> ~uniform. The knob to lift late-position accept when
                                       # the per-position curve decays faster than the released draft's.
SWA_WINDOW="${SWA_WINDOW:-128}"        # draft sliding-window context (--sliding-window). ★ 128 = ALIGN to
                                       # the released draft (config sliding_window=128). POLICY: full fidelity
                                       # to the released recipe — every divergent field gets aligned, we do NOT
                                       # keep our own value just because it ran. (Our old runs used the CLI
                                       # default 2048 = a divergence; now closed.) The dead `window_size` field
                                       # is a separate unused config knob, unrelated.
NONCAUSAL="${NONCAUSAL:-1}"            # --sliding-window-non-causal: block-internal attention NON-CAUSAL (every
                                       # draft slot sees ALL γ slots). ★ MUST be ON for DSV4-DSpark — the
                                       # vllm-ascend serve drafts NON-causally (deepseek_v4_dspark_proposer.py
                                       # `cad.causal=False`); training it OFF (the old CLI default) = causal block
                                       # = a train↔serve mismatch that FREEZES the pos2+ tail (ep0≈ep2.5 flat;
                                       # released, trained non-causal, gets 63/53). NONCAUSAL=0 only to reproduce
                                       # the old broken causal runs. (train.py help's "vLLM doesn't support" note
                                       # is STALE — our serve REQUIRES non-causal.)
SCHED_TYPE="${SCHED_TYPE:-cosine}"     # LR scheduler (--scheduler-type). cosine = align to the DeepSpec DSpark
                                       # trainer (was speculators' default linear).
WARMUP_RATIO="${WARMUP_RATIO:-0.04}"   # --scheduler-warmup-ratio; 0.04 (4%) = DeepSpec. Previously unset (~no
                                       # warmup). NB the 6e-4-NaN memory blames too-little warmup — this closes it.
LOSS_FN="${LOSS_FN:-{\"ce\":0.1,\"tv\":1.8}}"  # ce + TVD weights. ★ tv 1.8 (not 0.9): speculators `tv_loss` = TVD
                                       # = 1/2 of DeepSpec's L1 (=sum|p-q|=2*TVD, PR #648 chose the standard TVD
                                       # normalization), so tv 1.8 restores DeepSpec's effective ce:dist = 0.1:1.8
                                       # balance. Pure normalization-convention alignment. Set LOSS_FN=... to override.
TEACHER_DNORM="${TEACHER_DOUBLE_NORM:-0}"  # DSPARK_TEACHER_DOUBLE_NORM: =1 re-applies verifier_norm to the
                                       # (already post-norm) teacher hidden → norm(norm(h)) = the accidental
                                       # "double-norm bug" as a DELIBERATE knob. ★ 07-25 eval: it BEAT the
                                       # single-norm teacher by ~0.2 accept_len, ALL in the tail. It RESHAPES
                                       # the teacher on the HIDDEN (per-dim ~w² reweight, flips argmax ~16%) —
                                       # NOT the same as --kd-temperature (uniform logit sharpening). 0 = OFF
                                       # (single-norm, reproduction-faithful); set 1 to reproduce/A-B the win.
KD_TEMP="${KD_TEMPERATURE:-1.0}"        # --kd-temperature: teacher SHARPENING on the LOSS (teacher logits / T
                                       # before softmax). UNIFORM sharpening, argmax PRESERVED — a DIFFERENT
                                       # mechanism from TEACHER_DOUBLE_NORM (which reshapes on the hidden). A
                                       # separate technique to try; may or may not capture the double-norm gain.
                                       # 1.0 = OFF (default). ★ ICING — leave default for reproduction.
NOISE_STD="${NOISE_STD:-0.05}"          # --noise-std: uniform ±std noise added to target hidden states (aug).
                                       # 0.05 = current / DFlash-inherited default. ★ A/B knob: NOISE_STD=0 =
                                       # DeepSpec (it trains with ZERO hidden-state noise) — top candidate to
                                       # sharpen the pos2+ tail. Change ONE variable per run.
GROUPED="${DSPARK_GROUPED_MOE:-0}"
EP="${DSPARK_EP:-0}"
BF16EXPERTS="${BF16_EXPERTS:-auto}"     # AMP masters for EP experts. DEFAULT "auto": option A (upstream —
                                       # experts get fp32 masters) EXCEPT the faithful+EP=0 path, which
                                       # rank0-materialises the full unsharded model → fp32 experts OOM, so
                                       # auto-downgrades to option B there. Force with BF16_EXPERTS=1 (always
                                       # bf16 experts) or =0 (always fp32). Under EP=1 (the training norm) A fits.
RECOMPUTE="${RECOMPUTE:-0}"             # activation checkpointing: recompute each draft layer in backward
                                       # -> frees activation so MAX_ANCHORS scales past the memory wall
                                       # (the lever to slow an HS-bound step to the serve's HS rate).
COMPILE="${COMPILE:-0}"                # ★ SEED TECH: torch.compile'd experts (kills the ~42% grouped-GEMM
                                       # recompile, ~1.74x). REQUIRES the torch-2.12 stack (torch 2.12+cpu /
                                       # torch_npu 2.12rc1 / inductor_npu_ext / triton-ascend) — do NOT set
                                       # on the 2.10 main stack (desyncs train vs the 2.10 serve).
# Dataloader concurrency = HS fetch concurrency. DEFAULT 4 in REMOTE mode (HS_FETCH_BASE set): the
# remote HS fetch is expensive (~100 MB/sample prefill+HTTP through ONE serve MAXSEQS=64 + one sidecar
# process), so NPROC*NUM_WORKERS concurrent fetchers must stay near the serve's cap, else the first
# batch never assembles (all ranks I/O-block, AICore 0%). Shared-FS (no HS_FETCH_BASE) keeps 12 (local
# reads are cheap). Override with NUM_WORKERS=.
if [ -n "${HS_FETCH_BASE:-}" ]; then NUM_WORKERS="${NUM_WORKERS:-4}"; else NUM_WORKERS="${NUM_WORKERS:-12}"; fi
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
NOVAL="${NO_VAL:-1}"                    # ★ WELDED DEFAULT=1 (2026-07-30): cancel the per-epoch validation pass.
                                       # Val does SERIAL online HS generation for the 10% held-out split
                                       # (num_workers=0) which dominates the epoch AND wastes 10% of the data;
                                       # --no-validation trains on the FULL data. Pass NO_VAL=0 to re-enable val.
INITMOE="${INIT_MOE:-0}"               # INIT_MOE=1 -> warm-start the draft MoE (experts+router+shared) from
                                       # the verifier's target layers (draft n <- target_layer_ids[n]).
                                       # Trainable init; faithful only. A/B vs from-scratch.
INITATTN="${INIT_ATTN:-0}"             # INIT_ATTN=1  -> warm-start MLA core projections (wq_*/wkv/wo_*/sink)
INITHC="${INIT_HC:-0}"                 # INIT_HC=1    -> warm-start the two hyper-connections (attn_hc/ffn_hc)
INITNORM="${INIT_NORM:-0}"             # INIT_NORM=1  -> warm-start the two RMSNorms (attn_norm/ffn_norm)
INITLAYER="${INIT_LAYER:-0}"           # INIT_LAYER=1 -> WHOLE layer = attn+hc+norm+moe (coherent target layer)
INITNOROUTER="${INIT_MOE_NO_ROUTER:-0}"  # INIT_MOE_NO_ROUTER=1 -> with INIT_MOE/INIT_LAYER, leave the ROUTER at a
                                       # FRESH random init (experts + shared still warm-started). Un-collapse test:
                                       # start balanced instead of inheriting the verifier's pre-collapsed routing.
FROM_PRETRAINED="${FROM_PRETRAINED:-}"  # FROM_PRETRAINED=<dir> -> warm-start ALL draft weights from a saved
                                       # speculator checkpoint (a <N>/ ckpt dir: config.json + model.safetensors),
                                       # FRESH optimizer + FRESH cosine LR (NOT a resume). Takes precedence over
                                       # the INIT_* flags (train.py: --from-pretrained wins). ★ ABLATION USE:
                                       # warm-start from a prior CAUSAL run to test if its weights are a useful
                                       # init for non-causal training. ⚠️ --sliding-window-non-causal is read from
                                       # the LOADED config.json (NOT re-applied from CLI), so to train non-causal
                                       # on a causal ckpt you MUST first set "sliding_window_non_causal": true in
                                       # the copied dir's config.json — else NONCAUSAL=1 is silently ignored.

# ---- per-mode config ----
if [ "$MODE" = "faithful" ]; then
  NPROC="${NPROC:-8}"; LAYERS=3; EXPERTS=256
  PORTS="HCCL_NPU_SOCKET_PORT_RANGE=61000-62000 HCCL_HOST_SOCKET_PORT_RANGE=60000-61000"
  if [ "$EP" = "1" ]; then
    # EP: routed experts are Shard(0) DTensors (256/NPROC per card); grouped-GEMM runs on
    # the local slice. NO --init-on-meta (each rank builds only its 1/EP slice per-rank).
    # Checkpoints work under EP (DCP gathers the Shard(0) expert DTensors). DEFAULT = once per
    # epoch (CKPT_FREQ=1.0): EP save is DTensor-native (fast, no 768 per-expert gather), and
    # per-epoch keeps overhead minimal on the multi-day run. While DEBUGGING save/resume, set a
    # small CKPT_FREQ (e.g. 0.1 => every round(N*freq) steps) to hit the save/resume path fast.
    EXTRA="--checkpoint-freq ${CKPT_FREQ:-1.0}"
  else
    EXTRA="--init-on-meta --checkpoint-freq ${CKPT_FREQ:-1.0}"
    if [ "$GROUPED" = "1" ]; then
      echo "!! faithful uses per-expert FSDP; grouped-GEMM needs experts LOCAL. Turn on EP"
      echo "   instead:  DSPARK_EP=1 bash $0 faithful  (partitions experts + all-to-all)."
      echo "   Refusing grouped-GEMM on per-expert-FSDP faithful."; exit 2
    fi
  fi
elif [ "$MODE" = "reduced" ]; then
  NPROC="${NPROC:-1}"; LAYERS=1; EXPERTS=32; EXTRA=""; PORTS=""
  [ "$EP" = "1" ] && PORTS="HCCL_NPU_SOCKET_PORT_RANGE=61000-62000 HCCL_HOST_SOCKET_PORT_RANGE=60000-61000"
else
  echo "usage: $0 [reduced|faithful]"; exit 1
fi

# EP preconditions: >=2 cards and experts divisible by cards.
if [ "$EP" = "1" ]; then
  [ "$NPROC" -ge 2 ] || { echo "!! DSPARK_EP=1 needs NPROC>=2 (got $NPROC). e.g. DSPARK_EP=1 NPROC=2 bash $0 reduced"; exit 2; }
  [ $((EXPERTS % NPROC)) -eq 0 ] || { echo "!! EXPERTS=$EXPERTS not divisible by NPROC=$NPROC"; exit 2; }
fi

# NO_VAL=1 -> append --no-validation (skip the slow per-epoch val pass, train on full data).
if [ "$NOVAL" = "1" ]; then EXTRA="$EXTRA --no-validation"; fi
# INIT_*=1 -> append the matching --init-*-from-target warm-start flag(s).
if [ "$INITMOE" = "1" ]; then EXTRA="$EXTRA --init-moe-from-target"; fi
if [ "$INITATTN" = "1" ]; then EXTRA="$EXTRA --init-attn-from-target"; fi
if [ "$INITHC" = "1" ]; then EXTRA="$EXTRA --init-hc-from-target"; fi
if [ "$INITNORM" = "1" ]; then EXTRA="$EXTRA --init-norm-from-target"; fi
if [ "$INITLAYER" = "1" ]; then EXTRA="$EXTRA --init-layer-from-target"; fi
if [ "$INITNOROUTER" = "1" ]; then EXTRA="$EXTRA --init-moe-no-router"; fi
if [ -n "$FROM_PRETRAINED" ]; then EXTRA="$EXTRA --from-pretrained $FROM_PRETRAINED"; fi
# MARKOV_HEAD_TYPE=dflash2 -> the released DFlash2 selector's score, bias(a,h) = <A(a)*H(h), B>.
# ⚠ IGNORED SILENTLY under FROM_PRETRAINED: train.py:529 rebuilds the config from the
# checkpoint, and markov_head_type is NOT in DECODER_SHAPING_FLAGS, so it is neither applied
# nor rejected -- exactly the trap already documented above for --sliding-window-non-causal.
# To change the head type on a warm start you MUST edit the copied dir's config.json.
if [ -n "${MARKOV_HEAD_TYPE:-}" ]; then EXTRA="$EXTRA --markov-head-type $MARKOV_HEAD_TYPE"; fi
# SELECT_RANK=256 -> the ADDITIVE selection head. Zero-initialised: an exact no-op until
# trained, and ablatable by zeroing the term rather than retraining. Same FROM_PRETRAINED
# caveat as above -- on a warm start, set it in the copied dir's config.json.
if [ -n "${SELECT_RANK:-}" ]; then EXTRA="$EXTRA --select-rank $SELECT_RANK"; fi
# BLOCK_CONV_TAPS=2 -> the DFlash2 dynamic causal convolutions around each sublayer.
# Identity-initialised; same FROM_PRETRAINED silent-ignore caveat as the flags above.
if [ "$OPTIM" = "muon" ]; then
  [ -n "$MUON_LR" ] && EXTRA="$EXTRA --muon-lr $MUON_LR"
  EXTRA="$EXTRA --muon-adjust-lr-fn $MUON_ADJUST"
fi
[ -n "$MAX_STEPS" ] && EXTRA="$EXTRA --max-steps $MAX_STEPS"
if [ -n "${BLOCK_CONV_TAPS:-}" ]; then EXTRA="$EXTRA --block-conv-kernel-size $BLOCK_CONV_TAPS"; fi
if [ -n "${BLOCK_CONV_GROUP:-}" ]; then EXTRA="$EXTRA --block-conv-group-size $BLOCK_CONV_GROUP"; fi

# AMP experts: resolve "auto" -> downgrade to bf16 experts ONLY on the faithful+EP=0 path (rank0
# materialises the full unsharded model; fp32 experts would OOM). EP=1 / reduced keep option A.
if [ "$BF16EXPERTS" = "auto" ]; then
  if [ "$MODE" = "faithful" ] && [ "$EP" != "1" ]; then BF16EXPERTS=1; else BF16EXPERTS=0; fi
fi
if [ "$BF16EXPERTS" = "1" ]; then
  EXTRA="$EXTRA --bf16-experts"
  echo ">>> AMP: option B (experts stay bf16, no fp32 master) — memory path for faithful+EP=0"
else
  echo ">>> AMP: option A (experts get fp32 master, upstream #711) — fits under EP=1"
fi

# ---- COMPILE=1 on NPU: two box-dependent footguns the inductor_npu_ext AscendC path hits ----
# (1) umask: inductor's codecache REJECTS a group/world-writable kernel.so ("must not be writable
#     by group or others"). On a umask-002 box the AscendC build emits 0o775 and compile dies with an
#     opaque "Failed to build ascend kernel"; umask 0022 makes new .so 0o755. (This is exactly why a
#     umask-022 box compiled fine while a umask-002 box didn't — deterministic, not fork luck.)
# (2) spawn the compile worker: forking a torch_npu-initialised, multithreaded training process yields
#     flaky build failures (pytorch #148651 / #108586). spawn = clean fresh worker.
# (3) persistent compile cache on a LOCAL disk under COMPILE_CACHE_DIR (default $RUN/compile_cache).
#     Survives /tmp cleanup so kernels are REUSED across runs (a killed+relaunched run skips the full
#     AscendC rebuild warmup). Redirects all 3 cache layers at once — inductor codecache
#     (TORCHINDUCTOR_CACHE_DIR), triton-ascend (TRITON_CACHE_DIR), and the AscendC .npu_kernels_$USER
#     dir (which follows $TMPDIR). ★ MUST be a LOCAL disk, NOT /share: inductor file-LOCKS the
#     codecache, so a networked FS makes compile slow/flaky (and contends with the training's /share
#     I/O). cache_size_limit default 1024 (DSPARK_COMPILE_CACHE_LIMIT), consumed in moe_compile.py.
if [ "$COMPILE" = "1" ]; then
  umask 0022
  export TORCHINDUCTOR_WORKER_START="${TORCHINDUCTOR_WORKER_START:-spawn}"
  COMPILE_CACHE_DIR="${COMPILE_CACHE_DIR:-$RUN/compile_cache}"
  mkdir -p "$COMPILE_CACHE_DIR/inductor" "$COMPILE_CACHE_DIR/triton" "$COMPILE_CACHE_DIR/tmp"
  export TORCHINDUCTOR_CACHE_DIR="$COMPILE_CACHE_DIR/inductor"
  export TRITON_CACHE_DIR="$COMPILE_CACHE_DIR/triton"
  export TMPDIR="$COMPILE_CACHE_DIR/tmp"
  export DSPARK_COMPILE_CACHE_LIMIT="${DSPARK_COMPILE_CACHE_LIMIT:-1024}"
  echo ">>> COMPILE=1: umask 0022 | WORKER_START=$TORCHINDUCTOR_WORKER_START | cache_dir=$COMPILE_CACHE_DIR (local disk) | cache_limit=$DSPARK_COMPILE_CACHE_LIMIT"
fi

# ---- preflight ----
if [ -f "$CANN_ENV" ]; then
  set +e; source "$CANN_ENV"; set -e         # env scripts return nonzero benignly
else
  echo "WARN: CANN env not found at $CANN_ENV — pass CANN_ENV=/path/to/900env_npu.sh"
fi
mkdir -p "$RUN" 2>/dev/null || { echo "!! cannot create RUN=$RUN (different box user?) — override e.g. RUN=\$HOME/dspark_austin/run"; exit 1; }
rm -f "$HS_DIR"/hs_*.safetensors*                     # stale dumps collide with data row idx
# proxy-bypass hosts DERIVED from the endpoint (+ optional remote HS sidecar) so this works on
# BOTH A2 (endpoint 115/116, shared FS) and A3 (endpoint 182 + HS_FETCH_BASE 182, no shared FS).
# The old hardcoded 115/116 list silently sent A3's 182 traffic through the MITM proxy → HS fetch hung.
_ep_host="$(printf '%s' "$ENDPOINT"        | sed -E 's#^https?://([^:/]+).*#\1#')"
_fb_host="$(printf '%s' "${HS_FETCH_BASE:-}" | sed -E 's#^https?://([^:/]+).*#\1#')"
export no_proxy="127.0.0.1,localhost,${_ep_host}${_fb_host:+,$_fb_host}"
export NO_PROXY="$no_proxy"
if ! curl -sf --noproxy '*' "$ENDPOINT/models" >/dev/null 2>&1; then
  echo "WARN: serve not reachable at $ENDPOINT — start 115/116 first (or set ENDPOINT=)."
fi

TS="$(date +%Y%m%d_%H%M%S)"
# NB: build TAG with if-blocks, NOT `TAG="...$([ ] && echo ...)"` — under set -e a
# command substitution whose test fails returns nonzero and silently kills the script.
TAG="$MODE"
if [ "$EP" = "1" ]; then TAG="${TAG}_ep"; fi
if [ "$GROUPED" = "1" ]; then TAG="${TAG}_grouped"; fi
# --sliding-window-non-causal is action="store_true": add the flag ONLY when NONCAUSAL=1.
# if-block (NOT `&&`) so a NONCAUSAL=0 test can't nonzero-abort under set -e.
NONCAUSAL_FLAG=""
if [ "$NONCAUSAL" = "1" ]; then NONCAUSAL_FLAG="--sliding-window-non-causal"; fi
LOG="$RUN/${TAG}_${TS}.log"
# Fresh per-run save-path so we DON'T auto-resume a stale (possibly non-EP) checkpoint
# from ./output. Override SAVE_PATH=<dir> to resume a specific run.
SAVE_PATH="${SAVE_PATH:-$RUN/ckpt_${TAG}_${TS}}"

echo "==================================================================="
echo " DSV4-DSpark TRAIN  mode=$MODE  nproc=$NPROC  ${LAYERS}L x ${EXPERTS}E  lr=$LR  epochs=$EPOCHS  ep=$EP  grouped_moe=$GROUPED  recompute=$RECOMPUTE  compile=$COMPILE  noval=$NOVAL  init(moe/attn/hc/norm/layer)=$INITMOE/$INITATTN/$INITHC/$INITNORM/$INITLAYER"
echo " optimizer=$OPTIM  max_steps=${MAX_STEPS:-<all>}  muon_lr=${MUON_LR:-<10*lr>}  muon_adjust=$MUON_ADJUST"
echo " block=$BLOCK (all $BLOCK slots drafted = gamma = num_spec; sample_from_anchor=True)  seqlen=$SEQLEN  max_anchors=$MAX_ANCHORS  noncausal_block=$NONCAUSAL (=serve cad.causal=False)"
echo " draft-forward tokens = max_anchors*block = $((MAX_ANCHORS*BLOCK))  (anchor util = $MAX_ANCHORS/$SEQLEN)"
echo " verifier=$VERIFIER"
echo " data=$DATA"
echo " 📋 log -> $LOG   (rank0 mirror also in $RUN/train_*.log)"
echo " 💾 save -> $SAVE_PATH"
echo "==================================================================="

# hs_connectors 是 uv workspace member(上游 #735),普通 `pip install -e .` 解析不到,而
# src/speculators/train/data.py 无条件 import 它 -> 每个 rank 在 import 期就死。永久修法是
# `pip install --no-deps -e hs_connectors/`(已内建于 install_npu_env_dspark.sh);这里兜底,
# 使未装的环境也能直接跑。
export PYTHONPATH="$REPO_ROOT/hs_connectors/src${PYTHONPATH:+:$PYTHONPATH}"

# metrics.py 用 `logits.is_cuda` 判断能否走 Triton 融合核 —— 但 scripts/train.py 会 import
# torch_npu 的 transfer_to_npu,它把 NPU 张量的 .is_cuda 打成 True,判据因此在昇腾上恒真,
# 于是去调根本没有昇腾后端的 Triton 核。置 1 显式关掉融合路径,走 eager 损失。
export SPECULATORS_DISABLE_FUSED_LOSS="${SPECULATORS_DISABLE_FUSED_LOSS:-1}"

# ⚠ 从这里到 torchrun 是【一条命令】(`env VAR=… torchrun …`),每行以反斜杠续行。
# 中间不得插入任何语句或注释 —— 注释会吃掉该逻辑行的剩余部分,`env` 就变成"没有命令"、
# 只打印环境后退出(留下 nohup.out),torchrun 则脱离这段 env 前缀独立运行,于是
# DSPARK_HS_DUMP / PYTORCH_NPU_ALLOC_CONF 等【只存在于此处】的变量全部丢失,表现为
# "Response missing kv_transfer_params"(HS 走错路径)与显存 OOM(抗碎片没开)。
# 要加环境变量:加进下面的列表;要加 export:放到本注释【之前】。
nohup env \
  DSPARK_HS_DUMP=1 DSPARK_GROUPED_MOE="$GROUPED" DSPARK_EP="$EP" DSPARK_RECOMPUTE="$RECOMPUTE" DSPARK_COMPILE="$COMPILE" \
  DSPARK_TEACHER_DOUBLE_NORM="$TEACHER_DNORM" \
  DSPARK_MOE_BALANCE="${DSPARK_MOE_BALANCE:-0}" DSPARK_MOE_BALANCE_RATE="${DSPARK_MOE_BALANCE_RATE:-1e-3}" \
  DSPARK_LOG_EXPERT_LOAD="${DSPARK_LOG_EXPERT_LOAD:-0}" \
  PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}" \
  HCCL_CONNECT_TIMEOUT=1800 HCCL_EXEC_TIMEOUT=1800 $PORTS \
  torchrun --nproc_per_node "$NPROC" "${TRAIN_PY:-$REPO_ROOT/scripts/train.py}" \
    --speculator-type dsv4_dspark --served-model-name dsv4 \
    --num-layers "$LAYERS" --n-routed-experts "$EXPERTS" \
    --block-size "$BLOCK" --target-layer-ids 40 41 42 --max-anchors "$MAX_ANCHORS" \
    --dflash-decay-gamma "$DECAY_GAMMA" --sliding-window "$SWA_WINDOW" $NONCAUSAL_FLAG \
    --total-seq-len "$SEQLEN" --mask-token-id "$MASK_TOKEN" --noise-std "$NOISE_STD" \
    --kd-temperature "$KD_TEMP" \
    --draft-attn-impl sdpa --loss-fn "$LOSS_FN" \
    --scheduler-type "$SCHED_TYPE" --scheduler-warmup-ratio "$WARMUP_RATIO" \
    --optimizer "$OPTIM" --lr "$LR" --epochs "$EPOCHS" $EXTRA \
    --on-missing generate --on-generate delete \
    --num-workers "$NUM_WORKERS" --prefetch-factor "$PREFETCH_FACTOR" \
    --hidden-states-path "$HS_DIR" --vllm-endpoint "$ENDPOINT" \
    --verifier-name-or-path "$VERIFIER" --data-path "$DATA" \
    --save-path "$SAVE_PATH" --log-dir "$RUN" \
  > "$LOG" 2>&1 &

echo ">>> started PID $!  |  tail -f $LOG"
echo ">>> watch: profile/grad_norm (blow-up before NaN), profile/fwd_ms (grouped => spikes gone),"
echo ">>>        train/loss (non-zero, decreasing), NaN => kill immediately."
