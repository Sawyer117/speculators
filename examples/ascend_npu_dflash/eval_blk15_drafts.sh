#!/usr/bin/env bash
# BATCH eval of the BLOCK=15 draft line on the MAINLINE stack (vLLM 0.27.1 + vllm-ascend
# 4ce367a + CANN 9.2.0-beta1), into ONE master log.
#
# Sibling of eval_all_drafts.sh — that one stays as-is for the block5 / 176 line. This copy
# differs in exactly four places, each of which is a trap on the mainline stack:
#
#   1. serve script = serve_dsv4_a3_singlenode_specmethod.sh, NOT the plain one.
#      On 386530d12 `mtp` and `dspark` were the same path; on mainline `method` decides which
#      model class loads, and `dspark` is a first-class citizen with its own proposer. The
#      plain script asks for `mtp` and silently serves the wrong thing.
#   2. NUM_SPEC defaults to 15, not 5. DSpark block attention is NON-causal (cad.causal=False),
#      so num_speculative_tokens MUST be >= the draft's block_size. Below it there is no error,
#      only wrong numbers.
#   3. PREFETCH=0. On 0.27.1 `--safetensors-load-strategy prefetch` and multithreaded load
#      became mutually exclusive (they coexisted on 0.23.0).
#   4. TOKENIZER defaults under /home, not /share — this box has no /share.
#
#   # ⚠ export LD_LIBRARY_PATH BEFORE launching (mainline needs conda's libstdc++ ahead of
#   #   the system one; the serve script does not do it for you):
#   conda activate dspark-dsv4-serving
#   export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
#   nohup bash examples/ascend_npu_dflash/eval_blk15_drafts.sh > ~/eval_blk15_driver.log 2>&1 &
#   tail -f ~/eval_blk15_*/MASTER.log
#
# ⚠ THE BAR IS STACK-SPECIFIC. Released draft, num_spec=5:
#     old 176  (386530d12 / CANN 9.0.0)       gsm8k 4.665 / 5-set 4.4232
#     mainline (4ce367a  / CANN 9.2.0-beta1)  gsm8k 4.523 / 5-set 4.287
#   These entries run at num_spec=15, whose accept_len ceiling is 15 rather than 5, so they are
#   NOT directly comparable to either bar in magnitude. What IS comparable: tok/s end-to-end,
#   and the per-position conditional accept rate for positions 0-4.
#
# ⚠ MEASUREMENT SET: KEEP_WARMUP=1 (post-2026-08-13 default) = full 1319/500/164/257/80.
#   Do NOT mix these rows with pre-cutover 1309/490/154/247/70 rows.
#
# Env knobs (all optional):
#   PORT=7000  CONCURRENCY=48  DATASET=all  KEEP_WARMUP=1  MAX_NEW=2048  NUM_SPEC=15
#   DATASET=gsm8k          the cheap curve — enough to answer "converging or saturating?"
#   ONLY=<egrep pattern>   only entries whose label matches (e.g. ONLY='ep1p0|ep2p0|ep3p0')
#   SKIP_DONE=1            skip entries whose log already ended with FINAL SUMMARY
#   SERVE_TIMEOUT=1800  SETTLE=45  OUTDIR=~/eval_blk15_<TS>
#   HBM_WAIT=1800  HBM_FREE_MB=4096  PROC_WAIT=300
#     ★ 进程退干净 ≠ 显存回收完。崩溃退出后驱动实测要 ~12 分 40 秒,
#       而旧版只等 180 s 就起下一个 serve —— 2026-09-19 的 ep3 因此连挂七次。
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PORT="${PORT:-7000}"
CONCURRENCY="${CONCURRENCY:-48}"
DATASET="${DATASET:-all}"
KEEP_WARMUP="${KEEP_WARMUP:-1}"
MAX_NEW="${MAX_NEW:-2048}"
SERVE_TIMEOUT="${SERVE_TIMEOUT:-1800}"
NUM_SPEC="${NUM_SPEC:-15}"        # ⚠ 必须 >= 草稿的 block_size(非因果块注意力)
PREFETCH="${PREFETCH:-0}"         # ⚠ 0.27.1 上与多线程加载互斥
SETTLE="${SETTLE:-60}"   # fallback settle when npu-smi can't be read at all
# ⚠ vLLM renames its subprocesses to VLLM::EngineCore / VLLM::Worker -- UPPERCASE -- while the
#   API server is `vllm serve`, lowercase. `pkill -f` is CASE-SENSITIVE, so a lowercase-only
#   pattern kills the server and leaves the engine cores alive still holding every byte of HBM,
#   and the next serve then OOMs at weight load. Every kill/probe below uses -i.
PROCPAT="${PROCPAT:-vllm|serve_dsv4_a3_singlenode|EngineCore}"
KILL_PREFIX="${KILL_PREFIX:-}"       # set to 'sudo -n' if the serves need root to kill
SKIP_DONE="${SKIP_DONE:-1}"
ONLY="${ONLY:-}"
TS="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${OUTDIR:-$HOME/eval_blk15_$TS}"
MASTER="$OUTDIR/MASTER.log"
mkdir -p "$OUTDIR"

# Box ckpt root — same auto-detect as run_dspark_eval.sh (A3-176 / A2 / A3-nfs).
if [ -z "${CKPT_ROOT:-}" ]; then
  for _d in /home/canada_group_folder/ckpt /share/canada_group_folder/ckpt /mnt/nfs/canada_group_folder/ckpt; do
    [ -d "$_d" ] && CKPT_ROOT="$_d" && break
  done
fi
CKPT_ROOT="${CKPT_ROOT:-/home/canada_group_folder/ckpt}"
# ⚠ 必须在 CKPT_ROOT 之后:eval 客户端的 TOKENIZER 默认指向 /share,本机只有 /home。
export TOKENIZER="${TOKENIZER:-$CKPT_ROOT/DeepSeek-V4-Flash-bf16}"

export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"

# ---------------------------------------------------------------------------------------
# The work list: "label|dirname". An EMPTY dirname = AR baseline (no DRAFT -> no drafting;
# its tok/s is the speedup denominator).
#
# Ordered so a batch that dies early still delivered the headline: the converged endpoints
# first, then backwards down the curve. ep3p0 may not be on this box yet — a missing dir is
# reported and skipped, it does not kill the run.
# ---------------------------------------------------------------------------------------
ENTRIES=(
  "ep5p0-blk15|dsv4_dspark_blk15_ep5p0_vllm-77w"
  "ep4p0-blk15|dsv4_dspark_blk15_ep4p0_vllm-77w"
  "ep3p0-blk15|dsv4_dspark_blk15_ep3p0_vllm-77w"
  "ep2p0-blk15|dsv4_dspark_blk15_ep2p0_vllm-77w"
  "ep1p0-blk15|dsv4_dspark_blk15_ep1p0_vllm-77w"
)

# ENTRIES_OVERRIDE = 空格分隔的 "label|dirname",整份替换上面的清单。
# 上面那份是写死的 5 条;换一个 run(或者半 epoch 存点 ep2p5 这种)就对不上了,而对不上的
# 表现是「MISSING draft dir — SKIPPED」,一批跑完什么都没量到。export_run_ckpts.py 导完会把
# 可直接粘贴的这一串打出来,所以新导一批不必回来改这个文件。
if [ -n "${ENTRIES_OVERRIDE:-}" ]; then
  read -r -a ENTRIES <<< "$ENTRIES_OVERRIDE"
fi

say() { echo "$*" | tee -a "$MASTER"; }

serve_up() { curl -sf --noproxy '*' "http://localhost:$PORT/v1/models" >/dev/null 2>&1; }

# ⚠ Zombies must NOT count as alive. A killed vLLM worker whose parent never wait()s stays in
# the process table as `[VLLM::Worker] <defunct>` -- 24 of them, 17 days old, were sitting on the
# 176 box. They hold no NPU and no signal can touch them (only the parent reaping, or dying, ever
# clears one), so a plain pgrep made every teardown burn its full kill timeout: ~180 s x 2 per
# entry, ~1.8 h across an 18-entry batch.
procs_alive() {
  local p st
  for p in $(pgrep -if "$PROCPAT" 2>/dev/null); do
    st=$(ps -o stat= -p "$p" 2>/dev/null | tr -d ' ')
    case "$st" in
      '' | Z*) continue ;;    # already gone, or an unreapable <defunct> shell
      *)       return 0 ;;
    esac
  done
  return 1
}

# Max per-device HBM in use, in MB. Best-effort: `npu-smi info` prints a `used / total` cell
# per device; keep only cells whose total looks like memory (>1000) so the `0 / 0` AICore
# cells are ignored. Returns non-zero if npu-smi is absent or the output can't be parsed,
# in which case callers fall back to a fixed sleep.
npu_used_mb() {
  command -v npu-smi >/dev/null 2>&1 || return 1
  npu-smi info 2>/dev/null | grep -oE '[0-9]+ +/ +[0-9]+' \
    | awk -F'/' '{ u=$1+0; t=$2+0; if (t > 1000 && u > m) m = u } END { if (m == "") exit 1; print m+0 }'
}

# Is ANY process still holding an NPU? `npu-smi info` ends with a per-device process table; an
# idle device prints "No running processes found". This is the format-tolerant question to ask --
# an absolute HBM threshold is not, because the idle Memory-Usage baseline differs per box and a
# too-low threshold would stall every teardown for the full timeout.
#   0 = a device is still held   1 = nothing is holding one   2 = cannot tell (no npu-smi / odd output)
npu_procs_held() {
  command -v npu-smi >/dev/null 2>&1 || return 2
  local out; out="$(npu-smi info 2>/dev/null)" || return 2
  echo "$out" | grep -qi "Process id" || return 2
  echo "$out" | awk '
    /Process id/ { p = 1; next }
    p && /^\|[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]*\|[[:space:]]*[0-9]+/ { n++ }
    END { exit (n > 0 ? 0 : 1) }'
}

# HBM is released by the driver a beat AFTER the last process exits. Starting the next serve
# before that makes it OOM at weight load, which this driver would then log as "serve did not
# come up" -- i.e. a silently missing data point. pkill above already confirmed OUR processes
# are gone; this additionally catches a device pinned by somebody else's job.
# ★ 2026-09-22:这个函数原来只等【进程】退干净,等满 180 s 就走。这不够 —— 显存是
#   驱动在最后一个进程退出【之后】才回收的,而崩溃退出那条路上实测要 **~12 分 40 秒**。
#   进程表早就空了,`npu_procs_held` 返回"没人占",于是 sleep 20 就起下一个 serve,撞上
#   `ValueError: Free memory on device (6.08/61.27 GiB) on startup is less than desired`。
#   2026-09-19 的 ep3 就是这么连挂七次、最后一个数据点彻底丢掉的。
#   所以进程门之后必须再加一道【显存门】:轮询 npu-smi,等 used 落回空闲基线才放行。
HBM_WAIT="${HBM_WAIT:-1800}"        # 显存回落的最长等待(实测崩溃后 ~760 s)
HBM_FREE_MB="${HBM_FREE_MB:-4096}"  # used 低于它就算空闲(A3 单卡 61.27 GiB)
PROC_WAIT="${PROC_WAIT:-300}"       # 等进程放开设备的最长时间

wait_npu_free() {
  local waited=0 rc mb
  while [ "$waited" -lt "$PROC_WAIT" ]; do
    npu_procs_held; rc=$?
    [ "$rc" = 2 ] && { sleep "$SETTLE"; return 0; }       # can't tell → fixed settle
    [ "$rc" = 1 ] && break                                # nothing holding a device
    [ "$waited" = 60 ] && say "    waiting for an NPU still held by another process ..."
    sleep 10; waited=$((waited + 10))
  done

  # 显存门。拿不到 npu-smi 读数就退回固定 settle(和以前一样,不比以前差)。
  if ! mb=$(npu_used_mb); then
    sleep 20; say "    NPU clear after ${waited}s (npu-smi 读不到,按固定 settle 放行)"
    return 0
  fi
  local hb=0
  while [ "$mb" -gt "$HBM_FREE_MB" ] && [ "$hb" -lt "$HBM_WAIT" ]; do
    [ $((hb % 60)) = 0 ] && say "    等驱动回收显存 ... ${mb} MB 仍映射着(已等 ${hb}s / 上限 ${HBM_WAIT}s)"
    sleep 20; hb=$((hb + 20))
    mb=$(npu_used_mb) || { say "    npu-smi 读不到了,放行"; return 0; }
  done
  if [ "$mb" -gt "$HBM_FREE_MB" ]; then
    # 不静默放行 —— 下一个 serve 多半会 OOM,说清楚为什么,别让人对着"没起来"猜。
    say "    ⚠ 等满 ${HBM_WAIT}s 显存仍有 ${mb} MB 映射着(阈值 ${HBM_FREE_MB} MB)。"
    say "      下一个 serve 很可能 OOM 在权重加载。是不是有别人的任务占着这台机?"
  else
    say "    NPU clear:进程 ${waited}s + 显存回收 ${hb}s(max ${mb} MB 仍映射)"
  fi
  return 0
}

stop_serve() {
  # Nothing listening and no process alive → nothing to tear down, don't burn the settle time.
  serve_up || procs_alive || return 0
  say "    stopping serve ..."
  $KILL_PREFIX pkill -if "$PROCPAT" >/dev/null 2>&1
  for _ in $(seq 1 24); do procs_alive || break; sleep 5; done        # up to 120s graceful
  if procs_alive; then
    say "    (graceful stop timed out — escalating to SIGKILL)"
    $KILL_PREFIX pkill -9 -if "$PROCPAT" >/dev/null 2>&1
    for _ in $(seq 1 12); do procs_alive || break; sleep 5; done      # 60s more
  fi
  if procs_alive; then
    say "    !! processes STILL alive after SIGKILL — most likely owned by another user."
    say "       re-run the batch with KILL_PREFIX='sudo -n', or clear them by hand:"
    pgrep -aif "$PROCPAT" 2>/dev/null | head -10 | sed 's/^/       /' | tee -a "$MASTER"
  fi
  serve_up && say "    !! WARNING: something is STILL answering on :$PORT"
  wait_npu_free
  return 0
}

wait_ready() {
  local waited=0
  while [ "$waited" -lt "$SERVE_TIMEOUT" ]; do
    serve_up && return 0
    sleep 10; waited=$((waited + 10))
  done
  return 1
}

hms() { printf '%dh%02dm%02ds' $(($1/3600)) $((($1%3600)/60)) $(($1%60)); }

# ---------------------------------------------------------------------------------------
cd "$REPO_ROOT" || exit 1
BATCH_START=$SECONDS
: > "$MASTER"
say "################################################################################"
say "### DSV4-DSpark BATCH EVAL"
say "### started    : $(date '+%Y-%m-%d %H:%M:%S')"
say "### host       : $(hostname)"
say "### repo       : $REPO_ROOT @ $(git rev-parse --short HEAD 2>/dev/null) ($(git rev-parse --abbrev-ref HEAD 2>/dev/null))"
say "### ckpt root  : $CKPT_ROOT"
say "### settings   : DATASET=$DATASET CONCURRENCY=$CONCURRENCY KEEP_WARMUP=$KEEP_WARMUP MAX_NEW=$MAX_NEW NUM_PROMPTS=${NUM_PROMPTS:-0 (full)} NUM_SPEC=$NUM_SPEC PREFETCH=$PREFETCH TOKENIZER=$TOKENIZER"
say "### meas. set  : $([ "$KEEP_WARMUP" = "1" ] && echo 'FULL 1319/500/164/257/80 (post-2026-08-13)' || echo 'OLD 1309/490/154/247/70 (pre-cutover)')"
say "### entries    : ${#ENTRIES[@]}   outdir: $OUTDIR"
say "################################################################################"
say ""

IDX=0; NDONE=0; NFAIL=0; NSKIP=0
for E in "${ENTRIES[@]}"; do
  IDX=$((IDX + 1))
  LABEL="${E%%|*}"; DIRNAME="${E#*|}"
  [ -n "$ONLY" ] && ! echo "$LABEL" | grep -qE "$ONLY" && { NSKIP=$((NSKIP+1)); continue; }

  LOG="$OUTDIR/${IDX}_${LABEL}.log"
  DPATH=""; [ -n "$DIRNAME" ] && DPATH="$CKPT_ROOT/$DIRNAME"

  say "################################################################################"
  say "### [$IDX/${#ENTRIES[@]}] $LABEL"
  say "###   time   : $(date '+%Y-%m-%d %H:%M:%S')   (batch elapsed $(hms $((SECONDS-BATCH_START))))"
  say "###   draft  : ${DPATH:-<none — AR baseline, no speculative decoding>}"
  say "###   log    : $LOG"
  say "################################################################################"

  if [ -n "$DPATH" ] && [ ! -d "$DPATH" ]; then
    say "!! MISSING draft dir — SKIPPED"; say ""
    NFAIL=$((NFAIL+1)); continue
  fi
  if [ "$SKIP_DONE" = "1" ] && [ -s "$LOG" ] && grep -q "FINAL SUMMARY" "$LOG"; then
    say ">>> already complete in $LOG — SKIPPED"; say ""
    NSKIP=$((NSKIP+1)); continue
  fi

  T0=$SECONDS
  stop_serve
  say ">>> starting serve ..."
  if [ -n "$DPATH" ]; then
    DRAFT="$DPATH" NUM_SPEC="$NUM_SPEC" PREFETCH="$PREFETCH" \
      nohup bash examples/ascend_npu_dflash/serve_dsv4_a3_singlenode_specmethod.sh \
      > "$OUTDIR/${IDX}_${LABEL}.serve.log" 2>&1 &
  else
    NUM_SPEC="$NUM_SPEC" PREFETCH="$PREFETCH" \
      nohup bash examples/ascend_npu_dflash/serve_dsv4_a3_singlenode_specmethod.sh \
      > "$OUTDIR/${IDX}_${LABEL}.serve.log" 2>&1 &
  fi

  if ! wait_ready; then
    say "!! serve did not come up within ${SERVE_TIMEOUT}s — SKIPPING this entry"
    say "   tail of its serve log:"
    tail -20 "$OUTDIR/${IDX}_${LABEL}.serve.log" | sed 's/^/   /' | tee -a "$MASTER"
    stop_serve; say ""
    NFAIL=$((NFAIL+1)); continue
  fi
  say ">>> serve READY after $(hms $((SECONDS-T0)))  — running $DATASET ..."

  KEEP_WARMUP="$KEEP_WARMUP" DATASET="$DATASET" CONCURRENCY="$CONCURRENCY" \
  PORT="$PORT" MAX_NEW="$MAX_NEW" \
    bash examples/ascend_npu_dflash/run_dspark_eval.sh 2>&1 | tee "$LOG" | tee -a "$MASTER"
  RC=${PIPESTATUS[0]}

  if grep -q "FINAL SUMMARY" "$LOG"; then
    say "### RESULT [$IDX/${#ENTRIES[@]}] $LABEL : OK   (entry took $(hms $((SECONDS-T0))))"
    NDONE=$((NDONE+1))
  else
    say "### RESULT [$IDX/${#ENTRIES[@]}] $LABEL : FAILED (rc=$RC, no FINAL SUMMARY)"
    NFAIL=$((NFAIL+1))
  fi
  say ""
  stop_serve
done

stop_serve

# ---------------------------------------------------------------------------------------
# Consolidated tables. Parsed from each per-entry log's FINAL SUMMARY block, whose columns
# are: dataset samples turns tokens time throughput accept_len accept_rate.
# ---------------------------------------------------------------------------------------
emit_table() {                      # $1 = column index in the FINAL SUMMARY row, $2 = title
  say ""
  say "===================================================================================================="
  say "$2"
  say "===================================================================================================="
  printf '%-22s %9s %9s %10s %9s %10s %9s\n' \
    "draft" "gsm8k" "math500" "humaneval" "mbpp" "mt-bench" "mean" | tee -a "$MASTER"
  local i=0
  for E in "${ENTRIES[@]}"; do
    i=$((i + 1))
    local lbl="${E%%|*}" lg="$OUTDIR/${i}_${E%%|*}.log"
    [ -s "$lg" ] || continue
    grep -q "FINAL SUMMARY" "$lg" || continue
    # `n/a` rather than an em dash on purpose: awk pads %9s by BYTES, so a multi-byte
    # glyph would shift every following column. The AR baseline legitimately reports
    # accept_len as nan (num_drafts is 0, so the ratio is 0/0) -- print that as n/a and
    # suppress its mean, instead of the bare "nan ... 0.000" row it used to emit. Its
    # throughput row is unaffected and still carries the speedup denominator.
    awk -v L="$lbl" -v C="$1" '
      function show(x) { return (x == "" || x ~ /^[Nn][Aa][Nn]$/) ? "n/a" : x }
      /FINAL SUMMARY/     { insum = 1; next }
      insum && /^-----/   { rows = 1; next }
      rows && /^={10,}/   { rows = 0 }
      rows && NF >= 8     { v[$1] = $(C); if ($(C) ~ /^[Nn][Aa][Nn]$/) bad = 1; n++ }
      END {
        m  = (v["gsm8k"] + v["math500"] + v["humaneval"] + v["mbpp"] + v["mt-bench"]) / 5
        ms = (bad || n < 5) ? "n/a" : sprintf("%.3f", m)
        printf "%-22s %9s %9s %10s %9s %10s %9s\n",
               L, show(v["gsm8k"]), show(v["math500"]), show(v["humaneval"]),
               show(v["mbpp"]), show(v["mt-bench"]), ms
      }' "$lg" | tee -a "$MASTER"
  done
}

emit_table 7 "ACCEPT LENGTH  (mean = 5-dataset macro average)"
emit_table 6 "THROUGHPUT tok/s  (AR-baseline row = the speedup denominator)"

say ""
say "################################################################################"
say "### BATCH DONE  $(date '+%Y-%m-%d %H:%M:%S')   total $(hms $((SECONDS-BATCH_START)))"
say "###   ok=$NDONE  failed=$NFAIL  skipped=$NSKIP   of ${#ENTRIES[@]}"
say "###   master log : $MASTER"
say "###   per-entry  : $OUTDIR/<n>_<label>.log   (+ .serve.log for each)"
say "### Re-run just the failures with:  ONLY='<label>|<label>' OUTDIR=$OUTDIR bash $0"
say "################################################################################"
