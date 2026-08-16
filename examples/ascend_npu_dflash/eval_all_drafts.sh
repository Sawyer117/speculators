#!/usr/bin/env bash
# BATCH eval of every DSV4-DSpark draft we keep, into ONE master log.
#
# For each entry: stop whatever serve is running -> start serve_dsv4_a3_singlenode.sh with
# that DRAFT -> wait for ready -> run the full 5-dataset accept_len benchmark -> stop serve.
# Every line (banner + the wrapper's own output) is appended to a single master log so the
# draft<->result correspondence can never be lost, and a compact table is printed at the end.
#
#   nohup bash examples/ascend_npu_dflash/eval_all_drafts.sh > ~/eval_batch_driver.log 2>&1 &
#   tail -f ~/eval_batch_*/MASTER.log
#
# ⚠ MEASUREMENT SET. This runs KEEP_WARMUP=1 (the post-2026-08-13 default): the 10 warmup
#   prompts are warmed, the prefix cache is flushed a SECOND time, and then all N are measured
#   -- full 1319/500/164/257/80. Every ledger row written before 2026-08-13 is on the OLD
#   1309/490/154/247/70 set. That is fine and intended here: this batch re-measures BOTH A/B
#   arms and the released bar under the SAME new set, so every comparison inside this run is
#   self-consistent. Do NOT mix a row from this batch with a pre-cutover row.
#
# Env knobs (all optional):
#   PORT=7000  CONCURRENCY=48  DATASET=all  KEEP_WARMUP=1  MAX_NEW=2048
#   ONLY=<egrep pattern>   only run entries whose label matches (e.g. ONLY='ep5p0|released')
#   SKIP_DONE=1            skip entries whose per-draft log already ended with FINAL SUMMARY
#   SERVE_TIMEOUT=1800     seconds to wait for a serve to come up before giving up on it
#   SETTLE=45              seconds to let NPU memory drain between entries
#   OUTDIR=~/eval_batch_<TS>
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PORT="${PORT:-7000}"
CONCURRENCY="${CONCURRENCY:-48}"
DATASET="${DATASET:-all}"
KEEP_WARMUP="${KEEP_WARMUP:-1}"
MAX_NEW="${MAX_NEW:-2048}"
SERVE_TIMEOUT="${SERVE_TIMEOUT:-1800}"
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
OUTDIR="${OUTDIR:-$HOME/eval_batch_$TS}"
MASTER="$OUTDIR/MASTER.log"
mkdir -p "$OUTDIR"

# Box ckpt root — same auto-detect as run_dspark_eval.sh (A3-176 / A2 / A3-nfs).
if [ -z "${CKPT_ROOT:-}" ]; then
  for _d in /home/canada_group_folder/ckpt /share/canada_group_folder/ckpt /mnt/nfs/canada_group_folder/ckpt; do
    [ -d "$_d" ] && CKPT_ROOT="$_d" && break
  done
fi
CKPT_ROOT="${CKPT_ROOT:-/home/canada_group_folder/ckpt}"

export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"

# ---------------------------------------------------------------------------------------
# The work list: "label|dirname". An EMPTY dirname = AR baseline (serve started WITHOUT
# DRAFT, so no drafting happens and the tok/s columns are the speedup denominator).
# Ordered by value, not by epoch: the released bar and the converged endpoints run FIRST,
# so a batch that dies at hour 3 still delivered everything the headline needs.
#
# AR-baseline runs LAST, deliberately. The worklog's own note is that the denominator is
# DRAFT-INDEPENDENT -- measured once, reused for every row -- and a usable measurement
# already exists (460.7 / 612.3 / 320.9 / 669.0 / 446.8 tok/s, on the pre-cutover sample
# set). So it is the one entry whose absence blocks nothing, and it has already cost this
# batch one failed run: its serve died mid-stream with `Response ended prematurely` seven
# minutes in. Putting it at the tail keeps that failure from eating the front of a 9-hour
# unattended sweep. Drop it entirely with ONLY='released|ep[0-9]'.
# ---------------------------------------------------------------------------------------
ENTRIES=(
  "released|released_draft_bf16_standalone"
  "ep5p0-lossreduce|dsv4_dspark_ep5p0_lossreduce_vllm-77w"
  "ep4p5-ropefix|dsv4_dspark_ep4p5_ropefix_vllm-77w"
  "ep5p0-ropefix|dsv4_dspark_ep5p0_ropefix_vllm-77w"
  "ep4p5-lossreduce|dsv4_dspark_ep4p5_lossreduce_vllm-77w"
  "ep4p0-ropefix|dsv4_dspark_ep4p0_ropefix_vllm-77w"
  "ep3p5-ropefix|dsv4_dspark_ep3p5_ropefix_vllm-77w"
  "ep3p0-ropefix|dsv4_dspark_ep3p0_ropefix_vllm-77w"
  "ep3p0-lossreduce|dsv4_dspark_ep3p0_lossreduce_vllm-77w"
  "ep2p5-ropefix|dsv4_dspark_ep2p5_ropefix_vllm-77w"
  "ep2p0-ropefix|dsv4_dspark_ep2p0_ropefix_vllm-77w"
  "ep2p0-lossreduce|dsv4_dspark_ep2p0_lossreduce_vllm-77w"
  "ep1p5-ropefix|dsv4_dspark_ep1p5_ropefix_vllm-77w"
  "ep1p0-ropefix|dsv4_dspark_ep1p0_ropefix_vllm-77w"
  "ep1p0-lossreduce|dsv4_dspark_ep1p0_lossreduce_vllm-77w"
  "ep0p5-ropefix|dsv4_dspark_ep0p5_ropefix_vllm-77w"
  "ep0p5-lossreduce|dsv4_dspark_ep0p5_lossreduce_vllm-77w"
  "AR-baseline|"
)

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
wait_npu_free() {
  local waited=0 rc mb
  while [ "$waited" -lt 180 ]; do
    npu_procs_held; rc=$?
    [ "$rc" = 2 ] && { sleep "$SETTLE"; return 0; }       # can't tell → fixed settle
    [ "$rc" = 1 ] && break                                # nothing holding a device
    [ "$waited" = 60 ] && say "    waiting for an NPU still held by another process ..."
    sleep 10; waited=$((waited + 10))
  done
  sleep 20
  if mb=$(npu_used_mb); then say "    NPU clear after ${waited}s (max ${mb} MB still mapped)"
  else                       say "    NPU clear after ${waited}s"; fi
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
say "### settings   : DATASET=$DATASET CONCURRENCY=$CONCURRENCY KEEP_WARMUP=$KEEP_WARMUP MAX_NEW=$MAX_NEW"
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
    DRAFT="$DPATH" nohup bash examples/ascend_npu_dflash/serve_dsv4_a3_singlenode.sh \
      > "$OUTDIR/${IDX}_${LABEL}.serve.log" 2>&1 &
  else
    nohup bash examples/ascend_npu_dflash/serve_dsv4_a3_singlenode.sh \
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
