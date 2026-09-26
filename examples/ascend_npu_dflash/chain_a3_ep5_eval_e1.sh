#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# A3 流水线(实验分支版):试点到 ep5 → 停 → 导出 ep1–ep5 五个 ckpt → 评测 → 依次跑 E1 探针。
#
# 与基线分支上的 chain_a3_stop_export_eval_train.sh 的区别:
#   * 导出并评测 5 个 epoch 末 ckpt(不只 ep5),ep5 先测;
#   * 评测之后不起均衡新基线,而是跑 E1 的两组探针(ARM=A1 mask+因果、ARM=A2 真实 token+因果),
#     各约 6.1k 步、自己停;两组都跑完才算完;
#   * 可选:探针之后从【基线检出】起均衡新基线(THEN_BASELINE=1 BASELINE_REPO=<路径>)。
#
# ⚠ 必须从【实验分支的检出】运行(E1 探针要用本检出的代码;脚本会核对)。
#   不要在正跑着任务的那份检出里切分支 —— 另开一份:
#     git fetch origin && git worktree add ../speculators-exp exp/dsv4-dspark-block16
#
# 用法:
#   cd ../speculators-exp
#   DRY_RUN=1 bash examples/ascend_npu_dflash/chain_a3_ep5_eval_e1.sh    # 只做检查
#   nohup bash examples/ascend_npu_dflash/chain_a3_ep5_eval_e1.sh > ~/chain_e1.log 2>&1 &
#   tail -f ~/chain_e1.log
#
# 旋钮(都有默认值)
#   RUN_TS=20260925_143507  STOP_IDX=4(epoch4_end = 训满 5.0 epoch)  TAG=a3half
#   ONLY=0,1,2,3,4(导出哪些整数 ckpt 目录;0..4 = ep1..ep5)
#   EVAL_DATASET_EP5=all  EVAL_DATASET_REST=all(ep1–ep4;想省约两小时就设 gsm8k)
#   EXTRA_ENTRIES / EXTRA_CKPT_ROOT / EXTRA_DATASET=gsm8k(可选:顺带评测已在本机上的其它草稿,
#       格式同 ENTRIES_OVERRIDE,例如 'ep1p0-blk15-nobal|<目录名> ...')
#   PROBES="A1 A2"  STOP_AT_STEP=6100
#   START_AT=probes  跳过等待/停训练/导出/评测,直接从探针开始(前面几步已经做完、只需重跑探针时用)
#   THEN_BASELINE=0  BASELINE_REPO=  NEXT_LR=3e-4 NEXT_BAL=1 NEXT_BAL_RATE=2e-3 NEXT_EPOCHS=10 NEXT_ANCHORS=192
#   TRAIN_ENV=dspark-dsv4-train  SERVE_ENV=dspark-dsv4-serving  CANN_ENV=/home/a00652497/920env_npu.sh
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_TS="${RUN_TS:-20260925_143507}"
RUN_DIR="${RUN_DIR:-$HOME/dsv4_run}"
STOP_IDX="${STOP_IDX:-4}"
TAG="${TAG:-a3half}"
ONLY="${ONLY:-0,1,2,3,4}"
EVAL_DATASET_EP5="${EVAL_DATASET_EP5:-all}"
EVAL_DATASET_REST="${EVAL_DATASET_REST:-all}"
EXTRA_ENTRIES="${EXTRA_ENTRIES:-}"
EXTRA_CKPT_ROOT="${EXTRA_CKPT_ROOT:-/home/canada_group_folder/ckpt}"
EXTRA_DATASET="${EXTRA_DATASET:-gsm8k}"
PROBES="${PROBES:-A1 A2}"
STOP_AT_STEP="${STOP_AT_STEP:-6100}"
THEN_BASELINE="${THEN_BASELINE:-0}"
BASELINE_REPO="${BASELINE_REPO:-}"
NEXT_LR="${NEXT_LR:-3e-4}"
NEXT_BAL="${NEXT_BAL:-1}"
NEXT_BAL_RATE="${NEXT_BAL_RATE:-2e-3}"
NEXT_EPOCHS="${NEXT_EPOCHS:-10}"
NEXT_ANCHORS="${NEXT_ANCHORS:-192}"
TRAIN_ENV="${TRAIN_ENV:-dspark-dsv4-train}"
SERVE_ENV="${SERVE_ENV:-dspark-dsv4-serving}"
CANN_ENV="${CANN_ENV:-/home/a00652497/920env_npu.sh}"
TOKENIZER="${TOKENIZER:-/home/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16}"
HBM_FREE_MB="${HBM_FREE_MB:-4096}"
HBM_WAIT="${HBM_WAIT:-1800}"
DRY_RUN="${DRY_RUN:-0}"
START_AT="${START_AT:-wait}"

SAVE="$RUN_DIR/ckpt_faithful_ep_$RUN_TS"
TLOG="$RUN_DIR/faithful_ep_$RUN_TS.log"
MARK="$SAVE/epoch${STOP_IDX}_end"
# ⚠ [t]:本脚本自己的命令行里没有 scripts/train.py,但子 shell 可能有;[t] 让模式不匹配自己。
TRAIN_PAT='scripts/[t]rain.py'
TS="$(date +%Y%m%d_%H%M%S)"
OUT="$HOME/chain_e1_${TAG}_$TS"
mkdir -p "$OUT"

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
hr()  { echo "================================================================================"; }

conda_use() {   # 在当前 shell 里切 conda 环境
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh" || return 1
  conda activate "$1" || return 1
  [ "$(basename "${CONDA_PREFIX:-}")" = "$1" ] || return 1
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
}

npu_used_mb() {
  command -v npu-smi >/dev/null 2>&1 || return 1
  npu-smi info 2>/dev/null | grep -oE '[0-9]+ +/ +[0-9]+' \
    | awk -F'/' '{ u=$1+0; t=$2+0; if (t > 1000 && u > m) m = u } END { if (m == "") exit 1; print m+0 }'
}

wait_hbm_free() {
  local waited=0 mb
  while pgrep -f "$TRAIN_PAT" >/dev/null || pgrep -if 'vllm|EngineCore' >/dev/null; do
    sleep 10; waited=$((waited + 10))
    [ "$waited" -ge 600 ] && { say "⚠ 等进程退出等了 10 分钟还没退完,继续等显存"; break; }
  done
  waited=0
  while mb=$(npu_used_mb) && [ "$mb" -gt "$HBM_FREE_MB" ] && [ "$waited" -lt "$HBM_WAIT" ]; do
    [ $((waited % 120)) = 0 ] && say "   等驱动回收显存 ... ${mb} MB 仍映射(已等 ${waited}s)"
    sleep 20; waited=$((waited + 20))
  done
  mb=$(npu_used_mb) && say "   显存:${mb} MB(阈值 ${HBM_FREE_MB} MB,等了 ${waited}s)"
}

run_eval() {   # $1=标签 $2=CKPT_ROOT $3=ENTRIES $4=DATASET
  local name="$1" root="$2" entries="$3" ds="$4" dir="$OUT/eval_$1"
  say "   评测 [$name] dataset=$ds :$entries"
  ( conda_use "$SERVE_ENV" && cd "$REPO_ROOT" && \
    CKPT_ROOT="$root" TOKENIZER="$TOKENIZER" ENTRIES_OVERRIDE="$entries" \
    NUM_SPEC=15 DATASET="$ds" OUTDIR="$dir" SKIP_DONE=0 \
    CANN_ENV="$CANN_ENV" CONDA_ENV="$SERVE_ENV" \
    bash examples/ascend_npu_dflash/eval_blk15_drafts.sh ) > "$dir.driver.log" 2>&1
  say "   [$name] 结束(rc=$?)。全量:$dir/MASTER.log"
  [ -f "$dir/MASTER.log" ] && sed -n '/RESULT\|accept_len\|tok\/s\|throughput/p' "$dir/MASTER.log" | tail -20 | sed 's/^/   | /'
}

hr
echo "  A3 流水线(实验分支):ep5 → 停 → 导出 ep1–ep5 → 评测 → 探针 [$PROBES]"
echo "  训练 run   $RUN_TS   ($SAVE)"
echo "  导出       ONLY=$ONLY  tag=$TAG"
echo "  评测       ep5: $EVAL_DATASET_EP5   ep1–ep4: $EVAL_DATASET_REST   extra: ${EXTRA_ENTRIES:-无}"
echo "  探针       $PROBES  各停在 global_step ≥ $STOP_AT_STEP"
echo "  之后       THEN_BASELINE=$THEN_BASELINE ${BASELINE_REPO:+(从 $BASELINE_REPO)}"
echo "  输出       $OUT"
hr

# ── 0. 预检 ─────────────────────────────────────────────────────────────────
bad=0
[ -d "$SAVE" ] || { say "!! 找不到 $SAVE"; bad=1; }
[ -f "$TLOG" ] || { say "!! 找不到训练日志 $TLOG"; bad=1; }
[ -f "$CANN_ENV" ] || { say "!! 找不到 CANN_ENV=$CANN_ENV"; bad=1; }
[ -d "$TOKENIZER" ] || { say "!! 找不到 TOKENIZER=$TOKENIZER"; bad=1; }
for f in export_run_ckpts.py eval_blk15_drafts.sh launch_a3_e1_ar_ceiling.sh launch_a3_blk15_prestored.sh; do
  [ -f "$SCRIPT_DIR/$f" ] || { say "!! 缺 $SCRIPT_DIR/$f"; bad=1; }
done
grep -q 'DSPARK_BLOCK_INPUT' "$REPO_ROOT/src/speculators/models/dsv4_dspark/core.py" \
  || { say "!! 本检出没有 DSPARK_BLOCK_INPUT —— 不是实验分支的检出?"; bad=1; }
say "本检出:$REPO_ROOT  分支 $(git -C "$REPO_ROOT" branch --show-current 2>/dev/null)  $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null)"
for e in "$TRAIN_ENV" "$SERVE_ENV"; do
  conda env list 2>/dev/null | awk '{print $1}' | grep -x "$e" >/dev/null || { say "!! conda 环境 $e 不存在"; bad=1; }
done
if [ "$THEN_BASELINE" = 1 ]; then
  [ -f "$BASELINE_REPO/examples/ascend_npu_dflash/launch_a3_blk15_prestored.sh" ] \
    || { say "!! THEN_BASELINE=1 需要 BASELINE_REPO=<基线分支的检出>(现在是 '${BASELINE_REPO}')"; bad=1; }
fi
for p in $PROBES; do case "$p" in A1|A2) ;; *) say "!! PROBES 只认 A1 A2,收到 '$p'"; bad=1 ;; esac; done
case "$START_AT" in wait|probes) ;; *) say "!! START_AT 只认 wait 或 probes,收到 '$START_AT'"; bad=1 ;; esac
[ "$bad" = 0 ] || { say "预检没过,什么都没动。"; exit 2; }
if [ "$START_AT" = probes ]; then
  if pgrep -f "$TRAIN_PAT" >/dev/null; then
    say "!! START_AT=probes,但还有训练进程在跑 —— 先确认它该不该停,再起探针。"; exit 2
  fi
  say "START_AT=probes:跳过 1–3 步(等待、停训练、导出、评测),直接起探针 [$PROBES]"
  if [ "$DRY_RUN" = 1 ]; then say "DRY_RUN=1:预检完成,到此为止。"; exit 0; fi
else

if [ -L "$MARK" ]; then
  say "注意:$MARK 已经存在 —— 会立刻进入停训练那一步"
elif pgrep -f "$TRAIN_PAT" >/dev/null; then
  step=$(tail -c 200000 "$TLOG" | grep -aoE 'global_step=[0-9]+' | tail -1)
  say "训练在跑(${step:-读不到步数}),等 $MARK"
else
  say "!! 训练没在跑,而 $MARK 也不存在 —— 没有可等的东西。"; exit 2
fi
say "导出计划(干跑,不写任何东西):"
( conda_use "$TRAIN_ENV" && cd "$REPO_ROOT" && \
  python examples/ascend_npu_dflash/export_run_ckpts.py --run "$RUN_TS" --only "$ONLY" --tag "$TAG" --fresh-sec 0 2>&1 \
  | sed 's/^/   | /' | tail -30 )
say "导出目标所在盘的剩余空间(5 份草稿约 200 GB,按每份约 40 GB 估):"
for d in /home/canada_group_folder/ckpt /mnt/nfs/canada_group_folder/ckpt; do
  [ -d "$d" ] && df -h "$d" | tail -1 | sed "s|^|   $d  |"
done
if [ "$DRY_RUN" = 1 ]; then say "DRY_RUN=1:预检完成,到此为止。"; exit 0; fi

# ── 1. 等 ep5 存完 ───────────────────────────────────────────────────────────
last_beat=$SECONDS
until [ -L "$MARK" ]; do
  if ! pgrep -f "$TRAIN_PAT" >/dev/null; then
    sleep 60
    [ -L "$MARK" ] && break
    say "!! 训练进程没了,而 $MARK 一直没出现 —— 训练中途死了。整条中止。看 $TLOG 的结尾。"
    exit 1
  fi
  if [ $((SECONDS - last_beat)) -ge 1800 ]; then
    step=$(tail -c 200000 "$TLOG" | grep -aoE 'global_step=[0-9]+' | tail -1)
    say "   还在等 epoch${STOP_IDX}_end ... ${step:-?}"
    last_beat=$SECONDS
  fi
  sleep 60
done
hr; say "1. $MARK 出现 —— ep5 已存完"

# ── 2. 停训练 ────────────────────────────────────────────────────────────────
sleep 30
pkill -KILL -f "$TRAIN_PAT"
for _ in $(seq 60); do pgrep -f "$TRAIN_PAT" >/dev/null || break; sleep 5; done
say "2. 训练已停($SAVE/$STOP_IDX 完整,需要时可续跑)"

# ── 3. 导出 + 评测(失败不挡探针)──────────────────────────────────────────────
hr; say "3a. 导出 ckpt $ONLY(转换 + bit-exact 校验)"
EXP_LOG="$OUT/export.log"
( conda_use "$TRAIN_ENV" && cd "$REPO_ROOT" && \
  python examples/ascend_npu_dflash/export_run_ckpts.py --run "$RUN_TS" --only "$ONLY" \
    --tag "$TAG" --fresh-sec 0 --go ) > "$EXP_LOG" 2>&1
exp_rc=$?
tail -12 "$EXP_LOG" | sed 's/^/   | /'
ENTRIES=$(sed -n "s/.*ENTRIES_OVERRIDE='\([^']*\)'.*/\1/p" "$EXP_LOG" | tail -1)
FIRST_DIR=$(grep -E '^\s+✓ ' "$EXP_LOG" | awk '{print $3}' | head -1)
if [ -z "$ENTRIES" ] || [ -z "$FIRST_DIR" ] || [ ! -d "$FIRST_DIR" ]; then
  say "!! 导出没有产出任何可评测的草稿(rc=$exp_rc)—— 跳过评测。全量:$EXP_LOG"
else
  [ "$exp_rc" = 0 ] || say "⚠ 导出有失败项(rc=$exp_rc),只评测成功的那些"
  DRAFT_ROOT="$(dirname "$FIRST_DIR")"
  EP5=""; REST=""
  for e in $ENTRIES; do
    case "${e%%|*}" in ep5p0*) EP5="$e" ;; *) REST="$e $REST" ;; esac   # REST 倒序:ep4 → ep1
  done
  hr; say "3b. 评测(草稿根 $DRAFT_ROOT)"
  [ -n "$EP5" ] && run_eval ep5 "$DRAFT_ROOT" "$EP5" "$EVAL_DATASET_EP5"
  [ -n "${REST// /}" ] && run_eval ep1to4 "$DRAFT_ROOT" "${REST% }" "$EVAL_DATASET_REST"
fi
if [ -n "$EXTRA_ENTRIES" ]; then
  run_eval extra "$EXTRA_CKPT_ROOT" "$EXTRA_ENTRIES" "$EXTRA_DATASET"
fi

fi   # START_AT != probes

# ── 4. 探针 ─────────────────────────────────────────────────────────────────
for ARM in $PROBES; do
  hr; say "4. 探针 $ARM:等显存回收后起"
  wait_hbm_free
  if ! conda_use "$TRAIN_ENV"; then say "!! 切不到 $TRAIN_ENV —— 探针没起,整条中止"; exit 1; fi
  PLOG="$OUT/probe_$ARM.launch.log"
  ( cd "$REPO_ROOT" && ARM="$ARM" STOP_AT_STEP="$STOP_AT_STEP" \
    bash examples/ascend_npu_dflash/launch_a3_e1_ar_ceiling.sh ) 2>&1 | tee "$PLOG" | sed 's/^/   | /' | tail -25
  NEW_LOG=$(grep -oE '/[^ ]*/faithful_ep_[0-9_]+\.log' "$PLOG" | tail -1)
  if [ -z "$NEW_LOG" ]; then say "!! 探针 $ARM 没起来,看 $PLOG。整条中止。"; exit 1; fi
  sleep 600
  if ! pgrep -f "$TRAIN_PAT" >/dev/null; then
    say "!! 探针 $ARM 10 分钟内就退出了 —— 看 $NEW_LOG 的结尾。整条中止。"; exit 1
  fi
  grep -m1 '\[E1\]' "$NEW_LOG" | sed 's/^/   | /'
  say "   探针 $ARM 在跑:$NEW_LOG"
  last_beat=$SECONDS
  while pgrep -f "$TRAIN_PAT" >/dev/null; do
    if [ $((SECONDS - last_beat)) -ge 1800 ]; then
      step=$(tail -c 200000 "$NEW_LOG" | grep -aoE 'global_step=[0-9]+' | tail -1)
      say "   探针 $ARM ... ${step:-?} / $STOP_AT_STEP"
      last_beat=$SECONDS
    fi
    sleep 120
  done
  say "   探针 $ARM 结束:$(tail -1 "${NEW_LOG%.log}.e1.txt" 2>/dev/null)"
  echo "$ARM $NEW_LOG" >> "$OUT/probes.txt"
done

# ── 5. 可选:均衡新基线(从基线检出起)───────────────────────────────────────────
if [ "$THEN_BASELINE" = 1 ]; then
  hr; say "5. 起均衡新基线(从 $BASELINE_REPO)"
  wait_hbm_free
  if conda_use "$TRAIN_ENV"; then
    ( cd "$BASELINE_REPO" && \
      EPOCHS="$NEXT_EPOCHS" MAX_ANCHORS="$NEXT_ANCHORS" LR="$NEXT_LR" BF16_EXPERTS=0 HS_COUNT_SKIP=1 \
      DSPARK_MOE_BALANCE="$NEXT_BAL" DSPARK_MOE_BALANCE_RATE="$NEXT_BAL_RATE" \
      bash examples/ascend_npu_dflash/launch_a3_blk15_prestored.sh ) 2>&1 | tee "$OUT/baseline.launch.log" | tail -8
  else
    say "!! 切不到 $TRAIN_ENV —— 新基线没起"
  fi
fi

hr
say "流水线结束。导出 $EXP_LOG · 评测 $OUT/eval_*/MASTER.log · 探针 $OUT/probes.txt"
A0="$TLOG"; A1=$(awk '$1=="A1"{print $2}' "$OUT/probes.txt" 2>/dev/null); A2=$(awk '$1=="A2"{print $2}' "$OUT/probes.txt" 2>/dev/null)
if [ -n "$A1" ] && [ -n "$A2" ]; then
  say "比较三组(同一段步数):"
  echo "   python -u examples/ascend_npu_dflash/analyze_train_run.py $A2 --baseline $A1 $A0 --baseline-label A1 A0 --label A2 --skip 20 --max-step $STOP_AT_STEP --out ~/analysis_e1 2>&1 | tee ~/analyze_e1.log"
fi
