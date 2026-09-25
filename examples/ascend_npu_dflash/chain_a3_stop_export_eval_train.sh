#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# A3 流水线:在跑的训练到第 N 个 epoch 存完 → 停 → 导出这个 ckpt → 自动评测 → 起下一条训练。
#
# WHY
# ---
# 186 是唯一的卡,训练、导出、评测、下一条训练只能排队。每一段衔接都要人守着敲命令,而
# 这些点常常落在半夜。这个脚本把四段串起来,中间两次显存回收也替你等。
#
# 四段
#   1. 等 `<SAVE>/epoch<IDX>_end` 这个软链出现 —— trainer 在模型/优化器/调度器/训练状态【全部】
#      写完之后才建它(trainer.py:939),是「这个 epoch 存好了」最可靠的标志。等的期间训练若
#      自己死了(软链没出现、进程没了)就整条中止,不起下一条 —— 那得人来看。
#   2. 停训练(SIGKILL;ckpt 已完整,留着能续跑)。
#   3. 导出 + 评测:export_run_ckpts.py(转换 + bit-exact 校验)→ eval_blk15_drafts.sh
#      (自己起/停 serve,自己等显存回收)。这两步失败【不挡】第 4 步 —— 卡不该闲着,
#      ckpt 在盘上,之后可以手动补评。
#   4. 等显存回收,起下一条训练(launch_a3_blk15_prestored.sh)。
#
# 用法(随便开个 terminal,粘一次就不用管了):
#   cd <speculators 仓库> && nohup bash examples/ascend_npu_dflash/chain_a3_stop_export_eval_train.sh \
#       > ~/chain_a3.log 2>&1 &
#   tail -f ~/chain_a3.log
#   DRY_RUN=1 bash examples/ascend_npu_dflash/chain_a3_stop_export_eval_train.sh   # 只做检查、不等不停不起
#
# 旋钮(都有默认值)
#   RUN_TS=20260925_143507   STOP_IDX=4(= epoch4_end = 训满 5.0 epoch)   TAG=a3half
#   EVAL_DATASET=all(五个数据集;只要 gsm8k 就写 gsm8k)
#   NEXT_LR=3e-4  NEXT_BAL=1  NEXT_BAL_RATE=2e-3  NEXT_EPOCHS=10  NEXT_ANCHORS=192
#   TRAIN_ENV=dspark-dsv4-train  SERVE_ENV=dspark-dsv4-serving  CANN_ENV=/home/a00652497/920env_npu.sh
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_TS="${RUN_TS:-20260925_143507}"
RUN_DIR="${RUN_DIR:-$HOME/dsv4_run}"
STOP_IDX="${STOP_IDX:-4}"
TAG="${TAG:-a3half}"
EVAL_DATASET="${EVAL_DATASET:-all}"
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

SAVE="$RUN_DIR/ckpt_faithful_ep_$RUN_TS"
TLOG="$RUN_DIR/faithful_ep_$RUN_TS.log"
MARK="$SAVE/epoch${STOP_IDX}_end"
# ⚠ 模式里的 [t] 不能省:这个脚本自己的命令行里也有 scripts/train.py 这串字,不加 [t] 的
#   pkill / pgrep 会把自己也算进去 —— 杀掉自己,或者永远等不到「训练进程没了」。
TRAIN_PAT='scripts/[t]rain.py'
TS="$(date +%Y%m%d_%H%M%S)"
EVAL_OUT="$HOME/eval_chain_${TAG}_$TS"

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
hr()  { echo "================================================================================"; }

conda_use() {   # 在当前 shell 里切 conda 环境
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh" || return 1
  conda activate "$1" || return 1
  [ "$(basename "${CONDA_PREFIX:-}")" = "$1" ] || return 1
  # miniforge:环境自己的 libstdc++ 要赢过 /usr/lib64 那个(torch_npu 要 CXXABI_1.3.15)
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
}

npu_used_mb() {   # 各卡已用显存的最大值(MB);读不到返回非零
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

hr
echo "  A3 流水线:停在 epoch${STOP_IDX}_end → 导出 → 评测 → 起下一条训练"
echo "  训练 run   $RUN_TS   ($SAVE)"
echo "  评测       tag=$TAG  dataset=$EVAL_DATASET  env=$SERVE_ENV  → $EVAL_OUT"
echo "  下一条     LR=$NEXT_LR  BALANCE=$NEXT_BAL  RATE=$NEXT_BAL_RATE  EPOCHS=$NEXT_EPOCHS  ANCHORS=$NEXT_ANCHORS  env=$TRAIN_ENV"
hr

# ── 0. 预检 —— 半夜才发现路径错,就白等一天 ────────────────────────────────────
bad=0
[ -d "$SAVE" ] || { say "!! 找不到 $SAVE"; bad=1; }
[ -f "$TLOG" ] || { say "!! 找不到训练日志 $TLOG"; bad=1; }
[ -f "$CANN_ENV" ] || { say "!! 找不到 CANN_ENV=$CANN_ENV"; bad=1; }
[ -d "$TOKENIZER" ] || { say "!! 找不到 TOKENIZER=$TOKENIZER"; bad=1; }
for f in export_run_ckpts.py eval_blk15_drafts.sh launch_a3_blk15_prestored.sh; do
  [ -f "$SCRIPT_DIR/$f" ] || { say "!! 缺 $SCRIPT_DIR/$f"; bad=1; }
done
for e in "$TRAIN_ENV" "$SERVE_ENV"; do
  conda env list 2>/dev/null | awk '{print $1}' | grep -x "$e" >/dev/null || { say "!! conda 环境 $e 不存在"; bad=1; }
done
[ "$bad" = 0 ] || { say "预检没过,什么都没动。"; exit 2; }
if [ -L "$MARK" ]; then
  say "注意:$MARK 已经存在 —— 会立刻进入停训练那一步"
elif pgrep -f "$TRAIN_PAT" >/dev/null; then
  step=$(tail -c 200000 "$TLOG" | grep -oE 'global_step=[0-9]+' | tail -1)
  say "训练在跑(${step:-读不到步数}),等 $MARK"
else
  say "!! 训练没在跑,而 $MARK 也不存在 —— 没有可等的东西。"; exit 2
fi
say "导出计划(干跑,不写任何东西):"
( conda_use "$TRAIN_ENV" && cd "$REPO_ROOT" && \
  python examples/ascend_npu_dflash/export_run_ckpts.py --run "$RUN_TS" --tag "$TAG" 2>&1 | sed 's/^/   | /' | tail -25 )
if [ "$DRY_RUN" = 1 ]; then say "DRY_RUN=1:预检完成,到此为止。"; exit 0; fi

# ── 1. 等 epoch 存完 ──────────────────────────────────────────────────────────
last_beat=$SECONDS
until [ -L "$MARK" ]; do
  if ! pgrep -f "$TRAIN_PAT" >/dev/null; then
    sleep 60   # 刚好在存完、建软链的那几秒里退出?再看一眼
    [ -L "$MARK" ] && break
    say "!! 训练进程没了,而 $MARK 一直没出现 —— 训练中途死了。整条中止,不起下一条。"
    say "   看 $TLOG 的结尾。"
    exit 1
  fi
  if [ $((SECONDS - last_beat)) -ge 1800 ]; then
    step=$(tail -c 200000 "$TLOG" | grep -oE 'global_step=[0-9]+' | tail -1)
    say "   还在等 epoch${STOP_IDX}_end ... ${step:-?}"
    last_beat=$SECONDS
  fi
  sleep 60
done
hr; say "1. $MARK 出现 —— epoch 已存完"

# ── 2. 停训练 ────────────────────────────────────────────────────────────────
sleep 30
pkill -KILL -f "$TRAIN_PAT"
for _ in $(seq 60); do pgrep -f "$TRAIN_PAT" >/dev/null || break; sleep 5; done
say "2. 训练已停($SAVE/$STOP_IDX 完整,需要时可续跑)"

# ── 3. 导出 + 评测(失败不挡第 4 步)────────────────────────────────────────────
hr; say "3a. 导出 ckpt $STOP_IDX(转换 + bit-exact 校验)"
EXP_LOG="$EVAL_OUT.export.log"; mkdir -p "$(dirname "$EXP_LOG")"
# --fresh-sec 0:「10 分钟内动过的目录跳过」是防转到写了一半的 ckpt;这里软链已证明写完了。
( conda_use "$TRAIN_ENV" && cd "$REPO_ROOT" && \
  python examples/ascend_npu_dflash/export_run_ckpts.py --run "$RUN_TS" --only "$STOP_IDX" \
    --tag "$TAG" --fresh-sec 0 --go ) > "$EXP_LOG" 2>&1
exp_rc=$?
tail -20 "$EXP_LOG" | sed 's/^/   | /'
# 成功那行:「    ✓ ep5p0    /绝对/路径/dsv4_dspark_blk15_ep5p0_a3half_vllm-77w」
DRAFT_PATH=$(grep -E '^\s+✓ ' "$EXP_LOG" | awk '{print $3}' | tail -1)
ENTRY=$(sed -n "s/.*ENTRIES_OVERRIDE='\([^']*\)'.*/\1/p" "$EXP_LOG" | tail -1)
if [ "$exp_rc" != 0 ] || [ -z "$DRAFT_PATH" ] || [ ! -d "$DRAFT_PATH" ] || [ -z "$ENTRY" ]; then
  say "!! 导出没成功(rc=$exp_rc)—— 跳过评测,直接起下一条。全量日志:$EXP_LOG"
else
  say "3b. 评测 $ENTRY  (dataset=$EVAL_DATASET)"
  # 评测脚本按 $CKPT_ROOT/<目录名> 找草稿;导出工具可能把草稿放进 dsv4_dspark_drafts/ 子目录,
  # 所以 CKPT_ROOT 取【草稿实际所在的目录】,TOKENIZER 单独给(它的默认值是从 CKPT_ROOT 推的)。
  ( conda_use "$SERVE_ENV" && cd "$REPO_ROOT" && \
    CKPT_ROOT="$(dirname "$DRAFT_PATH")" TOKENIZER="$TOKENIZER" ENTRIES_OVERRIDE="$ENTRY" \
    NUM_SPEC=15 DATASET="$EVAL_DATASET" OUTDIR="$EVAL_OUT" SKIP_DONE=0 \
    CANN_ENV="$CANN_ENV" CONDA_ENV="$SERVE_ENV" \
    bash examples/ascend_npu_dflash/eval_blk15_drafts.sh ) > "$EVAL_OUT.driver.log" 2>&1
  say "   评测结束(rc=$?)。结果表:"
  if [ -f "$EVAL_OUT/MASTER.log" ]; then
    sed -n '/RESULT\|accept_len\|tok\/s\|throughput/p' "$EVAL_OUT/MASTER.log" | tail -30 | sed 's/^/   | /'
  else
    say "!! 没有 $EVAL_OUT/MASTER.log —— 看 $EVAL_OUT.driver.log"
  fi
fi

# ── 4. 等显存回收,起下一条训练 ─────────────────────────────────────────────────
hr; say "4. 等显存回收,然后起下一条训练"
wait_hbm_free
if ! conda_use "$TRAIN_ENV"; then say "!! 切不到 $TRAIN_ENV —— 下一条没起"; exit 1; fi
cd "$REPO_ROOT" || exit 1
EPOCHS="$NEXT_EPOCHS" MAX_ANCHORS="$NEXT_ANCHORS" LR="$NEXT_LR" BF16_EXPERTS=0 HS_COUNT_SKIP=1 \
DSPARK_MOE_BALANCE="$NEXT_BAL" DSPARK_MOE_BALANCE_RATE="$NEXT_BAL_RATE" \
  bash examples/ascend_npu_dflash/launch_a3_blk15_prestored.sh 2>&1 | tee "$EVAL_OUT.next_launch.log"
NEW_LOG=$(grep -oE '/[^ ]*/faithful_ep_[0-9_]+\.log' "$EVAL_OUT.next_launch.log" | head -1)
if [ -n "$NEW_LOG" ]; then
  say "   下一条训练的日志:$NEW_LOG —— 等 5 分钟核均衡横幅"
  sleep 300
  grep -m3 "MOE-BALANCE" "$NEW_LOG" | sed 's/^/   | /' \
    || say "   (5 分钟内还没打出 [MOE-BALANCE],模型可能还在加载;稍后 grep 一下)"
else
  say "!! 没拿到下一条训练的日志路径,看 $EVAL_OUT.next_launch.log"
fi
hr; say "流水线结束。导出 $EXP_LOG · 评测 $EVAL_OUT/ · 下一条 ${NEW_LOG:-?}"
