#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# E1 自回归上限对照(A3 · 预存 HS)。一次起一组,跑到 STOP_AT_STEP 自动停。
#
# 问题:草稿大小不变,如果块内真实 token 全部已知,贪心接受长度能高多少?
# 主读数是训练日志里的 hard_accept_len(训练块上逐位比 argmax,数到第一个错为止)。
#
#   A0  mask + 双向    = 现结构。不用新跑:A3 试点 faithful_ep_20260925_143507 前 6k 步的日志
#   A1  mask + 因果    = 因果对照        ARM=A1
#   A2  真实 token + 因果                 ARM=A2
#   A2 − A1 = 注意力模式相同时块内 token 的纯价值;A0 − A1 = 双向本身的价值。
#   A1 与 A2 必须都跑:只跑 A2 就分不清「token 的价值」和「双向 vs 因果」。
#
# 配对:配方逐项照抄试点(LR 2.8e-4、EPOCHS=10 的调度、anchor 192、种子 42),
# 于是数据顺序、锚点、噪声的随机流与试点一致,三组可以按步数逐点相减。
# ⚠ 所以这里 LR 是 2.8e-4,不是正式实验的 3e-4 —— 要和试点日志配对。
#
# 用法(在【实验分支的检出】里,训练环境已 activate):
#   ARM=A1 bash examples/ascend_npu_dflash/launch_a3_e1_ar_ceiling.sh
#   ... A1 停下后 ...
#   ARM=A2 bash examples/ascend_npu_dflash/launch_a3_e1_ar_ceiling.sh
#
# 旋钮:ARM(必填 A1|A2)  STOP_AT_STEP=6100(半数据 1 个 epoch 是 6117 步;
#       CKPT_FREQ=1 让第一次存档落在 6117,于是停在它之前 = 不写 41 GB 的 ckpt)
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

ARM="${ARM:-}"
STOP_AT_STEP="${STOP_AT_STEP:-6100}"
case "$ARM" in
  A1) BLOCK_INPUT=mask ;;
  A2) BLOCK_INPUT=true ;;
  *)  say "!! 要指定 ARM=A1(mask + 因果)或 ARM=A2(真实 token + 因果)"; exit 2 ;;
esac

# ── 让 python 用【这份检出】的 speculators ───────────────────────────────────
# 训练环境里的 speculators 是 editable 装的,指向当初跑安装脚本的那份检出(基线分支)。
# 从另一个工作树起训练,不改 PYTHONPATH 的话 import 到的是基线的代码 ——
# DSPARK_BLOCK_INPUT 会被【静默忽略】,A2 实际跑成 A1。所以先顶到最前面,再核一遍。
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
got=$(cd "$HOME" && TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -c \
  "import speculators, os; print(os.path.realpath(speculators.__file__))" 2>/dev/null)
case "$got" in
  "$(realpath "$REPO_ROOT")"/src/*) say "speculators 来自本检出:$got" ;;
  *) say "!! import 到的 speculators 不在本检出里:${got:-<导入失败>}"
     say "   期望在 $(realpath "$REPO_ROOT")/src/ 下。先 conda activate 训练环境。"; exit 2 ;;
esac
if ! grep -q 'DSPARK_BLOCK_INPUT' "$REPO_ROOT/src/speculators/models/dsv4_dspark/core.py"; then
  say "!! 本检出的 core.py 里没有 DSPARK_BLOCK_INPUT —— 不在实验分支上?"; exit 2
fi

# ── 配方:与试点一致,只改注意力与块输入 ─────────────────────────────────────
export EPOCHS=10 MAX_ANCHORS=192 LR=2.8e-4 BF16_EXPERTS=0 DSPARK_MOE_BALANCE=0
export NONCAUSAL=0 DSPARK_BLOCK_INPUT="$BLOCK_INPUT"
export CKPT_FREQ="${CKPT_FREQ:-1}" HS_COUNT_SKIP="${HS_COUNT_SKIP:-1}"

say "E1 $ARM:块输入=$DSPARK_BLOCK_INPUT  块内注意力=因果(NONCAUSAL=0)  停在 global_step ≥ $STOP_AT_STEP"
OUT=$(mktemp)
bash "$SCRIPT_DIR/launch_a3_blk15_prestored.sh" 2>&1 | tee "$OUT"
LOG=$(grep -oE '/[^ ]*/faithful_ep_[0-9_]+\.log' "$OUT" | tail -1)
rm -f "$OUT"
[ -n "$LOG" ] || { say "!! 没拿到训练日志路径,训练可能没起来(看上面的输出)"; exit 1; }

# ── 到步数自动停 ─────────────────────────────────────────────────────────────
MARK="${LOG%.log}.e1.txt"
{
  echo "E1 arm=$ARM DSPARK_BLOCK_INPUT=$DSPARK_BLOCK_INPUT NONCAUSAL=0 STOP_AT_STEP=$STOP_AT_STEP"
  echo "repo=$(realpath "$REPO_ROOT") sha=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null) branch=$(git -C "$REPO_ROOT" branch --show-current 2>/dev/null)"
  echo "pair_with=faithful_ep_20260925_143507 (A0,前 ${STOP_AT_STEP} 步)"
} > "$MARK"
nohup bash -c '
  LOG="$1"; STOP="$2"; MARK="$3"; sleep 120
  while pgrep -f "scripts/[t]rain.py" >/dev/null; do
    s=$(tail -c 200000 "$LOG" | grep -aoE "global_step=[0-9]+" | tail -1 | cut -d= -f2)
    if [ -n "$s" ] && [ "$s" -ge "$STOP" ]; then
      echo "stopped at global_step=$s  $(date "+%F %T")" >> "$MARK"
      pkill -KILL -f "scripts/[t]rain.py"; exit 0
    fi
    sleep 60
  done
  echo "training exited on its own before step $STOP  $(date "+%F %T")" >> "$MARK"
' _ "$LOG" "$STOP_AT_STEP" "$MARK" > /dev/null 2>&1 &
say "自动停止已挂上(PID $!)。记录:$MARK"
say "看进度:tail -f $LOG"
