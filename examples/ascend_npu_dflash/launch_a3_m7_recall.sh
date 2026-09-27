#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# M7:重排上限(A3 · 预存 HS)。在一个训好的草稿上测「正确 token 在不在前 K 名」,逐位置。
#
# 问题:草稿的 argmax 没选中时,正确 token 是否仍在它的前 2/4/8/16 名里?
# 读数(训练日志里,与 hard_accept_len 同一行):
#   position_{k}_recall{K}   第 k 位的 recall@K(对照 position_{k}_acc = recall@1)
#   oracle_accept_len_{K}    如果每一位都能从前 K 名里选对,贪心接受长度能到多少
# analyze_train_run.py 会汇总成 SELECTION HEADROOM 一段。
#
# 做法:FROM_PRETRAINED 从 ckpt 加载整个草稿(新优化器、新调度),LR=0 ⟹ 参数一步也不动;
#   TRAIN_PY 换成 recall_headroom_probe.py(包一层 compute_metrics,不改训练代码)。
#   跑到 STOP_AT_STEP 自动停。存档频率设成 1 个 epoch(第一次存档在 6117 步),所以不写 ckpt;
#   SAVE_PATH 是新目录,不碰被测 ckpt 所在的 run。
#
# 用法(在【实验分支的检出】里,训练环境已 activate,A3 空闲):
#   bash examples/ascend_npu_dflash/launch_a3_m7_recall.sh
#   CKPT=<某个 ckpt 目录> bash examples/ascend_npu_dflash/launch_a3_m7_recall.sh
#
# 旋钮:CKPT(默认试点 ep5 = ckpt_faithful_ep_20260925_143507/4)  STOP_AT_STEP=150  RECALL_KS=2,4,8,16
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

CKPT="${CKPT:-$HOME/dsv4_run/ckpt_faithful_ep_20260925_143507/4}"
STOP_AT_STEP="${STOP_AT_STEP:-150}"
[ -f "$CKPT/config.json" ] || { say "!! $CKPT 里没有 config.json —— 不是一个 ckpt 目录?"; exit 2; }
[ -f "$SCRIPT_DIR/recall_headroom_probe.py" ] || { say "!! 缺 recall_headroom_probe.py"; exit 2; }

# 与 E1 启动脚本同理:让 python 用本检出的 speculators,并核对。
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# 与 launch_a3_blk15_prestored.sh 相同的运行时环境,但要在 import 检查【之前】就位:
# miniforge 环境自带的 libstdc++ 必须排在系统那份前面(torch_npu 要 CXXABI_1.3.15),
# 只 conda activate 不够 —— 交互 shell 里直接跑本脚本时,import 会因此失败。
CANN_ENV="${CANN_ENV:-/home/a00652497/920env_npu.sh}"
# shellcheck source=/dev/null
[ -f "$CANN_ENV" ] && source "$CANN_ENV"
[ -n "${CONDA_PREFIX:-}" ] && export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
_imp_err=$(mktemp)
got=$(cd "$HOME" && TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -c \
  "import speculators, os; print(os.path.realpath(speculators.__file__))" 2>"$_imp_err")
case "$got" in
  "$(realpath "$REPO_ROOT")"/src/*) say "speculators 来自本检出:$got" ;;
  *) say "!! import 到的 speculators 不在本检出里:${got:-<导入失败>}"
     say "   python=$(command -v python)  CONDA_PREFIX=${CONDA_PREFIX:-<未激活>}"
     [ -s "$_imp_err" ] && { say "   报错末尾:"; tail -8 "$_imp_err" | sed 's/^/     /'; }
     rm -f "$_imp_err"; exit 2 ;;
esac
rm -f "$_imp_err"

unset DSPARK_BLOCK_INPUT
export FROM_PRETRAINED="$CKPT" LR=0 EPOCHS=10 MAX_ANCHORS=192 BF16_EXPERTS=0 DSPARK_MOE_BALANCE=0
export INIT_LAYER=0 INIT_MOE_NO_ROUTER=0 CKPT_FREQ=1 HS_COUNT_SKIP=1
export TRAIN_PY="$SCRIPT_DIR/recall_headroom_probe.py"
export RECALL_KS="${RECALL_KS:-2,4,8,16}"

say "M7:被测 ckpt = $CKPT   LR=0   停在 global_step ≥ $STOP_AT_STEP   RECALL_KS=$RECALL_KS"
OUT=$(mktemp)
bash "$SCRIPT_DIR/launch_a3_blk15_prestored.sh" 2>&1 | tee "$OUT"
LOG=$(grep -oE '/[^ ]*/faithful_ep_[0-9_]+\.log' "$OUT" | tail -1)
rm -f "$OUT"
[ -n "$LOG" ] || { say "!! 没拿到训练日志路径,训练可能没起来(看上面的输出)"; exit 1; }

MARK="${LOG%.log}.m7.txt"
echo "M7 recall headroom  ckpt=$CKPT  stop_at=$STOP_AT_STEP  repo=$(realpath "$REPO_ROOT") sha=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null)" > "$MARK"
nohup bash -c '
  LOG="$1"; STOP="$2"; MARK="$3"; sleep 120
  while pgrep -f "scripts/[t]rain.py|recall_headroom_[p]robe.py" >/dev/null; do
    s=$(tail -c 200000 "$LOG" | grep -aoE "global_step=[0-9]+" | tail -1 | cut -d= -f2)
    if [ -n "$s" ] && [ "$s" -ge "$STOP" ]; then
      echo "stopped at global_step=$s  $(date "+%F %T")" >> "$MARK"
      pkill -KILL -f "recall_headroom_[p]robe.py"; exit 0
    fi
    sleep 30
  done
  echo "training exited on its own before step $STOP  $(date "+%F %T")" >> "$MARK"
' _ "$LOG" "$STOP_AT_STEP" "$MARK" > /dev/null 2>&1 &
say "自动停止已挂上(PID $!)。记录:$MARK"
say "停下后汇总:python -u examples/ascend_npu_dflash/analyze_train_run.py $LOG --skip 5 2>&1 | tee ~/analyze_m7.log"
