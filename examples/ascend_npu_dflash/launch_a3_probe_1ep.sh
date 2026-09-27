#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 结构探针(A3 · 预存 HS):每个臂跑满半数据 1 个 epoch(6,117 步),等 epoch0_end 存档写完再停。
#
# 为什么跑满并存档:训练日志里的 hard_accept_len 是即时读数,但下游任务上的效果只有在服务端
# 推理才看得到(要先改 vLLM 支持对应结构)。存下的 ckpt 就是那时要用的;对照是参照 run 的 ep1
# (试点 ep1:服务端 gsm8k 4.935,5 数据集均值 4.093)。
#
#   E10   Markov 头换成 rnn(MARKOV_HEAD_TYPE=rnn:块内递推一个状态,而不是只看前一个 token)
#   E11a  每层(除最后一层)后加一个低秩预测头 + 辅助损失(DSPARK_INTER_HEADS=aux)
#   E11b  E11a + 把各层头的猜测经零初始化门控注入下一层(DSPARK_INTER_HEADS=inject)
#   E11 的结构与开关见 src/speculators/models/dsv4_dspark/inter_heads.py 与 core.py。
#
# 配对:REF=pilot(默认)逐项照抄试点 faithful_ep_20260925_143507(LR 2.8e-4、EPOCHS=10 的调度、
#   anchor 192、种子 42、NONCAUSAL=1、不开均衡),与 A0 以及 E1 的 A1/A2 同一套数据顺序,
#   可以直接比 5601–6100 步的窗口均值。
#   REF=bal 照抄均衡基线 faithful_ep_20260927_145217(LR 3e-4、DSPARK_MOE_BALANCE=1、
#   rate 1e-3、不设目标 entropy)。
#   ⚠ E10 的 rnn 头用 PyTorch 默认初始化,建模型时多消耗全局随机数,之后的锚点与噪声不再与
#     A0 逐步对齐,只能按窗口均值比。E11 的头用私有生成器,不影响对齐。
#
# 用法(在【实验分支的检出】里,训练环境已 activate;整条挂 nohup,日志落盘):
#   ARMS="E11a E11b E10" nohup bash examples/ascend_npu_dflash/launch_a3_probe_1ep.sh \
#     > ~/probe_queue_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# 队列:各臂依次跑。每个臂开跑前等机器空出来 —— 没有活着的训练 / vLLM / 流水线
# (chain_a3_*、eval_blk15_drafts)进程,且显存降到 HBM_FREE_MB 以下。所以均衡基线的
# 评测还没跑完时就可以挂上,它会等评测结束再起。某个臂失败不挡后面的臂。
#
# 旋钮:ARMS(必填)  REF=pilot|bal  STOP_IDX=0(第几个 epoch 末存完就停;0 = ep1)
#       DSPARK_INTER_AUX_WEIGHT(0.2)  DSPARK_INTER_RANK(256)  HBM_FREE_MB=4096  DRY_RUN=1(只预检)
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
hr()  { echo "================================================================================"; }

ARMS="${ARMS:-}"
REF="${REF:-pilot}"
STOP_IDX="${STOP_IDX:-0}"
HBM_FREE_MB="${HBM_FREE_MB:-4096}"
DRY_RUN="${DRY_RUN:-0}"

arm_env() {   # 一个臂相对参照 run 唯一改动的环境变量
  case "$1" in
    E10)  echo "MARKOV_HEAD_TYPE=rnn" ;;
    E11a) echo "DSPARK_INTER_HEADS=aux" ;;
    E11b) echo "DSPARK_INTER_HEADS=inject" ;;
    *)    return 1 ;;
  esac
}
[ -n "$ARMS" ] || { say "!! 要指定 ARMS,如 ARMS=\"E11a E11b E10\""; exit 2; }
for _a in $ARMS; do
  arm_env "$_a" >/dev/null || { say "!! 不认识的臂 $_a(可选 E10 | E11a | E11b)"; exit 2; }
done
case "$REF" in pilot|bal) ;; *) say "!! REF 只能是 pilot 或 bal"; exit 2 ;; esac

# ── 让 python 用【这份检出】的 speculators(理由同 launch_a3_e1_ar_ceiling.sh)──────────
# 训练环境里的 speculators 是 editable 装的,指向基线分支的检出;不顶 PYTHONPATH 的话
# DSPARK_INTER_HEADS 会被【静默忽略】,E11 实际跑成参照 run。
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
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
if ! grep -q 'DSPARK_INTER_HEADS' "$REPO_ROOT/src/speculators/models/dsv4_dspark/core.py"; then
  say "!! 本检出的 core.py 里没有 DSPARK_INTER_HEADS —— 不在实验分支上?"; exit 2
fi

# ── 配方:参照 run 逐项照抄,只改臂自己的那一个变量 ─────────────────────────────
# 这几个若从 shell 里带进来会悄悄改掉配方(续跑、限步、热启动、换头),一律清掉。
for _v in MAX_STEPS SAVE_PATH FROM_PRETRAINED MARKOV_HEAD_TYPE DSPARK_INTER_HEADS \
          DSPARK_BLOCK_INPUT DSPARK_MOE_BALANCE_TARGET; do
  if [ -n "${!_v:-}" ]; then say "注意:清掉 shell 里带进来的 $_v=${!_v}"; fi
  unset "$_v"
done
export EPOCHS=10 MAX_ANCHORS=192 BF16_EXPERTS=0 NONCAUSAL=1
export CKPT_FREQ=1 HS_COUNT_SKIP="${HS_COUNT_SKIP:-1}"
if [ "$REF" = pilot ]; then
  export LR=2.8e-4 DSPARK_MOE_BALANCE=0
  PAIR="faithful_ep_20260925_143507(试点,A0;E1 的 A1/A2 同一数据顺序)"
else
  export LR=3e-4 DSPARK_MOE_BALANCE=1 DSPARK_MOE_BALANCE_RATE=1e-3
  PAIR="faithful_ep_20260927_145217(均衡基线)"
fi
TRAIN_PAT='scripts/[t]rain.py'
SHA=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null)
BRANCH=$(git -C "$REPO_ROOT" branch --show-current 2>/dev/null)

hr
say "探针队列:$ARMS"
say "参照:REF=$REF → $PAIR"
say "配方:LR=$LR EPOCHS=$EPOCHS(调度)MAX_ANCHORS=$MAX_ANCHORS NONCAUSAL=$NONCAUSAL DSPARK_MOE_BALANCE=$DSPARK_MOE_BALANCE CKPT_FREQ=$CKPT_FREQ"
say "每个臂跑到 epoch${STOP_IDX}_end 存完就停;检出 $REPO_ROOT @ ${SHA:0:8} ($BRANCH)"
[ "$DRY_RUN" = 1 ] && { say "DRY_RUN=1:预检完成,到此为止。"; exit 0; }

# ── 机器空了没有 ─────────────────────────────────────────────────────────────
npu_used_mb() {
  command -v npu-smi >/dev/null 2>&1 || return 1
  npu-smi info 2>/dev/null | grep -oE '[0-9]+ +/ +[0-9]+' \
    | awk -F'/' '{ u=$1+0; t=$2+0; if (t > 1000 && u > m) m = u } END { if (m == "") exit 1; print m+0 }'
}
# 活着的(非僵尸)匹配进程是否存在:被杀的 vLLM worker 常以 <defunct> 留在进程表里,不占卡。
alive() {   # $1 = pgrep -f 的模式,其余 = 额外的 pgrep 选项
  local pat=$1 p st; shift
  for p in $(pgrep "$@" -f "$pat" 2>/dev/null); do
    st=$(ps -o stat= -p "$p" 2>/dev/null | tr -d ' ')
    case "$st" in ''|Z*) ;; *) return 0 ;; esac
  done
  return 1
}
busy() {
  alive "$TRAIN_PAT" && { echo "训练"; return 0; }
  alive '[v]llm|[E]ngineCore|[h]s_dump_daemon' -u "$USER" -i && { echo "vLLM"; return 0; }
  alive '[c]hain_a3_|[e]val_blk15_drafts' -u "$USER" && { echo "流水线"; return 0; }
  return 1
}
wait_free() {
  local waited=0 what mb
  while what=$(busy); do
    [ $((waited % 1800)) = 0 ] && say "   等机器空出来:还有${what}进程在跑(已等 $((waited / 60)) 分钟)"
    sleep 60; waited=$((waited + 60))
  done
  waited=0
  while mb=$(npu_used_mb) && [ "$mb" -gt "$HBM_FREE_MB" ]; do
    [ $((waited % 300)) = 0 ] && say "   等驱动回收显存 ... ${mb} MB 仍映射(已等 ${waited}s)"
    if [ "$waited" -ge 3600 ]; then
      say "!! 等了 1 小时显存还在 ${mb} MB(阈值 ${HBM_FREE_MB} MB),不起这个臂"; return 1
    fi
    sleep 20; waited=$((waited + 20))
  done
  mb=$(npu_used_mb) && say "   显存:${mb} MB(阈值 ${HBM_FREE_MB} MB)"
  return 0
}

# ── 一个臂:起训练 → 等 epoch 末存完 → 停 ─────────────────────────────────────
run_arm() {
  local arm=$1 kv out log save mark note last_beat step
  kv=$(arm_env "$arm")
  hr; say "探针 $arm:$kv"
  wait_free || return 1
  out=$(mktemp)
  env "$kv" bash "$SCRIPT_DIR/launch_a3_blk15_prestored.sh" 2>&1 | tee "$out"
  log=$(grep -oE '/[^ ]*/faithful_ep_[0-9_]+\.log' "$out" | tail -1)
  rm -f "$out"
  [ -n "$log" ] || { say "!! $arm 没拿到训练日志路径,训练可能没起来(看上面的输出)"; return 1; }
  save="$(dirname "$log")/ckpt_$(basename "${log%.log}")"
  mark="$save/epoch${STOP_IDX}_end"
  note="${log%.log}.probe.txt"
  {
    echo "probe arm=$arm $kv REF=$REF STOP_IDX=$STOP_IDX"
    echo "recipe LR=$LR EPOCHS=$EPOCHS MAX_ANCHORS=$MAX_ANCHORS NONCAUSAL=$NONCAUSAL DSPARK_MOE_BALANCE=$DSPARK_MOE_BALANCE CKPT_FREQ=$CKPT_FREQ" \
         "DSPARK_INTER_AUX_WEIGHT=${DSPARK_INTER_AUX_WEIGHT:-<默认 0.2>} DSPARK_INTER_RANK=${DSPARK_INTER_RANK:-<默认 256>}"
    echo "repo=$(realpath "$REPO_ROOT") sha=$SHA branch=$BRANCH"
    echo "pair_with=$PAIR"
    echo "started $(date '+%F %T')  log=$log"
  } > "$note"
  say "记录:$note"
  say "看进度:tail -f $log"

  sleep 120
  last_beat=$SECONDS
  until [ -L "$mark" ]; do
    if ! alive "$TRAIN_PAT"; then
      sleep 60
      [ -L "$mark" ] && break
      say "!! $arm 的训练进程没了,而 $mark 一直没出现 —— 训练中途死了。看 $log 的结尾。"
      echo "died before epoch${STOP_IDX}_end  $(date '+%F %T')" >> "$note"
      return 1
    fi
    if [ $((SECONDS - last_beat)) -ge 1800 ]; then
      step=$(tail -c 200000 "$log" | grep -aoE 'global_step=[0-9]+' | tail -1)
      say "   $arm 还在等 epoch${STOP_IDX}_end ... ${step:-?}"
      last_beat=$SECONDS
    fi
    sleep 60
  done
  sleep 30
  pkill -KILL -f "$TRAIN_PAT"
  for _ in $(seq 60); do alive "$TRAIN_PAT" || break; sleep 5; done
  step=$(tail -c 200000 "$log" | grep -aoE 'global_step=[0-9]+' | tail -1)
  echo "stopped after epoch${STOP_IDX}_end (${step:-?})  $(date '+%F %T')  ckpt=$save/$STOP_IDX" >> "$note"
  say "$arm 完成:ckpt $save/$STOP_IDX,日志 $log"
  return 0
}

done_arms=""; failed_arms=""
for _a in $ARMS; do
  if run_arm "$_a"; then done_arms="$done_arms $_a"; else failed_arms="$failed_arms $_a"; fi
done
hr
say "队列结束。完成:${done_arms:- 无}   失败:${failed_arms:- 无}"
say "每个臂的记录在 <训练日志>.probe.txt;训练日志与 provenance 在 ${RUN:-$HOME/dsv4_run}"
