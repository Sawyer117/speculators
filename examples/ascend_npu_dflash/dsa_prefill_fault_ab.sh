#!/usr/bin/env bash
# A3 上 `SparseAttnSharedkv` aicore 故障的根因 A/B —— 一次无人值守跑完,输出一张对照表。
#
# WHY
# ---
# 现象:vllm-ascend pin `4ce367a7d`,DeepSeek-V4-Flash bf16,DP2×TP8/EP16,纯 prefill
# (max_tokens=1)、并发 64 → plog 报 `fault kernel_name=SparseAttnSharedkv_*`,
# `fftsplus aivector error, error code = 0x800000`,多 chip 多 core;PTA 侧表现为
# `AclQueryEventRecordedStatus` 507015(异步故障的延迟暴露,报错位置≠出事位置)。
# 并发 1 稳定。
#
# 关键事实:**同一台 A3、同样 DP2×TP8/EP16、同样并发 64、同样 --no-enable-prefix-caching
# + max-model-len 8192 + graph 模式**,在旧 pin `386530d12` 上跑过 gsm8k 全量 96.59%
# (1274/1319,0 error)和 6 小时 31,916 行 rollout(0 error)。所以**拓扑/配置/并发都不是
# 变量**。剩下的变量只有三个:
#
#   1. pin           386530d12 → 4ce367a7d
#   2. aux 通路      我们自己加的 HS dumper + 强开 target-only aux 捕获
#   3. 流量形状      当年是 prefill+decode(max 3072),现在是纯 prefill(max_tokens=1)
#
# 这个脚本一次砍掉 2 和 3,把 1 逼成唯一嫌疑(或者反过来推翻它)。每个臂 = 一次干净重启
# + 同一批 Arrow 行 + 同样并发,只差一个变量。
#
# 臂
# --
#   A  复现对照      HS_DUMP=1  max_tokens=1   MAXBATCHTOK=8192   期望:崩(不崩就说明不稳定复现)
#   B  纯 prefill?   HS_DUMP=1  max_tokens=64  MAXBATCHTOK=8192   不崩 ⟹ 「整批 prefill」是必要条件
#   C  aux 必要?     HS_DUMP=0  max_tokens=1   MAXBATCHTOK=8192   崩   ⟹ 与我们的 dumper 无关
#   D  剂量(选做)   HS_DUMP=1  max_tokens=1   MAXBATCHTOK=2048   不崩 ⟹ 与单步 prefill token 数成剂量关系
#
# B 不崩 = 顺带解释了「当年 rollout 为什么没事」——一个实验回答两个问题,这是它排第一的原因。
# C 的结论直接决定上游报告里要不要提我们的 dumper(要是 C 也崩,报告就是纯上游的事)。
#
# ⚠ 这个脚本【不报速率】。它只回答「崩/不崩」。崩溃过程的计时没有意义——2026-09-19 已经
#   被这个坑咬过一次(256 条里 236 条报错,工具照样打出「0.11 行/s → 81.8 天」)。
#
# 前置
# ----
#   * 两个仓库都 pull 到最新;vllm-ascend 侧在 `feat/dsv4-dumpers-mrv2`(A/B/D 臂需要 HS
#     dumper;C 臂不需要,但同一个 build 跑完三臂才可比,所以别中途换分支)。
#   * 盘上要有 Arrow 数据集(自动找,找不到用 ARROW= 指定)。
#   * 跑之前不要有别的 vllm 活着 —— 脚本自己会 pkill,但别在跑的时候手动起别的。
#
# 用法
# ----
#   bash dsa_prefill_fault_ab.sh                       # 跑 A B C(默认)
#   ARMS="A B" bash dsa_prefill_fault_ab.sh            # 只跑两臂
#   HBM_WAIT=3600 bash dsa_prefill_fault_ab.sh         # 显存放得更慢就再加大(默认 1800s)
#   ARMS="A B C D" N=256 CONC=64 bash dsa_prefill_fault_ab.sh
#
#   建议 nohup:每臂要等一次 543GB 权重加载,三臂大约 1.5–2 小时。
#     nohup bash examples/ascend_npu_dflash/dsa_prefill_fault_ab.sh > ~/dsa_ab.log 2>&1 &
#     tail -f ~/dsa_ab.log
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ⚠ 一旦 source 过 portproxy_remote.sh,http_proxy 会把 localhost 也劫走。脚本自己的 curl
#   带了 --noproxy '*',但 dsv4_fire_hs_dumps.py 用的 openai 客户端只认 no_proxy 环境变量,
#   否则请求被送去公司代理、回来一张 HTML 错误页(openai.InternalServerError: <!doctype html>)。
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"

ARMS="${ARMS:-A B C}"
# ★★ 2026-09-20 实测:同一个臂、同一套配置,一次 256 条全过(21.9s / 0 错),下一次
#   231 错 + 12 次 SparseAttnSharedkv 故障。**这个故障是间歇的。** 于是:
#     崩   = 阳性,可信(这个条件下确实能触发)
#     不崩 = 阴性,【不可信】(可能只是这次没撞上)
#   所以每个臂必须重复若干次,阴性才有意义。REPEAT=1 只够拿阳性结论。
REPEAT="${REPEAT:-3}"
# ★★ 固定开销每轮 ~19 分钟(清场 12.5 + 加载编译 2.3 + 等端口),而一轮 256 条只打 10 秒。
#   靠反复重启来攒触发机会是把 99% 的时间花在开销上。BURSTS = 在【同一个服务实例】上
#   连打几轮,一轮崩了就停。实测 A 臂单轮 256 条的复现率只有 ~20%,BURSTS=8(=2048 条)
#   能把单次起服务的触发概率拉到 ~83%,而代价只是 +80 秒。
BURSTS="${BURSTS:-8}"
GRACE="${GRACE:-60}"                 # 先 SIGTERM 让 vllm 自己收尾,等这么久再 SIGKILL
N="${N:-256}"
CONC="${CONC:-64}"
PORT="${PORT:-7000}"
ENDPOINT="http://localhost:$PORT/v1"
OUT="${OUT:-$HOME/dsa_fault_ab}"
START_ROW="${START_ROW:-0}"          # 所有臂打同一批行 —— 负载必须逐条相同
# ★★ HS 文件是按 Arrow【行号】命名的(hs_<行号>.safetensors),训练侧也按行号去找。
#   所以 pilot 的 id 段必须落在数据集行数【之外】,否则:
#     (1) 写出去的 hs_<n> 看着像第 n 行的 HS,其实是第 START_ROW+k 行的 —— 静默投毒;
#     (2) 本脚本开打前会 rm 掉该段的旧文件,那就是在删真正的训练 HS。
#   2026-09-20 实测:ID_BASE 忘了定义,$((ID_BASE+...)) 当 0 算,id 段落到 256/512 ——
#   正好砸在真实行号范围里。下面加了硬闸,宁可退出也不让它再发生。
ID_BASE="${ID_BASE:-900000}"         # 每次 fire 用 [ID_BASE + k*N, +N) 一段,k 全局递增
ROWS_FULL="${ROWS_FULL:-772684}"     # 数据集行数;id 段必须整段在它之上
READY_TIMEOUT="${READY_TIMEOUT:-3600}"   # 543GB 权重加载 + 禁用缓存后的一次真编译,给足
SETTLE="${SETTLE:-25}"               # 打完到读 plog 之间的等待(plog 落盘 + 异步故障浮出来)
# ★ 这三个等待宁可长,不可短 —— 等长了人可以 Ctrl-C / kill,等短了就要人整晚盯着
#   手动重来。这台 A3 崩溃之后显存回落尤其慢(实测崩完瞬间还占着 61042/65536 MiB)。
KILL_WAIT="${KILL_WAIT:-180}"        # pkill 之后最多等多久确认进程真的没了
PORT_WAIT="${PORT_WAIT:-300}"        # 进程没了但端口还没放开(TIME_WAIT)时再等多久
HBM_WAIT="${HBM_WAIT:-1800}"         # 等卡上显存回落多久。崩溃后驱动侧释放很慢,给足半小时
HBM_FREE_MB="${HBM_FREE_MB:-4096}"   # 单 die 已用显存低于这个数才算「卡是空的」(空卡通常几百 MiB)
SERVE_SH="${SERVE_SH:-$SCRIPT_DIR/serve_dsv4_a3_singlenode_specmethod.sh}"
LOGDIR="${ASCEND_PROCESS_LOG_PATH:-$HOME/ascend/log}"
DSPARK_HS_DIR="${DSPARK_HS_DIR:-/home/canada_group_folder/dataset/dsv4_hs_dump}"

# ★ 全程锁死 DSA_OVERLAP=0。之前 DSA_OVERLAP=1 时故障落在 _mla_prolog_multistream,
#   关掉之后同一个错跑到 SparseAttnSharedkv —— 那是异步故障换了个暴露点,不是修好了。
#   要对比 SparseAttnSharedkv 这个签名,三个臂就必须都在 0 上,否则臂之间不可比。
DSA_OVERLAP="${DSA_OVERLAP:-0}"

# ★ 全臂禁用编译缓存。A/B/D 臂带 aux、C 臂不带,而 aux 改的是模型返回签名,却【进不了
#   编译缓存的 key】(set_aux_hidden_state_layers 在 get_model 之后才调)。留着缓存,
#   先跑的臂会把图塞进缓存、后跑的臂直接命中错误的图 —— 臂之间互相污染,对照全废。
#   代价是每个臂多一次真编译;换来的是每个臂跑的确实是它自己那张图。
export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"

if [ -z "${ARROW:-}" ]; then
  for d in /home/canada_group_folder/dataset/arrow* \
           /share/canada_group_folder/dataset/*/arrow* \
           /share/canada_group_folder/dataset/arrow*; do
    [ -f "$d/dataset_info.json" ] && ARROW="$d" && break
  done
fi

hms() { printf '%02d:%02d:%02d' $(($1/3600)) $((($1%3600)/60)) $(($1%60)); }
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
serve_up() { curl -sf --noproxy '*' "$ENDPOINT/models" >/dev/null 2>&1; }

# vLLM 给自己的进程改名:setproctitle(f"{VLLM_PROCESS_NAME_PREFIX}::{name}") ->
# VLLM::APIServer_0 / VLLM::DPCoordinator / VLLM::EngineCore_DP0 / VLLM::Worker,
# vllm-ascend 侧还有 VLLMWorker_DP / VLLM_DP_Coordinator。都带 VLLM,但别只赌这一点:
# 多列几个名字,漏掉一个残留进程 = 下一次起服务在 gloo rendezvous 上莫名其妙地挂。
KILLPAT="${KILLPAT:-vllm|EngineCore|APIServer|ApiServer|DPCoordinator|VLLMWorker|dspark_hs}"

# 还活着的目标进程(排除自己和自己的父 shell,否则 pgrep -f 会把本脚本也数进去)
_alive() { pgrep -i -u "$USER" -f "$KILLPAT" 2>/dev/null | grep -vx "$$" | grep -vx "$PPID"; }

# ★ 终极判据:卡上还剩多少显存被占。进程列表可能漏(名字对不上、D 状态卡在驱动里),
#   **显存不会骗人**。2026-09-19 实测:pkill 之后 pgrep 干净,起服务却报
#   `Free memory on device (22.42/61.27 GiB) ... less than desired (0.9, 55.14 GiB)`
#   —— 38.85 GiB 还被残留的 VLLMWorker_DP(各 41056 MiB)攥着。
#   npu-smi 的 HBM-Usage 列形如 `3536 / 65536`;只取分母 >= 30000 的那组(HBM,不是
#   旁边那列小的 Memory(MB)),返回所有 die 里最大的已用值(MB)。
_npu_used_mb() {
  command -v npu-smi >/dev/null 2>&1 || { echo -1; return; }
  npu-smi info 2>/dev/null \
    | grep -oE '[0-9]+ */ *[0-9]{5,}' \
    | awk -F'/' '{gsub(/ /,""); if ($2+0>=30000 && $1+0>mx) mx=$1+0} END{print mx+0}'
}

_port_free() {
  python - "$PORT" <<'PYEOF' 2>/dev/null
import socket, sys
s = socket.socket()
try:
    s.bind(("0.0.0.0", int(sys.argv[1]))); sys.exit(0)
except OSError:
    sys.exit(1)
finally:
    s.close()
PYEOF
}

# ★ 清场必须【可验证】。第一版是 pkill 之后 sleep 60 就当干净了 —— 2026-09-19 下一次起
#   服务在 `torch.distributed.new_group(backend="gloo")` 上报
#   `Failed to recv, got 0 bytes`(rendezvous 对端没了),典型的残留/端口没放开。
#   睡固定秒数不是清场,是许愿。
cleanup_verified() {
  local tag="$1"
  # ★ 先 SIGTERM。实测 SIGKILL 之后显存要 12 分 40 秒才被驱动收回(每轮都付),而 vllm
  #   自己有 graceful shutdown(日志里的 `[shutdown] send sigterm to process ...`),
  #   让它自己退能快得多。等 GRACE 秒还赖着再 -9 —— 最坏多花 60 秒,最好省 10 分钟。
  if [ -n "$(_alive)" ]; then
    pkill -TERM -i -u "$USER" -f "$KILLPAT" >/dev/null 2>&1
    local g=0
    while [ "$g" -lt "$GRACE" ] && [ -n "$(_alive)" ]; do sleep 2; g=$((g + 2)); done
    [ -n "$(_alive)" ] && say "    SIGTERM 后 $(hms $g) 还没退干净,转 SIGKILL"
  fi
  pkill -9 -i -u "$USER" -f "$KILLPAT" >/dev/null 2>&1
  local t=0 n
  while [ "$t" -lt "$KILL_WAIT" ]; do
    n=$(_alive | wc -l)
    [ "$n" -eq 0 ] && break
    sleep 2; t=$((t + 2))
  done
  # 宽限一次再判:上面那轮和这一行之间进程可能刚好退出,直接判失败会误杀一整臂。
  if [ "$(_alive | wc -l)" -gt 0 ]; then sleep 3; fi
  local left; left=$(_alive)
  if [ -n "$left" ]; then
    local shown; shown=$(echo "$left" | while read -r pid; do
      ps -o pid=,etime=,args= -p "$pid" 2>/dev/null | cut -c1-140; done)
    if [ -z "$shown" ]; then
      say "    (pgrep 数到残留但 ps 已查不到 —— 它们在这几秒里退干净了,继续)"
    else
      say "!! 清场未完成($tag):还有进程没死:"
      echo "$shown"
      return 1
    fi
  fi
  # 端口:进程没了不代表端口放开了(TIME_WAIT / 别人占着)。等的时候要出声 ——
  # 静默地等一分钟,和卡死在用户眼里没有区别。
  local pt=0 pb=0
  while ! _port_free && [ "$pt" -lt "$PORT_WAIT" ]; do
    [ "$pt" -eq 0 ] && say "    端口 $PORT 还被占着,等它放开(最多 $(hms "$PORT_WAIT"))..."
    sleep 2; pt=$((pt + 2)); pb=$((pb + 2))
    if [ "$pb" -ge 60 ]; then pb=0; say "    ... 端口还没放开($(hms $pt))"; fi
  done
  [ "$pt" -gt 0 ] && _port_free && say "    端口 $PORT 在 $(hms $pt) 后放开"
  if ! _port_free; then
    say "!! 清场未完成($tag):端口 $PORT 仍被占用($(hms $pt) 没放开)。"
    command -v ss >/dev/null && ss -ltnp 2>/dev/null | grep ":$PORT " | head -3
    return 1
  fi
  # 泄漏的 IPC:刚才那次崩溃日志里就有「7 leaked semaphore / 1 leaked shared_memory」
  local shm; shm=$(find /dev/shm -maxdepth 1 -user "$USER" \
                   \( -name 'psm_*' -o -name '*vllm*' -o -name 'torch_*' \) 2>/dev/null | wc -l)
  if [ "$shm" -gt 0 ]; then
    say "    清掉 $shm 个残留的 /dev/shm 对象(进程已全部确认退出,不会误删活着的)"
    find /dev/shm -maxdepth 1 -user "$USER" \
      \( -name 'psm_*' -o -name '*vllm*' -o -name 'torch_*' \) -delete 2>/dev/null
  fi
  # ★ 显存:pkill 之后驱动侧释放要时间,进程没了不等于卡空了。等它回落,别一次性判死。
  local used; used=$(_npu_used_mb)
  if [ "$used" -ge 0 ] 2>/dev/null; then
    local ht=0 hb=0
    while [ "$used" -gt "$HBM_FREE_MB" ] && [ "$ht" -lt "$HBM_WAIT" ]; do
      [ "$ht" -eq 0 ] && say "    卡上还占着 ${used} MiB,等显存回落(最多 $(hms "$HBM_WAIT"))..."
      sleep 10; ht=$((ht + 10)); used=$(_npu_used_mb)
      # 心跳:半小时的等待不出声和卡死没区别。每分钟一行,只说还没清完就够了。
      hb=$((hb + 10))
      if [ "$hb" -ge 60 ]; then hb=0; say "    ... 显存还没清空($(hms $ht))"; fi
    done
    if [ "$used" -gt "$HBM_FREE_MB" ]; then
      say "!! 清场未完成($tag):卡上仍有 ${used} MiB 被占(阈值 ${HBM_FREE_MB} MiB,等了 $(hms $ht))。"
      say "   占用一直没往下走 = 有进程没杀掉;还在往下走 = 这台机器放得慢,加大 HBM_WAIT 重跑。"
      say "   带着这些残留起服务,会在 worker init 报"
      say "   'Free memory on device (...) is less than desired GPU memory utilization' —— 那不是配置问题。"
      npu-smi info 2>/dev/null | grep -iE 'vllm|python|process id' | head -20
      return 1
    fi
    [ "$ht" -gt 0 ] && say "    显存在 $(hms $ht) 后回落到 ${used} MiB"
  fi
  if [ "$used" -ge 0 ] 2>/dev/null; then
    say "    清场已核实:无残留进程,端口 $PORT 可绑定,卡上占用 ${used} MiB"
  else
    say "    清场已核实:无残留进程,端口 $PORT 可绑定(没有 npu-smi,跳过显存检查)"
  fi
  return 0
}

# 单实例 —— 用 flock,不用进程名。
# ★ 试过按进程名 pgrep 自己,反复误报:`$(...)` 命令替换 fork 出的子 shell cmdline
#   和本体一模一样(实测本体 2731224 / 子 shell 2731230),连启动它的那层
#   `bash .../dsa_prefill_fault_ab.sh` 也会被数进来。想靠 PGID 或父子链把「自己人」
#   摘出去,在嵌套/非交互 shell 下都不稳。flock 是内核级的文件锁,不看进程名、
#   不看血缘,进程一退内核自动释放 —— 没有歧义,也不会留下要手工删的陈旧锁。
LOCK="${LOCK:-$HOME/.dsa_prefill_fault_ab.lock}"
exec 9>"$LOCK" || { echo "!! 打不开锁文件 $LOCK"; exit 2; }
if command -v flock >/dev/null 2>&1; then
  if ! flock -n 9; then
    echo "!! 已经有一个实例持有 $LOCK —— 两个 A/B 同时跑会互相把对方的服务当成自己的、"
    echo "   再互相 pkill,拿到的数全是噪声。"
    echo "   先杀掉:pkill -9 -f dsa_prefill_fault_ab.sh"
    exit 2
  fi
else
  echo ">>> 注意:没有 flock,跳过单实例保护。手工确认没有别的实例在跑。"
fi
echo $$ >&9
# ⚠ 上一轮那个【旧版本】脚本不持有这把锁(它根本没有锁),所以它拦不住。跑之前
#   自己确认一眼:pgrep -af dsa_prefill_fault_ab.sh —— 只该看到你刚起的这一个。

mkdir -p "$OUT"
RESULTS="$OUT/results.tsv"
: > "$RESULTS"

echo "================================================================================"
echo "  A3 SparseAttnSharedkv 故障 A/B"
echo "  臂         $ARMS"
echo "  ARROW      ${ARROW:-<找不到,用 ARROW= 指定>}"
echo "  serve      $SERVE_SH"
echo "  HS_DIR     $DSPARK_HS_DIR"
echo "  流量       每臂 $REPEAT 次起服务 × $BURSTS 连打 × $N 条 @ 并发 $CONC = $((REPEAT*BURSTS*N)) 条/臂"
echo "             Arrow 行 [$START_ROW, $((START_ROW+N)))(每轮同一批,负载逐条相同)"
echo "  DSA_OVERLAP=$DSA_OVERLAP  VLLM_DISABLE_COMPILE_CACHE=$VLLM_DISABLE_COMPILE_CACHE (都锁死,否则臂之间不可比)"
echo "  plog       $LOGDIR"
echo "  产物       $OUT"
echo "================================================================================"

if [ -z "$ARROW" ] || [ ! -f "$ARROW/dataset_info.json" ]; then
  echo "!! Arrow 数据集不可用:${ARROW:-<空>}(要有 dataset_info.json)。用 ARROW= 指定。"; exit 2
fi
[ -f "$SERVE_SH" ] || { echo "!! 找不到 serve 脚本:$SERVE_SH"; exit 2; }
[ -d "$LOGDIR" ] || echo "!! 警告:plog 目录 $LOGDIR 不存在 —— 算子级判据会全部落空。先确认 ASCEND_PROCESS_LOG_PATH。"

# ── 一个臂 ────────────────────────────────────────────────────────────────────
# $1 臂名  $2 HS_DUMP(1/0)  $3 max_tokens  $4 MAXBATCHTOK  $5 一句话说明
run_arm() {
  local arm="$1" hsdump="$2" maxtok="$3" mbt="$4" desc="$5"
  local slog="$OUT/serve_$arm.log" flog="$OUT/fire_$arm.log"
  local t_arm=$SECONDS

  echo
  echo "################################################################################"
  say "臂 $arm —— $desc"
  say "    HS_DUMP=$hsdump  max_tokens=$maxtok  MAXBATCHTOK=$mbt"
  echo "################################################################################"

  say "清场 ..."
  if ! cleanup_verified "臂 $arm 起跑前"; then
    say "!! 起跑前清不干净,这一臂【作废】—— 带着残留起服务只会得到一堆看不懂的 gloo 报错。"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$arm" "$desc" "清场失败" "-" "-" "残留未清干净" "$slog" >> "$RESULTS"
    ARM_FAILED_START=$((ARM_FAILED_START + 1))
    return
  fi

  # 故障判据的时间基准:只认这一刻【之后】写进 plog 的东西,不然会把上一个臂的账算过来
  local T0; T0=$(date +%s)

  say "起服务 → $slog"
  if [ "$hsdump" = "1" ]; then
    MAXBATCHTOK="$mbt" DSA_OVERLAP="$DSA_OVERLAP" HS_DUMP=1 DSPARK_HS_DIR="$DSPARK_HS_DIR" \
      nohup bash "$SERVE_SH" > "$slog" 2>&1 &
  else
    # C 臂:纯 target 服务,不设 HS_DUMP → 不装 dumper 钩子、不强开 aux 通路
    MAXBATCHTOK="$mbt" DSA_OVERLAP="$DSA_OVERLAP" \
      nohup bash "$SERVE_SH" > "$slog" 2>&1 &
  fi
  local spid=$!   # serve 脚本 exec 成 vllm,所以这个 PID 就是引擎:它没了 = 起失败

  # ★ 2026-09-19 教训:第一版只会干等 READY_TIMEOUT(40 分钟),服务 2 分钟就崩了却一行不吭,
  #   用户盯着 tail -f 看一片空白。**起失败必须立刻报,并且把致命 traceback 打到主日志里** ——
  #   要去翻另一个文件才知道出了什么事,等于没报。
  local t0=$SECONDS ready=0 why=""
  while [ $((SECONDS - t0)) -lt "$READY_TIMEOUT" ]; do
    if serve_up; then ready=1; break; fi
    # (a) 进程没了 = 引擎起失败。最可靠的判据,没有误报
    if ! kill -0 "$spid" 2>/dev/null; then why="引擎进程已退出"; break; fi
    # (b) 进程还在但日志里已经有致命行 —— 比等进程退出更快
    if grep -qE 'Engine core initialization failed|EngineCore failed to start|^ *(ValueError|RuntimeError|AssertionError|ImportError|OSError):' "$slog" 2>/dev/null; then
      why="日志出现致命错误"; break
    fi
    # (c) 心跳:别让 tail -f 看起来像卡死
    if [ $(( (SECONDS - t0) % 60 )) -lt 10 ] && [ $((SECONDS - t0)) -ge 60 ]; then
      say "    ... 等 READY $(hms $((SECONDS-t0)))  | $(tail -1 "$slog" 2>/dev/null | cut -c1-120)"
    fi
    sleep 10
  done
  if [ "$ready" != "1" ]; then
    [ -n "$why" ] || why="等满 $(hms "$READY_TIMEOUT") 仍未就绪"
    say "!! 臂 $arm 起不来($why,用时 $(hms $((SECONDS-t0))))—— 这一臂【作废】,不是「不崩」"
    # ★ 多进程崩溃是【级联】:一个 worker 先死,其余在 gloo rendezvous 上报
    #   `Failed to recv, got 0 bytes` —— 那是后果不是原因。所以先给【最早】那段。
    local first_err
    first_err=$(grep -nE 'ERROR|Traceback \(most recent call last\)' "$slog" 2>/dev/null | head -1 | cut -d: -f1)
    echo "---------------- ★ 最早的错误(首因,第 ${first_err:-?} 行起)----------------"
    [ -n "$first_err" ] && sed -n "${first_err},$((first_err + 40))p" "$slog" | cut -c1-200
    echo "---------------- 各类错误各取一条 ----------------"
    grep -ohE '(ValueError|RuntimeError|AssertionError|ImportError|OSError|DistNetworkError|HCCL[A-Za-z]*Error|RuntimeError)[^\n]{0,140}' \
      "$slog" 2>/dev/null | sed 's/  */ /g' | sort -u | head -8
    echo "---------------- 末尾 12 行 ----------------"
    tail -12 "$slog" 2>/dev/null | cut -c1-200
    echo "------------------------------------------------"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$arm" "$desc" "起不来" "-" "-" "$why" "$slog" >> "$RESULTS"
    cleanup_verified "臂 $arm 收尾" || true
    ARM_FAILED_START=$((ARM_FAILED_START + 1))
    return
  fi
  say "服务 READY($(hms $((SECONDS-t0))))"

  : > "$flog"
  rm -rf "$OUT/dumps_$arm"
  local errs=0 fired_tot=0 b e
  for b in $(seq 1 "$BURSTS"); do
  # ★ id 段必须每次都新。2026-09-20 实测:id-base 写死 901000,第二轮 231 条请求报错,
  #   却照样打印 `collected 256/256` —— 收到的是上一轮留在 DSPARK_HS_DIR 里的旧文件。
  #   一个假的「全收齐」比没有数更糟。所以段号全局递增,并且开打前把这一段清空。
  RUN_SEQ=$((RUN_SEQ + 1))
  local idbase=$((ID_BASE + RUN_SEQ * N))
  # 硬闸:整段都必须在数据集行号之外,否则停机。这条不是洁癖 —— 越过去就是往生产 HS
  # 目录里投毒(内容对不上行号),而且开打前那个 rm 会删掉真的训练 HS。
  if [ "$idbase" -le "$ROWS_FULL" ]; then
    say "!! id 段 [$idbase, $((idbase + N))) 落在数据集行号范围内(共 $ROWS_FULL 行)——"
    say "   这会往 $DSPARK_HS_DIR 里写出【行号与内容对不上】的 HS,并且删掉同号的真文件。"
    say "   停。把 ID_BASE 调到 > $ROWS_FULL(默认 900000)再跑。"
    exit 3
  fi
  seq "$idbase" $((idbase + N - 1)) \
    | sed "s|^|$DSPARK_HS_DIR/hs_|; s|\$|.safetensors|" | xargs -r rm -f 2>/dev/null

  local collect=()
  [ "$hsdump" = "1" ] || collect=(--no-collect)
  ENDPOINT="$ENDPOINT" ARROW="$ARROW" HS_DIR="$DSPARK_HS_DIR" \
    python "$SCRIPT_DIR/dsv4_fire_hs_dumps.py" \
      --out "$OUT/dumps_$arm" --n "$N" --concurrency "$CONC" \
      --id-base "$idbase" --start-row "$START_ROW" \
      --max-tokens "$maxtok" "${collect[@]}" >> "$flog" 2>&1
  e=$(sed -n 's/.*errors=\([0-9]*\).*/\1/p' "$flog" | tail -1); e="${e:-0}"
  errs=$((errs + e)); fired_tot=$((fired_tot + N))
  say "    第 $b/$BURSTS 轮(id 段 $idbase+):$(tail -1 "$flog" | cut -c1-96)"
  if ! serve_up; then say "    ★ 引擎在第 $b 轮死了 —— 停止连打"; break; fi
  done
  say "打完:累计 $fired_tot 条,errors=$errs"

  say "等 ${SETTLE}s 让 plog 落盘(507015 是异步故障,来得比请求晚)..."
  sleep "$SETTLE"

  # 服务还活着吗 —— 引擎死掉本身就是最硬的判据
  local alive="活"; serve_up || alive="死"

  # ★ 算子级故障:只数 T0 之后改动过的 plog。`fault kernel_name=` 才是权威,
  #   PTA 报的 507015 位置是异步暴露点,不能当出事位置。
  local faults="0" kinds="-"
  if [ -d "$LOGDIR" ]; then
    kinds=$(find "$LOGDIR" -name '*.log' -newermt "@$T0" 2>/dev/null \
            | xargs -r grep -ohE 'fault kernel_name=[^ ,]*' 2>/dev/null \
            | sed 's/fault kernel_name=//; s/_[0-9a-f]\{16,\}.*//' | sort | uniq -c | sort -rn \
            | awk '{printf "%s×%s ", $1, $2}')
    faults=$(find "$LOGDIR" -name '*.log' -newermt "@$T0" 2>/dev/null \
             | xargs -r grep -ohE 'fault kernel_name=' 2>/dev/null | wc -l)
    [ -n "$kinds" ] || kinds="-"
  fi
  local e507; e507=$(grep -c '507015\|AclQueryEventRecordedStatus' "$slog" 2>/dev/null)
  e507="${e507:-0}"   # grep -c 无命中时自己就打印 0 并返回 1,别再 `|| echo 0` —— 那会变成两行

  say "结果:引擎=$alive  fire errors=$errs  算子故障=$faults 次  507015=$e507 次  $kinds"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$arm" "$desc" "$alive" "$errs" "$faults" "$kinds" "507015×$e507" >> "$RESULTS"

  cleanup_verified "臂 $arm 收尾" || true
  say "臂 $arm 用时 $(hms $((SECONDS-t_arm)))"
}

T_ALL=$SECONDS
RUN_SEQ=0
ARM_FAILED_START=0
# 起不来通常是配置/构建问题,不是这一臂特有的 —— 后面的臂几乎必然同样起不来。默认第一次
# 起失败就停,别让人回来发现空等了两小时。STOP_ON_START_FAIL=0 可以强行跑完全部臂。
STOP_ON_START_FAIL="${STOP_ON_START_FAIL:-1}"
for arm in $ARMS; do
  for rep in $(seq 1 "$REPEAT"); do
    [ "$REPEAT" -gt 1 ] && say "===== 臂 $arm 第 $rep/$REPEAT 次 ====="
    case "$arm" in
      A) run_arm A 1 1  8192 "复现对照(纯 prefill + dumper)" ;;
      B) run_arm B 1 64 8192 "混入 decode —— 纯 prefill 是不是必要条件" ;;
      C) run_arm C 0 1  8192 "去掉 dumper/aux —— aux 是不是必要条件" ;;
      D) run_arm D 1 1  2048 "单步 prefill token 砍到 1/4 —— 剂量关系" ;;
      *) echo "!! 未知的臂:$arm(只认 A B C D)"; break ;;
    esac
    [ "$ARM_FAILED_START" -gt 0 ] && break
  done
  if [ "$ARM_FAILED_START" -gt 0 ] && [ "$STOP_ON_START_FAIL" = "1" ]; then
    echo
    say "!! 服务起不来,后面的臂大概率一样 —— 就地停,不空烧时间。"
    say "   先按上面的 traceback 修好起服务这一步,再重跑整个脚本。"
    say "   确实想跑完全部臂:STOP_ON_START_FAIL=0 bash ..."
    break
  fi
done

echo
echo "================================================================================"
echo "  对照表   (总用时 $(hms $((SECONDS-T_ALL))))"
echo "================================================================================"
printf '%-4s %-38s %-6s %-8s %-8s %s\n' 臂 说明 引擎 fire错 算子故障 故障算子
awk -F'\t' '{printf "%-4s %-38s %-6s %-8s %-8s %s\n", $1, $2, $3, $4, $5, $6}' "$RESULTS"
echo
echo "按臂汇总(崩 = 引擎死 或 算子故障>0):"
awk -F'\t' '{n[$1]++; if ($3=="死" || ($5+0)>0) k[$1]++}
  END{for (a in n) printf "  臂 %s: 崩 %d/%d 次\n", a, k[a]+0, n[a]}' "$RESULTS" | sort
echo
echo "怎么读   ★ 这个故障是【间歇】的(实测同配置一次 0 错、一次 231 错):"
echo "         崩 = 阳性,可信。不崩 = 阴性,只在 $REPEAT 次全都不崩时才值得采信,"
echo "         而且 $REPEAT 次也只是弱证据 —— 要硬结论就加大 REPEAT 或 N。"
echo "  A 崩 + B 不崩          ⟹ 「整批纯 prefill」是必要条件。顺带解释了旧栈 rollout"
echo "                            (prefill+decode)为什么没事。上游报告要写清这个触发形状。"
echo "  A 崩 + C 也崩          ⟹ 与我们的 HS dumper 无关,纯上游问题,报告里不用提 dumper。"
echo "  A 崩 + C 不崩          ⟹ aux 捕获是必要条件。先验我们强开 use_aux_hidden_state_outputs"
echo "                            那段(model_runner load_model 里的 +1 层号)对不对。"
echo "  A 不崩                 ⟹ 复现不稳定,上面所有对比都不成立。加大 N 或重跑,别往下推结论。"
echo "  D 不崩                 ⟹ 与单步 prefill token 数成剂量关系 = 该 kernel 在大 prefill 批下越界。"
echo
echo "  ⚠ 「起不来」≠「不崩」——那一臂作废,先看对应的 serve_<臂>.log。"
echo "  ⚠ 链路层(error cqe / EI0002 / link down)全 0 不代表没事,两层是独立的。"
echo "     细节:bash $SCRIPT_DIR/npu_hang_diag.sh"
echo
echo "产物:$OUT  (serve_<臂>.log / fire_<臂>.log / results.tsv)"
echo "机器已清场,没有服务在跑。恢复生产:"
echo "  DSA_OVERLAP=0 HS_DUMP=1 DSPARK_HS_DIR=$DSPARK_HS_DIR \\"
echo "    nohup bash $SERVE_SH > ~/serve_hsdump.log 2>&1 &"
