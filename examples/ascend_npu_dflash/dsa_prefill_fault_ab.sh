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
N="${N:-256}"
CONC="${CONC:-64}"
PORT="${PORT:-7000}"
ENDPOINT="http://localhost:$PORT/v1"
OUT="${OUT:-$HOME/dsa_fault_ab}"
START_ROW="${START_ROW:-0}"          # 所有臂打同一批行 —— 负载必须逐条相同
READY_TIMEOUT="${READY_TIMEOUT:-2400}"   # 543GB 权重加载,给足
SETTLE="${SETTLE:-25}"               # 打完到读 plog 之间的等待(plog 落盘 + 异步故障浮出来)
KILL_WAIT="${KILL_WAIT:-60}"         # pkill 之后等 HBM 真正释放
SERVE_SH="${SERVE_SH:-$SCRIPT_DIR/serve_dsv4_a3_singlenode_specmethod.sh}"
LOGDIR="${ASCEND_PROCESS_LOG_PATH:-$HOME/ascend/log}"
DSPARK_HS_DIR="${DSPARK_HS_DIR:-/home/canada_group_folder/dataset/dsv4_hs_dump}"

# ★ 全程锁死 DSA_OVERLAP=0。之前 DSA_OVERLAP=1 时故障落在 _mla_prolog_multistream,
#   关掉之后同一个错跑到 SparseAttnSharedkv —— 那是异步故障换了个暴露点,不是修好了。
#   要对比 SparseAttnSharedkv 这个签名,三个臂就必须都在 0 上,否则臂之间不可比。
DSA_OVERLAP="${DSA_OVERLAP:-0}"

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

mkdir -p "$OUT"
RESULTS="$OUT/results.tsv"
: > "$RESULTS"

echo "================================================================================"
echo "  A3 SparseAttnSharedkv 故障 A/B"
echo "  臂         $ARMS"
echo "  ARROW      ${ARROW:-<找不到,用 ARROW= 指定>}"
echo "  serve      $SERVE_SH"
echo "  HS_DIR     $DSPARK_HS_DIR"
echo "  流量       每臂 $N 条 @ 并发 $CONC,Arrow 行 [$START_ROW, $((START_ROW+N)))"
echo "  DSA_OVERLAP=$DSA_OVERLAP (全程锁死,否则臂之间不可比)"
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
  # 每臂独立 id 段,文件不串;900000 起 —— 落在数据集 772,684 行之外,训练进程不会当成自己的 HS 删掉
  ARM_SEQ=$((ARM_SEQ + 1))
  local idbase=$((900000 + 1000 * ARM_SEQ))
  local t_arm=$SECONDS

  echo
  echo "################################################################################"
  say "臂 $arm —— $desc"
  say "    HS_DUMP=$hsdump  max_tokens=$maxtok  MAXBATCHTOK=$mbt"
  echo "################################################################################"

  say "清场 ..."
  pkill -9 -i -u "$USER" -f 'vllm|EngineCore' >/dev/null 2>&1
  sleep "$KILL_WAIT"

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

  local t0=$SECONDS ready=0
  while [ $((SECONDS - t0)) -lt "$READY_TIMEOUT" ]; do
    if serve_up; then ready=1; break; fi
    sleep 10
  done
  if [ "$ready" != "1" ]; then
    say "!! 服务 $(hms $((SECONDS-t0))) 没起来 —— 这一臂作废(不是「不崩」!)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$arm" "$desc" "起不来" "-" "-" "-" "$slog" >> "$RESULTS"
    pkill -9 -i -u "$USER" -f 'vllm|EngineCore' >/dev/null 2>&1
    return
  fi
  say "服务 READY($(hms $((SECONDS-t0))))"

  say "打 $N 条 @ 并发 $CONC(max_tokens=$maxtok)→ $flog"
  local collect=()
  [ "$hsdump" = "1" ] || collect=(--no-collect)
  ENDPOINT="$ENDPOINT" ARROW="$ARROW" HS_DIR="$DSPARK_HS_DIR" \
    python "$SCRIPT_DIR/dsv4_fire_hs_dumps.py" \
      --out "$OUT/dumps_$arm" --n "$N" --concurrency "$CONC" \
      --id-base "$idbase" --start-row "$START_ROW" \
      --max-tokens "$maxtok" "${collect[@]}" > "$flog" 2>&1
  tail -3 "$flog"

  local errs; errs=$(sed -n 's/.*errors=\([0-9]*\).*/\1/p' "$flog" | tail -1); errs="${errs:-?}"

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

  pkill -9 -i -u "$USER" -f 'vllm|EngineCore' >/dev/null 2>&1
  say "臂 $arm 用时 $(hms $((SECONDS-t_arm)))"
}

T_ALL=$SECONDS
ARM_SEQ=0
for arm in $ARMS; do
  case "$arm" in
    A) run_arm A 1 1  8192 "复现对照(纯 prefill + dumper)" ;;
    B) run_arm B 1 64 8192 "混入 decode —— 纯 prefill 是不是必要条件" ;;
    C) run_arm C 0 1  8192 "去掉 dumper/aux —— aux 是不是必要条件" ;;
    D) run_arm D 1 1  2048 "单步 prefill token 砍到 1/4 —— 剂量关系" ;;
    *) echo "!! 未知的臂:$arm(只认 A B C D)" ;;
  esac
done

echo
echo "================================================================================"
echo "  对照表   (总用时 $(hms $((SECONDS-T_ALL))))"
echo "================================================================================"
printf '%-4s %-38s %-6s %-8s %-8s %s\n' 臂 说明 引擎 fire错 算子故障 故障算子
awk -F'\t' '{printf "%-4s %-38s %-6s %-8s %-8s %s\n", $1, $2, $3, $4, $5, $6}' "$RESULTS"
echo
echo "怎么读"
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
