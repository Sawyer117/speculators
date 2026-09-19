#!/usr/bin/env bash
# HS 批量生产看门狗 —— 崩了自己爬起来接着干,直到把该存的行存完。
#
# WHY
# ---
# 「预存 HS」这件事的收益是:两台 serve 机腾出来参训,DP 8→24 卡,每个 epoch 的墙钟缩到
# 1/3。唯一的障碍是 A3 上那个间歇的 `SparseAttnSharedkv` aicore 越界(2026-09-20 实测
# 复现率 ~11–18% / 256 条),它会打死引擎。
#
# **但这个故障挡不住我们。** 它是间歇的、崩了重启就能继续、而全量 dump 本来就要跑几天。
# 所以不必等上游修:加个看门狗,崩一次爬起来一次,按【盘上已有的文件】算出还缺哪些行,
# 从缺口接着打。上游 issue 归 issue,生产线今天就能开。
#
# 契约(别改):**文件名就是 Arrow 行号** —— 训练侧 `data.py:346` 按行号找
# `hs_<row>.safetensors`。所以这里必须 `--id-base R --start-row R`(id == row),
# 和 pilot 那种「用 900000+ 偏移避开真实行号」正好相反。
#
# 只写不删。任何删除动作都不在这个脚本里 —— 删 HS 是训练侧 rolling delete 的事。
#
# 用法
# ----
#   nohup bash hs_dump_daemon.sh > ~/hs_daemon.log 2>&1 &
#   tail -f ~/hs_daemon.log
#
#   MAX_ROWS=200000 bash hs_dump_daemon.sh        # 只存前 20W 行(盘放不下全量时)
#   ROW_START=0 ROW_END=100000 bash hs_dump_daemon.sh
#   CHUNK=512 CONC=32 bash hs_dump_daemon.sh      # 调块大小 / 并发
#   RESERVE_GB=500 bash hs_dump_daemon.sh         # 盘剩这么多就停
#
# 停:Ctrl-C 或 kill;下次再起会从盘上已有的文件接着算,不会重打。
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/npu_cleanup_lib.sh"

# ⚠ openai 客户端只认 no_proxy 环境变量;不设会被公司代理劫走,回来一张 HTML 错误页。
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"

PORT="${PORT:-7000}"
ENDPOINT="http://localhost:$PORT/v1"
SERVE_SH="${SERVE_SH:-$SCRIPT_DIR/serve_dsv4_a3_singlenode_specmethod.sh}"
DSPARK_HS_DIR="${DSPARK_HS_DIR:-/home/canada_group_folder/dataset/dsv4_hs_dump}"
CHUNK="${CHUNK:-256}"                # 一次 fire 打多少行
CONC="${CONC:-64}"                   # 并发(≤ serve 的 --max-num-seqs)
ROW_START="${ROW_START:-0}"
ROWS_FULL="${ROWS_FULL:-772684}"
ROW_END="${ROW_END:-$ROWS_FULL}"
MAX_ROWS="${MAX_ROWS:-0}"            # >0 = 本次最多【新增】这么多行(盘配额)
RESERVE_GB="${RESERVE_GB:-300}"      # dump 盘剩余低于此值就停,别把盘写满
READY_TIMEOUT="${READY_TIMEOUT:-3600}"
MAX_RESTARTS="${MAX_RESTARTS:-200}"  # 连续起不来这么多次就放弃(不是崩溃次数上限)
DSA_OVERLAP="${DSA_OVERLAP:-0}"
LOG="${LOG:-$HOME/hs_daemon_serve.log}"

hms() { printf '%02d:%02d:%02d' $(($1/3600)) $((($1%3600)/60)) $(($1%60)); }
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
serve_up() { curl -sf --noproxy '*' "$ENDPOINT/models" >/dev/null 2>&1; }

# Arrow 自动定位(和 pilot 用同一套已知位置)
if [ -z "${ARROW:-}" ]; then
  for d in /home/canada_group_folder/dataset/arrow* \
           /share/canada_group_folder/dataset/*/arrow* \
           /share/canada_group_folder/dataset/arrow*; do
    [ -f "$d/dataset_info.json" ] && ARROW="$d" && break
  done
fi

# 单实例:两个 daemon 同时跑会互相 pkill 对方的服务。用 flock(内核级,不看进程名)。
LOCK="${LOCK:-$HOME/.hs_dump_daemon.lock}"
exec 9>"$LOCK" || { echo "!! 打不开锁文件 $LOCK"; exit 2; }
if command -v flock >/dev/null 2>&1; then
  flock -n 9 || { echo "!! 已经有一个 hs_dump_daemon 在跑(锁 $LOCK)。先停掉它。"; exit 2; }
fi
echo $$ >&9

echo "================================================================================"
echo "  HS 批量生产看门狗"
echo "  ARROW        ${ARROW:-<找不到,用 ARROW= 指定>}"
echo "  DSPARK_HS_DIR $DSPARK_HS_DIR   (文件名 = Arrow 行号,只写不删)"
echo "  行范围       [$ROW_START, $ROW_END)   块 $CHUNK 行   并发 $CONC"
echo "  本次上限     $([ "$MAX_ROWS" -gt 0 ] && echo "$MAX_ROWS 行" || echo '不限')   盘保留 ${RESERVE_GB} GB"
echo "  serve        $SERVE_SH   (DSA_OVERLAP=$DSA_OVERLAP)"
echo "  serve 日志   $LOG        (每次重启【追加】,不覆盖 —— 崩溃现场要留住)"
echo "================================================================================"
[ -n "$ARROW" ] && [ -f "$ARROW/dataset_info.json" ] || { echo "!! Arrow 不可用:${ARROW:-<空>}"; exit 2; }
[ -f "$SERVE_SH" ] || { echo "!! 找不到 serve 脚本:$SERVE_SH"; exit 2; }
mkdir -p "$DSPARK_HS_DIR" 2>/dev/null

# ── 盘 ────────────────────────────────────────────────────────────────────────
free_gb() { df -BG --output=avail "$DSPARK_HS_DIR" 2>/dev/null | tail -1 | tr -dc '0-9'; }

# ── 已完成的行:扫一次目录,拿到 hs_<n>.safetensors 里的 n ────────────────────
# 77W 个文件时这一步要几秒,所以只在启动和每次重启后扫,块内不扫。
declare -A DONE
scan_done() {
  DONE=()
  local n cnt=0
  while read -r n; do DONE["$n"]=1; cnt=$((cnt + 1)); done < <(
    ls "$DSPARK_HS_DIR" 2>/dev/null | sed -n 's/^hs_\([0-9]\+\)\.safetensors$/\1/p')
  echo "$cnt"
}

chunk_done() {   # $1=起始行 —— 整块都在盘上才算做完
  local r end="$(( $1 + CHUNK ))"; [ "$end" -gt "$ROW_END" ] && end="$ROW_END"
  for ((r = $1; r < end; r++)); do [ -n "${DONE[$r]:-}" ] || return 1; done
  return 0
}

DONE_CNT=$(scan_done)
say "盘上已有 $DONE_CNT 个 HS 文件,剩余空间 $(free_gb) GB"

# ── 主循环 ────────────────────────────────────────────────────────────────────
T0=$SECONDS
ADDED=0 CRASHES=0 START_FAILS=0 SERVE_GEN=0
row=$ROW_START
serve_pid=""

start_serve() {
  say "清场后起服务(第 $((SERVE_GEN + 1)) 次)..."
  if ! cleanup_verified "起服务前"; then
    say "!! 清不干净,60s 后重试"; sleep 60; return 1
  fi
  SERVE_GEN=$((SERVE_GEN + 1))
  echo "===== serve generation $SERVE_GEN @ $(date) =====" >> "$LOG"
  DSA_OVERLAP="$DSA_OVERLAP" HS_DUMP=1 DSPARK_HS_DIR="$DSPARK_HS_DIR" \
    nohup bash "$SERVE_SH" >> "$LOG" 2>&1 &
  serve_pid=$!
  local t0=$SECONDS
  while [ $((SECONDS - t0)) -lt "$READY_TIMEOUT" ]; do
    serve_up && { say "服务 READY($(hms $((SECONDS - t0))))"; return 0; }
    if ! kill -0 "$serve_pid" 2>/dev/null; then
      say "!! 服务进程退出了,最早的错误:"
      local fl; fl=$(grep -nE 'ERROR|Traceback' "$LOG" | tail -1 | cut -d: -f1)
      [ -n "$fl" ] && sed -n "${fl},$((fl + 25))p" "$LOG" | cut -c1-180
      return 1
    fi
    if [ $(( (SECONDS - t0) % 120 )) -lt 10 ] && [ $((SECONDS - t0)) -ge 120 ]; then
      say "    ... 等 READY $(hms $((SECONDS - t0)))"
    fi
    sleep 10
  done
  say "!! 等满 $(hms "$READY_TIMEOUT") 仍未 READY"
  return 1
}

trap 'say "收到中断,清场后退出"; cleanup_verified "退出" >/dev/null 2>&1; exit 0' INT TERM

while [ "$row" -lt "$ROW_END" ]; do
  # 停止条件
  if [ "$MAX_ROWS" -gt 0 ] && [ "$ADDED" -ge "$MAX_ROWS" ]; then
    say "已达本次上限 $MAX_ROWS 行,停。"; break
  fi
  fg=$(free_gb); fg="${fg:-0}"
  if [ "$fg" -lt "$RESERVE_GB" ]; then
    say "!! 盘只剩 ${fg} GB(保留线 ${RESERVE_GB} GB),停。"; break
  fi
  if [ "$START_FAILS" -ge "$MAX_RESTARTS" ]; then
    say "!! 连续起不来 $START_FAILS 次,放弃。看 $LOG"; break
  fi

  if chunk_done "$row"; then row=$((row + CHUNK)); continue; fi

  if ! serve_up; then
    if ! start_serve; then START_FAILS=$((START_FAILS + 1)); continue; fi
    START_FAILS=0
    DONE_CNT=$(scan_done)          # 重启后重扫一次,崩溃那批可能写了一半
  fi

  end=$((row + CHUNK)); [ "$end" -gt "$ROW_END" ] && end="$ROW_END"
  n=$((end - row))
  # ★ id == row(训练侧按行号找文件);--no-collect 不再复制一份(否则占盘翻倍)
  ENDPOINT="$ENDPOINT" ARROW="$ARROW" HS_DIR="$DSPARK_HS_DIR" \
    python "$SCRIPT_DIR/dsv4_fire_hs_dumps.py" \
      --out /dev/null --n "$n" --concurrency "$CONC" \
      --id-base "$row" --start-row "$row" --no-collect >> "$LOG" 2>&1
  errs=$(sed -n 's/.*errors=\([0-9]*\).*/\1/p' "$LOG" | tail -1); errs="${errs:-0}"

  # 这一块实际落了多少文件(只认盘上的,不认返回码)
  got=0
  for ((r = row; r < end; r++)); do
    if [ -s "$DSPARK_HS_DIR/hs_$r.safetensors" ]; then DONE["$r"]=1; got=$((got + 1)); fi
  done
  ADDED=$((ADDED + got))

  el=$((SECONDS - T0)); rate=$(awk -v a="$ADDED" -v e="$el" 'BEGIN{printf "%.2f", a/(e>0?e:1)}')
  left=$((ROW_END - row - got))
  eta=$(awk -v l="$left" -v r="$rate" 'BEGIN{printf "%.1f", (r>0? l/r/3600 : -1)}')
  say "行 [$row,$end) → 落盘 $got/$n  errors=$errs | 累计 +$ADDED  $rate 行/s  剩 $left 行 ≈ ${eta}h  崩 $CRASHES 次  盘剩 ${fg}GB"

  if ! serve_up; then
    CRASHES=$((CRASHES + 1))
    say "★ 引擎死了(第 $CRASHES 次)—— 清场重启,从缺口接着打。不跳过这一块。"
    kf=$(find "${ASCEND_PROCESS_LOG_PATH:-$HOME/ascend/log}" -name '*.log' -newermt '-10 min' 2>/dev/null \
         | xargs -r grep -ohE 'fault kernel_name=[^ ,]*' 2>/dev/null \
         | sed 's/fault kernel_name=//; s/_[0-9a-f]\{16,\}.*//' | sort | uniq -c | tr '\n' ' ')
    [ -n "$kf" ] && say "  plog 故障算子:$kf"
    continue      # 不推进 row —— 没落盘的行下一轮会被 chunk_done 判为未完成并重打
  fi

  [ "$got" -eq "$n" ] && row="$end"   # 整块齐了才推进;缺了就原地重打
done

say "================================================================================"
say "结束:本次新增 $ADDED 行,崩溃 $CRASHES 次,起服务 $SERVE_GEN 次,用时 $(hms $((SECONDS - T0)))"
say "盘上现有 $(scan_done) 个 HS 文件,剩余 $(free_gb) GB"
say "serve 全量日志(含每次崩溃现场):$LOG"
cleanup_verified "收尾" >/dev/null 2>&1
say "机器已清场。"
