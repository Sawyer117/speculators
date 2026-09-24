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
#   ARROW=<hs_subset_arrow.py 抽出来的新 Arrow> bash hs_dump_daemon.sh   # 只存一部分行
#
# ★ 2026-09-25:默认把一次 forward 的 token 数限在 MAXBATCHTOK=512,并加质量闸。
#   HS 一旦预存就冻进语料了,算错了没有任何下游检查能发现。所以起服务后、以及每 GATE_EVERY
#   块,都用 prefill_noise_probe.py 把同一串长 token 重复 prefill 几次,只看高置信位置
#   (margin > 2 nat)的 argmax 翻不翻 —— 平局位置翻转是浮点抖动,无害;高置信位置翻了就是
#   算出了不同的分布。超过 GATE_MAX_CONF% 就停产。换配置(比如 ENABLE_EP=0 MAXBATCHTOK=8192)
#   也照样过这道闸。
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
DSPARK_HS_DIR="${DSPARK_HS_DIR:-}"   # 空 = /home/canada_group_folder/dataset/dsv4_hs_store/<Arrow 目录名>
CHUNK="${CHUNK:-256}"                # 一次 fire 打多少行
CONC="${CONC:-64}"                   # 并发(≤ serve 的 --max-num-seqs)
ROW_START="${ROW_START:-0}"
ROWS_FULL="${ROWS_FULL:-}"          # 空 = 读 Arrow 的行数
ROW_END="${ROW_END:-}"
MAX_ROWS="${MAX_ROWS:-0}"            # >0 = 本次最多【新增】这么多行(盘配额)
RESERVE_GB="${RESERVE_GB:-300}"      # dump 盘剩余低于此值就停,别把盘写满
READY_TIMEOUT="${READY_TIMEOUT:-3600}"
MAX_RESTARTS="${MAX_RESTARTS:-200}"  # 连续起不来这么多次就放弃(不是崩溃次数上限)
DSA_OVERLAP="${DSA_OVERLAP:-0}"
export MAXBATCHTOK="${MAXBATCHTOK:-512}"   # ★ 见文件头。serve 脚本读这个变量
GATE_MAX_CONF="${GATE_MAX_CONF:-1.0}"      # 高置信档(margin>2 nat)翻转率上限(%)
GATE_EVERY="${GATE_EVERY:-40}"             # 每多少块复查一次(0 = 只在起服务后查)
GATE_N="${GATE_N:-4}"                      # 闸用几行(都取长度 ≥ GATE_SEQ 的)
GATE_SEQ="${GATE_SEQ:-2048}"
GATE_BG="${GATE_BG:-4}"                    # 闸测量时的背景负载路数 —— 模拟 dump 时的批次拼装
GATE_LOG="${GATE_LOG:-$HOME/hs_daemon_gate.log}"
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

[ -n "${ARROW:-}" ] && [ -f "$ARROW/dataset_info.json" ] || { echo "!! Arrow 不可用:${ARROW:-<空>}"; exit 2; }
# ★ 每个 Arrow 一个目录:文件名 = 行号,换了 Arrow 行号的含义就变了,混放会把别的行当成已完成。
[ -n "$DSPARK_HS_DIR" ] || DSPARK_HS_DIR="/home/canada_group_folder/dataset/dsv4_hs_store/$(basename "$ARROW")"
if [ -z "$ROWS_FULL" ]; then
  ROWS_FULL=$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -c "
from datasets import load_from_disk; d = load_from_disk('$ARROW')
print(d.num_rows if hasattr(d, 'num_rows') else d[next(iter(d))].num_rows)" 2>/dev/null)
  [ -n "$ROWS_FULL" ] || { echo "!! 读不出 $ARROW 的行数(用 ROWS_FULL= 指定)"; exit 2; }
fi
ROW_END="${ROW_END:-$ROWS_FULL}"

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
echo "  serve        $SERVE_SH   (DSA_OVERLAP=$DSA_OVERLAP  MAXBATCHTOK=$MAXBATCHTOK  ENABLE_EP=${ENABLE_EP:-<serve 默认 1>})"
echo "  质量闸       笃定档翻转 ≤ ${GATE_MAX_CONF}%   每 $GATE_EVERY 块复查   $GATE_N 行 × $GATE_SEQ token  bg=$GATE_BG   → $GATE_LOG"
echo "  serve 日志   $LOG        (每次重启【追加】,不覆盖 —— 崩溃现场要留住)"
echo "================================================================================"
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

# ── 目录里已有的文件真是这个 Arrow 的吗?抽 3 个比 token_ids ─────────────────────
# 文件名只是个行号。别的 Arrow、别的 pilot 留下的同名文件会被当成「已完成」永远跳过,
# 训练时才发现 token 对不上。开工前花几秒挡掉。
if [ "$DONE_CNT" -gt 0 ]; then
  if ! TORCH_DEVICE_BACKEND_AUTOLOAD=0 python - "$DSPARK_HS_DIR" "$ARROW" <<'PYEOF'
import os, random, re, sys
from datasets import load_from_disk
from safetensors import safe_open
d, arrow = sys.argv[1], sys.argv[2]
ds = load_from_disk(arrow)
ds = (ds if hasattr(ds, "num_rows") else ds[next(iter(ds))]).with_format(None)
rows = [int(m.group(1)) for f in os.listdir(d) if (m := re.fullmatch(r"hs_(\d+)\.safetensors", f))]
bad = 0
for r in random.Random(0).sample(rows, min(3, len(rows))):
    if r >= len(ds):
        print(f"   hs_{r}: 行号超出这个 Arrow({len(ds)} 行)"); bad += 1; continue
    with safe_open(os.path.join(d, f"hs_{r}.safetensors"), "pt") as fh:
        tok = fh.get_tensor("token_ids").tolist()
    if tok != list(ds[r]["input_ids"]):
        print(f"   hs_{r}: token_ids 和 Arrow 第 {r} 行对不上"); bad += 1
sys.exit(1 if bad else 0)
PYEOF
  then
    say "!! $DSPARK_HS_DIR 里的文件不属于 $ARROW。换一个空目录(DSPARK_HS_DIR=),别混放。"
    exit 2
  fi
  say "抽查已有文件:token_ids 与 Arrow 一致 ✅"
fi

# ── 质量闸:这台服务现在算得对不对 ────────────────────────────────────────────
# 0 = PASS;非 0 = 不许产。判不了(样本不够、探针报错)也算不许 —— 冻进语料的数据宁缺勿错。
gate() {
  local tag="$1" rc start
  start=$(( (RANDOM * 32768 + RANDOM) % (ROWS_FULL > 1 ? ROWS_FULL : 1) ))
  echo "===== gate [$tag] @ $(date)  start_row=$start =====" >> "$GATE_LOG"
  # 先写到单独的文件再追加:只从【这一次】的输出里取判决,不会读到上一次的 GATE 行。
  ENDPOINT="$ENDPOINT" TORCH_DEVICE_BACKEND_AUTOLOAD=0 python "$SCRIPT_DIR/prefill_noise_probe.py" \
    --arrow "$ARROW" --n "$GATE_N" --seq-len "$GATE_SEQ" --min-len "$GATE_SEQ" \
    --start-row "$start" --repeat 3 --bg "$GATE_BG" --gate-max-conf "$GATE_MAX_CONF" \
    --label "hs_dump_daemon $tag" > "$GATE_LOG.cur" 2>&1
  rc=$?
  cat "$GATE_LOG.cur" >> "$GATE_LOG"
  local line; line=$(grep -E '^GATE ' "$GATE_LOG.cur" | tail -1)
  say "质量闸 [$tag]:${line:-没出判决(rc=$rc),见 $GATE_LOG}"
  return $rc
}

# ── 主循环 ────────────────────────────────────────────────────────────────────
T0=$SECONDS
ADDED=0 CRASHES=0 START_FAILS=0 SERVE_GEN=0 CHUNKS_SINCE_GATE=0
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
    if ! gate "serve#$SERVE_GEN" && ! { say "闸没过,60s 后换一批行再测一次"; sleep 60; gate "serve#$SERVE_GEN 复测"; }; then
      say "!! 质量闸不过 —— 这台服务现在产的 HS 不能要。停产。明细:$GATE_LOG"
      break
    fi
    CHUNKS_SINCE_GATE=0
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

  CHUNKS_SINCE_GATE=$((CHUNKS_SINCE_GATE + 1))
  if [ "$GATE_EVERY" -gt 0 ] && [ "$CHUNKS_SINCE_GATE" -ge "$GATE_EVERY" ]; then
    CHUNKS_SINCE_GATE=0
    if ! gate "row $row" && ! { sleep 60; gate "row $row 复测"; }; then
      say "!! 质量闸不过 —— 上一次过闸之后产的 $GATE_EVERY 块($((GATE_EVERY * CHUNK)) 行)都要当可疑。停产。明细:$GATE_LOG"
      break
    fi
  fi
done

say "================================================================================"
say "结束:本次新增 $ADDED 行,崩溃 $CRASHES 次,起服务 $SERVE_GEN 次,用时 $(hms $((SECONDS - T0)))"
say "盘上现有 $(scan_done) 个 HS 文件,剩余 $(free_gb) GB"
say "serve 全量日志(含每次崩溃现场):$LOG"
cleanup_verified "收尾" >/dev/null 2>&1
say "机器已清场。"
