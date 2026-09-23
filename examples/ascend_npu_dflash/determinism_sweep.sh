#!/usr/bin/env bash
# 温度 0 不可复现 —— 把还没排除的变量一次扫完,出一张表。
#
# WHY
# ---
# 2026-09-20,A3 186,温度 0 / 并发 1 / prefix cache 关 / 同一 prompt 连打 3 次,
# 量**完全一致前缀**(贪心是混沌的,第一个 token 一分叉后面全不同,所以逐 token 一致率
# 是误导性指标,该看第一次分叉在哪):
#
#     prompt-len            256    512   1024   2048
#     新栈 4ce367a+0.27.1   32/32   2/32  0/32   0/32     ← 断崖
#     老栈 386530d+0.23.0   19/32  12/32  8/32   6/32     ← 平滑退化
#
# **两套栈都不确定** ⟹ 不是 vllm-ascend 的 pin。而两套栈共用的东西还有:
#
#   * `--async-scheduling`  —— 华为官方 DeepSeek-V4-Flash 六个配方一个都没用。
#     调度跑在设备执行前面,而这个 pin 的 dsa_v1.py 把 SAS 的每核任务分配写进一个
#     **常驻共享 buffer**,下一步的 host 写入可能赶在上一步 kernel 读完之前。
#   * **DP2** —— 并发 1 时另一个 replica 在跑 **dummy batch**,而 MoE 的 all-to-all 是
#     **跨 DP** 的。dummy 的形状每步不同 → all-to-all 的 payload 不同 → 归约结果不同。
#     这能同时解释「同输入不同输出」和「上下文越长越明显」(token 越多,跨 DP 归约越多)。
#   * CANN 9.2.0-beta1 —— ns=5 那条线的栈是 vLLM 0.23.0 + `386530d12` + **CANN 9.0.0**。
#     ⚠ 2026-09-22 更正:我原来在这里写「这台机当年验证 gsm8k 96.59% 用的是 9.0.0」——
#     **没有任何记录把 96.59% 归到 186**(它指向 176 / rollout 机)。186 是第二台 A3 评测机,
#     它自己被记录的成绩是 0.28.0+main 的 92.04%(已否)。⟹ 在 186 上装 9.0.0 不是
#     「恢复已验证的好配置」,而是一个全新组合 —— 186 从来没有过一份已验证干净的配置。
#
# 每个臂 = 已核实清场 → 起服务 → 在若干长度上量确定性 → 收工。全程无人值守。
#
# USAGE
# -----
#   nohup bash determinism_sweep.sh > ~/det_sweep.log 2>&1 &
#   ARMS="base noasync dp1" LENS="256 1024 2048" bash determinism_sweep.sh
#   CONDA_ENV=dsv4-oldstack CANN_ENV=/home/a00652497/920env_npu.sh bash determinism_sweep.sh
#
# 臂
#   base     现状(对照)
#   noasync  ASYNC_SCHED=0    —— 去掉 --async-scheduling
#   dp1      DP=1 TP=16       —— 去掉 DP(EP 仍是 16,显存只会更宽松:dense 切 16 份)
#   both     两个一起去掉
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/npu_cleanup_lib.sh"

export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"

PORT="${PORT:-7000}"
ENDPOINT="http://localhost:$PORT/v1"
SERVE_SH="${SERVE_SH:-$SCRIPT_DIR/serve_dsv4_a3_singlenode_specmethod.sh}"
ARMS="${ARMS:-base noasync dp1}"
LENS="${LENS:-256 512 1024 2048}"
NROWS="${NROWS:-3}"
REPEAT="${REPEAT:-3}"
GEN="${GEN:-32}"
READY_TIMEOUT="${READY_TIMEOUT:-3600}"
OUT="${OUT:-$HOME/det_sweep}"
ID_BASE="${ID_BASE:-980000}"
DSA_OVERLAP="${DSA_OVERLAP:-0}"

# ★ 2026-09-23:探针原来是裸 `python`。它得跑在【装了 datasets/pyarrow/requests 的那个
#   env】里,而不是系统 python:从未激活 conda 的 shell 里启动时,`python` 根本不存在
#   (这台机只有 /usr/bin/python3),或者存在但缺依赖 —— 两种都是整臂作废。
PROBE_PY="${PROBE_PY:-}"
if [ -z "$PROBE_PY" ]; then
  if [ -n "${CONDA_ENV:-}" ] && command -v conda >/dev/null 2>&1; then
    PROBE_PY="conda run --no-capture-output -n ${CONDA_ENV} python"
  else
    PROBE_PY="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo python)"
  fi
fi

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

LOCK="${LOCK:-$HOME/.determinism_sweep.lock}"
exec 9>"$LOCK" || { echo "!! 打不开锁 $LOCK"; exit 2; }
command -v flock >/dev/null 2>&1 && { flock -n 9 || { echo "!! 已有实例在跑"; exit 2; }; }

mkdir -p "$OUT"; RES="$OUT/results.tsv"; : > "$RES"
echo "================================================================================"
echo "  温度 0 确定性扫描   臂: $ARMS   长度: $LENS"
echo "  ARROW      ${ARROW:-<找不到>}"
echo "  serve      $SERVE_SH   (CONDA_ENV=${CONDA_ENV:-<脚本默认>})"
echo "  探针解释器 $PROBE_PY"
# 两份日志被并排粘进聊天时,OUT 目录名往往丢了。把决定 MoE 走哪条路的开关直接写进
# 表头 —— 「这个数是哪个配置跑出来的」在这个项目里已经错过不止一次。
echo "  FUSED_MC2  ${VLLM_ASCEND_ENABLE_FUSED_MC2:-<未设,serve 脚本默认 1 = MegaMoe 开>}"
echo "  DSA_OVERLAP $DSA_OVERLAP"
echo "  每点       $NROWS 行 × $REPEAT 次重复,各生成 $GEN 个 token"
echo "  ★ 指标 = 完全一致前缀(贪心混沌,逐 token 一致率是误导性指标)"
echo "================================================================================"
[ -n "$ARROW" ] && [ -f "$ARROW/dataset_info.json" ] || { echo "!! Arrow 不可用"; exit 2; }

# 服务起不来时把【有用的】那几行捞出来。
# ★ 2026-09-22:第一版是「从第一个 ERROR 往下打 26 行 | cut -c1-180」。两处都错:
#   1. Python 的异常在 traceback 的【末尾】,不在开头 —— 26 行经常正好停在它前面,
#      于是日志里全是栈帧、没有原因。dp1 臂就是这么报的,什么都没说明。
#   2. vLLM 每行带 `(Worker_TP13_EP13 pid=…) ERROR 09-22 03:20:33 [multiproc_executor.py:912] `
#      的前缀,占掉 ~100 列,`cut -c1-180` 之后只剩 80 列内容,路径都是断的。
#   现在:剥掉前缀,开头给 12 行上下文,再单独把末尾的异常行捞出来。
dump_first_error() {
  local slog="$1" fl
  fl=$(grep -naE 'ERROR|Traceback' "$slog" 2>/dev/null | head -1 | cut -d: -f1)
  if [ -z "$fl" ]; then
    say "    日志里没有 ERROR/Traceback —— 打末尾 25 行:"
    tail -25 "$slog" | cut -c1-220
    return
  fi
  say "    最早的错误(第 $fl 行起,已剥掉 worker 行前缀):"
  sed -n "${fl},$((fl + 11))p" "$slog" | sed -E 's/^\([^)]*\) *//; s/^(ERROR|WARNING|INFO) [0-9:. -]+\[[^]]*\] *//' | cut -c1-220
  say "    ★ 异常行(traceback 的末尾才是原因):"
  grep -aE '^\S*(Error|Exception)\b|Error:|Exception:|out of memory|OOM|CANN|EZ[0-9]|aclnn' "$slog" \
    | sed -E 's/^\([^)]*\) *//; s/^(ERROR|WARNING|INFO) [0-9:. -]+\[[^]]*\] *//' \
    | grep -avE '^(File |Traceback|  )' | tail -6 | cut -c1-260
  say "    全量日志:$slog"
}

RUN=0
run_arm() {
  local arm="$1"; shift
  local slog="$OUT/serve_$arm.log"
  say "===== 臂 $arm :$* ====="
  if ! cleanup_verified "臂 $arm 起跑前"; then
    say "!! 清不干净,这一臂作废"; printf '%s\t清场失败\t-\n' "$arm" >> "$RES"; return
  fi
  # ⚠ `export "$@"` 在【没有参数】时(base 臂就是)不是空操作 —— 它是「打印所有导出变量」。
  #   2026-09-22 实测:整个环境被 dump 进了 ~/det_sweep.log,其中包括
  #   `HTTPS_PROXY=http://<user>:<pass>@...` —— 代理密码明文落盘。必须先判非空。
  #   (`redact_log.py` 现在也会擦这类凭据,但别指望脱敏兜底,先别写出去。)
  ( [ $# -gt 0 ] && export "$@"
    DSA_OVERLAP="$DSA_OVERLAP" nohup bash "$SERVE_SH" > "$slog" 2>&1 & )
  local t0=$SECONDS
  while [ $((SECONDS - t0)) -lt "$READY_TIMEOUT" ]; do
    serve_up && break
    if ! pgrep -i -u "$USER" -f 'vllm|EngineCore' >/dev/null 2>&1 && [ $((SECONDS-t0)) -gt 120 ]; then
      say "!! 服务进程没了。"
      dump_first_error "$slog"
      printf '%s\t起不来\t-\n' "$arm" >> "$RES"; cleanup_verified "收尾" >/dev/null 2>&1; return
    fi
    [ $(( (SECONDS-t0) % 120 )) -lt 10 ] && [ $((SECONDS-t0)) -ge 120 ] && say "    ... 等 READY $(hms $((SECONDS-t0)))"
    sleep 10
  done
  serve_up || { say "!! 等满 $READY_TIMEOUT s 仍未 READY"; dump_first_error "$slog"
                printf '%s\t起不来\t-\n' "$arm" >> "$RES"
                cleanup_verified "收尾" >/dev/null 2>&1; return; }
  say "服务 READY($(hms $((SECONDS-t0))))"

  local row="$arm"
  for L in $LENS; do
    RUN=$((RUN + 1))
    local o
    # ★ TORCH_DEVICE_BACKEND_AUTOLOAD=0 —— 探针是个 HTTP 客户端 + tokenizer,不需要 NPU。
    #   不关的话 `import torch` 会自动去加载 torch_npu,而这个进程里没 source CANN,直接
    #   `Failed to load the backend extension: torch_npu`,四个长度全部取不到数(2026-09-23 实测)。
    #   而且就算能加载也不该加载:探针占卡会和正在测的 serve 抢资源。
    o=$(ENDPOINT="$ENDPOINT" ARROW="$ARROW" TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
        $PROBE_PY "$SCRIPT_DIR/corpus_provenance_check.py" \
          --n "$NROWS" --gen "$GEN" --repeat "$REPEAT" --prompt-len "$L" \
          --id-base $((ID_BASE + RUN * 1000)) 2>&1)
    # 取「完全一致前缀最短」——贪心下这才是有意义的指标
    local pre; pre=$(echo "$o" | sed -n 's/.*完全一致前缀最短 \([0-9]*\).*/\1/p' | tail -1)
    local tok; tok=$(echo "$o" | sed -n 's/.*逐 token 一致 \([0-9.]*\)%.*/\1/p' | tail -1)
    [ -n "$pre" ] || { pre="?"; say "    L=$L 没取到数,原始输出:"; echo "$o" | tail -5; }
    say "    L=$L  完全一致前缀 $pre/$GEN   (逐token ${tok:-?}%)"
    row="$row\t$L:$pre/$GEN"
    if ! serve_up; then say "    ★ 服务在 L=$L 时死了,这一臂后面的点跳过"; break; fi
  done
  printf '%b\n' "$row" >> "$RES"
  cleanup_verified "臂 $arm 收尾" >/dev/null 2>&1
}

T0=$SECONDS
for a in $ARMS; do
  case "$a" in
    base)    run_arm base ;;
    noasync) run_arm noasync ASYNC_SCHED=0 ;;
    dp1)     run_arm dp1 DP=1 TP=16 ;;
    both)    run_arm both ASYNC_SCHED=0 DP=1 TP=16 ;;
    *) echo "!! 未知的臂:$a(只认 base noasync dp1 both)" ;;
  esac
done

echo
echo "================================================================================"
echo "  完全一致前缀(越接近 $GEN/$GEN 越好;$GEN/$GEN = 完全可复现)   总用时 $(hms $((SECONDS-T0)))"
echo "================================================================================"
cat "$RES"
cat <<'EOF'

怎么读
  某一臂在所有长度上都 32/32  ⟹ 那个变量就是根因,而且立刻有生产解。
  base 也 32/32               ⟹ 这一轮没复现(它是概率性的),加大 --n / --repeat 重跑,
                                 别把「这次没撞上」当成「修好了」。
  全都不是                    ⟹ 剩下 CANN 9.2.0-beta1(这台当年验证用的是 9.0.0,
                                 而 900env_npu.sh 现在已经不在这台机上了)。换它要重编
                                 算子、可能连 torch_npu 一起,那是另一个量级的工作。
EOF
