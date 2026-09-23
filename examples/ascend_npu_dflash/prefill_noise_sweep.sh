#!/usr/bin/env bash
# 纯 prefill 可复现性 —— 一条命令:清场 → 起服务 → 量本底噪声 → 收工 → 出表。
#
# WHY
# ---
# `determinism_sweep.sh` 量的是 prefill + 32 步 decode,一次翻转会被 decode 放大成一整段,
# 而且读数取的是 9 次比较里最差的那次(min 统计,高方差:同一配置两次测量 2048 那格分别
# 是 6 和 20)。它能比大小,给不出物理量。
#
# 这个脚本量的是训推两端真正共用的东西:**同一串 token,只 prefill,重复 R 次,比数值**。
#   * 训练侧 —— HS 是整条语料一次 prefill 抓的;
#   * 部署侧 —— 投机验证是 `[上次接受的 token + γ 个草稿]` 一次 forward,prefill 形状。
#   decode 只出现在语料生成那一次(176 老栈),那份语料现在是固定输入,不在回路里。
#
# 默认跑两臂,**这两臂的对比才是重点**:
#   bg0  并发 1,批次组成固定
#   bg4  测量期间后台打 4 路无关请求,批次组成每步都变
# 两臂 A 差不多 ⟹ 批次组成不是机制;bg4 明显更差 ⟹ 输出取决于同批里还有谁,
# 那是正确性 bug,而且同时解释「上下文越长越差」和「隔天数字不一样」。
#
# USAGE
#   CONDA_ENV=dspark-dsv4-serving CANN_ENV=/home/a00652497/920env_npu.sh \
#   ARROW=/home/canada_group_folder/dataset/arrow_0730_77w_dedup \
#     nohup bash prefill_noise_sweep.sh > ~/prefill_noise.log 2>&1 &
#
#   VLLM_ASCEND_ENABLE_FUSED_MC2=0 ... bash prefill_noise_sweep.sh    # 换 MoE 路径再测一遍
#   BGS="0" SEQLEN=512 N=4 bash prefill_noise_sweep.sh                # 想快就砍这几个
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/npu_cleanup_lib.sh"

export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"

PORT="${PORT:-7000}"
ENDPOINT="http://localhost:$PORT/v1"
SERVE_SH="${SERVE_SH:-$SCRIPT_DIR/serve_dsv4_a3_singlenode_specmethod.sh}"
OUT="${OUT:-$HOME/prefill_noise}"
N="${N:-8}"
SEQLEN="${SEQLEN:-2048}"
REPEAT="${REPEAT:-3}"
TOPK="${TOPK:-5}"
BGS="${BGS:-0 4}"
BG_LEN="${BG_LEN:-512}"
READY_TIMEOUT="${READY_TIMEOUT:-3600}"
DSA_OVERLAP="${DSA_OVERLAP:-0}"

if [ -z "${ARROW:-}" ]; then
  for d in /home/canada_group_folder/dataset/arrow* \
           /share/canada_group_folder/dataset/*/arrow* \
           /share/canada_group_folder/dataset/arrow*; do
    [ -f "$d/dataset_info.json" ] && ARROW="$d" && break
  done
fi

# 探针要跑在装了 datasets/openai 的 env 里。裸 `python` 在未激活 conda 的 shell 上
# 要么不存在(这台机只有 /usr/bin/python3),要么缺依赖 —— 两种都是整轮作废。
PROBE_PY="${PROBE_PY:-}"
if [ -z "$PROBE_PY" ]; then
  if [ -n "${CONDA_ENV:-}" ] && command -v conda >/dev/null 2>&1; then
    PROBE_PY="conda run --no-capture-output -n ${CONDA_ENV} python"
  else
    PROBE_PY="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo python)"
  fi
fi

hms() { printf '%02d:%02d:%02d' $(($1/3600)) $((($1%3600)/60)) $(($1%60)); }
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
serve_up() { curl -sf --noproxy '*' "$ENDPOINT/models" >/dev/null 2>&1; }

LOCK="${LOCK:-$HOME/.prefill_noise_sweep.lock}"
exec 9>"$LOCK" || { echo "!! 打不开锁 $LOCK"; exit 2; }
command -v flock >/dev/null 2>&1 && { flock -n 9 || { echo "!! 已有实例在跑"; exit 2; }; }

mkdir -p "$OUT"
echo "================================================================================"
echo "  纯 prefill 可复现性扫描"
echo "  ARROW       ${ARROW:-<找不到>}"
echo "  serve       $SERVE_SH   (CONDA_ENV=${CONDA_ENV:-<脚本默认>})"
echo "  探针解释器  $PROBE_PY"
echo "  FUSED_MC2   ${VLLM_ASCEND_ENABLE_FUSED_MC2:-<未设,serve 默认 1 = MegaMoe>}"
echo "  DSA_OVERLAP $DSA_OVERLAP"
echo "  规模        $N 行 × 前 $SEQLEN token × 重复 $REPEAT 次   top-$TOPK   臂: $BGS"
echo "================================================================================"
[ -n "$ARROW" ] && [ -f "$ARROW/dataset_info.json" ] || { echo "!! Arrow 不可用"; exit 2; }

if ! cleanup_verified "起跑前"; then
  echo "!! 清不干净,退出"; exit 1
fi

say "起服务 ..."
SLOG="$OUT/serve.log"
( DSA_OVERLAP="$DSA_OVERLAP" nohup bash "$SERVE_SH" > "$SLOG" 2>&1 & )
T0=$SECONDS
while [ $((SECONDS - T0)) -lt "$READY_TIMEOUT" ]; do
  serve_up && break
  if ! pgrep -i -u "$USER" -f 'vllm|EngineCore' >/dev/null 2>&1 && [ $((SECONDS-T0)) -gt 120 ]; then
    say "!! 服务进程没了。日志末尾:"; tail -40 "$SLOG" | cut -c1-220
    cleanup_verified "收尾" >/dev/null 2>&1; exit 1
  fi
  sleep 10
done
serve_up || { say "!! 等满 ${READY_TIMEOUT}s 仍未 READY"; tail -40 "$SLOG" | cut -c1-220
              cleanup_verified "收尾" >/dev/null 2>&1; exit 1; }
say "服务 READY($(hms $((SECONDS-T0))))"

# ★ prefix cache 必须是关的,否则第二次 prefill 直接命中 KV,一致性是构造出来的。
if grep -aq -- "--no-enable-prefix-caching" "$SLOG"; then
  say "已确认 serve 带 --no-enable-prefix-caching"
else
  say "⚠⚠ 日志里没看到 --no-enable-prefix-caching —— 如果 prefix cache 是开的,"
  say "    下面所有「完全一致」都是缓存命中,不是可复现性。先查清楚再信这组数。"
fi
grep -a "additional-config" "$SLOG" | tail -1

for BG in $BGS; do
  say "===== 臂 bg=$BG ====="
  $PROBE_PY "$SCRIPT_DIR/prefill_noise_probe.py" \
    --endpoint "$ENDPOINT" --arrow "$ARROW" \
    --n "$N" --seq-len "$SEQLEN" --repeat "$REPEAT" --topk "$TOPK" \
    --bg "$BG" --bg-len "$BG_LEN" \
    --csv "$OUT/positions_bg${BG}.csv" \
    --label "bg=$BG seq=$SEQLEN" 2>&1 | tee "$OUT/report_bg${BG}.txt"
  serve_up || { say "!! 服务在 bg=$BG 之后死了,后面的臂跳过"; break; }
done

cleanup_verified "收尾" >/dev/null 2>&1

echo
echo "================================================================================"
echo "  汇总(完整报告在 $OUT/report_bg*.txt,逐位置明细在 positions_bg*.csv)"
echo "================================================================================"
for BG in $BGS; do
  f="$OUT/report_bg${BG}.txt"
  [ -s "$f" ] || continue
  echo "--- bg=$BG ---"
  sed -n '/^A\. 本底噪声/,/^$/p;/^D\. 语料一致率/,/^$/p;/^★ 判读/,/^$/p' "$f"
done
