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
# ★ 2026-09-23:第一轮每行只有一个长度,「行身份」和「序列长度」完全混在一起 ——
#   短行翻 2%、长行翻 90%,但那也可能是那几行本身难。SEQLENS 把【同一批行】截到不同
#   长度重复测,才能把长度单独摘出来。
SEQLENS="${SEQLENS:-256 512 1024 2048}"
SEQLEN="${SEQLEN:-}"          # 兼容老用法:设了就只跑这一个长度
[ -n "$SEQLEN" ] && SEQLENS="$SEQLEN"
REPEAT="${REPEAT:-3}"
TOPK="${TOPK:-5}"
BGS="${BGS:-0 4}"
BG_LEN="${BG_LEN:-512}"
READY_TIMEOUT="${READY_TIMEOUT:-3600}"
DSA_OVERLAP="${DSA_OVERLAP:-0}"
# ★ 决定性的那一臂。位置 0–128 在 2048 长的行里翻 53%,在 262 长的行里只翻 2% —— 因果
#   模型里前缀的输出不可能依赖后面的 token,所以变的不是位置,是【这次 forward 里有多少
#   token】。MoE 的分组 GEMM / all-to-all 正是按整批 token 决定行为的那条路。
#   把 max_num_batched_tokens 压到 512,2048 的 prompt 会被切成 4 块喂进去;翻转率若塌回
#   ~2%,机制就锁定了,而且这是个我们本来就有的生产旋钮。
MBTS="${MBTS:-8192 512}"

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
echo "  规模        $N 行 × 重复 $REPEAT 次 × top-$TOPK"
echo "  长度阶梯    $SEQLENS        (同一批行截到不同长度 —— 把长度和行身份摘开)"
echo "  MAXBATCHTOK $MBTS           (每个值一次 serve 重启;压小 = 强制分块)"
echo "  背景负载    $BGS"
echo "================================================================================"
[ -n "$ARROW" ] && [ -f "$ARROW/dataset_info.json" ] || { echo "!! Arrow 不可用"; exit 2; }

if ! cleanup_verified "起跑前"; then
  echo "!! 清不干净,退出"; exit 1
fi

run_one_serve() {                       # $1 = MAXBATCHTOK;起服务并等 READY,失败返回 1
  local MBT="$1"
  say "起服务(MAXBATCHTOK=$MBT)..."
  SLOG="$OUT/serve_mbt${MBT}.log"
  ( DSA_OVERLAP="$DSA_OVERLAP" MAXBATCHTOK="$MBT" nohup bash "$SERVE_SH" > "$SLOG" 2>&1 & )
  local T0=$SECONDS
  while [ $((SECONDS - T0)) -lt "$READY_TIMEOUT" ]; do
    serve_up && break
    if ! pgrep -i -u "$USER" -f 'vllm|EngineCore' >/dev/null 2>&1 && [ $((SECONDS-T0)) -gt 120 ]; then
      say "!! 服务进程没了。日志末尾:"
      tail -40 "$SLOG" | cut -c1-220
      return 1
    fi
    sleep 10
  done
  serve_up || { say "!! 等满 ${READY_TIMEOUT}s 仍未 READY"; tail -40 "$SLOG" | cut -c1-220; return 1; }
  say "服务 READY($(hms $((SECONDS-T0))))"
  # 把决定「这组数怎么解释」的几行原样抄进报告 —— prefix cache 开着的话重复 prefill 会
  # 命中 KV,一致性是构造出来的;分块大小则直接决定每次 forward 里有多少 token。
  grep -aoE "enable_prefix_caching=[A-Za-z]+|Chunked prefill is enabled with max_num_batched_tokens=[0-9]+" \
    "$SLOG" | sort -u | head -4 | sed 's/^/    /' 
  grep -a "additional-config" "$SLOG" | tail -1 | sed 's/^/    /'
  return 0
}

# 抽一个数出来:$1=报告文件 $2=起始行的正则
pick() { awk -v pat="$2" '$0 ~ pat {getline; print; exit}' "$1" | grep -oE '[0-9.]+%' | head -1; }
pick_same() { awk -v pat="$2" '$0 ~ pat {print; exit}' "$1" | grep -oE '[0-9.]+%' | head -1; }

RES="$OUT/summary.tsv"; : > "$RES"
printf '配置\t翻转率\t笃定档(>2nat)\tresponse段语料一致\n' >> "$RES"

for MBT in $MBTS; do
  if ! cleanup_verified "MBT=$MBT 起跑前"; then
    say "!! 清不干净,跳过 MBT=$MBT"; continue
  fi
  if ! run_one_serve "$MBT"; then
    say "!! MBT=$MBT 起不来,跳过这一臂"
    cleanup_verified "收尾" >/dev/null 2>&1; continue
  fi
  for BG in $BGS; do
    for L in $SEQLENS; do
      TAG="mbt${MBT}_bg${BG}_len${L}"
      say ""
      say "===== $TAG ====="
      F="$OUT/report_${TAG}.txt"
      $PROBE_PY "$SCRIPT_DIR/prefill_noise_probe.py" \
        --endpoint "$ENDPOINT" --arrow "$ARROW" \
        --n "$N" --seq-len "$L" --repeat "$REPEAT" --topk "$TOPK" \
        --bg "$BG" --bg-len "$BG_LEN" \
        --csv "$OUT/positions_${TAG}.csv" \
        --label "$TAG" 2>&1 | tee "$F"
      printf '%s\t%s\t%s\t%s\n' "$TAG" \
        "$(pick "$F" '^A\. 本底噪声')" \
        "$(pick_same "$F" '笃定')" \
        "$(pick_same "$F" 'response 段')" >> "$RES"
      serve_up || { say "!! 服务死在 $TAG,这一臂剩下的跳过"; break 2; }
    done
  done
  cleanup_verified "收尾 MBT=$MBT" >/dev/null 2>&1
done

echo
echo "================================================================================"
echo "  汇总(逐条报告 $OUT/report_*.txt,逐位置明细 positions_*.csv)"
echo "================================================================================"
column -t -s "$(printf '\t')" "$RES" 2>/dev/null || cat "$RES"
cat <<'EOF'

怎么读
  ★ 同一批行、只改 --seq-len ⟹ 把「行身份」和「序列长度」摘开。翻转率随长度上升
    而位置桶内部平坦,就坐实了「变的是一次 forward 里有多少 token」,而不是位置。
  ★ MAXBATCHTOK=512 把 2048 的 prompt 切成 4 块喂进去。翻转率若塌回短行那个 ~2%,
    机制就锁定在【每次 forward 的 token 数】—— MoE 的分组/all-to-all 正是按整批
    token 决定行为的那条路,注意力不是(它逐位置、因果)。
    而且这是个我们本来就有的生产旋钮:要可复现的数,就把它压小。
  ★ 笃定档(margin>2 nat)翻转 >5% ⟹ 那不是数值抖动,别拿那组数去判语料。
EOF
