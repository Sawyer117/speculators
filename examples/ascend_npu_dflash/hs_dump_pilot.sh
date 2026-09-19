#!/usr/bin/env bash
# 预存 HS 的可行性 pilot:打 N 行真 HS,量【速度 / 体积 / 压缩率】三个数,一次出结论。
#
# WHY. 「HS 能不能预存」决定一件大事:能预存,两台 serve 机就腾出来参训,DP 8→24 卡,
# 每个 epoch 的墙钟缩到 1/3(收益来自机器变多,不是 HS 变快 —— 训练日志说 fetch_frac
# 中位 0.010)。而这个判断一直卡在三个没量过的数上。这个脚本一次把三个都量了:
#
#   速度    打 N 行计时 → 行/s → 全量 dump 一次要几小时/几天
#   体积    从真文件的 safetensors 头读 shape(精确),不靠和 Arrow 交叉推算
#   压缩率  真文件上跑 gzip / 字节平面拆分,并量解压吞吐(在线 untar 的前提)
#
# ⚠ 这个 pilot 只量【字节和秒】,不验 HS 的【值】对不对。MRV2 的 HS dumper 从没上机
#   验过 —— 正式全量 dump 之前必须逐张量比对旧的 hs_*.safetensors(同 prompt、同
#   aux 层 [40,41,42]、同 pre/post-norm 约定、在 flashcomm all-gather 之后)。
#   拿几 TB 去存一批错的条件输入,是本项目已经踩过两次的坑。
#
# 前置:一台 HS-dump serve 活着。分支必须是 feat/dsv4-dumpers-mrv2 —— 它同时带
#   HS dumper 和 mega-MoE 的 OOM 修复;feat/dsv4-hs-dumper-mrv2 缺后者,权重加载就 OOM。
#
#   git -C <vllm-ascend> fetch fork && git -C <vllm-ascend> checkout feat/dsv4-dumpers-mrv2
#   HS_DUMP=1 DSPARK_HS_DIR=<dump 目录> \
#     nohup bash examples/ascend_npu_dflash/serve_dsv4_a3_singlenode_specmethod.sh \
#     > ~/serve_hsdump.log 2>&1 &
#   (HS_DUMP=1 与 DRAFT 互斥 —— HS 生产端只服务 target。)
#
# USAGE
#   ARROW=<arrow dir> HS_DIR=<serve 的 DSPARK_HS_DIR> bash hs_dump_pilot.sh
#   N=512 CONCURRENCY=64 OUT=~/hs_pilot bash hs_dump_pilot.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ⚠ 一旦 source 过 portproxy_remote.sh,http_proxy 就会把 localhost 也劫走。脚本自己的
#   curl 带了 --noproxy '*' 所以探活能过,但 dsv4_fire_hs_dumps.py 用的是 openai 客户端,
#   它认 http_proxy —— 于是请求被送到公司代理,回来一张 HTML 错误页,报
#   openai.InternalServerError: <!doctype html>...HIS Proxy Notification...
#   eval_blk15_drafts.sh 里一直有这行,这个脚本漏了。
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"

PORT="${PORT:-7000}"
ENDPOINT="${ENDPOINT:-http://localhost:$PORT/v1}"
N="${N:-256}"
CONCURRENCY="${CONCURRENCY:-64}"
# ⚠ 900000 > 772,684 行 —— 落在数据集之外,训练进程不会把这批当成自己的 HS 删掉。
#   正式全量 dump 时【不能】用偏移:trainer 是按 Arrow 行号找文件的。
ID_BASE="${ID_BASE:-900000}"
OUT="${OUT:-$HOME/hs_pilot}"
ROWS_FULL="${ROWS_FULL:-772684}"

# Arrow / HS 目录:给了就用,没给就在已知位置里找
if [ -z "${ARROW:-}" ]; then
  for d in /home/canada_group_folder/dataset/arrow* \
           /share/canada_group_folder/dataset/*/arrow* \
           /share/canada_group_folder/dataset/arrow*; do
    [ -f "$d/dataset_info.json" ] && ARROW="$d" && break
  done
fi
HS_DIR="${HS_DIR:-/home/canada_group_folder/dataset/dsv4_hs_dump}"

echo "================================================================================"
echo "  ENDPOINT   $ENDPOINT"
echo "  ARROW      ${ARROW:-<找不到,用 ARROW= 指定>}"
echo "  HS_DIR     $HS_DIR        (serve 的 DSPARK_HS_DIR,文件先落在这)"
echo "  OUT        $OUT           (收到这里)"
echo "  N          $N   并发 $CONCURRENCY   id-base $ID_BASE"
echo "================================================================================"
# 存在性也要查:只查非空的话,路径写错会一路跑到 fire 工具里才报,浪费一次起服务的等待
if [ -z "$ARROW" ] || [ ! -f "$ARROW/dataset_info.json" ]; then
  echo "!! Arrow 数据集不可用:${ARROW:-<空>}(要有 dataset_info.json)"; exit 2
fi

if ! curl -sf --noproxy '*' "$ENDPOINT/models" >/dev/null 2>&1; then
  echo "!! $ENDPOINT 没响应 —— HS-dump serve 没起来。看脚本顶部的起服务命令。"
  exit 2
fi
echo ">>> serve 在线"
mkdir -p "$OUT"

# ── 1) 打 N 行,计时 ──────────────────────────────────────────────────────────
echo; echo ">>> [1/3] 打 $N 行 ..."
T0=$SECONDS
ENDPOINT="$ENDPOINT" ARROW="$ARROW" HS_DIR="$HS_DIR" \
  python "$SCRIPT_DIR/dsv4_fire_hs_dumps.py" \
    --out "$OUT" --n "$N" --concurrency "$CONCURRENCY" --id-base "$ID_BASE" \
    2>&1 | tee "$OUT/.fire.log" || exit 1
ELAPSED=$((SECONDS - T0))
GOT=$(ls "$OUT"/hs_*.safetensors 2>/dev/null | wc -l)
[ "$GOT" -gt 0 ] || { echo "!! 一个文件都没收到 —— serve 的 DSPARK_HS_DUMP 开了吗?"; exit 1; }

# ★ 速率只有在【几乎没有错误】时才有意义。2026-09-19 实测教训:256 条里 236 条 500、
#   serve 中途死掉,脚本照样打出「0.11 行/s → 全量 81.8 天」,而那量的是崩溃过程不是吞吐。
#   一个看起来精确的错数,比没有数更糟 —— 它会被拿去做决策。
ERRS=$(sed -n 's/.*errors=\([0-9]*\).*/\1/p' "$OUT/.fire.log" | tail -1)
ERRS="${ERRS:-0}"
if [ "$ERRS" -gt $((N / 20)) ] || [ "$GOT" -lt $((N / 2)) ]; then
  echo
  echo "################################################################################"
  echo "!! 速率【不予报告】:$N 条里 $ERRS 条报错,只收到 $GOT 个文件。"
  echo "   这种情况下计时量的是崩溃过程,不是吞吐。先修崩溃再谈速度。"
  echo "   体积和压缩率不受影响(它们只看文件内容),下面照常给。"
  echo "################################################################################"
  SKIP_RATE=1
fi

if [ -z "${SKIP_RATE:-}" ]; then
echo
echo "================================================================================"
printf ">>> 速度:%s 行 / %s 秒 = %s 行/s\n" "$GOT" "$ELAPSED" \
  "$(awk -v g="$GOT" -v e="$ELAPSED" 'BEGIN{printf "%.2f", g/(e>0?e:1)}')"
awk -v g="$GOT" -v e="$ELAPSED" -v R="$ROWS_FULL" 'BEGIN{
  r = g/(e>0?e:1);
  printf ">>> 按此速率 dump 一次:  全量 %d 行 = %.1f 小时 (%.1f 天)\n", R, R/r/3600, R/r/86400;
  for (n = 300000; n >= 100000; n -= 100000)
    printf "                         %7d 行 = %.1f 小时\n", n, n/r/3600;
}'
echo "================================================================================"
fi

# ── 2) 体积 ───────────────────────────────────────────────────────────────────
echo; echo ">>> [2/3] 体积"
python "$SCRIPT_DIR/hs_capacity_probe.py" --arrow "$ARROW" --hs-dir "$OUT" || true

# ── 3) 压缩率 ─────────────────────────────────────────────────────────────────
echo; echo ">>> [3/3] 压缩率(真文件)"
python "$SCRIPT_DIR/hs_compress_bench.py" --hs-dir "$OUT" || true

echo
echo ">>> pilot 文件留在 $OUT($(du -sh "$OUT" 2>/dev/null | cut -f1))—— 量完可以删"
echo ">>> ⚠ 正式 dump 前先验 HS 的【值】:同 prompt 与旧 hs_*.safetensors 逐张量比对。"
