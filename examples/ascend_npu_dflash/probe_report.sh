#!/usr/bin/env bash
# 把今天几个【离线】诊断一次跑完,出一份报告。不占卡,可以和别的活并行。
#
# 做三件事,每件都自己找路径,找不到就说清楚缺什么而不是静默跳过:
#   1. aux_mix_probe —— released 和我们的草稿,main_proj 怎么加权那三层 aux hidden state。
#      找得到 HS dump 就一并算【真实贡献】‖W_b·h_b‖,而不只是权重范数(深层激活 RMS
#      更大,只看范数会系统性低估深层)。
#   2. positions_report —— 把 prefill_noise_sweep 的逐位置 CSV 按行/位置/margin 切开,
#      回答「失效是随序列长度来的,还是越过某条线之后才来的」。
#   3. serve 配置核查 —— prefix caching / chunked prefill / max_num_batched_tokens。
#      2026-09-23 那组数里 bg=4 反而比 bg=0 好一倍,而缓存命中率恰好随序列变长而升高、
#      随并发升高而降低 —— 所以这几个开关必须先确认,否则整组数的解释是悬的。
#
# USAGE
#   CONDA_ENV=dspark-dsv4-serving bash probe_report.sh
#
#   可覆盖:RELEASED= DRAFTS="d1 d2" HS= NOISE_DIR= CKPT_ROOT= OUT=
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$HOME/probe_report_$(date +%Y%m%d_%H%M%S).log}"
NOISE_DIR="${NOISE_DIR:-$HOME/prefill_noise}"

if [ -z "${CKPT_ROOT:-}" ]; then
  for d in /home/canada_group_folder/ckpt /share/canada_group_folder/ckpt /mnt/nfs/canada_group_folder/ckpt; do
    [ -d "$d" ] && CKPT_ROOT="$d" && break
  done
fi
CKPT_ROOT="${CKPT_ROOT:-/home/canada_group_folder/ckpt}"
RELEASED="${RELEASED:-$CKPT_ROOT/released_draft_bf16_standalone}"

# 探针要 safetensors + torch(bf16 权重 numpy 后端读不了)。裸 python 在未激活 conda 的
# shell 上要么不存在,要么缺依赖 —— 两种都是整轮白跑。
PY="${PY:-}"
if [ -z "$PY" ]; then
  if [ -n "${CONDA_ENV:-}" ] && command -v conda >/dev/null 2>&1; then
    PY="conda run --no-capture-output -n ${CONDA_ENV} python"
  else
    PY="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo python)"
  fi
fi

say() { echo "$*" | tee -a "$OUT"; }
hdr() { say ""; say "################################################################################"; say "## $*"; say "################################################################################"; }

: > "$OUT"
say "离线诊断报告   $(date '+%F %T')"
say "  CKPT_ROOT   $CKPT_ROOT"
say "  解释器      $PY"
say "  输出        $OUT"

# ── 1. 找我们自己的草稿 ───────────────────────────────────────────────────────
if [ -z "${DRAFTS:-}" ]; then
  DRAFTS="$(ls -d "$CKPT_ROOT"/*dspark*vllm* "$CKPT_ROOT"/*blk15* "$CKPT_ROOT"/*dspark_drafts/* 2>/dev/null \
            | grep -v "released" | sort -u | head -8 | tr '\n' ' ')"
fi
# ── 2. 找一份 HS dump(有它才是真实贡献,没有只能报权重范数)────────────────
if [ -z "${HS:-}" ]; then
  HS="$(find /home /share -maxdepth 6 -name 'hs_*.safetensors' -size +1k 2>/dev/null | head -1)"
fi

hdr "1. main_proj 怎么加权 3 层 aux hidden state"
if [ ! -d "$RELEASED" ]; then
  say "!! released 草稿不在 $RELEASED —— 这份是对照,没有它下面的对比没意义。"
  say "   $CKPT_ROOT 下叫 released* 的有:"
  ls -d "$CKPT_ROOT"/*released* 2>/dev/null | sed 's/^/     /' | tee -a "$OUT"
fi
if [ -z "$DRAFTS" ]; then
  say "!! 没找到我们自己的草稿(试过 *dspark*vllm* / *blk15* / *dspark_drafts/*)。"
  say "   $CKPT_ROOT 下现有的目录:"
  ls -d "$CKPT_ROOT"/*/ 2>/dev/null | head -25 | sed 's/^/     /' | tee -a "$OUT"
  say "   想指定就带 DRAFTS=\"<目录1> <目录2>\" 重跑。"
else
  say "   我们的草稿:$DRAFTS"
fi
if [ -n "$HS" ]; then
  say "   HS dump:$HS   (⟹ 会算真实贡献 ‖W_b·h_b‖)"
  HS_ARG=(--hs "$HS")
else
  say "   ⚠ 没找到 hs_*.safetensors —— 只能报权重范数。深层激活 RMS 更大会让它的权重"
  say "     偏小,所以【只看范数不能下结论】。有 HS 就带 HS=<文件> 重跑。"
  HS_ARG=()
fi
say ""
# shellcheck disable=SC2086
$PY "$SCRIPT_DIR/aux_mix_probe.py" "$RELEASED" $DRAFTS "${HS_ARG[@]}" 2>&1 | tee -a "$OUT"

hdr "2. prefill 噪声:失效随【长度】还是随【位置】"
CSVS="$(ls "$NOISE_DIR"/positions_bg*.csv 2>/dev/null | tr '\n' ' ')"
if [ -z "$CSVS" ]; then
  say "!! $NOISE_DIR 下没有 positions_bg*.csv —— 先跑 prefill_noise_sweep.sh。"
else
  # shellcheck disable=SC2086
  $PY "$SCRIPT_DIR/positions_report.py" $CSVS 2>&1 | tee -a "$OUT"
fi

hdr "3. serve 配置核查(缓存/分块,决定上面那组数怎么解释)"
SLOG="$NOISE_DIR/serve.log"
if [ ! -s "$SLOG" ]; then
  say "!! 没有 $SLOG"
else
  say "-- 命中的配置行 --"
  grep -aiE "prefix.?caching|chunked.?prefill|max_num_batched|long_prefill|max_num_seqs|enable_chunked" \
    "$SLOG" | sed -E 's/^\([^)]*\) *//' | sort -u | head -20 | sed 's/^/   /' | tee -a "$OUT"
  say ""
  say "-- additional-config --"
  grep -a "additional-config" "$SLOG" | tail -1 | sed 's/^/   /' | tee -a "$OUT"
fi

hdr "读法"
say "  ① L40/41/42 的【贡献占比】(不是权重范数)才是「哪层更重要」。released 的权重"
say "     范数是 43/29/28(浅层最大),但深层激活 RMS 更大,乘完可能完全是另一个样子。"
say "     真正值钱的是:我们的草稿和 released 差多少 —— 差很多 = 学到的是另一种组合。"
say "  ② 翻转率随长度单调、位置桶内部平坦  ⟹ 问题是【整条有多长】。"
say "     位置桶之间有断崖                  ⟹ 越过那条线才开始算错,那是根因坐标。"
say "     笃定档(margin>2 nat)翻转 >5%    ⟹ 不是数值抖动,别拿这组数判语料。"
say "  ③ prefix caching 若是开的,第二次 prefill 会命中缓存,而命中率随长度升高、随并发"
say "     降低 —— 那能同时解释「长序列更差」和「bg=4 反而更好」。必须先排除。"
say ""
say "全文:$OUT"
