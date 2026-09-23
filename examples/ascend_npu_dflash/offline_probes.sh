#!/usr/bin/env bash
# 所有【不占卡】的探针,一条命令跑完,一份日志。
#
# 三件事,都不碰 NPU,所以可以在 serve/训练跑着的时候随便跑:
#   1. main_proj 怎么加权三层 aux —— released vs 我们自己的草稿(aux_mix_probe.py)
#   2. prefill 噪声的逐位置切片 —— 失效随长度来还是随位置来(positions_report.py)
#   3. serve 的关键配置 —— prefix cache / chunked prefill 到底开没开
#
# 全部自动发现路径:草稿目录、HS dump、上一次 prefill_noise 的 CSV 和 serve.log。
# 找不到就说清楚找了哪里,不会静默跳过。
#
# USAGE
#   bash offline_probes.sh                       # 全自动
#   RELEASED=<dir> OURS="<dir1> <dir2>" HS=<file> bash offline_probes.sh   # 手动指定
#   PREFILL_OUT=~/prefill_noise bash offline_probes.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${LOG:-$HOME/offline_probes.log}"
PREFILL_OUT="${PREFILL_OUT:-$HOME/prefill_noise}"

# 探针只读权重和 CSV,不需要 NPU;但 import torch 会自动拉 torch_npu,没 source CANN
# 的 shell 里直接炸。脚本里的 python 都带这个。
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

PY="${PY:-}"
if [ -z "$PY" ]; then
  if [ -n "${CONDA_ENV:-}" ] && command -v conda >/dev/null 2>&1; then
    PY="conda run --no-capture-output -n ${CONDA_ENV} python"
  elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    PY="$CONDA_PREFIX/bin/python"
  else
    PY="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo python)"
  fi
fi

say() { echo "$*" | tee -a "$LOG"; }
hr()  { say "================================================================================"; }

: > "$LOG"
hr; say "  不占卡探针合集   $(date '+%Y-%m-%d %H:%M:%S')"; say "  python: $PY"; hr

# ── 找权重 ───────────────────────────────────────────────────────────────────
if [ -z "${CKPT_ROOT:-}" ]; then
  for d in /home/canada_group_folder/ckpt /share/canada_group_folder/ckpt /mnt/nfs/canada_group_folder/ckpt; do
    [ -d "$d" ] && CKPT_ROOT="$d" && break
  done
fi
CKPT_ROOT="${CKPT_ROOT:-/home/canada_group_folder/ckpt}"
say "ckpt 根目录: $CKPT_ROOT"

RELEASED="${RELEASED:-}"
if [ -z "$RELEASED" ]; then
  for c in "$CKPT_ROOT"/released_draft_bf16_standalone "$CKPT_ROOT"/*released*draft*; do
    [ -d "$c" ] && RELEASED="$c" && break
  done
fi

OURS="${OURS:-}"
if [ -z "$OURS" ]; then
  # 我们导出的 vLLM 格式草稿。按修改时间倒序,最多取 4 个,免得表太长。
  # ★ 两个通配符会匹配到同一个目录(*dspark*vllm* 和 *blk15* 常常都中),不去重的话
  #   同一个草稿在表里出现两遍,差值行也重复。awk 去重比 sort -u 好:保住 ls -dt 的时间序。
  OURS="$(ls -dt "$CKPT_ROOT"/*dspark*vllm* "$CKPT_ROOT"/*blk15* 2>/dev/null \
          | grep -v "released" | awk '!seen[$0]++' | head -4 | tr '\n' ' ')"
fi

say "released : ${RELEASED:-<没找到>}"
say "我们的   : ${OURS:-<没找到 —— $CKPT_ROOT 下没有 *dspark*vllm* / *blk15* 目录>}"
if [ -z "$OURS" ]; then
  say "           目录里现有的(前 20 个):"
  ls -1 "$CKPT_ROOT" 2>/dev/null | head -20 | sed 's/^/             /' | tee -a "$LOG"
fi

# ── 找一份 HS dump(有它才能算真实贡献,而不只是权重范数)──────────────────
HS="${HS:-}"
if [ -z "$HS" ]; then
  for base in "${DSPARK_HS_DIR:-}" "$HOME" /home/canada_group_folder /share/canada_group_folder /tmp; do
    [ -n "$base" ] && [ -d "$base" ] || continue
    HS="$(find "$base" -maxdepth 5 -name 'hs_*.safetensors' -type f 2>/dev/null | head -1)"
    [ -n "$HS" ] && break
  done
fi
if [ -n "$HS" ]; then
  say "HS dump  : $HS"
else
  say "HS dump  : <没找到>  ⚠ 只能报权重范数。深层激活 RMS 更大会压低它的权重,"
  say "           所以光看范数会系统性低估深层 —— 有 HS 才是真实贡献。"
  say "           有的话用 HS=<file> 指定。"
fi
say ""

# ── 1. main_proj 加权 ────────────────────────────────────────────────────────
hr; say "  1. main_proj 怎么加权三层 aux(layer 40/41/42)"; hr
if [ -z "$RELEASED" ] && [ -z "$OURS" ]; then
  say "!! 一个草稿都没找到,跳过"
else
  HSARG=(); [ -n "$HS" ] && HSARG=(--hs "$HS")
  # shellcheck disable=SC2086
  $PY "$SCRIPT_DIR/aux_mix_probe.py" $RELEASED $OURS "${HSARG[@]}" 2>&1 | tee -a "$LOG"
fi
say ""

# ── 2. prefill 噪声的逐位置切片 ──────────────────────────────────────────────
hr; say "  2. prefill 噪声:随长度还是随位置"; hr
CSVS="$(ls -1 "$PREFILL_OUT"/positions_bg*.csv 2>/dev/null | tr '\n' ' ')"
if [ -z "$CSVS" ]; then
  say "!! $PREFILL_OUT 下没有 positions_bg*.csv —— 先跑 prefill_noise_sweep.sh"
else
  # shellcheck disable=SC2086
  $PY "$SCRIPT_DIR/positions_report.py" $CSVS 2>&1 | tee -a "$LOG"
fi
say ""

# ── 3. serve 的关键配置 ──────────────────────────────────────────────────────
hr; say "  3. serve 配置:prefix cache / chunked prefill 开没开"; hr
say "   (prefix cache 若是开的,重复 prefill 会命中 KV 缓存,一致性是构造出来的;"
say "    而命中率随序列变长而升、随背景负载而降 —— 那正好能解释上一轮两处反常。)"
SLOGS="$(ls -1t "$PREFILL_OUT"/serve.log "$HOME"/det_sweep*/serve_base.log 2>/dev/null | head -3)"
if [ -z "$SLOGS" ]; then
  say "!! 找不到 serve 日志($PREFILL_OUT/serve.log)"
else
  for f in $SLOGS; do
    say "--- $f"
    grep -aiE "prefix_caching|enable_prefix|chunked_prefill|max_num_batched_tokens|long_prefill|num_scheduler_steps" "$f" \
      | sed -E 's/^\([^)]*\) *//' | sort -u | head -12 | sed 's/^/    /' | tee -a "$LOG"
    grep -a "additional-config" "$f" | tail -1 | sed 's/^/    /' | tee -a "$LOG"
  done
fi
say ""
hr; say "  全部输出已存到 $LOG"; hr
