#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# A3 单机(16 卡)上用【预存 HS】训 blk15 —— 不需要 serve。
#
# 薄包装:train_dsv4_dspark.sh 做全部实事,这里只钉默认值 + 开跑前的三道检查。
# 配方 = blk15 基线 faithful_ep_20260908_025936 逐项照抄,只有下面这几处因为规模不同而改:
#
#   项            基线(8×A2 在线 HS)     本脚本(16×A3 预存 HS)   为什么
#   数据          77W 全量                 随机一半(seed 0)         盘:全量 HS 19.6 TB
#   NPROC / EP    8 / EP8                  16 / EP16                 A3 单机 16 个逻辑卡
#   全局 batch    1×                       2×                        每步 16 个 rank
#   LR            2e-4                     2.8e-4                    √2 批量缩放(之前 A3 线同法定 3e-4)
#   步数/epoch    24,896                   ~6,200                    数据 ½ × 批量 2×
#   HS            在线生成、用完即删       预存,缺文件直接报错      HS_ON_MISSING=raise
#
# 其余(BLOCK/γ/锚点/Muon/噪声/非因果/warm-start/bf16 专家/不开均衡)与基线一致。
#
# 用法
#   bash launch_a3_blk15_prestored.sh                  # 正式跑
#   MAX_STEPS=100 bash launch_a3_blk15_prestored.sh    # 先跑 100 步:量单步时间、显存、HS 读取
#   LR=2e-4 bash launch_a3_blk15_prestored.sh          # 任何旗标都能用环境变量覆盖
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATA="${DATA:-/mnt/nfs/canada_group_folder/dataset/arrow_0730_77w_dedup_half_s0}"
export HS_DIR="${HS_DIR:-/mnt/nfs/canada_group_folder/dataset/dsv4_hs_store/arrow_0730_77w_dedup_half_s0}"
export HS_ON_MISSING="${HS_ON_MISSING:-raise}"
export VERIFIER="${VERIFIER:-/home/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16}"
export CANN_ENV="${CANN_ENV:-/home/a00652497/920env_npu.sh}"
export RUN="${RUN:-$HOME/dsv4_run}"
export ENDPOINT="${ENDPOINT:-http://127.0.0.1:7000/v1}"   # 预存模式下不会被访问;train.py 要求有值

# ── 规模相关 ────────────────────────────────────────────────────────────────
export NPROC="${NPROC:-16}" DSPARK_EP="${DSPARK_EP:-1}"
export LR="${LR:-2.8e-4}" EPOCHS="${EPOCHS:-5}" WARMUP_RATIO="${WARMUP_RATIO:-0.04}" SCHED_TYPE="${SCHED_TYPE:-cosine}"
# ── 与基线一致 ──────────────────────────────────────────────────────────────
export MAX_ANCHORS="${MAX_ANCHORS:-192}" SEQLEN="${SEQLEN:-3072}" BLOCK="${BLOCK:-15}" DECAY_GAMMA="${DECAY_GAMMA:-15}"
export MASK_TOKEN="${MASK_TOKEN:-128799}"
export OPTIM="${OPTIM:-muon}" MUON_ADJUST="${MUON_ADJUST:-match_rms_adamw}" MUON_HYBRID="${MUON_HYBRID:-0}"
export NONCAUSAL="${NONCAUSAL:-1}" SWA_WINDOW="${SWA_WINDOW:-128}" NOISE_STD="${NOISE_STD:-0.05}" KD_TEMP="${KD_TEMP:-1.0}"
export RECOMPUTE="${RECOMPUTE:-1}" COMPILE="${COMPILE:-0}" NO_VAL="${NO_VAL:-1}" CKPT_FREQ="${CKPT_FREQ:-0.5}"
export INIT_LAYER="${INIT_LAYER:-1}" INIT_MOE_NO_ROUTER="${INIT_MOE_NO_ROUTER:-1}" BF16_EXPERTS="${BF16_EXPERTS:-1}"
export DSPARK_MOE_BALANCE="${DSPARK_MOE_BALANCE:-0}"
export DSPARK_LOG_EXPERT_LOAD="${DSPARK_LOG_EXPERT_LOAD:-1}" DSPARK_LOG_EXPERT_LOAD_EVERY="${DSPARK_LOG_EXPERT_LOAD_EVERY:-50}"

# CANN 先 source(torch_npu 导入要它),再把环境自己的 lib 放最前 —— 顺序反了 CANN 会压到前面。
[ -f "$CANN_ENV" ] && source "$CANN_ENV"
# miniforge 环境:让环境自己的 libstdc++ 排在 /usr/lib64 那个前面。不加的话 torch_npu 导入就报
#   ImportError: /usr/lib64/libstdc++.so.6: version `CXXABI_1.3.15' not found
# (serve 脚本里同样的一行,见 serve_dsv4_a3_singlenode_specmethod.sh)
[ -n "${CONDA_PREFIX:-}" ] && export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# ── 检查 1:环境 —— 训练栈能导入(只导入,不碰卡)────────────────────────────
( cd "$HOME" && TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -c "
import torch, torch_npu, transformers, speculators
from speculators.train.data import ArrowDataset
print(f'    torch {torch.__version__}  torch_npu {torch_npu.__version__}  transformers {transformers.__version__}')
" ) || { say "!! 训练栈导入失败(先 conda activate 训练环境)"; exit 2; }

# ── 检查 2:dump 服务已经退了 —— 它还占着 16 张卡就别起 ─────────────────────────
if pgrep -u "$USER" -f 'vllm|EngineCore|hs_dump_daemon' >/dev/null 2>&1; then
  say "!! 还有 vllm / hs_dump_daemon 进程在跑 —— dump 没结束,或者没清场。先停它。"
  pgrep -u "$USER" -af 'vllm|EngineCore|hs_dump_daemon' | head -5
  exit 2
fi

# ── 检查 3:HS 齐不齐 —— 预存模式下缺一个文件就会在训练中途报错停掉 ───────────
if [ "$HS_ON_MISSING" = "raise" ]; then
  rows=$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -c "
from datasets import load_from_disk; d = load_from_disk('$DATA')
print(d.num_rows if hasattr(d, 'num_rows') else d[next(iter(d))].num_rows)" 2>/dev/null)
  [ -n "$rows" ] || { say "!! 读不出 $DATA 的行数"; exit 2; }
  say "数 HS 文件($HS_DIR,几十万个文件要十几秒)..."
  have=$( { find "$HS_DIR" -maxdepth 1 -name 'hs_*.safetensors' 2>/dev/null || true; } | wc -l)
  say "HS 文件 $have / Arrow 行数 $rows"
  if [ "$have" -lt "$rows" ]; then
    if [ "${ALLOW_PARTIAL:-0}" = "1" ]; then
      say "⚠ 缺 $((rows - have)) 个,ALLOW_PARTIAL=1 放行 —— 碰到缺的那一行训练会报错停下。"
    else
      say "!! 缺 $((rows - have)) 个 HS 文件。dump 跑完了吗?(tail ~/hs_daemon_full.log)"
      say "   只想先试几步、确定不会碰到缺的行:ALLOW_PARTIAL=1 MAX_STEPS=... 再起。"
      exit 2
    fi
  fi
fi

cat <<CFG
── A3 blk15 · 预存 HS ────────────────────────────────────────────────────────
  DATA      = $DATA
  HS_DIR    = $HS_DIR   (HS_ON_MISSING=$HS_ON_MISSING)
  VERIFIER  = $VERIFIER
  RUN       = $RUN      CANN_ENV = $CANN_ENV
  NPROC=$NPROC EP=$DSPARK_EP  LR=$LR  EPOCHS=$EPOCHS  WARMUP_RATIO=$WARMUP_RATIO  MAX_STEPS=${MAX_STEPS:-<全程>}
  BLOCK=$BLOCK γ=$DECAY_GAMMA  MAX_ANCHORS=$MAX_ANCHORS  SEQLEN=$SEQLEN  OPTIM=$OPTIM  BF16_EXPERTS=$BF16_EXPERTS
──────────────────────────────────────────────────────────────────────────────
  头 100 步看:profile/step_ms、mem/reserved_gb、profile/fetch_ms(读 NFS 的 HS)、
  grad_norm(step 800 附近 ~1 = INIT_LAYER 生效;~8 = warm-start 没生效)、loss 无 NaN。
──────────────────────────────────────────────────────────────────────────────
CFG

exec bash "$SCRIPT_DIR/train_dsv4_dspark.sh" faithful
