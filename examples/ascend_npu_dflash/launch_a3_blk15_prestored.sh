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
#   专家精度      bf16(option B)          fp32 主权重(option A)    EP16 下每卡 16 个专家 × 4 B = EP8 的
#                                                                     32 × 2 B,显存不变;更新在 fp32 里累加
#
# 其余(BLOCK/γ/锚点/Muon/噪声/非因果/warm-start/不开均衡)与基线一致。
# option A 和 Muon 不冲突:Muon 在 bf16 里做 Newton-Schulz,更新按参数自己的 dtype 加回去
# (muon_distributed.py `p_local.add_(update.to(p_local.dtype))`),fp32 主权重只会让累加更准。
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
export INIT_LAYER="${INIT_LAYER:-1}" INIT_MOE_NO_ROUTER="${INIT_MOE_NO_ROUTER:-1}" BF16_EXPERTS="${BF16_EXPERTS:-0}"
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
# 僵尸(<defunct>)不算:评测停 serve 后常留几个 [VLLM::Worker] <defunct>,不占卡、也杀不掉,
# 算进来的话流水线最后一步会永远起不来(eval_blk15_drafts.sh 的 procs_alive 同一个道理)。
_live=""
for _p in $(pgrep -u "$USER" -if 'vllm|EngineCore|hs_dump_daemon' 2>/dev/null); do
  case "$(ps -o stat= -p "$_p" 2>/dev/null | tr -d ' ')" in ''|Z*) ;; *) _live="$_live $_p" ;; esac
done
if [ -n "$_live" ]; then
  say "!! 还有 vllm / hs_dump_daemon 进程在跑 —— dump/评测没结束,或者没清场。先停它。"
  ps -o pid=,stat=,args= -p ${_live// /,} 2>/dev/null | cut -c1-160 | head -5
  exit 2
fi

# ── 检查 3:HS 齐不齐 —— 预存模式下缺一个文件就会在训练中途报错停掉 ───────────
if [ "$HS_ON_MISSING" = "raise" ]; then
  # 顺带核格式:训练侧 data.py 要 torch 格式(input_ids 取出来是 tensor)。python 格式的 Arrow 会在
  # 16 个 rank 加载完模型、第一个 batch 才报 `must be Tensor, not list` —— 在这里几秒就挡掉。
  read=$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -c "
from datasets import load_from_disk; d = load_from_disk('$DATA')
d = d if hasattr(d, 'num_rows') else d[next(iter(d))]
print(d.num_rows, d.format['type'])" 2>/dev/null)
  rows=${read%% *}; fmt=${read##* }
  [ -n "$rows" ] || { say "!! 读不出 $DATA 的行数"; exit 2; }
  if [ "$fmt" != "torch" ]; then
    say "!! $DATA 的格式是 '$fmt',训练要 torch。修(只改 state.json,不动数据):"
    say "   python $SCRIPT_DIR/hs_subset_arrow.py --arrow <源 Arrow> --out $DATA --repair-format"
    exit 2
  fi
  if [ -n "${HS_COUNT_SKIP:-}" ]; then
    have="$rows"; say "HS_COUNT_SKIP=1:跳过计数(信 hs_dump_daemon 结尾那行「盘上现有 N 个 HS 文件」)"
  else
    # 用 `ls -f`(不排序、不 stat),和 hs_dump_daemon 同一个数法:NFS 上几十万个文件,find 可能慢得多。
    say "数 HS 文件($HS_DIR,NFS 上几十万个文件,冷缓存可能要一两分钟;HS_COUNT_SKIP=1 可跳过)..."
    _t0=$SECONDS
    have=$( { ls -f "$HS_DIR" 2>/dev/null || true; } | grep -c '^hs_[0-9]*\.safetensors$' || true)
    say "HS 文件 $have / Arrow 行数 $rows   (数了 $((SECONDS - _t0))s)"
  fi
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
