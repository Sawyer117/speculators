#!/usr/bin/env bash
# 在这台 A3 上装一套【老栈】,只为回答一个问题:温度 0 的不可复现是不是 4ce367a 引入的。
#
# WHY
# ---
# 2026-09-20 实测(A3 / vllm-ascend `4ce367a7d` / vLLM v0.27.1 / DP2×TP8/EP16,
# 温度 0、并发 1、prefix cache 关、同一 prompt 连打 3 次):
#
#     prompt-len   256     320    384   448    512   1024   2048
#     最低一致率   100.0%  28.1%  3.1%  46.9%  9.4%  0.0%   0.0%
#
# ≤256 完全确定;~300 以上静默算错,严重程度随机。**这不是崩溃,是算错**——崩了还知道。
# 很多行「完全一致前缀 0/32」,即第一个生成的 token 就分叉,而第一个 token 来自 prefill
# 的最后一个位置 ⟹ 坏在 prefill,decode 只是继承了错的 KV。
#
# 老栈 `386530d12`(2026-07-14)在这台机上跑过 gsm8k 全量 96.59% 和 6 小时 0 error 的
# rollout。**它到底有没有同样的毛病,决定三件大事:**
#   1. 这是不是 4ce367a 的回归 —— 决定上游 issue 怎么写;
#   2. `arrow_0730_77w_dedup` 的长尾(>300 token 之后的部分)可不可信 —— 那是老栈产的,
#      而训练权重最大的恰恰是那一段;
#   3. 这个 pin 上产出的所有 eval 数字要不要重做。
#
# 三件套(Dockerfile 的 ARG VLLM_TAG 是权威):
#     386530d12  →  vLLM v0.23.0        ← 本脚本装这套
#     4ce367a7d  →  vLLM v0.27.1        ← 现在在跑的
# CANN 用 9.2(`920env_npu.sh`),和现行一致,**只留 vllm-ascend + vLLM 一个变量**。
#
# 顺带一个白送的对照:老 pin 上**没有我们的 HS dumper**,所以它同时也证否了
# 「dumper 扰动时序」这条。
#
# 安全
# ----
#   * 全新 conda 环境 + 全新目录,**一个字都不碰 `dspark-dsv4-serving` 和
#     `installation_027`**。跑砸了删掉重来即可。
#   * 拒绝在生产环境里运行(见下面的守卫)。
#   * 用的是仓库里那份 SSOT 安装脚本 `install_npu_env_dspark.sh`(它本来就照 v0.23.0
#     写的),这里只是按老 pin 备好目录再调它 —— 不重写安装逻辑。
#
# USAGE
# -----
#     bash examples/ascend_npu_dflash/install_oldstack_a3.sh          # 先看计划
#     GO=1 bash examples/ascend_npu_dflash/install_oldstack_a3.sh     # 真装(几小时)
#
# 装完它会把老栈的起服务命令 + 确定性测试命令打出来。
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

OLD_SHA="${OLD_SHA:-386530d12}"
VLLM_TAG="${VLLM_TAG:-v0.23.0}"
ROOT="${ROOT:-/home/a00652497/dsv4_eval/oldstack_023}"
ENV_NAME="${ENV_NAME:-dsv4-oldstack}"
PY_VER="${PY_VER:-3.11}"
CANN_ENV="${CANN_ENV:-/home/a00652497/920env_npu.sh}"
FORK="${FORK:-https://github.com/Sawyer117/vllm-ascend.git}"
PROD_ENVS="${PROD_ENVS:-dspark-dsv4-serving dsv4-eval-main}"
NEED_GB="${NEED_GB:-60}"
GO="${GO:-0}"

VLLM_DIR="$ROOT/installation/vllm-$VLLM_TAG"
VA_DIR="$ROOT/installation/vllm-ascend-$OLD_SHA"

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

echo "================================================================================"
echo "  A3 老栈安装(只为测温度 0 的可复现性)"
echo "  vllm-ascend   $OLD_SHA   (2026-07-14,gsm8k 96.59% 那套)"
echo "  vLLM          $VLLM_TAG      (Dockerfile ARG VLLM_TAG 认定)"
echo "  CANN          $CANN_ENV"
echo "  新环境        conda env '$ENV_NAME'  (python $PY_VER)"
echo "  新目录        $ROOT"
echo "  ⚠ 不碰        $PROD_ENVS / installation_027"
echo "================================================================================"

# ── 守卫 ─────────────────────────────────────────────────────────────────────
cur="${CONDA_DEFAULT_ENV:-}"
for p in $PROD_ENVS; do
  if [ "$cur" = "$p" ]; then
    echo "!! 你现在在生产环境 '$cur' 里。这个脚本会装一堆东西,**绝不能**在它里面跑。"
    echo "   先 conda deactivate(回 base)再来。"
    exit 2
  fi
done
[ -f "$CANN_ENV" ] || { echo "!! CANN env 不存在:$CANN_ENV"; exit 2; }
[ -f "$SCRIPT_DIR/install_npu_env_dspark.sh" ] || { echo "!! 找不到 SSOT 安装脚本"; exit 2; }

free_gb=$(df -BG --output=avail "$(dirname "$ROOT")" 2>/dev/null | tail -1 | tr -dc '0-9')
echo ">>> 目标盘剩余 ${free_gb:-?} GB(需要约 ${NEED_GB} GB)"
if [ -n "$free_gb" ] && [ "$free_gb" -lt "$NEED_GB" ]; then
  echo "!! 空间不够,先清。"; exit 2
fi

echo
echo "将要执行:"
echo "  1. conda create -n $ENV_NAME python=$PY_VER"
echo "  2. git clone $FORK → $VA_DIR ; git checkout $OLD_SHA"
echo "  3. source $CANN_ENV"
echo "  4. ROOT=$ROOT VLLM_DIR=$VLLM_DIR VA_DIR=$VA_DIR CANN_ENV=$CANN_ENV \\"
echo "       bash $SCRIPT_DIR/install_npu_env_dspark.sh"
echo "     (它自己 clone vLLM $VLLM_TAG,并从源码编译 V4/SAS 的 CANN 算子 —— 这步最久)"
echo
if [ "$GO" != "1" ]; then
  echo ">>> 这是计划。确认无误后:GO=1 bash $0"
  exit 0
fi

# ── 1. 新环境 ────────────────────────────────────────────────────────────────
say "1/4 建 conda 环境 $ENV_NAME"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | awk '{print $1}' | grep -qx "$ENV_NAME" \
  || conda create -y -n "$ENV_NAME" "python=$PY_VER" || exit 1
conda activate "$ENV_NAME" || exit 1
say "    python=$(python -V 2>&1)  env=$CONDA_DEFAULT_ENV"
[ "$CONDA_DEFAULT_ENV" = "$ENV_NAME" ] || { echo "!! 没切进 $ENV_NAME,停"; exit 1; }

# ── 2. 按老 SHA 备好 vllm-ascend ─────────────────────────────────────────────
# SSOT 脚本用 `git clone --branch <VA_BRANCH>`,而 --branch 不接受 commit SHA。
# 所以这里先把目录准备好(clone + checkout SHA),它看到 .git 存在就会跳过 clone。
say "2/4 取 vllm-ascend @ $OLD_SHA"
mkdir -p "$ROOT/installation" || exit 1
if [ ! -d "$VA_DIR/.git" ]; then
  git clone "$FORK" "$VA_DIR" || exit 1
fi
( cd "$VA_DIR" && git fetch --all -q && git checkout -q "$OLD_SHA" ) || exit 1
say "    $(git -C "$VA_DIR" log --oneline -1)"

# ── 3+4. 交给 SSOT ───────────────────────────────────────────────────────────
say "3/4 source CANN:$CANN_ENV"
# shellcheck disable=SC1090
source "$CANN_ENV"

say "4/4 调 SSOT 安装脚本(编译算子,几十分钟到几小时;全程输出到 $ROOT/install.log)"
mkdir -p "$ROOT"
ROOT="$ROOT" VLLM_DIR="$VLLM_DIR" VA_DIR="$VA_DIR" VA_BRANCH="$OLD_SHA" \
  CANN_ENV="$CANN_ENV" \
  bash "$SCRIPT_DIR/install_npu_env_dspark.sh" 2>&1 | tee "$ROOT/install.log"
rc=${PIPESTATUS[0]}
if [ "$rc" != "0" ]; then
  say "!! 安装失败(rc=$rc)。最后 30 行:"; tail -30 "$ROOT/install.log"; exit "$rc"
fi

# ── 事后:哪些 flag 老 vLLM 不认 ─────────────────────────────────────────────
say "检查我们 serve 脚本用的 flag 在 vLLM $VLLM_TAG 上认不认:"
H=$(vllm serve --help 2>/dev/null)
for f in --async-scheduling --tokenizer-mode --enable-expert-parallel --data-parallel-size-local \
         --no-enable-prefix-caching --additional-config --compilation-config --model-loader-extra-config; do
  echo "$H" | grep -q -- "$f" && echo "    ✅ $f" || echo "    ❌ $f  ← 老 vLLM 不认,起服务时要去掉"
done

echo
echo "================================================================================"
echo "  装好了。老栈 = vllm-ascend $OLD_SHA + vLLM $VLLM_TAG + CANN 9.2"
echo "================================================================================"
cat <<EOF
起服务(注意:老 pin **没有** HS dumper,所以别带 HS_DUMP —— 这也正好是 dumper 的对照):

  conda activate $ENV_NAME
  source $CANN_ENV
  CONDA_ENV=$ENV_NAME CANN_ENV=$CANN_ENV DSA_OVERLAP=0 \\
    nohup bash $SCRIPT_DIR/serve_dsv4_a3_singlenode_specmethod.sh > ~/serve_oldstack.log 2>&1 &
  until curl -sf --noproxy '*' http://localhost:7000/v1/models >/dev/null; do sleep 10; done; echo READY

★ 关键对照(和新栈一模一样的命令):

  for L in 256 512 1024 2048; do
    echo "## \$L"
    ENDPOINT=http://localhost:7000/v1 \\
    ARROW=/home/canada_group_folder/dataset/arrow_0730_77w_dedup \\
      python $SCRIPT_DIR/corpus_provenance_check.py \\
        --n 3 --gen 32 --repeat 3 --prompt-len \$L --id-base \$((994000 + \$L)) 2>&1 | grep "服务确定性"
  done

读法
  老栈全 100%  ⟹ 4ce367a 的回归。上游 issue 立得住,而且老栈产的语料可信。
  老栈也散     ⟹ 范围远超一个 pin —— 语料长尾和历史 eval 都要重新审,
                  这比"回归"严重得多,得先停下来盘影响面。
EOF
