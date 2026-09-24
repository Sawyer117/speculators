#!/usr/bin/env bash
# 一键装 A3(186)上的 DSV4-DSpark【训练】环境 —— 只装训练要的,不装 vllm / vllm-ascend。
#
# WHY
# ---
# 训练只 import torch + torch_npu + transformers + datasets + safetensors + speculators
# (+ hs_connectors),不 import vllm。所以不用走 install_npu_env_dspark.sh 那条从源码编 CANN
# 算子的长路,几分钟就装完。
#
# 版本照 109 的训练环境 dspark-dsv4-compile(2026-09-25 实测):
#   torch 2.12.0 · torch_npu 2.12.0rc1 · transformers 5.13.1 · datasets 5.0.0 ·
#   safetensors 0.8.0 · numpy 2.3.5
# 训练代码在这套上跑通过。老栈 env(dsv4-oldstack)不能拿来训:它的 transformers 是 5.17.0,
# 超出 speculators 要求的 <5.15.0;而且它是 dump 服务正在用的环境,不许动。
#
# 安全
#   * 只往 ENV_NAME 里装;名字撞上服务/评测环境直接拒绝。
#   * 不碰 NPU:只有 pip 和 import 检查,dump 跑着也能装。
#   * 可重入:环境已在就跳过 create,把缺的补上;钉死的版本若被 pip 挪动,最后会报出来并以非零退出。
#
# 用法
#   bash examples/ascend_npu_dflash/install_train_env_a3.sh
#   ENV_NAME=dspark-dsv4-train2 bash examples/ascend_npu_dflash/install_train_env_a3.sh
#   TORCH_VER=... TORCH_NPU_VER=... TRANSFORMERS_VER=... 覆盖版本
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ENV_NAME="${ENV_NAME:-dspark-dsv4-train}"
PY_VER="${PY_VER:-3.11}"
TORCH_VER="${TORCH_VER:-2.12.0}"
TORCH_NPU_VER="${TORCH_NPU_VER:-2.12.0rc1}"
TRANSFORMERS_VER="${TRANSFORMERS_VER:-5.13.1}"
DATASETS_VER="${DATASETS_VER:-5.0.0}"
SAFETENSORS_VER="${SAFETENSORS_VER:-0.8.0}"
NUMPY_VER="${NUMPY_VER:-2.3.5}"
TV_VER="${TV_VER:-0.27.0}"        # torchvision / torchaudio:speculators 的 pyproject 列了,但训练不 import;
TA_VER="${TA_VER:-2.12.0}"        #   装不上只告警
CANN_ENV="${CANN_ENV:-/home/a00652497/920env_npu.sh}"
PROXY_SH="${PROXY_SH:-/home/a00652497/portproxy_remote.sh}"
PROTECTED="${PROTECTED:-base dsv4-oldstack dspark-dsv4-serving dsv4-eval-main dsv4-main-cann91}"

TORCH_CPU_INDEX="${TORCH_CPU_INDEX:-https://download.pytorch.org/whl/cpu}"
HW_PYPI="https://mirrors.huaweicloud.com/repository/pypi/simple"
HW_ASCEND="https://mirrors.huaweicloud.com/ascend/repos/pypi"
IDX=(--extra-index-url "$HW_PYPI" --extra-index-url "$HW_ASCEND")

LOG="${LOG:-$HOME/install_train_env_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "$LOG") 2>&1
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

for p in $PROTECTED; do
  [ "$ENV_NAME" = "$p" ] && { say "!! ENV_NAME=$ENV_NAME 是服务/评测环境,不往里装。换个名字。"; exit 2; }
done

echo "================================================================================"
echo "  A3 训练环境   env=$ENV_NAME   python $PY_VER"
echo "  torch $TORCH_VER · torch_npu $TORCH_NPU_VER · transformers $TRANSFORMERS_VER"
echo "  datasets $DATASETS_VER · safetensors $SAFETENSORS_VER · numpy $NUMPY_VER"
echo "  speculators = $REPO_ROOT (editable)      全量日志 → $LOG"
echo "================================================================================"

# ── 0. 代理 + conda ─────────────────────────────────────────────────────────
if [ -f "$PROXY_SH" ]; then set +e; source "$PROXY_SH" >/dev/null 2>&1; set -e; say "代理:已 source $PROXY_SH"; fi
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || true)}"
[ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ] \
  || { say "!! 找不到 conda(CONDA_BASE=${CONDA_BASE:-<空>})。先让 conda 可用,或传 CONDA_BASE="; exit 2; }
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ── 1. 环境 ─────────────────────────────────────────────────────────────────
# 注意不用 grep -q:它读到就退出,awk 吃 SIGPIPE,pipefail 下整条判「假」,就会去 create 一个已存在的环境。
if conda env list | awk '{print $1}' | grep -x "$ENV_NAME" >/dev/null; then
  say "1. 环境 $ENV_NAME 已存在 —— 跳过 create,在它上面补齐"
else
  say "1. conda create -n $ENV_NAME python=$PY_VER"
  if ! conda create -y -n "$ENV_NAME" "python=$PY_VER"; then
    say "   联网 create 失败(代理/证书?)—— 用本地包缓存再试一次(--offline)"
    conda create -y --offline -n "$ENV_NAME" "python=$PY_VER"
  fi
fi
set +e; conda activate "$ENV_NAME"; set -e     # activate.d 里的脚本偶尔返回非零,别让 set -e 误杀
[ "$(basename "${CONDA_PREFIX:-}")" = "$ENV_NAME" ] || { say "!! conda activate $ENV_NAME 没生效"; exit 2; }
python -c "import sys; assert sys.version_info[:2] == tuple(map(int, '$PY_VER'.split('.'))), sys.version" \
  || { say "!! 激活后的 python 不是 $PY_VER"; exit 2; }
say "   python = $(command -v python)"

# ── 2. torch + torch_npu(pip 一律用 python -m pip,别让 PATH 上另一个 pip 装到别处)──
say "2. torch $TORCH_VER + torch_npu $TORCH_NPU_VER + numpy $NUMPY_VER"
python -m pip install -q -U pip setuptools wheel
# ★ torch 必须是【不带 CUDA】的版本。aarch64 上 PyPI 的 torch 包(2.12.0 是 426 MB)带 CUDA,和 torch_npu
#   装在一起导入就报 `Two accelerators cannot be used at the same time in PyTorch: npu and cuda`。
#   109 装的是 2.12.0+cpu —— 来自 PyTorch 的 CPU 源,这里照做;连不上才退回 PyPI,并由下面那道检查兜底。
#   公司代理对 download.pytorch.org 做 SSL 拦截(自签证书,pip 报 CERTIFICATE_VERIFY_FAILED);PyPI 和华为
#   镜像不受影响。所以只对 PyTorch 的两个下载域名跳过证书校验,其余源照常校验。
TORCH_TRUST=(--trusted-host download.pytorch.org --trusted-host download-r2.pytorch.org)
if ! python -m pip install "torch==$TORCH_VER" --index-url "$TORCH_CPU_INDEX" "${TORCH_TRUST[@]}" \
       --retries 2 --timeout 60; then
  say "   ⚠ CPU 源($TORCH_CPU_INDEX)装不上,退回 PyPI —— 下一步会查它带不带 CUDA"
  python -m pip install "${IDX[@]}" "torch==$TORCH_VER"
fi
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -c "
import sys, torch
print(f'   torch {torch.__version__}   torch.version.cuda = {torch.version.cuda}')
sys.exit(0 if torch.version.cuda is None else 1)" || {
  say "!! 装上的 torch 带 CUDA —— 会和 torch_npu 冲突(Two accelerators cannot be used at the same time)。"
  say "   先卸掉:python -m pip uninstall -y torch;确认能连上 $TORCH_CPU_INDEX 后重跑本脚本。"
  exit 1; }
# torch_npu 按已装的 torch 钉住,防止它的依赖解析把 torch 换掉
python -m pip install "${IDX[@]}" "torch-npu==$TORCH_NPU_VER" pyyaml \
  -c <(python -m pip freeze | grep -iE '^torch==')
python -m pip install "numpy==$NUMPY_VER"

# ── 3. 训练依赖 ──────────────────────────────────────────────────────────────
say "3. transformers / datasets / safetensors + 训练用的小依赖"
python -m pip install "${IDX[@]}" "transformers==$TRANSFORMERS_VER" "datasets==$DATASETS_VER" \
  "safetensors==$SAFETENSORS_VER"

# 先核对装上的就是要的版本(transformers/datasets 的依赖解析有可能顺手挪了 numpy 之类),
# 再把它们写成约束文件 —— 否则约束文件会把「已经被挪过的版本」当成标准。
python - <<PYEOF || { say "!! 装上的版本和要求的不一致,见上面"; exit 1; }
from importlib.metadata import version
want = {"torch": "$TORCH_VER", "torch-npu": "$TORCH_NPU_VER", "transformers": "$TRANSFORMERS_VER",
        "datasets": "$DATASETS_VER", "safetensors": "$SAFETENSORS_VER", "numpy": "$NUMPY_VER"}
bad = [f"{d}: 要 {w},装的是 {version(d)}" for d, w in want.items()
       if version(d).split("+")[0] != w]          # torch 可能带 +cpu 本地后缀
print("\n".join("   !! " + b for b in bad) or "   版本核对:全部与要求一致")
raise SystemExit(1 if bad else 0)
PYEOF

# 钉死的版本写成约束文件:后面每一步都带着它,pip 想挪 torch 之类就会直接报错,而不是悄悄升级。
CONSTRAINTS="$CONDA_PREFIX/train_constraints.txt"
python -m pip freeze | grep -iE '^(torch|torch-npu|torch_npu|transformers|numpy|datasets|safetensors)==' > "$CONSTRAINTS"
say "   约束文件 $CONSTRAINTS:"; sed 's/^/     /' "$CONSTRAINTS"

python -m pip install "${IDX[@]}" -c "$CONSTRAINTS" click huggingface-hub "loguru>=0.7.2,<=0.7.3" \
  "openai>=2.0.0" protobuf psutil "pydantic>=2.0.0" "pydantic-settings>=2.0.0" rich \
  "tqdm>=4.66.3,<=4.70.0" "typer>=0.12.0" tensorboard aiohttp packaging
if ! python -m pip install --no-deps "${IDX[@]}" "torchvision==$TV_VER" "torchaudio==$TA_VER" 2>/dev/null; then
  say "   ⚠ torchvision $TV_VER / torchaudio $TA_VER 装不上 —— 训练不 import 它们,只会让 pip check 抱怨,忽略"
fi

# ── 4. speculators + hs_connectors(editable,--no-deps:依赖上面已经按钉死的版本装好)─────
say "4. speculators + hs_connectors(editable)"
python -m pip install --no-deps -e "$REPO_ROOT/hs_connectors"
python -m pip install --no-deps -e "$REPO_ROOT"

# ── 5. 核对 —— 版本没被挪、能导入、speculators 指向这个仓库 ─────────────────────
say "5. 核对(只导入,不碰卡)"
moved=$(python -m pip freeze | grep -iE '^(torch|torch-npu|torch_npu|transformers|numpy|datasets|safetensors)==' \
        | sort | diff - <(sort "$CONSTRAINTS") || true)
if [ -n "$moved" ]; then
  say "!! 钉死的版本被挪动了:"; echo "$moved" | sed 's/^/     /'
  exit 1
fi
[ -f "$CANN_ENV" ] && { set +e; source "$CANN_ENV" >/dev/null 2>&1; set -e; }
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"   # miniforge:环境的 libstdc++ 要赢过 /usr/lib64
( cd "$HOME" && TORCH_DEVICE_BACKEND_AUTOLOAD=0 python - "$REPO_ROOT" <<'PYEOF'
import importlib, os, sys
from importlib.metadata import version
repo = os.path.realpath(sys.argv[1])
bad = 0
for m, dist in [("torch", "torch"), ("torch_npu", "torch-npu"), ("transformers", "transformers"),
                ("datasets", "datasets"), ("safetensors", "safetensors"), ("numpy", "numpy"),
                ("hs_connectors", "hs-connectors"), ("speculators", "speculators")]:
    try:
        mod = importlib.import_module(m)
        print(f"     {m:13s} {version(dist):22s} {os.path.dirname(mod.__file__)}")
    except Exception as e:  # noqa: BLE001
        print(f"     {m:13s} !! {type(e).__name__}: {str(e)[:160]}"); bad += 1
try:
    import speculators
    if not os.path.realpath(speculators.__file__).startswith(repo):
        print(f"  !! speculators 不是从 {repo} 导入的"); bad += 1
    from speculators.train.data import ArrowDataset  # noqa: F401
    import speculators.models.dsv4_dspark  # noqa: F401
    print("     训练侧 import(data / dsv4_dspark 模型)OK")
except Exception as e:  # noqa: BLE001
    print(f"  !! 训练侧 import 失败:{type(e).__name__}: {str(e)[:200]}"); bad += 1
sys.exit(1 if bad else 0)
PYEOF
) || { say "!! 导入核对没过,见上面。全量日志:$LOG"; exit 1; }

say "   pip check(只告警):"
python -m pip check 2>&1 | sed 's/^/     /' || true

echo "================================================================================"
say "装好了:conda activate $ENV_NAME"
echo "  下一步(dump 跑完、卡空出来之后):"
echo "    conda activate $ENV_NAME && cd $REPO_ROOT && \\"
echo "      MAX_STEPS=100 bash examples/ascend_npu_dflash/launch_a3_blk15_prestored.sh"
echo "================================================================================"
