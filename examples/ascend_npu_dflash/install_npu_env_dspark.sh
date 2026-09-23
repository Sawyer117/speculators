#!/usr/bin/env bash
# GOLD-STANDARD one-shot installer for the DSpark / DSV4-Flash Ascend-NPU stack
# (vLLM 0.23.0 + vllm-ascend BUILT FROM SOURCE), into the CURRENT (activated) py3.11 env.
#
# This script is the SINGLE SOURCE OF TRUTH for the DSpark/DSV4 env — the markdown
# install docs (ascend-npu-dspark-install.md / ascend-npu-dsv4-dspark-w8a8-inference.md) defer to it.
# Fix the recipe HERE, once; propagate to feature branches by merging main forward.
#
# Differs from the DFlash install_npu_env.sh (which is a separate, older stack):
#   * vLLM v0.23.0 (not v0.20.2)
#   * vllm-ascend built FROM SOURCE on the DSpark branch — compiles the V4/SAS CANN ops
#     (there is NO prebuilt wheel for it), so the host toolchain matters (see step 2).
#
# EVERY hard-won gotcha is baked in:
#   * numpy 2.3.5 forced LAST with --no-deps (triton-ascend pins numpy<2 and would
#     silently downgrade it). Set NUMPY_VER=1.26.4 for the conservative pin.
#   * TOOLCHAIN: system gcc + CANN's own lld/ccec/bisheng, ZERO conda compilers.
#     conda gxx hijacks CMake -> opbuild ABI fail; `export CC/CXX` breaks the AICPU
#     cross-compile. `patch` MUST be installed or the op build dies `exit 127`.
#   * rm -rf csrc/build before the vllm-ascend rebuild (a stale build reuses the wrong gcc).
#
# PREREQS: CANN 9.0.0 at OS level; you are INSIDE your py3.11 env; sudo/admin only to
#   `yum install` host `patch`/gcc if missing.
# USAGE:   bash examples/ascend_npu_dflash/install_npu_env_dspark.sh
# OVERRIDES (env):
#   ROOT=<dir>       code root holding installation/ + speculators/ (default: repo's ../..)
#   VLLM_TAG         vLLM tag to clone/verify (default v0.23.0). ★ MUST match what the
#                    vllm-ascend pin's Dockerfile ARG VLLM_TAG says — a mismatched pair
#                    builds and installs fine and then dies in `vllm serve --help`.
#   VLLM_DIR         existing vLLM checkout to build editable (default $ROOT/installation/vllm-$VLLM_TAG)
#   VA_DIR           existing vllm-ascend checkout   (default $ROOT/installation/vllm-ascend-v4)
#   VA_BRANCH        vllm-ascend branch if cloning fresh (default dspark-dsv4)
#   NUMPY_VER        default 2.3.5 (verified); set 1.26.4 for the conservative pin
#   CANN_ENV         path to ascend-toolkit/set_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"                 # the speculators checkout
ROOT="${ROOT:-$(cd "$REPO_ROOT/.." && pwd)}"                 # code root (installation/ + speculators/)
VLLM_TAG="${VLLM_TAG:-v0.23.0}"
VLLM_DIR="${VLLM_DIR:-$ROOT/installation/vllm-$VLLM_TAG}"
VA_DIR="${VA_DIR:-$ROOT/installation/vllm-ascend-v4}"
VA_BRANCH="${VA_BRANCH:-dspark-dsv4}"
NUMPY_VER="${NUMPY_VER:-2.3.5}"
CANN_ENV="${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
# SKIP_SPECULATORS=1 —— 不把 speculators 装进这个 env。serve 用不到它:它的 pyproject
#   只注册了 console_scripts,没有 vllm.general_plugins / vllm.platform_plugins,所以
#   vLLM 起服务时根本不会 import 它。要一个「只有推理栈」的干净对照环境时用这个。
#   探针要的 datasets/pyarrow/requests 照装。
SKIP_SPECULATORS="${SKIP_SPECULATORS:-0}"
# TRANSFORMERS_VER=<ver> —— 钉死 transformers;不设 = 沿用原行为(让 pip 自己解)。
#   2026-09-23 实测不钉的后果:新装的 env 拿到 PyPI 当天的 5.17.0,而 vllm-ascend 的
#   requirements 写的是 5.14.1、speculators 要求 <5.15.0 —— 两边都被违反,安装却照样
#   「成功」。第 7b 步现在会把这类违反单独打一遍。
TRANSFORMERS_VER="${TRANSFORMERS_VER:-}"
HW_PYPI="https://mirrors.huaweicloud.com/repository/pypi/simple"
HW_ASCEND="https://mirrors.huaweicloud.com/ascend/repos/pypi"
IDX=(--extra-index-url "$HW_PYPI" --extra-index-url "$HW_ASCEND")

echo "==================================================================="
echo " DSpark/DSV4 Ascend-NPU stack  (speculators=$REPO_ROOT, numpy=$NUMPY_VER)"
echo "==================================================================="

echo "== 0. sanity: py311 + source CANN =="
python -c "import sys; assert sys.version_info[:2]==(3,11), 'need py3.11, got %s'%sys.version" \
  || { echo "Activate your py3.11 env first."; exit 1; }
[ -f "$CANN_ENV" ] && { source "$CANN_ENV"; echo "sourced CANN: $CANN_ENV"; } \
  || echo "WARN: CANN set_env not at $CANN_ENV — set CANN_ENV=... (needed to compile the ops)"

echo "== 1. build deps + torch/torch-npu 2.10.0 + numpy $NUMPY_VER + CANN backfill =="
python -m pip install -U pip setuptools "setuptools-scm>=8" wheel packaging "cmake>=3.26" ninja jinja2 setuptools-rust pybind11
python -m pip install "${IDX[@]}" torch==2.10.0 torch-npu==2.10.0 pyyaml
python -m pip install "numpy==$NUMPY_VER"
# CANN op compiler (TBE/TVM) imports these DURING the build (step 4) — install BEFORE it.
python -m pip install decorator "scipy>=1.7.3" ml-dtypes attrs psutil pyyaml matplotlib openpyxl tornado
python -c "import torch, torch_npu, torchgen.model, numpy as n; print('torch', torch.__version__, '| numpy', n.__version__, '| npu', torch_npu.npu.is_available())"

echo "== 2. host toolchain: system gcc + CANN (NO conda compilers) =="
# The vllm-ascend op build (build_aclnn.sh) shells out to `patch` and wants SYSTEM gcc +
# CANN's own lld/ccec/bisheng. Conda gxx hijacks CMake (opbuild ABI fail); never export CC/CXX.
conda remove -y gxx_linux-aarch64 gcc_linux-aarch64 clang clangxx lld >/dev/null 2>&1 || true  # unhijack CMake
unset CC CXX || true
if ! command -v patch >/dev/null || ! command -v gcc >/dev/null || ! command -v make >/dev/null; then
  echo "installing host build utils (needs sudo)…"
  sudo yum install -y patch gcc gcc-c++ make || {
    echo "!! could not auto-install. Ask admin: sudo yum install -y patch gcc gcc-c++ make"
    echo "   (\`patch\` is the usual 'FAILED: [code=127]' culprit in the op build.)"; exit 1; }
fi
for t in gcc g++ make patch; do command -v "$t" >/dev/null || { echo "!! missing host tool: $t"; exit 1; }; done
echo "toolchain OK: gcc=$(command -v gcc) | patch=$(command -v patch) | lld=$(command -v lld 2>/dev/null || echo 'from CANN')"

echo "== 3. vLLM $VLLM_TAG (empty build, editable) =="
if [ -d "$VLLM_DIR/.git" ]; then
  # ★ 2026-09-23:目录存在就跳过 clone,于是【目录名和里面的版本可以不一致】。
  #   实际踩到:install_oldstack_a3.sh 传了 VLLM_DIR=.../vllm-v0.27.1 却没传 VLLM_TAG,
  #   这里把写死的 v0.23.0 克隆了进去 —— 装完是 vLLM 0.23.0 配 vllm-ascend 44bbd5ea3
  #   (它要 0.27.1),`vllm serve --help` 崩在 CLI 解析上。目录名在说谎,而且没人会怀疑它。
  _have="$(git -C "$VLLM_DIR" describe --tags --exact-match 2>/dev/null \
            || git -C "$VLLM_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
  if [ "$_have" != "$VLLM_TAG" ]; then
    echo "!! $VLLM_DIR 里是 vLLM '$_have',但本次要的是 '$VLLM_TAG'。" >&2
    echo "   目录名不代表内容。修法(会重新 clone + 重编):" >&2
    echo "     rm -rf '$VLLM_DIR'  然后带 VLLM_TAG=$VLLM_TAG 重跑" >&2
    exit 2
  fi
  echo "   复用已有 checkout:$VLLM_DIR ($_have)"
else
  git clone --depth 1 --branch "$VLLM_TAG" https://github.com/vllm-project/vllm "$VLLM_DIR"
fi
( cd "$VLLM_DIR" && TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_TARGET_DEVICE=empty \
    python -m pip install -e . --no-build-isolation -v )

echo "== 4. vllm-ascend @ $VA_BRANCH — FROM SOURCE (compiles the V4/SAS CANN ops) =="
[ -d "$VA_DIR/.git" ] || git clone --branch "$VA_BRANCH" --single-branch https://github.com/Sawyer117/vllm-ascend.git "$VA_DIR"
( cd "$VA_DIR" && rm -rf csrc/build && pip install -e . --no-deps --no-build-isolation -v )

echo "== 5. vllm-ascend runtime extras (--no-deps protects torch) =="
python -m pip install numba einops pandas msgpack
python -m pip install --no-deps torchvision==0.25.0 torchaudio==2.10.0 --extra-index-url "$HW_PYPI"
# triton-ascend REQUIRED (block_table slot-mapping kernel at runtime); it pins numpy<2 -> re-forced in step 7.
python -m pip install triton-ascend==3.2.1 "${IDX[@]}"

echo "== 6. speculators (--no-deps) + train/rollout deps =="
if [ "$SKIP_SPECULATORS" = "1" ]; then
  echo "   SKIP_SPECULATORS=1 —— 跳过 speculators,只装探针/评测要的依赖"
else
  python -m pip install --no-deps -e "$ROOT/speculators" 2>/dev/null || python -m pip install --no-deps -e "$REPO_ROOT"
fi
python -m pip install datasets loguru typer pydantic-settings tensorboard aiohttp
if [ -n "$TRANSFORMERS_VER" ]; then
  echo "   钉 transformers==$TRANSFORMERS_VER"
  python -m pip install "transformers==$TRANSFORMERS_VER"
fi

echo "== 7. FORCE numpy $NUMPY_VER (LAST pip op — triton-ascend<2 downgraded it) + verify =="
python -m pip install --no-deps "numpy==$NUMPY_VER"
NUMPY_VER="$NUMPY_VER" SKIP_SPECULATORS="$SKIP_SPECULATORS" python - <<'PY'
import os, numpy, torch, torch_npu, torchgen.model, vllm, vllm_ascend
want = os.environ["NUMPY_VER"]
print("numpy      ", numpy.__version__, "(want", want + ")", "OK" if numpy.__version__ == want else "!! MISMATCH")
print("torch      ", torch.__version__, "| vllm", vllm.__version__)
print("vllm_ascend", vllm_ascend.__file__)   # must be under your code root, not someone else's
if os.environ.get("SKIP_SPECULATORS") == "1":
    print("speculators <not installed — SKIP_SPECULATORS=1>")
else:
    import speculators
    print("speculators", speculators.__file__)
import transformers, tokenizers
print("transformers", transformers.__version__, "| tokenizers", tokenizers.__version__)
print("OK: DSpark/DSV4 stack imports cleanly")
PY

# ★ 「装完了」不等于「版本对」。pip 解不开依赖时只打一段 ERROR 然后继续,整个安装照样
#   以 0 退出,而那段 ERROR 早被后面几千行 pip 输出冲走。2026-09-23 实测:一个全新 env
#   里 transformers / torch-npu / triton-ascend / fastapi 四条 pin 同时被违反,所有步骤
#   都「成功」。所以最后单独再打一遍。
echo "== 7b. 依赖一致性(pip check)—— 不致命,但每一条都要看过 =="
python -m pip check || echo "   ↑ 每一条都是【已装版本 ≠ 某个包声明的 pin】。serve 起不来先查这里。"

echo "==================================================================="
echo " DONE. Expect: numpy $NUMPY_VER | torch 2.10.0 | vllm ${VLLM_TAG#v} | vllm-ascend from $VA_DIR"
echo " NOTE: serve/eval also needs the CANN 9.0.0 nnal/atb set_env sourced in a clean shell."
echo "==================================================================="
