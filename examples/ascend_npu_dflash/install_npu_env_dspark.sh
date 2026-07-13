#!/usr/bin/env bash
# GOLD-STANDARD one-shot installer for the DSpark / DSV4-Flash Ascend-NPU stack
# (vLLM 0.23.0 + vllm-ascend BUILT FROM SOURCE), into the CURRENT (activated) py3.11 env.
#
# This script is the SINGLE SOURCE OF TRUTH for the DSpark/DSV4 env — the markdown
# install docs (ascend-npu-dspark-install.md / ascend-npu-dsv4-dspark.md) defer to it.
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
#   VLLM_DIR         existing vLLM checkout to build editable (default $ROOT/installation/vllm-v0.23.0)
#   VA_DIR           existing vllm-ascend checkout   (default $ROOT/installation/vllm-ascend-v4)
#   VA_BRANCH        vllm-ascend branch if cloning fresh (default dspark-dsv4)
#   NUMPY_VER        default 2.3.5 (verified); set 1.26.4 for the conservative pin
#   CANN_ENV         path to ascend-toolkit/set_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"                 # the speculators checkout
ROOT="${ROOT:-$(cd "$REPO_ROOT/.." && pwd)}"                 # code root (installation/ + speculators/)
VLLM_DIR="${VLLM_DIR:-$ROOT/installation/vllm-v0.23.0}"
VA_DIR="${VA_DIR:-$ROOT/installation/vllm-ascend-v4}"
VA_BRANCH="${VA_BRANCH:-dspark-dsv4}"
NUMPY_VER="${NUMPY_VER:-2.3.5}"
CANN_ENV="${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
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

echo "== 3. vLLM v0.23.0 (empty build, editable) =="
[ -d "$VLLM_DIR/.git" ] || git clone --depth 1 --branch v0.23.0 https://github.com/vllm-project/vllm "$VLLM_DIR"
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
python -m pip install --no-deps -e "$ROOT/speculators" 2>/dev/null || python -m pip install --no-deps -e "$REPO_ROOT"
python -m pip install datasets loguru typer pydantic-settings tensorboard aiohttp

echo "== 7. FORCE numpy $NUMPY_VER (LAST pip op — triton-ascend<2 downgraded it) + verify =="
python -m pip install --no-deps "numpy==$NUMPY_VER"
NUMPY_VER="$NUMPY_VER" python - <<'PY'
import os, numpy, torch, torch_npu, torchgen.model, vllm, vllm_ascend, speculators
want = os.environ["NUMPY_VER"]
print("numpy      ", numpy.__version__, "(want", want + ")", "OK" if numpy.__version__ == want else "!! MISMATCH")
print("torch      ", torch.__version__, "| vllm", vllm.__version__)
print("vllm_ascend", vllm_ascend.__file__)   # must be under your code root, not someone else's
print("speculators", speculators.__file__)
print("OK: DSpark/DSV4 stack imports cleanly")
PY

echo "==================================================================="
echo " DONE. Expect: numpy $NUMPY_VER | torch 2.10.0 | vllm 0.23.0 | vllm-ascend from $VA_DIR"
echo " NOTE: serve/eval also needs the CANN 9.0.0 nnal/atb set_env sourced in a clean shell."
echo "==================================================================="
