#!/usr/bin/env bash
# One-shot install of the full Ascend-NPU stack for DFlash (train + serve + rollout),
# into the CURRENT (activated) Python 3.11 env. Bakes in every gotcha we hit:
#   1. vLLM PINNED to v0.20.2 (main builds 0.20.3.dev0 -> breaks vllm-ascend's
#      mla.prefill patch). Built fresh from the tag, not from a copied source tree.
#   2. vLLM built with VLLM_TARGET_DEVICE=empty (no CUDA kernels) + --no-build-isolation
#      (reuse env torch, don't re-download) + TORCH_DEVICE_BACKEND_AUTOLOAD=0
#      (so the build's `import torch` doesn't choke on torch_npu's deps).
#   3. numpy PINNED to 1.26.4 at the end (vLLM bumps it to 2.x which breaks
#      triton-ascend / scipy on NPU).
#   4. vllm-ascend 0.20.2rc1 via Huawei wheel; speculators installed --no-deps
#      (else it reinstalls CUDA torch + stock vllm and wipes torch-npu).
#
# PREREQS: CANN 9.0.0 toolkit installed at OS level; you are INSIDE your py311 env.
# USAGE:   bash examples/ascend_npu_dflash/install_npu_env.sh
# OVERRIDE: VLLM_DIR=/path/to/existing/vllm   (else a fresh v0.20.2 clone is made)
#           CANN_ENV=/path/to/ascend-toolkit/set_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"           # the speculators repo (this checkout)

CANN_ENV="${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
VLLM_DIR="${VLLM_DIR:-$REPO_ROOT/../vllm-v0.20.2}"     # fresh clone target if not provided
HW_PYPI="https://mirrors.huaweicloud.com/repository/pypi/simple"
HW_ASCEND="https://mirrors.huaweicloud.com/ascend/repos/pypi"
IDX=(--extra-index-url "$HW_PYPI" --extra-index-url "$HW_ASCEND")

echo "==================================================================="
echo " Ascend-NPU DFlash stack install  (speculators=$REPO_ROOT)"
echo "==================================================================="

echo "== 0. sanity: py311 + CANN =="
python -c "import sys; assert sys.version_info[:2]==(3,11), 'need py3.11, got %s'%sys.version" \
  || { echo "Activate your Python 3.11 env first."; exit 1; }
if [ -f "$CANN_ENV" ]; then source "$CANN_ENV"; echo "sourced CANN: $CANN_ENV"; \
  else echo "WARN: CANN set_env.sh not at $CANN_ENV — set CANN_ENV=... if your install differs"; fi

echo "== 1. build deps + torch/torch-npu 2.10.0 =="
python -m pip install -U pip setuptools "setuptools-scm>=8" wheel packaging "cmake>=3.26" ninja jinja2
python -m pip install "${IDX[@]}" torch==2.10.0 torch-npu==2.10.0 pyyaml

echo "== 2. vLLM @ v0.20.2 (empty build, no isolation) =="
if [ -n "${VLLM_DIR:-}" ] && [ -d "$VLLM_DIR/.git" ]; then
  echo "using existing vLLM at $VLLM_DIR -> checkout v0.20.2"
  git -C "$VLLM_DIR" fetch --depth 1 origin refs/tags/v0.20.2:refs/tags/v0.20.2 2>/dev/null \
    || git -C "$VLLM_DIR" fetch --tags
  git -C "$VLLM_DIR" checkout -f v0.20.2
else
  echo "cloning fresh vLLM v0.20.2 -> $VLLM_DIR"
  git clone --depth 1 --branch v0.20.2 https://github.com/vllm-project/vllm "$VLLM_DIR"
fi
( cd "$VLLM_DIR" && TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_TARGET_DEVICE=empty \
    python -m pip install -e . --no-build-isolation -v )

echo "== 3. vllm-ascend 0.20.2rc1 (prebuilt wheel) =="
python -m pip install "${IDX[@]}" vllm-ascend==0.20.2rc1

echo "== 4. CANN / profiler backfill (fresh-env Python deps) =="
python -m pip install decorator "scipy>=1.7.3" ml-dtypes attrs psutil pyyaml matplotlib openpyxl tornado

echo "== 5. speculators (--no-deps) + pure runtime/rollout deps =="
python -m pip install --no-deps -e "$REPO_ROOT"
python -m pip install datasets loguru typer pydantic-settings tensorboard aiohttp

echo "== 6. PIN numpy 1.26.4 (must be LAST — others bump it to 2.x) =="
python -m pip install "numpy==1.26.4"

echo "== 7. verify (full stack imports, torch_npu autoload ON) =="
python - <<'PY'
import numpy, scipy, torch, torch_npu, vllm, speculators
print("numpy    ", numpy.__version__)
print("scipy    ", scipy.__version__)
print("torch    ", torch.__version__)
print("torch_npu", getattr(torch_npu, "__version__", "loaded"))
print("vllm     ", vllm.__version__)
print("OK: full stack imports cleanly")
PY

echo "==================================================================="
echo " DONE. Expect: numpy 1.26.4 | torch 2.10.0 | vllm 0.20.2 (no .dev0)"
echo " Next: serve_qwen3_4b_nohup.sh + train_qwen3_4b_nohup.sh (see BASELINE.md)"
echo "==================================================================="
