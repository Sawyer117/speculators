#!/usr/bin/env bash
# One-shot, IDENTICAL-on-every-node setup for the DeepSeek-V4-Flash bf16 serve env.
#
# Run the SAME command on EVERY serve node (108, 109, and any future node) — paths are
# hardcoded so nodes never diverge (the 115/116 dspark_2026-vs-dspark_base mixup lesson).
#
#   # once per node:
#   git clone -b feat/dspark-confidence-head https://github.com/Sawyer117/speculators.git /home/a00652497/dspark_2026/speculators
#   CANN_ENV=<your CANN 9.0.0 set_env.sh> bash /home/a00652497/dspark_2026/speculators/examples/ascend_npu_dflash/setup_dsv4_env.sh
#
# PREREQS on the node (this script does NOT install them):
#   - CANN 9.0.0 installed + a source script (set CANN_ENV to it; it must put lld/ccec on PATH).
#   - host build utils: `sudo yum install -y patch gcc gcc-c++ make`  (patch is the easy-to-miss one).
#   - conda available; 8× NPU visible (`npu-smi info`).
#
# It builds: conda env `dspark-dsv4-base` (py3.11, torch/torch_npu 2.10.0, numpy 2.3.5),
# vLLM 0.23.0+empty, vllm-ascend @ `feat/dsv4-hs-dumper` (compiles the V4 CANN ops; our Plan B
# HS-dumper branch, based off dspark-dsv4 — pure-Python delta, same ops). Recipe = doc §4 verbatim,
# with consistent paths + the vllm-ascend ref PINNED to match 115/116.
# NB: no `set -u` — sourcing CANN/conda references unbound vars ($ZSH_VERSION) and would abort.
set -o pipefail

# ---- consistent config (override via env; defaults are the canonical layout) ----
ROOT="${ROOT:-/home/a00652497/dspark_2026}"
CANN_ENV="${CANN_ENV:-/home/a00652497/900env_npu.sh}"   # ★ CANN 9.0.0 source script — CONFIRM it exists on this node
CONDA_ENV="${CONDA_ENV:-dspark-dsv4-base}"
VA_REF="${VA_REF:-feat/dsv4-hs-dumper}"                  # pinned == 115/116 (Plan B HS-dumper branch)
SPEC_REF="${SPEC_REF:-feat/dspark-confidence-head}"      # speculators branch (serve + smoke + P3 driver + fixes)
HW="https://mirrors.huaweicloud.com"
INST="$ROOT/installation"; mkdir -p "$INST"
echo ">>> ROOT=$ROOT  CANN_ENV=$CANN_ENV  env=$CONDA_ENV  vllm-ascend=$VA_REF  speculators=$SPEC_REF"

# ---- source CANN + verify the toolchain (the exit-127 lesson: fail loud, don't guess) ----
[ -f "$CANN_ENV" ] || { echo "!! CANN source script not found: $CANN_ENV — set CANN_ENV=<set_env.sh>"; exit 2; }
# shellcheck disable=SC1090
source "$CANN_ENV"
for c in patch gcc lld ccec; do
  command -v "$c" >/dev/null || { echo "!! missing '$c' on PATH — build will exit-127. Fix (doc §4):"; \
    echo "   sudo yum install -y patch gcc gcc-c++ make ; re-source CANN"; \
    echo "   toolchain now:"; which gcc g++ lld ld.lld ccec bisheng patch; exit 2; }
done

# ---- conda env (NO conda compilers — they hijack CMake → ABI fail) ----
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -qw "$CONDA_ENV" || conda create -n "$CONDA_ENV" python=3.11 -y
conda activate "$CONDA_ENV"
conda remove -y gxx_linux-aarch64 gcc_linux-aarch64 clang clangxx lld 2>/dev/null; true
unset CC CXX

# ---- python deps + torch/torch_npu 2.10.0 ----
python -m pip install -U pip setuptools "setuptools-scm>=8" wheel packaging "cmake>=3.26" ninja jinja2 setuptools-rust pybind11
python -m pip install --extra-index-url $HW/repository/pypi/simple --extra-index-url $HW/ascend/repos/pypi torch==2.10.0 torch-npu==2.10.0 pyyaml
python -m pip install decorator "scipy>=1.7.3" ml-dtypes attrs psutil pyyaml matplotlib openpyxl tornado
# rollout / eval CLIENT deps — response_regeneration/script.py + Evaluator.py need these
# (serve itself doesn't; easy to miss until a rollout errors `No module named datasets`).
python -m pip install datasets aiohttp tqdm
python -c "import torch, torch_npu; print('torch', torch.__version__, torch_npu.__version__, torch_npu.npu.is_available(), torch_npu.npu.device_count())"

# ---- vLLM 0.23.0 (+empty) ----
[ -d "$INST/vllm-v0.23.0/.git" ] || git clone --depth 1 --branch v0.23.0 https://github.com/vllm-project/vllm.git "$INST/vllm-v0.23.0"
cd "$INST/vllm-v0.23.0"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_TARGET_DEVICE=empty python -m pip install -e . --no-build-isolation -v
python -c "import vllm; print('vllm', vllm.__version__)"

# ---- vllm-ascend @ pinned ref (compiles V4 CANN ops — the moment of truth) ----
[ -d "$INST/vllm-ascend-v4/.git" ] || git clone --branch "$VA_REF" https://github.com/Sawyer117/vllm-ascend.git "$INST/vllm-ascend-v4"
cd "$INST/vllm-ascend-v4"
git fetch origin && git checkout -B "$VA_REF" "origin/$VA_REF"
python -m pip install numba einops pandas msgpack
python -m pip install --no-deps torchvision==0.25.0 torchaudio==2.10.0 --extra-index-url $HW/repository/pypi/simple
python -m pip install triton-ascend==3.2.1 --extra-index-url $HW/repository/pypi/simple --extra-index-url $HW/ascend/repos/pypi
pip install --no-deps "numpy==2.3.5"    # triton-ascend silently downgrades numpy → force back (runs fine at 2.3.5)
rm -rf build csrc/build                  # MANDATORY between builds — CMake caches the compiler choice
pip install -e . --no-deps --no-build-isolation -v 2>&1 | tee ~/va_build.log
python -c "import numpy; print('numpy', numpy.__version__)"
python -c "import vllm_ascend; print('vllm_ascend OK', vllm_ascend.__file__)"

# ---- ATB/NNAL env GATE (the libatb.so / "Mki::Dl undefined symbol" lesson) ----
# The serve loads torch_npu's ATB ops, which needs THIS node's CANN 9.0.0 `nnal/atb/set_env.sh`
# sourced by your CANN_ENV — NOT a stale CANN 8.5.1 atb, and NOT only nnal/asdsip. Fail LOUD here
# so a bad 900env is caught at setup, not 20 minutes into a serve. (Also: the serve reads the file
# at its hardcoded CANN_ENV — edit THAT file, not a personal copy in your own $HOME.)
echo ""; echo ">>> ATB/NNAL gate (must pass, else serve dies with libatb.so / Mki undefined symbol):"
echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -qi '/nnal/atb/.*/lib' \
  || echo "   WARN: no nnal/atb lib dir on LD_LIBRARY_PATH — CANN_ENV ($CANN_ENV) probably doesn't source the 9.0.0 nnal/atb/set_env.sh"
if python -c "from torch_npu.op_plugin.atb import _atb_ops" 2>/dev/null; then
  echo "   atb ops OK"
else
  echo "!! ATB FAILED to load. Fix CANN_ENV ($CANN_ENV) — add THIS node's CANN 9.0.0 atb (verify the path with"
  echo "   'find <CANN> -name libatb.so'; use the 9.0.0 tree, NOT any CANN 8.5.1 copy):"
  echo "       source <CANN>/nnal/atb/set_env.sh"
  echo "   Re-verify in a FRESH shell that sourced ONLY that file:"
  echo "       echo \$LD_LIBRARY_PATH | tr : '\\n' | grep atb   # must show the atb lib dir"
  echo "       python -c 'from torch_npu.op_plugin.atb import _atb_ops; print(1)'"
  echo "   And confirm you edited the file the SERVE sources (its CANN_ENV), not a personal copy:"
  echo "       grep -nE 'atb|nnal' /home/a00652497/900env_npu.sh"
  exit 3
fi

# ---- speculators repo (no-op if you already cloned it to run this script) ----
[ -d "$ROOT/speculators/.git" ] || git clone --branch "$SPEC_REF" https://github.com/Sawyer117/speculators.git "$ROOT/speculators"
git -C "$ROOT/speculators" fetch origin && git -C "$ROOT/speculators" checkout -B "$SPEC_REF" "origin/$SPEC_REF"

echo ""
echo ">>> DONE on $(hostname). Consistency check — must be IDENTICAL on 108 & 109:"
echo "    vllm-ascend: $(git -C "$INST/vllm-ascend-v4" log --oneline -1)"
echo "    vllm       : $(python -c 'import vllm; print(vllm.__version__)')"
echo "    numpy      : $(python -c 'import numpy; print(numpy.__version__)')"
echo ">>> next (A2 pair): serve_dsv4_bf16_dualnode.sh (108=head, 109=worker)"
echo ">>> next (A3 single node): serve_dsv4_a3_singlenode.sh  (DP2/TP8/EP16; run under nohup)"
