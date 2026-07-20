#!/usr/bin/env bash
# One-shot A3 (single-node, 16 logical cards) setup for the DeepSeek-V4-Flash bf16 **EVAL SERVE**
# env — the #12006 DSpark rewrite stack (`dspark-dsv4-v3`), the one on which the released draft
# scores gsm8k accept_len 4.658. Run once on a FRESH A3 box; re-runnable (idempotent-ish).
#
#   # once per A3 node (clone this repo first, any branch with this script):
#   git clone -b feat/dsv4-dspark https://github.com/Sawyer117/speculators.git /home/a00652497/dspark_2026/speculators
#   CANN_ENV=/home/a00652497/900env_npu.sh \
#     bash /home/a00652497/dspark_2026/speculators/examples/ascend_npu_dflash/setup_dsv4_serve_a3.sh
#
# WHAT differs from setup_dsv4_env.sh (the A2 dumper env):
#   - Builds vllm-ascend from **upstream vllm-project PR #12006** (the DSpark REWRITE / eval-serve
#     stack), NOT our `feat/dsv4-hs-dumper` branch (that one is for TRAINING-side HS extraction).
#     #12006 = #12005 (eager correctness) + the python-only ACLGraph commit; requirements pin
#     torch==torch_npu==2.10.0 (same as the box, NO torch bump), and 83 csrc files changed vs base
#     so the CANN custom ops ARE recompiled here (the real work, ~30-40 min).
#   - env name `dspark-dsv4-serving` (override with CONDA_ENV=...). Keeps any -base env intact.
#   - A3 paths: weights live under /home/canada_group_folder (A3 has no /share).
#
# PREREQS (this script does NOT install them; the pre-flight checks + errors loudly):
#   - CANN 9.0.0 + a source script (CANN_ENV) that puts lld/ccec on PATH **and** sources the
#     9.0.0 nnal/atb/set_env.sh (NOT a stale 8.5.1 atb — the libatb.so / "Mki::Dl" lesson).
#   - host build utils: sudo yum install -y patch gcc gcc-c++ make   (patch is the easy miss).
#   - conda; 16 NPU logical cards visible (npu-smi info).
#   - proxy that resolves github.com / pypi (A3 is behind a MITM proxy; git/pip use http(s)_proxy).
# NB: no `set -u` — sourcing CANN/conda references unbound vars ($ZSH_VERSION) and would abort.
set -o pipefail

# ---- config (override via env) ----
ROOT="${ROOT:-/home/a00652497/dspark_2026}"
CANN_ENV="${CANN_ENV:-/home/a00652497/900env_npu.sh}"      # ★ CANN 9.0.0 source script
CONDA_ENV="${CONDA_ENV:-dspark-dsv4-serving}"              # eval serve env (#12006)
VA_UPSTREAM="${VA_UPSTREAM:-https://github.com/vllm-project/vllm-ascend.git}"
VA_PR="${VA_PR:-12006}"                                    # DSpark rewrite PR (eager+ACLGraph)
VA_COMMIT="${VA_COMMIT:-386530d12}"                        # pin the #12006 head we validated (4.658);
                                                          # set VA_COMMIT="" to take the live PR head.
HW="https://mirrors.huaweicloud.com"
INST="$ROOT/installation"; mkdir -p "$INST"
VA_DIR="$INST/vllm-ascend-serving"
echo ">>> A3 SERVE SETUP  ROOT=$ROOT  env=$CONDA_ENV  vllm-ascend=PR#$VA_PR @ ${VA_COMMIT:-<live head>}"

# ---- pre-flight: CANN + toolchain (fail loud; the exit-127 lesson) ----
[ -f "$CANN_ENV" ] || { echo "!! CANN source script not found: $CANN_ENV — set CANN_ENV=<set_env.sh>"; exit 2; }
# shellcheck disable=SC1090
source "$CANN_ENV"
for c in patch gcc lld ccec; do
  command -v "$c" >/dev/null || { echo "!! missing '$c' on PATH — build will exit-127. Fix:"; \
    echo "   sudo yum install -y patch gcc gcc-c++ make ; re-source CANN"; \
    which gcc g++ lld ld.lld ccec bisheng patch; exit 2; }
done

# ---- conda env (NO conda compilers — they hijack CMake → ABI fail) ----
source "$(conda info --base)/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
conda env list | grep -qw "$CONDA_ENV" || conda create -n "$CONDA_ENV" python=3.11 -y
conda activate "$CONDA_ENV"
conda remove -y gxx_linux-aarch64 gcc_linux-aarch64 clang clangxx lld 2>/dev/null; true
unset CC CXX

# ---- python deps + torch/torch_npu 2.10.0 ----
python -m pip install -U pip setuptools "setuptools-scm>=8" wheel packaging "cmake>=3.26" ninja jinja2 setuptools-rust pybind11
python -m pip install --extra-index-url $HW/repository/pypi/simple --extra-index-url $HW/ascend/repos/pypi torch==2.10.0 torch-npu==2.10.0 pyyaml
python -m pip install decorator "scipy>=1.7.3" ml-dtypes attrs psutil pyyaml matplotlib openpyxl tornado
# eval CLIENT deps (Evaluator.py / gsm8k) — serve itself doesn't need them but eval does
python -m pip install datasets aiohttp tqdm
python -c "import torch, torch_npu; print('torch', torch.__version__, torch_npu.__version__, torch_npu.npu.is_available(), torch_npu.npu.device_count())"

# ---- vLLM 0.23.0 (+empty) — SAME vllm #12006 pins; do NOT rebuild if present ----
[ -d "$INST/vllm-v0.23.0/.git" ] || git clone --depth 1 --branch v0.23.0 https://github.com/vllm-project/vllm.git "$INST/vllm-v0.23.0"
cd "$INST/vllm-v0.23.0"
# ★ guard on the INSTALLED dist (pip show), NOT `import vllm` — the import check FALSE-PASSES here
# because cwd IS the vllm source dir, so `import vllm` picks up ./vllm/ (note the "_version" warning)
# and the real install is skipped → later `from vllm import envs` in vllm_ascend dies with
# "No module named 'vllm'". pip show reflects the actual editable install regardless of cwd.
python -m pip show vllm >/dev/null 2>&1 || TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_TARGET_DEVICE=empty python -m pip install -e . --no-build-isolation -v
( cd /tmp && python -c "import vllm; print('vllm', vllm.__version__)" )   # verify from a neutral cwd

# ---- vllm-ascend @ PR #12006 (compiles V4 CANN ops — the moment of truth, ~30-40 min) ----
if [ ! -d "$VA_DIR/.git" ]; then
  git clone "$VA_UPSTREAM" "$VA_DIR"
fi
cd "$VA_DIR"
git fetch origin "pull/$VA_PR/head:dspark-dsv4-v3"
git checkout -B dspark-dsv4-v3 dspark-dsv4-v3
[ -n "$VA_COMMIT" ] && git checkout "$VA_COMMIT" 2>/dev/null || echo ">>> (using live PR#$VA_PR head)"
echo ">>> vllm-ascend at: $(git log --oneline -1)"
# ★ regex is a BUILD-TIME dep of the CANN op codegen (csrc/.../ascendc_impl_build.py `import regex`).
# The serve env installs no transformers (which normally drags regex in), so add it explicitly here —
# else build_aclnn dies with "No module named 'regex'" and vllm_ascend never builds.
python -m pip install numba einops pandas msgpack regex
python -m pip install --no-deps torchvision==0.25.0 torchaudio==2.10.0 --extra-index-url $HW/repository/pypi/simple
python -m pip install triton-ascend==3.2.1 --extra-index-url $HW/repository/pypi/simple --extra-index-url $HW/ascend/repos/pypi
pip install --no-deps "numpy==2.3.5"    # triton-ascend silently downgrades numpy → force back
rm -rf build csrc/build                  # MANDATORY between builds — CMake caches the compiler choice
pip install -e . --no-deps --no-build-isolation -v 2>&1 | tee ~/va_serve_build.log
python -c "import numpy; print('numpy', numpy.__version__)"
python -c "import vllm_ascend; print('vllm_ascend OK', vllm_ascend.__file__)"

# ---- ATB/NNAL env GATE (libatb.so / "Mki::Dl undefined symbol" lesson) ----
echo ""; echo ">>> ATB/NNAL gate (must pass, else serve dies with libatb.so / Mki undefined symbol):"
echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -qi '/nnal/atb/.*/lib' \
  || echo "   WARN: no nnal/atb lib dir on LD_LIBRARY_PATH — CANN_ENV ($CANN_ENV) probably doesn't source the 9.0.0 nnal/atb/set_env.sh"
if python -c "from torch_npu.op_plugin.atb import _atb_ops" 2>/dev/null; then
  echo "   atb ops OK"
else
  echo "!! ATB FAILED to load. Fix CANN_ENV ($CANN_ENV) to source THIS node's CANN 9.0.0 nnal/atb/set_env.sh"
  echo "   (find <CANN> -name libatb.so ; use the 9.0.0 tree, NOT a CANN 8.5.1 copy), then re-run in a FRESH shell."
  exit 3
fi

echo ""
echo ">>> DONE on $(hostname)."
echo "    vllm-ascend: $(git -C "$VA_DIR" log --oneline -1)"
echo "    vllm       : $(python -c 'import vllm; print(vllm.__version__)')"
echo "    numpy      : $(python -c 'import numpy; print(numpy.__version__)')"
echo ">>> next: serve  →  CONDA_ENV=$CONDA_ENV MODEL=/home/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16 \\"
echo "            DRAFT=/home/canada_group_folder/ckpt/released_draft_bf16_standalone NUM_SPEC=5 EAGER=1 \\"
echo "            nohup bash examples/ascend_npu_dflash/serve_dsv4_a3_singlenode.sh > ~/dsv4_a3.log 2>&1 &"
