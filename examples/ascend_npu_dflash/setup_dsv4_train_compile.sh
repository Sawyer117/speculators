#!/usr/bin/env bash
# One-shot TRAINING env for the DSV4-DSpark draft on a fresh A3 box (e.g. 176), on the
# **torch-2.12 COMPILE stack** — the seed-tech: torch.compile'd grouped-GEMM experts
# (~1.74×, bit-exact vs eager). NO vLLM / vllm-ascend — training doesn't need them (they're
# serve/eval only), so this is much lighter than the serve env (no CANN-op csrc build).
#
#   # once on the box (clone first to get this script + the speculators source it installs editable):
#   git clone -b feat/dsv4-dspark https://github.com/Sawyer117/speculators.git \
#       /home/a00652497/dspark_austin/speculators
#   CANN_ENV=/home/a00652497/900env_npu.sh \
#       bash /home/a00652497/dspark_austin/speculators/examples/ascend_npu_dflash/setup_dsv4_train_compile.sh
#
# Recipe = docs/deployment/ascend-npu-dsv4-dspark-compile-recompile.md (the 2.12 stack section,
# validated in the `dspark-dsv4-compile` clone via test_compile_grouped_mm.py). The doc builds by
# CLONING the 2.10 base then swapping torch; a FRESH box has no base to clone, so we build the whole
# thing from scratch here (same end state).
#
# ⚠️ TWO version/source points to CROSS-CHECK against the working `dspark-dsv4-compile` you built:
#   (1) torch_npu==2.12.0rc1 comes from the Ascend pip source (set TORCH_NPU_IDX if yours differs);
#   (2) inductor_npu_ext lives on **gitcode.com** — if 176 can't reach gitcode, mirror torchair to
#       github and pass TORCHAIR_URL=..., or scp the checkout in.
# NB: no `set -u` — sourcing CANN/conda references unbound vars ($ZSH_VERSION) and would abort.
set -o pipefail

# ---- config (override via env) ----
ROOT="${ROOT:-/home/a00652497/dspark_austin}"
CANN_ENV="${CANN_ENV:-/home/a00652497/900env_npu.sh}"        # CANN 9.0.0 source (torch_npu 2.12rc1 runs on it)
CONDA_ENV="${CONDA_ENV:-dspark-dsv4-compile}"                # matches the env you validated compile in
SPEC_REF="${SPEC_REF:-feat/dsv4-dspark}"
TORCHAIR_URL="${TORCHAIR_URL:-https://gitcode.com/Ascend/torchair.git}"   # ★ gitcode — see note (2)
TORCHAIR_COMMIT="${TORCHAIR_COMMIT:-3c9418c2}"               # the inductor_npu_ext commit the doc pins
NUMPY_VER="${NUMPY_VER:-2.3.5}"
CPU_IDX="${CPU_IDX:-https://download.pytorch.org/whl/cpu}"   # ★ torch 2.12.0 +cpu wheel lives here
HW="https://mirrors.huaweicloud.com"
TORCH_NPU_IDX="${TORCH_NPU_IDX:-$HW/ascend/repos/pypi}"      # torch_npu 2.12.0rc1 source
INST="$ROOT/installation"; mkdir -p "$INST"
echo ">>> 176 TRAIN(compile) env=$CONDA_ENV  torch 2.12.0+cpu / torch_npu 2.12.0rc1  ROOT=$ROOT"

# ---- pre-flight: CANN (torch_npu runtime) + gcc. NO lld/ccec/patch — training compiles no CANN ops. ----
[ -f "$CANN_ENV" ] || { echo "!! CANN source not found: $CANN_ENV — set CANN_ENV=<set_env.sh>"; exit 2; }
# shellcheck disable=SC1090
source "$CANN_ENV"
command -v gcc >/dev/null || { echo "!! gcc missing on PATH (sudo yum install -y gcc gcc-c++ make)"; exit 2; }

# ---- conda env (fresh py3.11; NO conda compilers) ----
source "$(conda info --base)/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
conda env list | grep -qw "$CONDA_ENV" || conda create -n "$CONDA_ENV" python=3.11 -y
conda activate "$CONDA_ENV"
conda remove -y gxx_linux-aarch64 gcc_linux-aarch64 clang clangxx lld 2>/dev/null; true
unset CC CXX

# ---- build deps ----
python -m pip install -U pip setuptools "setuptools-scm>=8" wheel packaging ninja jinja2 pybind11

# ---- torch 2.12.0 +cpu (★ MUST be the +cpu wheel via the CPU index — the default aarch64 wheel is
#      +cu130 and makes torch_npu 2.12.0rc1 crash at import) + torch_npu 2.12.0rc1 (Ascend source) ----
# ★ the box proxy MITMs pytorch.org with a self-signed cert (SSLCertVerificationError); trust the
#   pytorch hosts. download.pytorch.org 302-redirects to download-r2.pytorch.org, so trust BOTH.
#   (override the full set via PIP_TRUSTED if your proxy differs.)
PIP_TRUSTED="${PIP_TRUSTED:---trusted-host download.pytorch.org --trusted-host download-r2.pytorch.org --trusted-host pypi.org --trusted-host files.pythonhosted.org}"
python -m pip install "torch==2.12.0" --index-url "$CPU_IDX" $PIP_TRUSTED
# ★ HARD GATE: if torch is not +cpu (e.g. SSL still failed → torch_npu pulled +cu130), STOP now —
#   otherwise torch_npu drags cuda-toolkit + nvidia-* and the whole stack is wrong.
python -c "import torch,sys; v=torch.__version__; print('torch', v); sys.exit(0 if '+cpu' in v else 1)" \
  || { echo '!! torch is NOT +cpu — fix the pytorch.org SSL/proxy (PIP_TRUSTED / REQUESTS_CA_BUNDLE) and rerun; do NOT continue.'; exit 4; }
python -m pip install "torch_npu==2.12.0rc1" --extra-index-url "$TORCH_NPU_IDX"
python -m pip install decorator "scipy>=1.7.3" ml-dtypes attrs psutil pyyaml
# do NOT `from torch_npu.contrib import transfer_to_npu` — it crashes on this stack; plain import only.
python -c "import torch, torch_npu; print('torch', torch.__version__, '| torch_npu', torch_npu.__version__, '| npu', torch_npu.npu.is_available())"

# ---- inductor_npu_ext (Ascend torchair experimental) = the AscendC Inductor codegen backend ----
# ★ --no-deps MANDATORY: it declares torch>=2.8.0 and would otherwise pull torch-2.13 + cuda-toolkit +
#   nvidia-* (a CUDA torch) and clobber torch_npu.
if [ ! -d "$INST/torchair/.git" ]; then
  git clone "$TORCHAIR_URL" "$INST/torchair" || { echo "!! clone failed ($TORCHAIR_URL). If 176 can't reach gitcode, mirror torchair to github and re-run with TORCHAIR_URL=<that>."; exit 3; }
fi
( cd "$INST/torchair" && git fetch --all -q && git checkout "$TORCHAIR_COMMIT" )
python -m pip install -e "$INST/torchair/experimental/_inductor_npu_ext/python/" --no-deps

# ---- Ascend Triton (the inductor codegen needs triton-ascend, NOT the cuda-triton) ----
python -m pip uninstall -y triton 2>/dev/null; true
python -m pip install triton-ascend --extra-index-url "$HW/repository/pypi/simple" --extra-index-url "$HW/ascend/repos/pypi"

# ---- speculators (editable, --no-deps) + transformers + train/rollout/HS-fetch deps ----
python -m pip install --no-deps -e "$ROOT/speculators"
python -m pip install "transformers>=4.56.1,<5.14.0"
python -m pip install datasets safetensors requests aiohttp tqdm loguru typer pydantic-settings tensorboard matplotlib

# ---- numpy 2.3.5 forced LAST (triton-ascend pins numpy<2 and would downgrade it) ----
python -m pip install --no-deps "numpy==$NUMPY_VER"

# ---- verify (the gates that actually matter for compile training) ----
python - <<PY
import torch, torch_npu, numpy, speculators, transformers
print("torch       ", torch.__version__, "(want 2.12.0+cpu — NOT +cu130)")
print("torch_npu   ", torch_npu.__version__, "| npu available:", torch_npu.npu.is_available())
import inductor_npu_ext
print("inductor_npu_ext OK:", inductor_npu_ext.__file__)
print("numpy       ", numpy.__version__, "(want $NUMPY_VER)")
print("transformers", transformers.__version__)
print("speculators ", speculators.__file__)
PY
echo ""
echo ">>> DONE (176 compile-train env $CONDA_ENV)."
echo "    torch : $(python -c 'import torch; print(torch.__version__)')  (must end in +cpu)"
echo "    npu   : $(python -c 'import torch_npu; print(torch_npu.npu.is_available())')"
echo ">>> enable compile in training:  COMPILE=1  (train_dsv4_dspark.sh → DSPARK_COMPILE=1)"
echo ">>> HS-over-HTTP (no shared FS w/ 182 serve): set on the trainer  HS_FETCH_BASE=http://182:9009"
echo "    HS_DIR=<182's DSPARK_HS_DIR string>  ENDPOINT=http://182:7000/v1  (see hs_sidecar.py on 182)"
