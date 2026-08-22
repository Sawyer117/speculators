#!/usr/bin/env bash
# Installer for the DSV4-Flash **W8A8 single-node serve** stack: vLLM v0.27.1 +
# vllm-ascend UPSTREAM main, pinned. A SEPARATE env from the DSpark training/HS stack.
#
# ⚠ THIS IS NOT A REPLACEMENT for install_npu_env_dspark.sh. That script stays the SSOT for
# the training / HS-dump / eval stack (vLLM v0.23.0 + our fork's dspark-dsv4 branch) and must
# keep working -- `austin` is feeding hidden states to a live training run. Install this into
# a NEW conda env; never re-run it over an existing one.
#
# WHY A NEW STACK AT ALL. Three upstream merges we need are all from the last 36 hours and
# none are in our v0.23.0 base:
#   #12968  2026-08-20  MRV2 supports DSV4 DSpark        877 -> 1192 tok/s, engine-side only
#   #14490  2026-08-21  fix dsv4 quant NAME mismatch     hit directly by a quantized deploy
#   #14696  2026-08-21  fix DSpark spec-decode on MRV2   two crashes: NoneType hidden buffer
#                                                        in the profiling dummy run, and a
#                                                        missing block_size in SWA indices
# ⚠ #14696 merged the morning this was written, so MRV2+DSpark has ~zero soak time. The pin
# below is the FLOOR (its merge commit). Keep MRV1 as the fallback if the serve misbehaves.
#
# ⚠ KNOWN-BROKEN UPSTREAM: the official single-node DSpark command errors with
#   "Can't determine cudagraph shapes that are both a multiple of 6 (num_speculative_tokens+1)
#    ... and 4 (tensor_parallel_size) required by sequence parallelism"   (issue #14260, OPEN)
# Their workaround is num_speculative_tokens 5 -> 7, which we CANNOT use: our draft is trained
# at block_size 5 and emits exactly 5. Use serve_dsv4_a2_singlenode_w8a8.sh, which picks a TP
# that divides num_spec+1 instead. See that script for the arithmetic.
#
# DIFFERS FROM install_npu_env_dspark.sh in exactly these places:
#   vLLM         v0.23.0                    -> v0.27.1        (vllm-ascend main's Dockerfile pin)
#   vllm-ascend  Sawyer117 fork/dspark-dsv4 -> UPSTREAM main @ VA_COMMIT (no fork patches needed:
#                                              the HS-dump patches are training-only)
#   torch-npu    2.10.0                     -> 2.10.0.post4   (main's requirements.txt)
#   triton-ascend 3.2.1                     -> 3.2.2
#   CANN         9.0.0                      -> 9.1.0          ⚠ OS-level, NOT pip. See step 0.
#
# PREREQS: a py3.11 env you just created and activated; CANN at OS level.
# USAGE:   # ⚠ -c conda-forge --override-channels is REQUIRED on these boxes: the Anaconda
#          # `defaults` channel answers HTTP 403 Forbidden (repo.anaconda.com is licence-gated),
#          # and conda leaves NO env behind when it fails, so a plain `conda create` looks like
#          # it worked until `conda activate` says EnvironmentNameNotFound.
#          # ⚠ ask for `pip` EXPLICITLY: conda-forge's python does not pull it in the way
#          # the defaults channel's does, and step 1 then dies with "No module named pip".
#          conda create -n dsv4-w8a8 -c conda-forge --override-channels python=3.11 pip -y
#          conda activate dsv4-w8a8
#          bash examples/ascend_npu_dflash/install_npu_env_dsv4_w8a8.sh
# OVERRIDES (env):
#   ROOT / VLLM_DIR / VA_DIR   as in the sibling script
#   VA_COMMIT   vllm-ascend commit to pin (default: the #14696 merge commit = the floor)
#   VA_REPO     default upstream; point at the fork only if you need fork patches
#   NUMPY_VER   default 2.3.5
#   CANN_ENV    path to ascend-toolkit/set_env.sh. ⚠ NOT /usr/local on these boxes -- CANN sits
#               under the shared account, e.g.
#               CANN_ENV=/home/a00652497/CANN/9.0.0.0430/ascend-toolkit/set_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"                 # the speculators checkout
ROOT="${ROOT:-$(cd "$REPO_ROOT/.." && pwd)}"                 # code root (installation/ + speculators/)
VLLM_DIR="${VLLM_DIR:-$ROOT/installation/vllm-v0.27.1}"
VA_DIR="${VA_DIR:-$ROOT/installation/vllm-ascend-main}"
VA_REPO="${VA_REPO:-https://github.com/vllm-project/vllm-ascend.git}"
# FLOOR = the #14696 merge commit. Anything older crashes DSpark on MRV2.
VA_COMMIT="${VA_COMMIT:-4ce367a7d12db55c3dbe9b670eff52b2e14b3b9a}"
NUMPY_VER="${NUMPY_VER:-2.3.5}"
CANN_ENV="${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
HW_PYPI="https://mirrors.huaweicloud.com/repository/pypi/simple"
HW_ASCEND="https://mirrors.huaweicloud.com/ascend/repos/pypi"
IDX=(--extra-index-url "$HW_PYPI" --extra-index-url "$HW_ASCEND")

echo "==================================================================="
echo " DSV4-Flash W8A8 serve stack  (vLLM v0.27.1 + vllm-ascend ${VA_COMMIT:0:12}, numpy=$NUMPY_VER)"
echo "==================================================================="

echo "== 0. sanity: py311 + source CANN =="
python -c "import sys; assert sys.version_info[:2]==(3,11), 'need py3.11, got %s'%sys.version" \
  || { echo "Activate your py3.11 env first."; exit 1; }
# ⚠ ABORT on a MIXED CANN shell. set_env.sh PREPENDS to PATH/LD_LIBRARY_PATH rather than
# replacing, so sourcing 9.1.0 into a shell that already sourced 9.0.0 leaves the older ccec
# and headers ahead of the newer ones. The build then takes ~40 minutes and fails in exactly
# the place it failed before, which reads like "the upgrade did not help" and is not.
# Cost of getting this wrong is 40 minutes; cost of the check is instant. Use a FRESH shell.
_cann_root="$(cd "$(dirname "$(dirname "$CANN_ENV")")" 2>/dev/null && pwd || true)"
if [ -n "${ASCEND_HOME_PATH:-}" ] && [ -n "$_cann_root" ]; then
  case "$ASCEND_HOME_PATH" in
    "$_cann_root"*) : ;;
    *)
      echo "!! this shell ALREADY has a different CANN sourced:"
      echo "     already active : $ASCEND_HOME_PATH"
      echo "     you asked for  : $CANN_ENV"
      echo "   set_env.sh prepends, so the old one would still win and the op build would fail"
      echo "   ~40 minutes from now. Open a FRESH shell, activate the env, and re-run:"
      echo "     conda activate \$(basename \"${CONDA_PREFIX:-<env>}\")"
      echo "     CANN_ENV=$CANN_ENV bash \$0"
      exit 1 ;;
  esac
fi
[ -f "$CANN_ENV" ] && { source "$CANN_ENV"; echo "sourced CANN: $CANN_ENV"; } \
  || echo "WARN: CANN set_env not at $CANN_ENV — set CANN_ENV=... (needed to compile the ops)"
# vllm-ascend main builds its image on CANN 9.1.0; our other envs are on 9.0.0. The ops compile
# against CANN headers, so a mismatch surfaces as an opbuild failure in step 4, not here.
# Warn loudly rather than abort: 9.0.0 may well work, but if step 4 dies this is the first suspect.
# Read the version from where CANN ACTUALLY is, not a hardcoded /usr/local: on these boxes it
# lives under the shared account, e.g. /home/a00652497/CANN/9.0.0.0430/ascend-toolkit. set_env.sh
# exports ASCEND_HOME_PATH, so derive from that and fall back to the path itself.
CANN_VER="$(cat "${ASCEND_HOME_PATH:-/nonexistent}/version.cfg" 2>/dev/null | tr -d ' \n' || true)"
[ -n "$CANN_VER" ] || CANN_VER="${ASCEND_HOME_PATH:-unknown}"
echo "CANN reported: ${CANN_VER}   (vllm-ascend main is built on 9.1.0)"
case "$CANN_VER" in *9.1.0*) : ;; *) echo "⚠ NOT 9.1.0 — if the op build in step 4 fails, upgrade CANN first." ;; esac

# ⚠ conda-forge envs need their OWN libstdc++ to win over the system one. conda-forge's
# libsqlite is built with the ICU extension, so `import sqlite3` pulls libicui18n.so.78, which
# needs CXXABI_1.3.15 -- newer than /usr/lib64/libstdc++.so.6. The env already ships
# libstdcxx-16.1.0; it just loses the search order, because CANN's set_env puts system paths
# ahead of it. Prepending $CONDA_PREFIX/lib fixes it. Symptom without this, at the END of a
# long pip log and easy to miss under the dependency-conflict noise:
#   ImportError: /usr/lib64/libstdc++.so.6: version `CXXABI_1.3.15' not found
#   RuntimeError: Failed to load the backend extension: torch_npu
[ -n "${CONDA_PREFIX:-}" ] && export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

echo "== 1. build deps + torch/torch-npu 2.10.0 + numpy $NUMPY_VER + CANN backfill =="
# Self-heal a pip-less env (conda-forge python ships without it) instead of dying here.
python -m pip --version >/dev/null 2>&1 || { echo "no pip in this env — bootstrapping via ensurepip"; python -m ensurepip --upgrade; }
python -m pip install -U pip setuptools "setuptools-scm>=8" wheel packaging "cmake>=3.26" ninja jinja2 setuptools-rust pybind11
python -m pip install "${IDX[@]}" torch==2.10.0 torch-npu==2.10.0.post4 pyyaml
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

echo "== 3. vLLM v0.27.1 (empty build, editable) =="
[ -d "$VLLM_DIR/.git" ] || git clone --depth 1 --branch v0.27.1 https://github.com/vllm-project/vllm "$VLLM_DIR"
( cd "$VLLM_DIR" && TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_TARGET_DEVICE=empty \
    python -m pip install -e . --no-build-isolation -v )

echo "== 4. vllm-ascend @ ${VA_COMMIT:0:12} — FROM SOURCE (compiles the V4/SAS CANN ops) =="
# Pin a COMMIT, not a branch: main moves several times a day and #14696 landed only hours
# before this pin, so "whatever main is today" is not a reproducible stack.
[ -d "$VA_DIR/.git" ] || git clone "$VA_REPO" "$VA_DIR"
( cd "$VA_DIR" && git fetch origin && git checkout --detach "$VA_COMMIT" && git --no-pager log --oneline -1 )
( cd "$VA_DIR" && rm -rf csrc/build && pip install -e . --no-deps --no-build-isolation -v )

echo "== 5. vllm-ascend runtime extras (--no-deps protects torch) =="
python -m pip install numba einops pandas msgpack
python -m pip install --no-deps torchvision==0.25.0 torchaudio==2.10.0 --extra-index-url "$HW_PYPI"
# triton-ascend REQUIRED (block_table slot-mapping kernel at runtime); it pins numpy<2 -> re-forced in step 7.
python -m pip install triton-ascend==3.2.2 "${IDX[@]}"

echo "== 5b. align transformers with what vllm-ascend pins =="
# vLLM 0.27.1 asks for transformers>=5.5.3 and pulls the newest; vllm-ascend pins an EXACT
# version. Both constraints are satisfiable at vllm-ascend's pin, so honour it -- the DSV4
# model/config code lives in vllm-ascend and there is no reason to run it on a transformers
# it was never tested against. Read the pin from the installed package rather than hardcoding
# it, so this stays correct when VA_COMMIT moves.
# (The fastapi conflict is NOT resolvable the same way: vllm-ascend wants <0.124.0 while vLLM
#  0.27.1 requires >=0.133.0 -- upstream's own pins contradict. Keep vLLM's; it serves the HTTP
#  API. Symptom if that ever bites: serve returns 500 with a `_IncludedRouter` traceback.)
TF_REQ="$(python - <<'PY2'
import importlib.metadata as m
try:
    reqs = m.requires("vllm_ascend") or []
except Exception:
    reqs = []
print(next((r.split(";")[0].strip() for r in reqs if r.startswith("transformers")), ""))
PY2
)"
if [ -n "$TF_REQ" ]; then
  echo "vllm-ascend pins: $TF_REQ"
  python -m pip install --no-deps "$TF_REQ" || echo "WARN: could not apply $TF_REQ — continuing"
else
  echo "no transformers pin found in vllm_ascend metadata — leaving as installed"
fi

echo "== 6. speculators (--no-deps) + train/rollout deps =="
python -m pip install --no-deps -e "$ROOT/speculators" 2>/dev/null || python -m pip install --no-deps -e "$REPO_ROOT"
# hs_connectors is a uv WORKSPACE MEMBER of speculators (its own pyproject.toml + src/), and a
# hard dependency of speculators.train.data -- so --no-deps skips it and `import speculators`
# dies. Worse, from the repo root Python finds the hs_connectors/ DIRECTORY as an implicit
# namespace package and reports "cannot import name FileTransfer ... (unknown location)",
# which reads like a broken install rather than a missing one.
[ -f "$REPO_ROOT/hs_connectors/pyproject.toml" ] \
  && python -m pip install --no-deps -e "$REPO_ROOT/hs_connectors" \
  || echo "note: no hs_connectors workspace member here — skipping"
python -m pip install datasets loguru typer pydantic-settings tensorboard aiohttp

echo "== 7. FORCE numpy $NUMPY_VER (LAST pip op — triton-ascend<2 downgraded it) + verify =="
python -m pip install --no-deps "numpy==$NUMPY_VER"
NUMPY_VER="$NUMPY_VER" python - <<'PY'
import os, numpy, torch, torch_npu, torchgen.model, vllm, vllm_ascend
want = os.environ["NUMPY_VER"]
print("numpy      ", numpy.__version__, "(want", want + ")", "OK" if numpy.__version__ == want else "!! MISMATCH")
print("torch      ", torch.__version__, "| vllm", vllm.__version__)
print("vllm_ascend", vllm_ascend.__file__)   # must be under your code root, not someone else's
import transformers
print("transformers", transformers.__version__)
# speculators is OPTIONAL in a serve-only env -- it is needed to CONVERT our own trained draft,
# not to serve. Its import pulls the training stack (hs_connectors, datasets), so a failure
# here must not condemn a serving install that is otherwise complete. Report and continue.
try:
    import speculators
    print("speculators", speculators.__file__)
except Exception as exc:
    print("speculators  NOT importable:", type(exc).__name__, exc)
    print("             (fine for SERVING; needed only to convert our own draft)")
print("OK: vLLM + vllm-ascend import cleanly and the ascend platform plugin registers")
PY

echo "==================================================================="
echo " DONE. Expect: numpy $NUMPY_VER | torch 2.10.0 | vllm 0.27.1 | vllm-ascend ${VA_COMMIT:0:12}"
echo " NEXT: bash examples/ascend_npu_dflash/serve_dsv4_a2_singlenode_w8a8.sh"
echo " NOTE: serve also needs the CANN nnal/atb set_env sourced in a CLEAN shell."
echo "==================================================================="
