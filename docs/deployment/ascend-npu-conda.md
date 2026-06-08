# Deploying Speculators on Ascend NPU (conda + pip)

End-to-end guide for running **Speculators** draft models with **vLLM** on a
**Huawei Ascend NPU** server, using a **conda** environment (no Docker).

> Verified version chain (June 2026). Ascend installs are extremely
> version-sensitive — do **not** mix versions across the columns below.

## 1. Version compatibility matrix

| Component       | Pinned version          | Notes                                              |
|-----------------|-------------------------|----------------------------------------------------|
| speculators     | `0.5.0` (latest)        | needs `torch 2.9–2.11`, `transformers 4.56.1–<5.7` |
| vLLM            | `0.20.2`                | the version vllm-ascend v0.20.2rc1 targets         |
| vllm-ascend     | `0.20.2rc1` (latest)    | auto-installs `torch-npu`                           |
| torch / torch-npu | `2.10.0`              | auto-installed by vllm-ascend — don't install yourself |
| triton-ascend   | `3.2.1`                 | install LAST, separately (see §6e)                 |
| **CANN**        | **`9.0.0`**             | + NNAL 9.0.0 (provides `libatb.so`)                |
| Python          | **`3.10` or `3.11`**    | must be `>=3.10, <3.12` — **do NOT use 3.12/3.13**  |

If you must stay on **CANN 8.5.1**, use the stable chain instead:
vllm-ascend `0.18.0` + vLLM `0.18.0` + torch-npu `2.9.0` (also compatible with
speculators 0.5.0). The rest of this guide assumes the latest chain (CANN 9.0.0).

## 2. Prerequisites

Driver/firmware + CANN 9.0.0 + NNAL 9.0.0 must already be installed. Verify:

```bash
# NPU visible?
npu-smi info

# CANN version (expect 9.0.0)
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg
```

If `npu-smi info` fails, fix the driver before continuing — nothing below will work.

## 3. Create the conda environment

```bash
conda create -n vllm-ascend python=3.11 -y
conda activate vllm-ascend
python -m pip install --upgrade pip
```

## 4. Source the CANN environment (do this in EVERY new shell)

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

> Tip: append these two lines to the env's activate hook so they run automatically:
> `echo 'source /usr/local/Ascend/ascend-toolkit/set_env.sh' >> $CONDA_PREFIX/etc/conda/activate.d/ascend.sh`

## 5. Install system build deps

```bash
# Ubuntu/Debian
sudo apt-get update -y && sudo apt-get install -y gcc g++ cmake libnuma-dev git curl wget jq
# or RHEL/openEuler (note: gcc-c++ not g++, numactl-devel not libnuma-dev):
# sudo yum install -y gcc gcc-c++ cmake numactl-devel git curl wget jq
```

## 6. Install vLLM + vllm-ascend

> ⚠️ **The install procedure differs by CPU architecture.** On `aarch64` there is
> no prebuilt `vllm` wheel, so a plain `pip install vllm` falls back to a source
> build that assumes CUDA and dies with `AssertionError: CUDA_HOME is not set`.
> Pick the section that matches `uname -m`.

### 6a. (China networks only, optional) Speed up downloads

```bash
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

### 6b. Install vLLM

**On x86_64** — a CPU wheel exists, install directly:

```bash
# let pip find the CPU torch build (x86_64 only)
pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/"
pip install vllm==0.20.2
```

**On aarch64** — build from source with an empty target device so vLLM does NOT
try to compile CUDA kernels (the NPU kernels come from vllm-ascend, not vLLM):

```bash
git clone --depth 1 --branch v0.20.2 https://github.com/vllm-project/vllm
cd vllm
VLLM_TARGET_DEVICE=empty pip install -e .
cd ..
```

> `VLLM_TARGET_DEVICE=empty` is the whole trick — without it the build looks for
> `CUDA_HOME` and fails. The `empty` build skips kernel compilation, so it's fast.

### 6c. Install vllm-ascend (pulls torch==2.10.0 + torch-npu==2.10.0 automatically)

**Recommended: install the prebuilt wheel.** Pass *both* Huawei Cloud indexes —
vllm-ascend lives on `repository/pypi/simple`, but its dep `triton-ascend==3.2.1`
lives on the separate `ascend/repos/pypi` index, so you need both or the resolve
fails:

```bash
pip install vllm-ascend==0.20.2rc1 \
  --extra-index-url https://mirrors.huaweicloud.com/repository/pypi/simple \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
```

A binary wheel skips compilation entirely, so this also dodges the source-build
trap described next.

> Do **not** install `torch` / `torch-npu` yourself — vllm-ascend pins them to
> `2.10.0`. If you pre-install torch you'll fight its resolver.

### 6c-alt. Editable source install of vllm-ascend (only if you'll modify it)

Use this **only if you need to edit vllm-ascend code**. To just run speculators,
stay with the wheel above. The source path needs CANN sourced (step 4) and the
build toolchain (step 5), and it actually compiles the NPU C++ ops, so it's slow.

**The trap:** `pip install -e .` first installs `[build-system].requires` (which
includes `triton-ascend==3.2.1`) in an **isolated build env that ignores
command-line `--extra-index-url` and only reads global pip config**. With only the
tuna mirror in global config it fails with
`Could not find a version that satisfies triton-ascend==3.2.1` — even though
`pip install triton-ascend==3.2.1 --extra-index-url .../ascend/repos/pypi` works.
The version is correct; the isolated env just can't see the ascend index.

Pick **one** of these two reliable methods:

**Method 1 — disable build isolation (most deterministic).** Pre-install every
build dependency into the current env from both indexes, then build with
`--no-build-isolation` so pip reuses them instead of spawning a blind isolated env:

```bash
# 1) install ALL build deps (from pyproject [build-system].requires) into THIS env
pip install \
  --extra-index-url https://mirrors.huaweicloud.com/repository/pypi/simple \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  attrs "cmake>=3.26" decorator einops googleapis-common-protos numpy packaging pip \
  pybind11 pyyaml scipy pandas pandas-stubs psutil "setuptools>=64" "setuptools-scm>=8" \
  transformers==5.5.3 torch==2.10.0 torch-npu==2.10.0 torchvision wheel msgpack quart \
  numba "xgrammar>=0.1.30" "fastapi<0.124.0" "compressed_tensors>=0.11.0" \
  arctic-inference==0.1.1 triton-ascend==3.2.1

# 2) build editable WITHOUT isolation (reuses the deps above — won't re-resolve)
git clone --depth 1 --branch v0.20.2rc1 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git submodule update --init --recursive
pip install -e . --no-build-isolation
cd ..
```

**Method 2 — put the ascend index in global config** so the isolated build env
inherits it (simpler command, but the isolated env re-downloads build deps each
time):

```bash
pip config set global.extra-index-url \
  "https://mirrors.huaweicloud.com/repository/pypi/simple https://mirrors.huaweicloud.com/ascend/repos/pypi"
git clone --depth 1 --branch v0.20.2rc1 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git submodule update --init --recursive
pip install -e .
cd ..
```

> If Method 2 still fails to find `triton-ascend==3.2.1` (some pip/build-backend
> versions don't propagate global config into the isolated env), fall back to
> Method 1 — `--no-build-isolation` removes the isolated env from the equation
> entirely.

### 6d. Backfill CANN's Python dependencies (fresh conda env)

CANN ships its own Python packages (`te`, `auto-tune`, `opc-tool`, `superkernel`,
`ms-service-profiler`, …) that power op compilation. A **fresh** conda env doesn't
have the scientific-Python libs they declare, so after the install above you'll
see a wall of `pip` dependency-conflict warnings like
`te 0.4.0 requires decorator, which is not installed`. The vllm-ascend install
still "Successfully installed", but op compilation can fail later — backfill them:

```bash
pip install decorator "scipy>=1.7.3" ml-dtypes tornado absl-py attrs psutil pyyaml
```

Optional — only if you'll use the `ms-service-profiler` performance tool:

```bash
pip install matplotlib "pandas~=2.2" openpyxl
```

### 6e. Install triton-ascend — LAST, after everything else

```bash
pip install triton-ascend==3.2.1 \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
```

> Install `triton-ascend` **last**; installing it earlier lets later packages
> re-pin its deps and break it.

## 7. Verify the NPU stack BEFORE touching speculators

```bash
npu-smi info
pip show vllm vllm-ascend torch torch-npu triton-ascend 2>/dev/null | grep -E "^(Name|Version)"
python3 -c "import vllm, torch, torch_npu; print('vllm', vllm.__version__, '| torch', torch.__version__)"
```

Expect `vllm 0.20.2`, `vllm-ascend 0.20.2rc1`, `torch 2.10.0`, `torch-npu 2.10.0`,
`triton-ascend 3.2.1`, and no import errors. If `import torch_npu` complains about
`libatb.so`, you forgot to `source` the NNAL env (step 4).

Optional smoke test with a tiny model:

```bash
python3 -c "
from vllm import LLM, SamplingParams
llm = LLM(model='Qwen/Qwen2.5-0.5B-Instruct', enforce_eager=True)
print(llm.generate(['Hello from'], SamplingParams(max_tokens=16))[0].outputs[0].text)
"
```

## 8. Install Speculators — WITHOUT breaking the NPU stack ⚠️

Speculators lists `torch`, `torchaudio`, `torchvision`, `transformers` (and pulls
`vllm` transitively) as plain dependencies. On a default index that **reinstalls
the CUDA/CPU torch and stock vLLM, wiping out your torch-npu + vllm-ascend setup.**
So install with `--no-deps`, then add only the safe dependencies — everything
**except** `torch` / `torchaudio` / `torchvision` / `transformers` / `vllm`, which
are already provided by vllm-ascend (torch 2.10.0, transformers 5.5.3, …).

### 8a. From PyPI (release, not editable)

```bash
pip install --no-deps speculators==0.5.0
```

### 8b. Editable / from source (when you'll modify speculators code)

```bash
git clone https://github.com/<you>/speculators.git
cd speculators
git fetch --tags          # needed so setuptools-git-versioning resolves a version
pip install -e . --no-deps
cd ..
```

> `--no-deps` only skips runtime deps; the editable build isolation needs just
> `setuptools` + `setuptools-git-versioning`, so it won't pull torch. Don't use
> `--depth 1` for the clone — without tags the version resolves to `0.0.0`.

### 8c. Add the safe dependencies (both install paths)

```bash
pip install \
  "click" "datasets>=4.0.0,<=4.8.5" "huggingface-hub" "loguru>=0.7.2,<=0.7.3" \
  "numpy>=2.0.0,<=2.4.2" "openai>=2.0.0" "protobuf" "psutil" \
  "pydantic>=2.0.0" "pydantic-settings>=2.0.0" "rich" "safetensors" \
  "tqdm>=4.66.3,<=4.67.3" "typer>=0.12.0"
```

> ⚠️ Watch the output of this command:
> - **numpy** — speculators pins `<=2.4.2`, but the NPU stack may have installed a
>   newer numpy. If pip downgrades it and `import torch_npu` then breaks, drop the
>   `numpy` entry above and keep the stack's version.
> - **If pip tries to touch `torch` / `transformers` / `vllm`, stop** — none of
>   them are in the list above and they must not be reinstalled.

> Building your own draft models? Add the training extras the same careful way
> (`pip install --no-deps ...` for anything that re-pins torch).

## 9. Verify Speculators + the NPU stack survived

```bash
speculators --version
python3 -c "import speculators, torch, torch_npu, vllm, vllm_ascend; print('OK — torch', torch.__version__, '| vllm', vllm.__version__)"
```

`torch` must still report **2.10.0** and `vllm_ascend` must still import (if it
doesn't, vLLM loses its NPU platform and any `LLM(...)` call dies with
`Device string must not be empty`). If torch flipped to a non-`+`/CUDA build,
step 8 reinstalled torch — fix with:
`pip install --force-reinstall --no-deps torch-npu==2.10.0 torch==2.10.0`.

> Run this from a directory that does **not** contain a `vllm/` or `speculators/`
> folder (e.g. `cd /tmp`). Running next to a source clone shadows the installed
> package and you'll see `cannot import name 'LLM' from 'vllm' (unknown location)`.

## 10. Run speculative decoding on the NPU

A Speculators-format model carries its own `speculator_config`, so a plain
`vllm serve` picks up the draft model automatically:

```bash
# online server
vllm serve RedHatAI/Qwen3-8B-speculator.eagle3 --enforce-eager
```

EAGLE/EAGLE3 specifics on Ascend (offline example):

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,          # raise to use more NPUs for the base model
    enforce_eager=True,              # recommended on NPU for spec-decode
    speculative_config={
        "method": "eagle3",
        "model": "<your speculators eagle3 checkpoint>",
        "draft_tensor_parallel_size": 1,   # EAGLE draft MUST be TP=1
        "num_speculative_tokens": 2,
    },
)
print(llm.generate(["The capital of France is"], SamplingParams(max_tokens=32))[0].outputs[0].text)
```

Key NPU constraints:
- The EAGLE draft model must run with `draft_tensor_parallel_size=1`.
- Prefer `enforce_eager=True` for spec-decode on NPU.
- Match `num_speculative_tokens` to how the speculator was trained.

## 11. Common errors & fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `libatb.so: cannot open shared object file` | NNAL env not sourced | re-run step 4 |
| `import torch_npu` fails / torch is CUDA build | step 8 reinstalled torch | `pip install --force-reinstall --no-deps torch-npu==2.10.0 torch==2.10.0` |
| `npu-smi: command not found` inside env | driver path not on PATH | source CANN env (step 4); confirm driver installed |
| vLLM ignores the draft model | model isn't Speculators-format / wrong vLLM | confirm vLLM `0.20.2`; check the model's `config.json` has `speculator_config` |
| Python version error on `pip install vllm-ascend` | Python 3.12+/3.13 | recreate env with `python=3.11` |

---

*Generated for the Ascend NPU deployment of vllm-project/speculators. Versions
current as of June 2026 — re-check the matrix when upgrading any single component.*
