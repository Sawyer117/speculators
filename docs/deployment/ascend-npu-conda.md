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
| torch / torch-npu | `2.10.0`              | inside speculators' allowed range ✅               |
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
sudo apt-get update -y && sudo apt-get install -y gcc g++ cmake libnuma-dev git curl
# or RHEL/openEuler:
# sudo yum install -y gcc g++ cmake numactl-devel git curl
```

## 6. Install vLLM + vllm-ascend

```bash
# On x86_64 only: let pip find the CPU torch build (skip on aarch64)
pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/"

# 1) vLLM core (CPU/empty build — the NPU kernels come from vllm-ascend)
pip install vllm==0.20.2

# 2) vllm-ascend — this pulls torch-npu==2.10.0 automatically
pip install \
  --extra-index-url https://mirrors.huaweicloud.com/repository/pypi/simple \
  vllm-ascend==0.20.2rc1
```

## 7. Verify the NPU stack BEFORE touching speculators

```bash
npu-smi info
python3 -c "import vllm, torch, torch_npu; print('vllm', vllm.__version__, '| torch', torch.__version__)"
```

Expect `vllm 0.20.2 | torch 2.10.0` and no import errors. If `import torch_npu`
complains about `libatb.so`, you forgot to `source` the NNAL env (step 4).

Optional smoke test with a tiny model:

```bash
python3 -c "
from vllm import LLM, SamplingParams
llm = LLM(model='Qwen/Qwen2.5-0.5B-Instruct', enforce_eager=True)
print(llm.generate(['Hello from'], SamplingParams(max_tokens=16))[0].outputs[0].text)
"
```

## 8. Install Speculators — WITHOUT breaking the NPU stack ⚠️

`pip install speculators` lists `torch` and `vllm` as plain dependencies. On a
default index that **reinstalls the CUDA/CPU torch and stock vLLM, wiping out
your torch-npu + vllm-ascend setup.** Install with `--no-deps`, then add only the
safe dependencies (everything except torch/torchvision/torchaudio/vllm/transformers,
which are already provided by vllm-ascend).

```bash
# 1) speculators itself, no dependency resolution
pip install --no-deps speculators==0.5.0

# 2) its remaining deps, explicitly EXCLUDING torch*/vllm/transformers
pip install \
  "click" "datasets<=4.8.4,>=4.0.0" "huggingface-hub" \
  "loguru<=0.7.3,>=0.7.2" "numpy<=2.4.2,>=2.0.0" "openai>=2.0.0" \
  "protobuf" "psutil" "pydantic>=2.0.0" "pydantic-settings>=2.0.0" \
  "rich" "safetensors" "setuptools" "tqdm<=4.67.3,>=4.66.3" "typer>=0.12.0"
```

> Building your own draft models? Add the training extras the same careful way
> (`pip install --no-deps ...` for anything that re-pins torch).

## 9. Verify Speculators + the NPU stack survived

```bash
speculators --version
python3 -c "import speculators, torch, torch_npu, vllm; print('OK — torch', torch.__version__, '| vllm', vllm.__version__)"
```

`torch` must still report **2.10.0**. If it flipped to a non-`+`/CUDA build,
step 8 reinstalled torch — fix with:
`pip install --force-reinstall --no-deps torch-npu==2.10.0 torch==2.10.0`.

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
