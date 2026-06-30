# Ascend NPU — DSpark Qwen3 Inference Baseline (Install & Run)

> **Fork-only, team-internal — not for upstream.** Sets up DeepSeek **DSpark**
> speculative-decoding **inference** (no training) for Qwen3 on Ascend NPU, using
> DeepSeek's released DSpark draft checkpoint.
>
> This guide is **from-scratch on a bare machine** (no env cloning) and **bakes in every
> pit we hit** — do the steps in order and you won't re-trip them.

## What this baseline is

- Reproduce **DSpark** (DFlash's successor; higher accept length) **inference** on Ascend.
- **Qwen3 path needs NO custom Ascend kernels** — SparseMLA / TiLang HC / fused deepseek
  kernels are all **DeepSeek-V4-only**. Qwen3 (GQA) = the DFlash backbone (already merged on
  vllm-ascend) **+ a per-step low-rank Markov logit bias**. That's the whole delta.
- Built on **vllm-ascend PR #11153 + our 3 bug fixes** (branch below).
- **New env baseline** (vLLM 0.23.0), replacing the old 0.20.2rc1 DFlash setup — we follow
  upstream, no backport. **CANN stays 9.0.0; only vLLM + vllm-ascend move.**

## Verified stack (matches vllm-ascend main's pins)

| component   | version                                       | note |
|-------------|-----------------------------------------------|------|
| Python      | **3.11**                                       | matches the proven torch-npu wheel |
| CANN        | **9.0.0** / 910b                               | **unchanged** from the DFlash box |
| torch       | **2.10.0**                                     | pinned by vllm-ascend main `requirements.txt` |
| torch-npu   | **2.10.0**                                     | pinned by vllm-ascend main `requirements.txt` |
| numpy       | **1.26.4** (pinned)                            | 2.x breaks triton-ascend / scipy on NPU |
| vLLM        | **v0.23.0**, built `VLLM_TARGET_DEVICE=empty`  | → "0.23.0+empty" |
| vllm-ascend | `Sawyer117/vllm-ascend` @ **`dspark-qwen3-npu`** (`a9709b8e`) | PR #11153 + 3 fixes |
| draft ckpt  | `deepseek-ai/dspark_qwen3_4b_block7`           | **must edit `config.json`** — step 5 |

## Project layout

vLLM + vllm-ascend live under `installation/`, siblings of the speculators checkout:

```
<root>/dspark/                 # e.g. /home/<user>/2026/dspark
├── speculators/               # this fork (run scripts in examples/ascend_npu_dflash/)
└── installation/
    ├── vllm-v0.23.0/          # vLLM source (editable) — step 3
    └── vllm-ascend/           # fork @ dspark-qwen3-npu (editable / PYTHONPATH) — step 4
```

---

## 0. Foundation (driver + CANN)

```bash
npu-smi info | head -20                       # driver OK → lists the 910b cards
source <ascend-toolkit>/set_env.sh            # your box's real CANN paths
source <nnal>/atb/set_env.sh
echo "ASCEND_HOME=$ASCEND_HOME_PATH"          # non-empty → CANN sourced
```
If `npu-smi` missing → driver not installed (admin: `Ascend-hdk npu-driver + firmware` .run).
If CANN missing → install CANN 9.0.0 toolkit + nnal first.

## 1. Fresh py3.11 env (from scratch — do NOT clone)

```bash
conda create -n dspark-base python=3.11 -y
conda activate dspark-base
```

## 2. Build deps + torch/torch_npu 2.10.0 + numpy

```bash
# build tooling (note setuptools-rust — needed by vLLM 0.23.0's build, step 3)
python -m pip install -U pip setuptools "setuptools-scm>=8" wheel packaging \
  "cmake>=3.26" ninja jinja2 setuptools-rust pybind11

# torch + torch_npu 2.10.0 via Huawei mirrors
python -m pip install \
  --extra-index-url https://mirrors.huaweicloud.com/repository/pypi/simple \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  torch==2.10.0 torch-npu==2.10.0 pyyaml

# ⚠️ GOTCHA: a fresh env has no numpy, and torch_npu's profiler imports it on load.
python -m pip install "numpy==1.26.4"

# ⚠️ GOTCHA: vllm-ascend is built FROM SOURCE (step 4), and the CANN op compiler (TBE/TVM)
#   imports decorator/scipy/etc. DURING that build. Install the CANN backfill NOW — BEFORE
#   any source build — or the op compile dies with `ModuleNotFoundError: No module named 'decorator'`.
python -m pip install decorator "scipy>=1.7.3" ml-dtypes attrs psutil pyyaml matplotlib openpyxl tornado
python -m pip install "numpy==1.26.4"     # re-pin (scipy/etc. may bump numpy to 2.x)

# verify torch_npu loads and sees the NPU
python -c "import torch, torch_npu; print('torch', torch.__version__, '| torch_npu', torch_npu.__version__); print('npu available:', torch_npu.npu.is_available(), '| devices:', torch_npu.npu.device_count())"
# expect: torch 2.10.0 | torch_npu 2.10.0 | npu available: True | devices: 8
```

## 3. Build vLLM v0.23.0 (+empty) into installation/

```bash
cd <root>/dspark/installation
git clone --depth 1 --branch v0.23.0 https://github.com/vllm-project/vllm.git vllm-v0.23.0
cd vllm-v0.23.0

# ⚠️ GOTCHA: vLLM 0.23.0's build imports setuptools_rust (0.20.2 didn't). With
#   --no-build-isolation it must already be in the env — we installed it in step 2.
# empty target (no CUDA) + reuse env torch 2.10.0 + don't autoload torch_npu during build:
TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_TARGET_DEVICE=empty \
  python -m pip install -e . --no-build-isolation -v

# ⚠️ GOTCHA: vLLM bumps numpy to 2.x → re-pin 1.26.4 (ignore the pip dep-conflict warning)
python -m pip install "numpy==1.26.4"

python -c "import torch, torch_npu, vllm, numpy; print('vllm', vllm.__version__, '| torch', torch.__version__, '| torch_npu', torch_npu.__version__, '| numpy', numpy.__version__)"
# expect: vllm 0.23.0+empty | torch 2.10.0 | torch_npu 2.10.0 | numpy 1.26.4
```

## 4. Install vllm-ascend (DSpark branch) into installation/

```bash
cd <root>/dspark/installation
git clone https://github.com/Sawyer117/vllm-ascend.git
cd vllm-ascend
git checkout dspark-qwen3-npu     # = PR #11153 + fixes (commit a9709b8e)
```
**Must build from source** — there is no prebuilt wheel for our main-based branch, so a
PYTHONPATH overlay won't work (the CANN custom ops must be compiled). The CANN backfill from
step 2 must already be installed (the op compiler imports `decorator`/`scipy`/…).
```bash
# --no-deps protects torch; --no-build-isolation reuses env torch_npu to compile the ops.
# Heavy: compiles CANN ops + downloads protobuf etc. from gitcode (needs network).
pip install -e . --no-deps --no-build-isolation -v

# vllm-ascend runtime extras that --no-deps skipped and vLLM didn't already pull.
# numba: EPLB module imports it at engine-core startup.
# torchvision/torchaudio: vllm-ascend patches qwen3_5 → pulls qwen3_vl → transformers
#   qwen2_vl image processor → torchvision. --no-deps so it doesn't drag in a different torch.
python -m pip install numba einops pandas msgpack
python -m pip install --no-deps torchvision==0.25.0 torchaudio==2.10.0 \
  --extra-index-url https://mirrors.huaweicloud.com/repository/pypi/simple
# triton-ascend is REQUIRED (not optional): vllm-ascend's block_table slot-mapping is a triton
# kernel used every step at runtime — without it you get "'function' object is not subscriptable".
python -m pip install triton-ascend==3.2.1 \
  --extra-index-url https://mirrors.huaweicloud.com/repository/pypi/simple \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
# clang-15: triton-ascend JIT-compiles kernels with it (the repo Dockerfile installs clang-15 for this).
conda install -y -c conda-forge clang=15 clangxx=15 lld=15
python -m pip install "numpy==1.26.4"     # re-pin after the build + extras

# verify (also exercises our DSpark patch import). Note: vllm_ascend has NO __version__
# attribute — just confirm it imports without a traceback. A trailing torch_npu
# "ERR99999 ... Device:-1" on process exit is harmless teardown noise.
python -c "import vllm_ascend; print('vllm_ascend import OK')"
pip show vllm-ascend | grep -E "Name|Version"
```
If the op compile dies on a missing module → it's a CANN backfill dep; install it, then
**clear and rebuild**: `rm -rf csrc/build && pip install -e . --no-deps --no-build-isolation -v`.

## 5. Draft checkpoint + FIX its architecture ⚠️ (required)

```bash
huggingface-cli download deepseek-ai/dspark_qwen3_4b_block7 --local-dir ./dspark_qwen3_4b_block7
```
vLLM (even latest `main`) has **no `Qwen3DSparkModel`** registered (DSpark is still open PR
`vllm#46995`). The DSpark draft loads via the **DFlash** path, so edit `config.json`:
```jsonc
"architectures": ["DFlashDraftModel"]     // was ["Qwen3DSparkModel"]
```
Leave `markov_head_type`, `target_layer_ids`, `mask_token_id`, `markov_rank`, … untouched — the
vllm-ascend patch attaches the Markov/confidence heads when it sees `markov_head_type`. Without
this edit the server **won't start** (`architectures ... are not supported`).

## 6. Serve + eval (reuse the DFlash harness)

Serve directly (the run scripts predate this stack; a plain `vllm serve` is simplest). Three
DSpark-specific knobs **matter**:
- **`"num_speculative_tokens": 7`** — the `block7` ckpt drafts **7** tokens (DSpark paper config),
  NOT block-1=6. Setting 6 follows the DFlash convention but drops the 7th draft position (~0.58 of
  the accept length): gsm8k accept **5.66 @ 6 vs ~6.2 @ 7**. Always set it to the ckpt's block size.
- **`"method": "dflash"`** in `--speculative-config` — DSpark MUST run on the DFlash proposer
  (`AscendDflashProposer`), which sets up the HS pipeline + `_next_token_ids` the Markov head
  needs. Without it vLLM defaults to the generic `draft_model` proposer → `AttributeError:
  ... has no attribute '_next_token_ids'` in dummy_run.
- **`--enforce-eager`** — simplest path (disables torch.compile). Accept length is identical to
  graph mode. For **~5× throughput**, install triton-ascend and **drop `--enforce-eager`** to run
  full cudagraph — correct accept there needs the cudagraph-safe Markov fix (vllm-ascend PR #11153).

```bash
export ASCEND_RT_VISIBLE_DEVICES=0 VLLM_USE_V1=1
export TARGET=/path/to/Qwen3-4B
export DRAFT=/path/to/dspark_qwen3_4b_block7      # the architecture-edited one
nohup vllm serve "$TARGET" \
  --trust-remote-code --tensor-parallel-size 1 \
  --max-num-seqs 64 --max-model-len 8192 \
  --enforce-eager \
  --speculative-config "{\"model\":\"$DRAFT\",\"num_speculative_tokens\":7,\"method\":\"dflash\",\"draft_tensor_parallel_size\":1}" \
  --host 0.0.0.0 --port 30000 \
  > dspark_serve.log 2>&1 &
tail -f dspark_serve.log            # wait for "Application startup complete"
```
Then eval with the client (serve already running on :30000):
```bash
cd <root>/dspark/speculators/examples/ascend_npu_dflash
pip install datasets requests       # Evaluator deps, if missing
bash run_eval.sh                     # accept length + throughput per benchmark
```
Accept length: gsm8k **~6.2** @ `num_speculative_tokens=7` (Qwen3-4B; DSpark paper Qwen3-8B 6.30 @
7 vs plain DFlash 4.91). Graph mode (drop `--enforce-eager`) adds ~5× throughput at the same accept.

---

## Gotchas log (the pits we hit — 2026-06-30)

1. **Fresh env has no numpy** → `torch_npu` profiler `ModuleNotFoundError: numpy` on import.
   Install `numpy==1.26.4` right after torch/torch_npu. (`install_npu_env.sh` installs it last;
   step-by-step you hit it early.)
2. **numpy must be 1.26.4, not 2.x** — 2.x breaks triton-ascend/scipy on NPU. vLLM bumps it to
   2.x, so **re-pin after the vLLM build**.
3. **vLLM 0.23.0 build needs `setuptools-rust`** (0.20.2 didn't). With `--no-build-isolation` it
   must be in the env first (added to step 2's build deps).
4. **Use `--no-build-isolation` + `TORCH_DEVICE_BACKEND_AUTOLOAD=0` + `VLLM_TARGET_DEVICE=empty`**
   for the vLLM build (reuse env torch 2.10.0; no CUDA; don't choke on torch_npu autoload).
5. **torch/torch_npu stay at 2.10.0** — vllm-ascend main `requirements.txt` pins exactly these,
   so no upgrade despite jumping vLLM 0.20.2 → 0.23.0. CANN 9.0.0 unchanged.
6. **DSpark checkpoint architecture** = `Qwen3DSparkModel`, unregistered everywhere → edit
   `config.json` `architectures` → `["DFlashDraftModel"]` (step 5).
7. **vllm-ascend has NO prebuilt wheel for our branch** → build from source, which compiles the
   CANN custom ops. The op compiler (TBE/TVM) imports `decorator` (+ scipy/ml-dtypes/…), so the
   **CANN backfill must be installed BEFORE the build** (now in step 2), not after. Symptom if
   you skip it: `ModuleNotFoundError: No module named 'decorator'` mid op-compile. After fixing,
   `rm -rf csrc/build` before rebuilding (the failed ninja target is cached).
8. **The C++ extension needs `pybind11`** (`python -m pybind11 --cmakedir` runs late in CMake
   configure). It's in step 2's build deps now, so a clean run never hits this.

## What's in the branch (vs raw PR #11153)

7 fixes on top of PR #11153 (commits `a9709b8e`, `3681038b`, `636a07e1`, `401725db`, `1a9e9d2f`):
1. **`hf_confif` → `hf_config`** (×2, `spec_decode/llm_base_proposer.py`) — the typo raised
   `AttributeError` during `hasattr()` arg-eval, crashing the proposer's else-branch for **all**
   requests, not just DSpark.
2. **`_linear_output(...)` (undefined) → `confidence, _ = self.proj(...)`** — vLLM linear layers
   return `(output, bias)`; unpack it. (Confidence head is dormant at inference; now correct.)
3. **`dspark_markov_rank` → `markov_rank`** — read the actual released-checkpoint field.
4. **patch `DFlashQwen3Model.__init__`, not `._init`** — PR #11153 wrapped `._init`, but vLLM
   v0.23.0's `DFlashQwen3Model` uses the standard `__init__` (no `_init` helper); the stale
   reference crashed vllm-ascend at import (`AttributeError: ... has no attribute '_init'`).
5. **`confidence_head.proj` `bias=False` → `bias=True`** — the released checkpoint has a
   `confidence_head.proj.bias` weight; without the bias param, weight loading dies with
   `KeyError: 'confidence_head.proj.bias'`.
6. **guard `_next_token_ids` in `_run_merged_draft`** — in graph mode the cudagraph-capture
   dummy_run runs the Markov loop before any real `prepare()` sets `_next_token_ids` → use it
   when set, else `0` (capture only needs shapes). Eager dodged this (its profiling run takes a
   different branch). Required for `--enforce-eager`-OFF (graph) mode.
7. **Markov logits from `sample_hidden_states` + `num_spec`** (not `last_hidden_states` +
   `num_spec+1`) — the branch reshaped `compute_logits(last_hidden_states)` to `[batch, num_spec+1,
   vocab]`, assuming `num_input_tokens == batch*(num_spec+1)` (true in real decode, so eager
   worked). cudagraph capture uses an arbitrary `num_tokens` → `view(-1)` mis-inferred the vocab
   (`147188` vs `151936`) → `cannot broadcast` on `logits[:, idx] += markov_bias`. Use
   `sample_hidden_states` (the canonical draft positions, also used by the non-DSpark branch),
   reshape to `num_spec`. Required for graph mode. **Verify accept length matches the eager run.**

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `ModuleNotFoundError: numpy` on `import torch_npu` | fresh env — `pip install numpy==1.26.4` |
| `ModuleNotFoundError: setuptools_rust` building vLLM | `pip install setuptools-rust`, retry (step 3) |
| `cargo`/`rustc not found` building vLLM | `conda install -y -c conda-forge rust`, retry |
| `ModuleNotFoundError: decorator` during vllm-ascend op compile | CANN backfill missing before build — install it (step 2 list), then `rm -rf csrc/build` + rebuild |
| `No module named pybind11` / `pybind11 --cmakedir` failed | `pip install pybind11`, re-run build (incremental — won't recompile the ops) |
| `ModuleNotFoundError: numba` at engine-core startup | vllm-ascend EPLB needs it — `pip install numba einops pandas msgpack`, re-pin numpy 1.26.4 |
| `ModuleNotFoundError: torchvision` at startup (via qwen3_5→qwen3_vl) | `pip install --no-deps torchvision==0.25.0 torchaudio==2.10.0` (--no-deps to keep torch 2.10.0) |
| `'vllm' object has no attribute 'qkv_rmsnorm_rope'` (compile pass) | from-scratch build lacks the triton-ascend fusion op — serve with `--enforce-eager` (or install triton-ascend) |
| `AttributeError: ... has no attribute '_next_token_ids'` (dummy_run) | DSpark ran on the wrong proposer — add `"method": "dflash"` to `--speculative-config` |
| `TypeError: 'function' object is not subscriptable` (block_table.compute_slot_mapping, at first request) | triton-ascend missing — `pip install triton-ascend==3.2.1` + `conda install -c conda-forge clang=15` |
| `Model architectures ['Qwen3DSparkModel'] are not supported` | step 5 not done — edit config.json → `["DFlashDraftModel"]` |
| weight-load `missing/unexpected keys` for `markov_*`/`confidence_*` | head weight-name mapping — report; may need a loader tweak |
| accept length ≈ plain DFlash | wrong `NUM_SPEC_TOKENS`, or DSpark path inactive (`markov_head_type` missing) |

## Notes

- **DSpark is unmerged everywhere** (2026-06-30): vLLM `#46995` (GPU) open; vllm-ascend `#11153`
  (Qwen3, this) open; `#11066` (DeepSeek-V4) draft. No vLLM release/main registers `Qwen3DSparkModel`.
- **Confidence head is dormant at inference** in both the NPU PR and the GPU reference — this is
  **Markov-only DSpark v1**. Confidence-scheduled dynamic block length = v2.
- **DeepSeek-V4-Flash DSpark is a separate, much heavier path** (needs SparseMLA / TiLang Ascend
  kernels; vllm-ascend `#11066`). This doc is Qwen3 only.
