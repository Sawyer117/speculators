# Ascend NPU — DeepSeek-V4-Flash DSpark Inference (Setup & Status)

> **Fork-only, team-internal — WIP.** Reproduce DeepSeek **DSpark** speculative-decoding
> **inference** for **DeepSeek-V4-Flash** on Ascend NPU. Sibling of the Qwen3 DSpark docs
> (`ascend-npu-dspark-install.md` / `ascend-npu-dspark-report.md`); this file is the **V4** line,
> kept on its own branch so the two don't tangle.
>
> **Status (2026-07-03):** environment building on `dspark-dsv4-base`; drafter extracted (13 GB);
> serve + eval pending. Update the checkboxes in §8 as it progresses.

---

## 0. Goal & the three moving parts

Three pieces, one environment:

- **Target** = `DeepSeek-V4-Flash-w8a8-mtp` (INT8, ~300 GB) — the big MoE verifier. The proven NPU
  path (Eco-Tech / vllm-ascend official image).
- **Draft** = DeepSeek's released **DSpark** module (3 MTP blocks + Markov head), **extracted** from
  the 166 GB `deepseek-ai/DeepSeek-V4-Flash-DSpark` checkpoint (see §2).
- **Runtime** = vllm-ascend + **PR #11196** (QwertyJack) — native V4 DSpark support (`method: mtp`).

One conda env, `dspark-dsv4-base`, runs BOTH the AR/MTP baseline AND DSpark, so the numbers are
same-box comparable (the discipline from the Qwen3 report).

---

## 1. Precision — get this straight (the FP4 "blocker" is a non-issue)

**Who is whose parent.** DeepSeek trains/releases **natively in FP8** (V3 card: *"Since FP8 training
is natively adopted … we only provide FP8 weights"* + `fp8_cast_bf16.py`). So for DeepSeek, **FP8 is
the parent; a bf16 copy is the child** (an *upcast* — same information, 2× the bytes). This is the
reverse of Llama/Qwen (which release bf16 as the parent). A "full bf16" DeepSeek checkpoint is bf16
*storage* holding *fp8-precision content* — running it gains compatibility, not accuracy.

**What the DSpark checkpoint actually stores** (measured via safetensors headers — trust the
tensors, not `config.json`'s `expert_dtype: fp4` label):

| component | dtype | note |
|---|---|---|
| MoE experts `w1/w2/w3` | **INT8 (`I8`)** + `F8_E8M0` block scale | MX-style block-scaled int8 |
| MLA attention `wq/wkv/wo` | **FP8 `e4m3`** + `F8_E8M0` scale | MXFP8 |
| `markov_w1/w2`, `confidence`, `gate`, all norms | **BF16** | |
| `embed`, `head`, final `norm`, `hc_head_*` | **BF16** / FP32 (hc) | |

**There is NO FP4 anywhere** (trunk layer 18 and draft mtp blocks both = INT8 experts + FP8 attn).
So the "my card only does FP8/INT8, FP4 is a blocker" fear **does not apply**. And you do **not**
need to hand-dequant to bf16 (see §3 — #11196 loads the quantized weights directly).

**INT8 (w8a8) accuracy cost** — Eco-Tech's w8a8 vs the official FP8/FP4 numbers:

| bench | mode | official | w8a8 | Δ |
|---|---|---|---|---|
| GPQA-Diamond | Non-Think | 71.2 | 71.21 | +0.0 |
| MMLU-Pro | Non-Think | 83.0 | 82.85 | −0.15 |
| MMLU-Pro | Max | 86.2 | 85.86 | −0.34 |

≈ **0.2 pt, within noise** (their PTQ = QuaRot + FlexSmoothQuant + per-channel/per-token int8). And
for spec decode the target quant is **self-consistent**: accept length is measured against the same
int8 target, so it does not affect the speedup math — only the target's absolute quality (which the
table shows is barely touched). Only benchmarked on GPQA/MMLU-Pro; hard long-CoT tasks
(LiveCodeBench/HMMT) were not re-tested at w8a8 and are typically more quant-sensitive — verify on
your own eval if it matters.

---

## 2. Drafter extraction — 13 GB, not 166 GB

The 166 GB `DeepSeek-V4-Flash-DSpark` checkpoint = the full V4-Flash target **plus** a small DSpark
tail. The tail is perfectly file-separable (mapped via `model.safetensors.index.json`):

| drafter piece | shard | size |
|---|---|---|
| `mtp.0.*` (block 1, incl. `main_proj`) | `model-00046-of-00048` | 3.61 GB |
| `mtp.1.*` (block 2) | `model-00047-of-00048` | 3.56 GB |
| `mtp.2.*` (block 3 + `markov_head` + `confidence_head`) | `model-00048-of-00048` | 3.69 GB |
| shared `embed.weight` | `model-00001-of-00048` | 1.06 GB |
| shared `head` + `norm` + `hc_head_*` | `model-00045-of-00048` | 1.06 GB |
| **total** | **5 shards** | **12.98 GB** (vs 166.9 GB — 13×) |

Shards 46/47/48 are **100 % pure `mtp`** (zero waste); shard 1 = only `embed`; shard 45 = only
`head/norm`. Each `mtp` block is a full V4 MoE block (MLA + 256 experts + shared + hc). DSpark config:
`dspark_block_size=5`, `dspark_target_layer_ids=[40,41,42]`, `dspark_markov_rank=256`,
`dspark_noise_token_id=128799`.

**Download only the 5 shards** — either `hf download <5 shard filenames> + json + inference/` (use a
mirror/token to dodge anonymous throttling; the small files come fast, big shards stall unauth), or
selective git-lfs:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark   # or hf-mirror.com
cd DeepSeek-V4-Flash-DSpark
git lfs pull --include="model-00001-of-00048.safetensors,model-00045-of-00048.safetensors,model-00046-of-00048.safetensors,model-00047-of-00048.safetensors,model-00048-of-00048.safetensors"
find . -name "model-*-of-00048.safetensors" -size -1k -delete   # drop the 43 un-pulled pointer stubs
```

Then **trim `model.safetensors.index.json` to the present shards** (else the loader looks for the 43
missing files):

```bash
python3 - <DRAFT_DIR> <<'PY'
import json, glob, os, sys
d = sys.argv[1]
idx = json.load(open(f"{d}/model.safetensors.index.json"))
present = {os.path.basename(p) for p in glob.glob(f"{d}/*.safetensors")}
wm = {k:v for k,v in idx["weight_map"].items() if v in present}
json.dump({"metadata":{"total_size":sum(os.path.getsize(f'{d}/{s}') for s in present)},"weight_map":wm},
          open(f"{d}/model.safetensors.index.json","w"))
print("trimmed ->", len(wm), "tensors,", sorted(present))
PY
```

**No dequant, no architecture edit.** #11196's `load_weights` consumes the INT8/FP8 + `float8_e8m0fnu`
scales directly (`_draft_quant_config` returns the quant config by default; `.view(torch.uint8)` on
e8m0 scales; expert-aware loaders). The optional `dspark_mtp_dequantized_to_bf16: true` flag is an
escape hatch for a pre-bf16 checkpoint — **don't set it**. V4 DSpark is auto-detected by
`model_type == "deepseek_v4" && dspark_block_size` → routed to `method="mtp"`, so the Qwen3 trick of
editing `architectures → DFlashDraftModel` is **not** needed here.

---

## 3. Environment `dspark-dsv4-base`

Same recipe as the Qwen3 DSpark env, **one change**: vllm-ascend → the V4 DSpark branch. #11196 is
against `main` and guards on **vLLM 0.23.0** (`vllm_version_is("0.23.0")`), i.e. the *same* vLLM as
the Qwen3 env — the discrepancy is small.

| component | version | note |
|---|---|---|
| Python | 3.11 | |
| CANN | new (≥ 9.0.0; main's Dockerfile base = `cann:9.0.0-910b`) | V4 kernels compiled during the vllm-ascend build |
| torch / torch-npu | 2.10.0 | main `requirements.txt` pin |
| numpy | **2.3.5** (force after triton-ascend) | triton-ascend 3.2.1 pulls it down to 1.26.4; re-force `pip install --no-deps numpy==2.3.5` — verified fine at 2.3.5 |
| vLLM | v0.23.0 `VLLM_TARGET_DEVICE=empty` | "0.23.0+empty" |
| vllm-ascend | `Sawyer117/vllm-ascend @ dspark-dsv4` (= QwertyJack `qwertyjack/deepseek-v4-dspark-main`, `6cdb99e`) | PR #11196 |
| triton-ascend | 3.2.1 + clang-15 + gxx_linux-aarch64 | runtime slot-mapping kernel + JIT headers |

**Env-as-template / read-only-server workflow.** `dspark-dsv4-base` is the clean base; clone per
member (`conda create --clone dspark-dsv4-base -n dspark-dsv4-<name>`) to avoid pollution — but note
editable installs share the source tree, so a member who *edits* vllm-ascend also needs their own
source checkout + re-point `pip install -e`. **The serve box is pull-only**: it clones
`Sawyer117:dspark-dsv4` read-only; fixes are pushed from a networked box (patches → push to the
fork → server `git pull`). Never push without the owner's OK.

---

## 4. Build commands

```bash
export ROOT=/abs/path/dspark-dsv4 && mkdir -p "$ROOT/installation"
# CANN + nnal already sourced; npu-smi shows 8 cards.

conda create -n dspark-dsv4-base python=3.11 -y && conda activate dspark-dsv4-base

# deps + torch/torch_npu 2.10.0 + CANN backfill (numpy floats to 2.3.5 via torch)
python -m pip install -U pip setuptools "setuptools-scm>=8" wheel packaging "cmake>=3.26" ninja jinja2 setuptools-rust pybind11
python -m pip install --extra-index-url https://mirrors.huaweicloud.com/repository/pypi/simple \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi torch==2.10.0 torch-npu==2.10.0 pyyaml
python -m pip install decorator "scipy>=1.7.3" ml-dtypes attrs psutil pyyaml matplotlib openpyxl tornado
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch_npu.npu.is_available(), torch_npu.npu.device_count())"  # expect ... True 8

# vLLM 0.23.0 (+empty)
cd "$ROOT/installation"
git clone --depth 1 --branch v0.23.0 https://github.com/vllm-project/vllm.git vllm-v0.23.0 && cd vllm-v0.23.0
TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_TARGET_DEVICE=empty python -m pip install -e . --no-build-isolation -v
python -c "import vllm; print(vllm.__version__)"   # 0.23.0+empty

# vllm-ascend V4 DSpark (read-only clone of the fork branch)
cd "$ROOT/installation"
git clone --branch dspark-dsv4 --single-branch https://github.com/Sawyer117/vllm-ascend.git vllm-ascend-v4 && cd vllm-ascend-v4

# runtime extras FIRST (the build doesn't import them; clang/gxx are runtime-only → conda goes LAST)
python -m pip install numba einops pandas msgpack
python -m pip install --no-deps torchvision==0.25.0 torchaudio==2.10.0 --extra-index-url https://mirrors.huaweicloud.com/repository/pypi/simple
python -m pip install triton-ascend==3.2.1 --extra-index-url https://mirrors.huaweicloud.com/repository/pypi/simple --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
# ⚠️ triton-ascend 3.2.1 metadata REQUIRES numpy<2 → it silently DOWNGRADES numpy 2.3.5 → 1.26.4.
#    Force it back (verified to run fine at 2.3.5; the "requires numpy<2" warning is expected/ignored):
pip install --no-deps "numpy==2.3.5"

# compile vllm-ascend WITH numpy 2.3.5 (compiles the V4 CANN ops — the moment of truth)
pip install -e . --no-deps --no-build-isolation -v

# conda clang/gxx LAST — runtime triton JIT only; doing it BEFORE the build can perturb the env / error
conda install -y -c conda-forge clang=15 clangxx=15 lld=15 gxx_linux-aarch64

python -c "import numpy; print(numpy.__version__)"          # confirm still 2.3.5 (re-force if conda moved it)
python -c "import vllm_ascend; print('vllm_ascend import OK')"
```

**Ordering rule:** runtime extras (numba/einops/pandas/msgpack/torchvision/triton-ascend) + the
numpy re-force go BEFORE `pip install -e .`; **conda clang/gxx goes LAST** (after the compile) — doing
the conda step earlier can perturb the build env and error.

Harmless warnings: `ms-service-profiler` / `schedule-search` (CANN profiling tools, unused at
inference); `opencv-python-headless requires numpy>=2` (opencv is vLLM's multimodal dep — never
imported for V4 text, so 1.26.4/2.3.5 either way is fine); `triton-ascend requires numpy<2` (we
override to 2.3.5 on purpose). If the op compile dies on a missing python module → install it, then
`rm -rf csrc/build` and rebuild. If it dies on a missing CANN symbol → the new CANN may lack a V4 op
(the real risk; capture the error).

---

## 5. Serve — AR / MTP baseline (official vllm-ascend V4-Flash recipe, **A2 64G×8**)

Needs the **~300 GB `DeepSeek-V4-Flash-w8a8-mtp` target** (not the 13 GB draft). Env (Atlas 800 A2):

```bash
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export OMP_PROC_BIND=false OMP_NUM_THREADS=8 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACL_OP_INIT_MODE=1 VLLM_ASCEND_ENABLE_FLASHCOMM1=1 USE_MULTI_GROUPS_KV_CACHE=1
export TASK_QUEUE_ENABLE=1 HCCL_OP_EXPANSION_MODE="AIV" HCCL_BUFFSIZE=512 USE_MULTI_BLOCK_POOL=1
```

A2 = single node, **TP8 / DP1** (A3 128G×8 differs: DP4/TP4, method name `deepseek_mtp`, other env):

```bash
MODEL=/path/DeepSeek-V4-Flash-w8a8-mtp
vllm serve "$MODEL" --served-model-name dsv4 \
  --data-parallel-size 1 --tensor-parallel-size 8 --enable-expert-parallel \
  --quantization ascend \
  --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --enable-auto-tool-choice \
  --max-model-len 135168 --max-num-seqs 16 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.92 --block-size 128 \
  --safetensors-load-strategy prefetch --enable-chunked-prefill --enable-prefix-caching --async-scheduling \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":16}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[2,4,6,8,10,12,14,16,18,20,22,24,32,36,40]}' \
  --additional-config '{"enable_cpu_binding":true,"multistream_overlap_shared_expert":false}' \
  --port 7000
# MTP baseline: add   --speculative-config '{"num_speculative_tokens":1,"method":"mtp"}'   (A2 = "mtp"; A3 = "deepseek_mtp")
# AR baseline:  omit --speculative-config entirely.
# Reaching "Application startup complete" = V4 kernels run on the new CANN → env moment-of-truth passed.
```

Smoke test (bypass the netentsec proxy): `curl --noproxy '*' http://localhost:7000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"dsv4","messages":[{"role":"user","content":"Who are you?"}],"max_tokens":128,"temperature":0}'`

---

## 6. Serve — DSpark (the point)

Same target serve, but point the speculative draft at the **extracted 13 GB draft** with
`method: mtp`, `num_speculative_tokens: 5` (= `dspark_block_size`). Keep the draft `config.json`
as-is (no architecture edit, no dequant flag). Exact flag wiring to be confirmed on first run:

```bash
  --speculative-config '{"model":"/path/dsv4-dspark-draft","num_speculative_tokens":5,"method":"mtp"}'
```

---

## 7. Eval (reuse the DFlash harness — it's model-agnostic)

```bash
cd examples/ascend_npu_dflash
TARGET=dsv4 EVAL_PORT=7000 DATASET=gsm8k bash run_eval.sh      # smoke
TARGET=dsv4 EVAL_PORT=7000 DATASET=all   bash run_eval.sh      # full
```
Record, same box / same graph config: **AR → tok/s** (accept nan, the denominator); **MTP → tok/s +
accept** (num_spec 1 → accept ≤ 2); **DSpark → tok/s + accept** (num_spec 5). The three form the
AR / MTP / DSpark table that answers "how much does the Markov head beat plain MTP" — same shape as
the Qwen3 DFlash-vs-DSpark table.

---

## 8. Status (2026-07-03) — BLOCKED on a vLLM-0.23/main w8a8 regression

Env built fine (V4 CANN ops compiled on CANN **9.0.0.0512**). Two bugs on serve (A2 TP8/DP1):

1. **[FIXED] ACL-graph capture crash** — DSA `_forward_o_proj` fed a **2D `wo_a.weight`** (w8a8 loads
   some o_proj weights 2D) to `npu_transpose_batchmatmul`, which needs 3D →
   *"Dimension out of range … got 2"*. Fix = plain `torch.matmul` o_proj for `n_local_groups==1`
   (TP8). The ONLY code change; on `Sawyer117/vllm-ascend@dspark-dsv4` commit `6036507`.

2. **[BLOCKED] Garbage output** — coherent English words stuck repeating ("X as X as…"), eager AND
   graph. **Ruled out:** o_proj (bulletproof matmul still garbles), sliding window (traced identical),
   act_fn (#11184 present), the DSpark HS-capture in `_forward` (guarded, no-op for base), prefix
   caching (`--no-enable-prefix-caching` didn't fix), config (matched official verbatim), and the
   checkpoint (works on the official image).

**Root cause (proven, not guessed): NOT CANN — a vLLM-0.23/main w8a8 regression.** The official
`quay.io/ascend/vllm-ascend:v0.22.1rc1` image runs this exact w8a8-mtp model **coherently** on
**CANN 9.0.0** (same as us) + **vLLM v0.22.1** + vllm-ascend **v0.22.1rc1**. We differ ONLY in
**vLLM 0.23.0** and **vllm-ascend #11196/main** (~150 commits past rc1, incl. `1c57c6a0a` fused-moe
refactor, `55b8cd0d8` SFA o_proj TP weights). Same CANN → **don't reinstall CANN.** QwertyJack
validated #11196 in **bf16** → dtype-independent code is fine → the bug is the **w8a8 path on
vLLM-0.23/main**. Exact regression not findable by static read (~150-commit bisect).

**Next:** (A) get a working w8a8 baseline on the **v0.22.1 stack** (the `:v0.22.1rc1` image, or build
vllm-ascend @ `v0.22.1rc1` + vLLM 0.22.1 on the same CANN 9.0.0); (B) **ask QwertyJack** whether
#11196 supports w8a8 or only bf16 — the authoritative answer. (bf16 needs ~16 cards; we have 8 → OOM.)
Old `:deepseekv4` image (5/6) = CANN **8.5.1** + vLLM **0.18** — a different OLD stack, not comparable.

## 9. References

- Draft ckpt: `deepseek-ai/DeepSeek-V4-Flash-DSpark` (166 GB; extract 5 shards → 13 GB, §2).
- Target ckpt: `DeepSeek-V4-Flash-w8a8-mtp` (Eco-Tech / vllm-ascend on ModelScope), INT8 w8a8.
- Runtime PR: **vllm-project/vllm-ascend #11196** (QwertyJack) — mirrored to
  `Sawyer117/vllm-ascend @ dspark-dsv4` (`6cdb99e`).
- Official serve: vllm-ascend DeepSeek-V4-Flash tutorial (v0.18.0); images
  `quay.io/ascend/vllm-ascend:deepseekv4` (A2) / `:deepseekv4-a3` (A3).
- Sibling Qwen3 line: `ascend-npu-dspark-install.md`, `ascend-npu-dspark-report.md`.
