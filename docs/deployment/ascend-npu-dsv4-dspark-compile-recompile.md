# DSV4-DSpark: killing the grouped-GEMM recompile (torch.compile on NPU) — **WIP**

> **Status: WIP / investigating.** The training run is healthy and producing on `torch 2.10`
> (`dspark-dsv4-austin`); this doc tracks the *optional* compile optimization that would recover
> ~42% of wall-clock. It is being validated in an **isolated cloned env** (`dspark-dsv4-compile`)
> and does NOT block the main run.

## 1. The problem — RECOMPILE-bound after HS was solved

Bottleneck history (measured, `analyze_train_run.py --skip 30`):

| stage | HS fetch | grouped-GEMM recompile | note |
|---|---|---|---|
| MoE eager loop | — | — | opt_ms ~28s (per-expert FSDP) |
| EP + fused MoE (Graft A) | **57%** | 34% | MoE fast → training *outran the serve* → HS-bound |
| + `RECOMPUTE=1` + `MAX_ANCHORS=384` | **0.1%** ✅ | **42.5%** ⬅ | anchor slowed the step to the serve's rate → HS solved; **recompile now #1** |

`npu_grouped_matmul` recompiles per unique routed-token count; at `anchors=384` the counts are big
and varied, so `DSPARK_MOE_BUCKET=512` leaves many distinct shapes → 53 spikes of **21–60s each**,
every ~12 steps = **42% of wall-clock** (1.7× slowdown). Verdict from the analyzer: **RECOMPILE-bound
(fwd 42% vs HS 0%) → compile + `maybe_mark_dynamic` is the permanent fix, ~42% recoverable.**

## 2. Fix candidates

| option | mechanism | pro | con |
|---|---|---|---|
| **bucketing** (`DSPARK_MOE_BUCKET`) | round token count to a bucket → fewer shapes | cheap, no dependency | only *reduces* recompiles; **adds padding memory** (we're at 92% HBM); need to drop anchor to fit |
| **compile** (torchtitan-npu) | `torch.compile` + `maybe_mark_dynamic(x,0)` → one shape-generic kernel | **zero recompile + zero padding** (memory-friendly) | needs `inductor_npu_ext` + a **matched torch/CANN** stack |

We're memory-tight + recompile-bound → **compile is the right fix** (bucketing's padding hurts at 92%).

## 3. The compile approach (provenance)

- torchtitan-npu compiles the grouped-swiglu experts fwd: `torch._grouped_mm` + `npu_swiglu`
  (`torchtitan_npu/converters/kernels/gmm.py:_run_experts_grouped_mm`), wrapped with
  `torch.compile(backend="inductor")` + `torch._dynamo.maybe_mark_dynamic(x, 0)`
  (`models/deepseek_v4/parallelize.py:_patch_grouped_mm_compile`, ~L985-1018).
- `aten::_grouped_mm` → `npu_grouped_matmul` via a `PrivateUse1` impl (`torchtitan_npu/ops/_grouped_mm.py`).
- Also sets `torch._dynamo.config.capture_scalar_outputs = True` (parallelize.py:1027).
- NPU codegen backend = **`inductor_npu_ext`** (Ascend torchair experimental), which hooks Inductor's
  Codegen → AutoFuse → AscendC kernels (`docs/torch_compile.md`). Not built into torch_npu.

## 4. Version requirement (出处) — **this is the crux**

`torchtitan-npu/requirements.txt`:
```
torch==2.12.0+cpu
torch_npu==2.12.0rc1
```
`inductor_npu_ext`: `git clone https://gitcode.com/Ascend/torchair.git && git checkout 3c9418c2`
then `pip install -e experimental/_inductor_npu_ext/python/`.

**Our box training env is `torch 2.10`.** `inductor_npu_ext@3c9418c2` patches torch **2.12**'s Inductor;
on 2.10 the Inductor internals differ → CUDA-centric passes run unpatched → crash (see §6).

## 5. Install (in the CLONED env only)

```bash
conda create -n dspark-dsv4-compile --clone dspark-dsv4-austin
conda activate dspark-dsv4-compile
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu   # ★ MUST be +cpu — torch_npu
                                              # 2.12.0rc1 is built for +cpu; the default aarch64 wheel is
                                              # +cu130 and breaks transfer_to_npu (_apply_patches ABI mismatch)
pip install torch_npu==2.12.0rc1              # from the Ascend pip source
# do NOT use transfer_to_npu (torchtitan-npu doesn't; it crashes on this stack). Plain `import torch_npu`
# registers the "npu" device + torch.npu.* — that's all we need.
cd ~/torchair && git checkout 3c9418c2
pip install -e experimental/_inductor_npu_ext/python/ --no-deps   # ★ --no-deps: else it drags torch 2.13 + cuda-toolkit + nvidia-*
cd -
# ★ torch_npu's _inductor is Triton-based -> needs the ASCEND Triton, not the CUDA triton that
# --force-reinstall pulled ("0 active drivers"). CANN 9.0.0 -> triton-ascend 3.2.x. Match your base env.
pip uninstall triton -y && pip install triton-ascend
import inductor_npu_ext  # (in code) engages AscendC codegen — see the test; torchtitan-npu entry.py:92
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.is_available())"
```
**★ `--no-deps` is mandatory** — inductor_npu_ext declares `torch>=2.8.0`, so plain install pulls
`torch-2.13.0 + cuda-toolkit==13.0.3 + nvidia-* + triton` (a CUDA torch) and clobbers torch_npu.

Test: `python examples/ascend_npu_dflash/test_compile_grouped_mm.py`
(GO/NO-GO: parity vs eager + do varying token counts avoid recompile?).

## 6. Attempt log (chronological)

1. **torch 2.10, run test** → crash in Dynamo: `_device_supports_tma()` →
   `torch.cuda.get_device_capability() >= (9,0)` where get_device_capability() is `None` on NPU →
   `'>=' NoneType vs tuple`. (torch's Triton/TMA probe assumes CUDA.)
2. **Added shim** (`test_compile_grouped_mm.py`: force get_device_capability→(0,0), set
   `capture_scalar_outputs=True`) → got PAST Dynamo into **Inductor codegen**, then crash:
   `_inductor/fx_passes/post_grad.py:move_constructors_to_cuda` → `get_gpu_type()` →
   `assert len(avail_gpus) <= 1`. (A CUDA-centric Inductor post-grad pass, not patched by
   inductor_npu_ext on torch 2.10.)
3. **Root cause = version mismatch** (2.10 vs required 2.12). `inductor_npu_ext --force-reinstall`
   tried to pull **torch 2.13.0 + CUDA** (its `torch>=2.8.0` dep) → must use `--no-deps`.
4. **torch 2.12 + torch_npu 2.12.0rc1 installed** (`inductor_npu_ext --no-deps`). `import torch_npu;
   torch.npu.is_available()` → **True** ⇒ **CANN gate PASSED** (2.12.0rc1 runs on the box CANN). BUT
   `pip install torch==2.12.0+cpu` resolved to **`2.12.0+cu130`** (no aarch64 `+cpu` wheel on the default
   index) → `from torch_npu.contrib import transfer_to_npu` crashes at import:
   `_apply_patches() takes 0 positional arguments but 1 was given` (torch_npu 2.12.0rc1 is built for the
   **+cpu** torch, not +cu130; [web](https://github.com/BrightXiaoHan/pytorch-npu/)).
5. **Fixes**: (a) **torchtitan-npu does NOT use `transfer_to_npu`** (grep: 0 hits) — removed it from the
   test; plain `import torch_npu` registers the `npu` device + `torch.npu.*`. (b) install torch
   **2.12.0+cpu** (the build torch_npu 2.12.0rc1 expects), not `+cu130`, via the CPU index.
6. **torch 2.12.0+cpu in** → eager baseline runs, transfer_to_npu gone. But `torch.compile` crashed:
   `torch_npu.utils._dynamo.register_inductor_npu` → torch_npu's **built-in `_inductor` (Triton backend)**
   → `RuntimeError: 0 active drivers` (only CUDA-triton 3.7.1 is installed; no Ascend-Triton driver).
   **Fix**: `import inductor_npu_ext` (torchtitan-npu `entry.py:92` does exactly this) engages the
   AutoFuse/AscendC codegen and bypasses the Triton path — we'd installed inductor_npu_ext but never
   *imported* it. Also needed **`triton-ascend`** (CANN 9.0.0 → 3.2.x) — the CUDA-triton pulled earlier
   gave "0 active drivers".
7. **triton-ascend + `import inductor_npu_ext` in → compile WORKS shape-generically** ✅:
   `dynamo stats {unique_graphs: 1}`, first token count compiles (~1.6s), **all other counts ~3.8ms
   (no per-shape recompile)** — the 42% recompile IS killable by compile, `maybe_mark_dynamic` works on
   NPU. Initial parity looked wrong (rel~1.3) but that was a **test bug** (inputs regenerated per loop →
   eager vs compiled ran on *different* random data); fixed to reuse the same inputs. Re-verifying
   parity. ⟵ *we are here.*

## 7. Risks / open questions

- **CANN compat**: `torch_npu==2.12.0rc1` may require a newer CANN than the box has → import fails.
  Hard system-level blocker if so.
- **Training migration**: even if compile is GREEN on torch 2.12, the *training* must also run on
  torch 2.12 + torch_npu 2.12rc1 to benefit (the whole EP/fused/recompute stack was validated on 2.10).
  Separate migration + re-validation effort. Prove compile viability FIRST (cheap, in the clone).

## 8. Fallback (if compile proves non-viable on this box)

Bucketing at a lower anchor to fit memory:
```bash
RECOMPUTE=1 MAX_ANCHORS=320 SEQLEN=3072 DSPARK_MOE_BUCKET=2048 DSPARK_EP=1 bash examples/ascend_npu_dflash/train_dsv4_dspark.sh faithful
```
Bigger bucket → fewer shapes → recompiles amortize; drop anchor to 320 so the extra padding fits.
Partial (est. 42% → ~15%), no dependency. Or accept the 42% (run is producing).

## Related
- Test: `examples/ascend_npu_dflash/test_compile_grouped_mm.py`
- Bucketing / empty-rank / Graft A: `src/speculators/models/dsv4_dspark/backbone/moe_grouped_gemm.py`
- Recompute: `src/speculators/models/dsv4_dspark/core.py` (`DSPARK_RECOMPUTE`) + `train_dsv4_dspark.sh` (`RECOMPUTE`)
- Full pipeline: [ascend-npu-dsv4-dspark-pipeline.md](./ascend-npu-dsv4-dspark-pipeline.md)
