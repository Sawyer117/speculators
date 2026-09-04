# Newton-Schulz kernel handoff — the Muon hot spot on Ascend 910B1

**Ask:** replace one batched iteration with a faster kernel. Target is a **~2.4×** cut,
which is **~20% off the whole training step**. Everything below is measured on the box
unless it says "estimate".

- Reference / oracle / benchmark: [`examples/ascend_npu_dflash/muon_ns_kernel_reference.py`](https://github.com/Sawyer117/speculators/blob/feat/dsv4-dspark-block16/examples/ascend_npu_dflash/muon_ns_kernel_reference.py)
- Production call site: [`src/speculators/train/muon_distributed.py`](https://github.com/Sawyer117/speculators/blob/feat/dsv4-dspark-block16/src/speculators/train/muon_distributed.py) → `zeropower_via_newtonschulz5`
- Branch `feat/dsv4-dspark-block16`, commit `38a64ca9`

```bash
git clone https://github.com/Sawyer117/speculators.git && cd speculators
git checkout feat/dsv4-dspark-block16
python examples/ascend_npu_dflash/muon_ns_kernel_reference.py          # small shapes
python examples/ascend_npu_dflash/muon_ns_kernel_reference.py --full   # real shapes
```

The script needs only `torch`. It runs on whatever accelerator `torch.accelerator`
reports, CUDA included — you do not need an Ascend box to develop against it.

## 1. Why this is worth doing

Measured on 8× Ascend 910B1, DSV4-DSpark draft (3 layers × 256 experts, EP8, bf16
experts), run `ckpt_faithful_ep_20260905_022910`:

| | AdamW | Muon |
|---|---|---|
| `opt_ms` | 99 | **992** |
| `step_ms` | 2040 | **2940** |

The optimizer is **a third of every training step**, and essentially all of it is this
one iteration. Muon is not optional here: it carries one momentum buffer where AdamW
carries two moments, and on this expert stack the AdamW footprint does not fit in 64 GB.
So the iteration has to get cheaper rather than go away.

## 2. The math (verbatim from production)

```python
COEFF_PRIMARY   = (3.4445, -4.7750, 2.0315)
COEFF_SECONDARY = (2.0, -1.5, 0.5)      # only when hybrid_ns=True, last 2 steps

x = grad.bfloat16()                      # [E, m, n]
if x.shape[-2] > x.shape[-1]:            # iteration wants WIDE matrices
    x = x.transpose(1, 2)
x = x / (torch.linalg.norm(x, dim=(-2, -1), keepdim=True) + eps)

a, b, c = COEFF_PRIMARY
for i in range(steps):                   # steps = 5
    if hybrid_ns and i >= steps - 2:
        a, b, c = COEFF_SECONDARY
    gram = torch.bmm(x, x.transpose(1, 2))        # [E, m, m]   SYMMETRIC
    poly = b * gram + c * torch.bmm(gram, gram)   # [E, m, m]   SYMMETRIC
    x    = a * x + torch.bmm(poly, x)             # [E, m, n]
```

## 3. Shapes that matter

**Only one shape matters: `[E=32, 2048, 4096]` bf16, 5 iterations, 9 calls per step.**

Those nine are the routed-expert stacks: 3 layers × {w1, w2, w3}. `E=32` is the per-rank
expert count (256 experts / EP8). `w2` is stored `[32, 4096, 2048]` and the tall/wide
transpose brings it to the same shape, so all nine are identical work.

The optimizer also runs 37 "matrix-route" 2D parameters through the same function, but
after the transpose their gram is `min(m, n)` on a side and they are negligible — the
largest, `markov_w1/w2`, are `[129280, 256]`, so their gram is 256×256. **Do not tune for
those.**

## 4. Where the time goes

Per stack, per iteration:

```
bmm(x, x^T)       2*32*2048*2048*4096 = 1.10e12 FLOPs
bmm(gram, gram)   2*32*2048^3         = 0.55e12
bmm(poly, x)      2*32*2048*2048*4096 = 1.10e12
                                        --------
                                        2.75e12
× 5 iterations × 9 stacks             = 1.24e14 FLOPs / step / rank
```

`1.24e14 / 0.992 s` = **125 TFLOPS**, about **33%** of the 910B1's ~376 TFLOPS bf16 peak.
For GEMMs this large (batch 32, M=N=2048, K=4096) a tuned dense bmm normally lands well
above a third of peak, so **there is headroom before any algorithmic change.**

## 5. Lead 1 — the two big GEMMs are symmetric (1.43× FLOPs)

`gram = x @ xᵀ` is symmetric by construction, and the square of a symmetric matrix is
symmetric, so `gram @ gram` is too. A general bmm computes all n² outputs where only the
triangle carries information. **torch cannot express this — there is no batched SYRK.**

```
bmm(x, x^T)      1.10e12 -> 0.55e12    SYRK
bmm(gram, gram)  0.55e12 -> 0.275e12   symmetric squared
bmm(poly, x)     1.10e12 -> 1.10e12    SYMM: same FLOPs, half the reads of poly
                 2.75e12 -> 1.93e12    = 1.43x
```

This premise is **checked, not asserted**. `verify_structure()` in the reference measures
`max|A − Aᵀ| / max|A|` for both and prints it; at the production shape both come back
exactly `0.00e+00`.

## 6. Lead 2 — fuse the two epilogues (~14%)

`poly = b*gram + c*gram2` and `x = a*x + (poly@x)` are two extra full round trips.
Per iteration per stack: `gram` is 268 MB in bf16, `x` is 537 MB, total traffic ≈ 4.8 GB.
Over 5 iterations × 9 stacks that is **~216 GB/step**, ≈ **135 ms** at 1.6 TB/s — about
14% of the 992 ms, nearly all of which folds into the two GEMM epilogues (`axpby` on the
`gram@gram` output, `axpby` on the `poly@x` output).

## 7. Lead 3 — accumulate fp32, store bf16

Input and output are bf16 and must stay so. The accumulator is the kernel's choice, and
fp32 accumulation is what keeps the 5-step schedule where the coefficients expect it.

## 8. Already ruled out — do not spend time here

Reassociating `poly @ x` as `b*(gram@x) + c*(gram@(gram@x))` costs **3.30e12** against the
current **2.75e12**. Because n=2048 < m=4096, forming `gram²` is cheaper than applying
`gram` to `x` twice. **The current form is already FLOP-optimal.**

Lowering `steps` from 5 to 3 is also closed: measured on these exact shapes, the best
coefficient schedule at 3 steps reaches `mean|σ−1|` of only 0.21, i.e. barely
orthogonalized. It is not a knob.

## 9. Correctness bar

**Not elementwise agreement — `mean|σ − 1|` of the result.**

The iteration exists to push the singular values toward 1, and the training consumes that
spectrum. A kernel that differs from the reference in the last bf16 bits is fine; one that
matches elementwise while moving `mean|σ − 1|` has changed the training run. `check()`
prints both numbers.

Reference values, 5 steps, `COEFF_PRIMARY`, at `[32, 2048, 4096]`: **`mean|σ−1| ≈ 0.165`**.
That reproduces a figure this repo measured independently for the same schedule, so the
reference really is the math the trainer runs.

Two more invariants:

- `COEFF_PRIMARY` deliberately does **not** converge to 1: as a map on singular values,
  `p(x) = a x + b x³ + c x⁵` has `p(1) = 0.7010`, so 1 is not a fixed point. **By design.**
  Do not "fix" it and do not reorder the polynomial.
- Output must be the orthogonalized tensor in the **input dtype and layout** (the
  tall/wide transpose is undone inside the function).

## 10. Target backend — needs confirming before you start

The training runs on **Ascend 910B1 / CANN / `torch_npu`**, not CUDA. Before committing to
Triton, confirm which of these the toolchain supports on this box:

- `triton-ascend` (Huawei's Triton port) — maturity here is **unverified**; check it first.
- A CANN / AscendC custom operator — heavier, but the supported path.
- `torch.compile` over the loop: this repo has banked an unrelated ~1.74× from compiling
  the expert GEMMs, pending a torch-2.12 stack. Whether it helps *this* loop is untried.

Develop against the reference on CUDA if that is easier — the math and shapes are
identical, and only the final numbers have to come from the Ascend box.

## 11. Definition of done

1. `check()` passes at `[32, 2048, 4096]`: `mean|σ−1|` no worse than 0.165.
2. `bench()` shows a real speedup at that shape.
3. Handles `steps=5` and the `hybrid_ns=True` variant (last 2 iterations swap
   coefficients).
4. Correct for a batch of any `E` and both orientations (the caller transposes tall
   inputs, but do not assume `m < n` if you touch the wrapper).
5. `opt_ms` on the box drops from 992 — that is the number that decides it.
