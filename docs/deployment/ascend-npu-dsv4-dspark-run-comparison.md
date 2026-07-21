# DSV4-DSpark training runs — essential differences (17W-5EP vs A2 vs A3)

_Last updated: 2026-07-21._

Reference table for the three DSV4-DSpark draft-training runs we compare. Companion to the
eval ledger (`ascend-npu-dsv4-dspark-eval-results.md`) — that one holds serve-side accept-length;
this one holds the **training-config** differences so cross-run curves are read correctly.

| Axis | **17W-5EP (old stack)** | **A2 (today, control)** | **A3 (today, full-align)** |
|---|---|---|---|
| **Data** | 17W (177k rows, wrong-default small set) | **77W** | **77W** |
| **Window (SWA)** | 2048 | 2048 (old) | **128** (aligned to released) |
| **Scheduler / warmup** | old (unaligned) | old | **cosine / 0.04** |
| **tv (distill) weight** | 0.9 | 0.9 | **1.8** (= DeepSpec L1 balance) |
| **LR** | 2e-4 | 2e-4 | **3e-4** (raised from 2e-4, √2 batch-scaling) |
| _— below: affects speed/memory, not quality —_ | | | |
| **Experts precision** | old stack | **bf16** (option-B) | **fp32** (option-A) |
| **WORLD_SIZE / DP** | 8 | **8** (measured `[rank7]`) | **16** (measured `WORLD_SIZE=16` / `[rank15]`) |
| **Mesh** | — | FSDP8 + EP8 | **FSDP16 + EP16** (1D mesh, FSDP = EP = world_size) |
| **Global batch** | 8·b | 8·b | **16·b = 2×** |
| **steps/epoch** | **~4,900** (measured: 24,512 / 5) | **~21,500** (computed) | **~10,700** (computed) |
| **Epochs** | 5 | 5 | **10** (compensates the half steps/epoch → same total updates as A2) |
| **Total updates** | ~24,500 | ~107k | ~107k (= A2) |
| **Stack / compile** | torch 2.10, no compile | 2.12, COMPILE=1 | 2.12, COMPILE=1 |

## How to read cross-run curves

1. **Only the first four rows drive "accept_len at the same step"**: data, window, scheduler, tv.
   The rest (precision / EP / DP / compile) only move speed and memory.
2. **Do NOT compare at the same step.** A3 is DP16 (batch 2×), so at any given step it has already
   consumed 2× the samples of A2/17W. Compare by **samples seen** (A3 step × 2 ↔ A2 step) or by
   **final value**, never raw step-for-step.
3. **Training accept_len is not comparable across these axes, and does not predict eval.**
   `epoch4-17w` trained with a healthy-looking tail yet collapsed at eval (gsm8k pos3/4 = 19/5).
   The verdict is always **convert + eval → serve-side pos3/4**.

## steps/epoch derivation

Only 17W-5EP crossed full epochs, so it is the measured anchor (24,512 steps / 5 epochs ≈ 4,900).
The 77W runs are extrapolated:

```
steps/epoch(run) = steps/epoch(17W) × [rows(run) / rows(17W)] × [world_size(17W) / world_size(run)]
```

- rows: 17W ≈ 177,000, 77W ≈ 775,965 (ratio ≈ 4.38) — the per-rank batch cancels out of the ratio.
- A2-77W (ws 8): 4,900 × 4.38 × (8/8) ≈ **21,500**
- A3-77W (ws 16): 4,900 × 4.38 × (8/16) ≈ **10,700**

Consistency check: both 77W runs were still in epoch 0 when sampled (A2 @ 10,809 < 21,500;
A3 @ 2,208 < 10,700), which matches the absence of any epoch boundary in their logs.
(If the 77W Arrow row count differs from ~776k after prep, scale both 77W numbers proportionally.)
