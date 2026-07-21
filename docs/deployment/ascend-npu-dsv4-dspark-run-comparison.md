# DSV4-DSpark training runs — essential differences (17W-5EP vs A2 vs A3)

_Last updated: 2026-07-21._

Reference table for the three DSV4-DSpark draft-training runs we compare. Companion to the
eval ledger (`ascend-npu-dsv4-dspark-eval-results.md`) — that one holds serve-side accept-length;
this one holds the **training-config** differences so cross-run curves are read correctly.

| Axis | **17W-5EP** | **A2** | **A3** |
|---|---|---|---|
| **Data** | 17W | 77W | 77W |
| **Window (SWA)** | 2048 | 2048 | 128 |
| **Scheduler / warmup** | linear | linear | cosine / 0.04 |
| **tv weight** | 0.9 | 0.9 | 1.8 |
| **LR** | 2e-4 | 2e-4 | 3e-4 |
| **Expert master (AMP)** | — | bf16 | fp32 |
| **Mesh (FSDP=EP=ws)** | 8-card | FSDP8 + EP8 | FSDP16 + EP16 |
| **Global batch** | 8·b | 8·b | 16·b |
| **steps/epoch** | ~4,900 | ~21,500 | ~10,700 |
| **Stack / compile** | 2.10, no compile | 2.12, COMPILE=1 | 2.12, COMPILE=1 |

_Rows 1–5 drive quality; rows 6–10 affect speed/memory only._

> **Expert master (AMP)** is the fp32-master / optimizer-state precision only — the expert
> **forward is bf16 in every run AND at serve** (FSDP2 casts to bf16 for compute; the fp32
> master is used solely in the optimizer step). So option-A (A3) keeps train/serve forward
> consistency (bf16 = bf16); fp32 buys precise updates, not a different forward. Both A2 and A3
> use MAX_ANCHORS=512, so the modest per-step fwd gap is 16-way vs 8-way collective overhead.

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
